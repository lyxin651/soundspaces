"""Fail-closed entry validation for O1 result analysis.

This module binds a real O1 analysis to one audited scientific manifest and
one independently validated run root.  It deliberately validates metadata and
cache integrity only; it never reads ASR hypotheses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from active_audition.a4.cache import CacheStore
from active_audition.a4.cache_resume import (
    CacheExpectedManifest,
    COMPLETION_MARKER_VALID,
    reconcile_cache_manifest,
    read_completion_marker,
)
from active_audition.a4.smoke_manifest import (
    ENGINEERING_SMOKE_MANIFEST_SCHEMA_VERSIONS,
    EngineeringSmokeManifest,
)
from active_audition.o1.manifest import (
    O1ExploratoryManifest,
    O1ManifestError,
    O1_MANIFEST_SCHEMA_VERSION,
    O1_MANIFEST_SCHEMA_VERSION_V2,
)
from active_audition.o1.landscape import O1_DRY_RUN_KIND, O1_REAL_KIND


ANALYSIS_ENTRY_SCHEMA_VERSION = "active-asr-o1-analysis-entry-v1"
O1_RUN_ROOT_MANIFEST_MISMATCH = "O1_RUN_ROOT_MANIFEST_MISMATCH"


class O1AnalysisEntryError(ValueError):
    """Raised when the requested analysis inputs are not one authority set."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise O1AnalysisEntryError("{}: cannot read {}".format(O1_RUN_ROOT_MANIFEST_MISMATCH, path)) from exc
    if not isinstance(value, Mapping):
        raise O1AnalysisEntryError("{}: {} is not an object".format(O1_RUN_ROOT_MANIFEST_MISMATCH, path))
    return value


def load_manifest_by_schema(path: Path) -> Tuple[Any, str]:
    """Parse exactly one known manifest type; never probe loaders heuristically."""

    payload = _json(path)
    schema = payload.get("schema_version")
    try:
        if schema in (O1_MANIFEST_SCHEMA_VERSION, O1_MANIFEST_SCHEMA_VERSION_V2):
            return O1ExploratoryManifest.from_payload(payload), "o1"
        if schema in ENGINEERING_SMOKE_MANIFEST_SCHEMA_VERSIONS:
            return EngineeringSmokeManifest.from_payload(payload), "engineering"
    except (ValueError, KeyError, TypeError) as exc:
        raise O1AnalysisEntryError("invalid {} manifest".format(schema)) from exc
    raise O1AnalysisEntryError("unknown manifest schema: {!r}".format(schema))


def resolve_analysis_kind(manifest: Any, manifest_type: str, requested: Optional[str]) -> str:
    """Apply the explicit manifest/analysis-kind compatibility contract."""

    if manifest_type == "o1":
        if manifest.schema_version == O1_MANIFEST_SCHEMA_VERSION_V2:
            manifest.require_scientific_noise_audit()
            inferred = O1_REAL_KIND
            allowed = (O1_REAL_KIND,)
        else:
            inferred = O1_DRY_RUN_KIND
            allowed = (O1_DRY_RUN_KIND,)
    elif manifest_type == "engineering":
        inferred = O1_DRY_RUN_KIND
        allowed = (O1_DRY_RUN_KIND,)
    else:
        raise O1AnalysisEntryError("unknown manifest type")
    kind = inferred if requested is None else requested
    if kind not in allowed:
        if kind == O1_REAL_KIND:
            raise O1AnalysisEntryError("O1_SCIENTIFIC_MANIFEST_REQUIRED")
        raise O1AnalysisEntryError("analysis kind is incompatible with manifest schema")
    return kind


def _require_summary_binding(summary: Mapping[str, Any], manifest: O1ExploratoryManifest, name: str) -> None:
    if summary.get("manifest_id") != manifest.manifest_id or summary.get("manifest_sha256") != manifest.manifest_sha256:
        raise O1AnalysisEntryError("{}: {} manifest identity mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH, name))


def validate_scientific_run_root(
    manifest: O1ExploratoryManifest,
    manifest_path: Path,
    run_root: Path,
    cache_root: Path,
) -> Dict[str, Any]:
    """Validate summaries, exact cache set, and marker without reading ASR results."""

    try:
        manifest.require_scientific_noise_audit()
    except O1ManifestError as exc:
        raise O1AnalysisEntryError("O1_SCIENTIFIC_MANIFEST_REQUIRED") from exc
    if manifest.schema_version != O1_MANIFEST_SCHEMA_VERSION_V2:
        raise O1AnalysisEntryError("O1_SCIENTIFIC_MANIFEST_REQUIRED")
    root = Path(run_root)
    required = (
        "mixture_stage_summary.json",
        "o1_component_snr_summary.json",
        "asr_stage_summary.json",
        "mixture_expected_manifest.json",
        "final_cache_expected_manifest.json",
    )
    for name in required:
        if not (root / name).is_file():
            raise O1AnalysisEntryError("{}: missing {}".format(O1_RUN_ROOT_MANIFEST_MISMATCH, name))
    mixture_summary = _json(root / "mixture_stage_summary.json")
    component_summary = _json(root / "o1_component_snr_summary.json")
    asr_summary = _json(root / "asr_stage_summary.json")
    for summary, name in ((mixture_summary, "mixture"), (component_summary, "component-snr"), (asr_summary, "asr")):
        _require_summary_binding(summary, manifest, name)
    if mixture_summary.get("expected") != 768 or mixture_summary.get("valid") != 768:
        raise O1AnalysisEntryError("{}: mixture cardinality mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    if component_summary.get("records") != 768 or component_summary.get("result_dependent_pose_selection") is not False:
        raise O1AnalysisEntryError("{}: component-SNR summary mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    if asr_summary.get("expected") != 2304 or asr_summary.get("valid") != 2304:
        raise O1AnalysisEntryError("{}: ASR cardinality mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    component_path = root / "o1_component_snr.jsonl"
    if sum(1 for line in component_path.read_text(encoding="utf-8").splitlines() if line.strip()) != 768:
        raise O1AnalysisEntryError("{}: component-SNR record count mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    mixture_expected = CacheExpectedManifest.from_payload(_json(root / "mixture_expected_manifest.json"))
    final_expected = CacheExpectedManifest.from_payload(_json(root / "final_cache_expected_manifest.json"))
    if dict(mixture_expected.expected_counts) != {"rir": 384, "mixture": 768, "asr": 0}:
        raise O1AnalysisEntryError("{}: mixture expected manifest mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    if dict(final_expected.expected_counts) != {"rir": 384, "mixture": 768, "asr": 2304}:
        raise O1AnalysisEntryError("{}: final expected manifest mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    if final_expected.infrastructure_contract_sha256 != manifest.infrastructure_contract_sha256:
        raise O1AnalysisEntryError("{}: cache contract mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    store = CacheStore(str(cache_root))
    reconciliation = reconcile_cache_manifest(final_expected, store)
    if not reconciliation.complete or any(reconciliation.rebuild_counts.values()):
        raise O1AnalysisEntryError("{}: final cache reconciliation is incomplete".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    marker = read_completion_marker(store, final_expected)
    if marker is None or reconciliation.completion_marker_status != COMPLETION_MARKER_VALID:
        raise O1AnalysisEntryError("{}: completion marker is not valid".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    if marker.expected_manifest_id != final_expected.manifest_id or marker.reconciliation_record_id != reconciliation.record_id:
        raise O1AnalysisEntryError("{}: completion marker binding mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    return {
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_file_sha256": sha256_file(manifest_path),
        "final_cache_manifest_id": final_expected.manifest_id,
        "final_cache_manifest_sha256": final_expected.manifest_sha256,
        "completion_marker_id": marker.marker_id,
        "completion_marker_status": COMPLETION_MARKER_VALID,
        "asr_stage_summary_sha256": sha256_file(root / "asr_stage_summary.json"),
        "diagnostic_component_snr_sha256": sha256_file(component_path),
        "expected_counts": dict(final_expected.expected_counts),
        "valid_counts": dict(reconciliation.reuse_counts),
        "reconciliation_record_id": reconciliation.record_id,
        "reconciliation_record_sha256": reconciliation.record_sha256,
    }


__all__ = [
    "ANALYSIS_ENTRY_SCHEMA_VERSION",
    "O1AnalysisEntryError",
    "O1_RUN_ROOT_MANIFEST_MISMATCH",
    "load_manifest_by_schema",
    "resolve_analysis_kind",
    "sha256_file",
    "validate_scientific_run_root",
]
