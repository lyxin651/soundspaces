"""Machine-readable Step 2C provenance collection and final-lock checks."""

import hashlib
import importlib.metadata
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

from .schema import CLIP_DURATION_SEC, SAMPLE_RATE_HZ, SCHEMA_VERSION


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceError(ValueError):
    """Raised when provenance is missing or contains a placeholder."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "UNAVAILABLE"


def _rlr_resource() -> Mapping[str, Any]:
    candidates = []
    try:
        import sys

        candidates.extend(Path(item) / "habitat_sim/_ext/libRLRAudioPropagation.so" for item in sys.path)
    except Exception:
        pass
    for path in candidates:
        if path.is_file():
            return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"path": None, "size_bytes": None, "sha256": None, "status": "NOT_FOUND"}


def _hrtf_resource(hrtf_path: Optional[str] = None, enclosing: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    if hrtf_path:
        path = Path(hrtf_path).resolve()
        if not path.is_file():
            raise ProvenanceError("HRTF path does not exist: {}".format(path))
        return {
            "mode": "external_file", "path": str(path),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        }
    enclosing = dict(enclosing or {})
    return {
        "mode": "embedded_or_not_exposed",
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "status": "NOT_EXPOSED_BY_INSTALLED_HABITAT_SIM",
        "enclosing_implementation": enclosing,
    }


def build_provenance_template(
    repo_root: str,
    hrtf_path: Optional[str] = None,
    *,
    random_seed: Optional[int] = None,
) -> Mapping[str, Any]:
    """Collect Core provenance without pretending Step 2A/2B are complete."""

    if random_seed is None:
        raise ProvenanceError("random_seed must be injected by the resolved build/regression config")
    root = Path(repo_root).resolve()
    ontology = root / "registries/ontology.yaml"
    if not ontology.is_file():
        raise ProvenanceError("ontology registry is missing: {}".format(ontology))
    commit = _git_commit(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_code_commit": commit,
        "soundspaces_commit": commit,
        "habitat_sim_version": _package_version("habitat-sim", "habitat_sim"),
        "rlraudio_propagation": _rlr_resource(),
        "hrtf": _hrtf_resource(hrtf_path, _rlr_resource()),
        "foa_contract_version": "P0-B@3e3c21ab0151a0b0d8e8d1e0946e9fa015c7bf92",
        "ontology_sha256": sha256_file(ontology),
        "ray_config": {"indirectRayCount": 5000, "sourceRayCount": 200},
        "indirectRayCount": 5000,
        "sourceRayCount": 200,
        "materials_mode": "OFF",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "clip_duration_sec": CLIP_DURATION_SEC,
        "random_seed": int(random_seed),
        "source_registry_sha256": "PENDING_STEP_2A",
        "source_split_version": "PENDING_STEP_2A",
        "scene_registry_sha256": "PENDING_STEP_2B",
        "scene_split_version": "PENDING_STEP_2B",
    }


def validate_core_provenance_template(value: Mapping[str, Any]) -> None:
    expected = {
        "source_registry_sha256": "PENDING_STEP_2A",
        "source_split_version": "PENDING_STEP_2A",
        "scene_registry_sha256": "PENDING_STEP_2B",
        "scene_split_version": "PENDING_STEP_2B",
    }
    for key, pending in expected.items():
        if value.get(key) != pending:
            raise ProvenanceError("Core provenance field {} must remain {}".format(key, pending))


def _require_sha(value: Any, name: str) -> None:
    text = str(value)
    if not SHA256_RE.fullmatch(text) or text == "0" * 64:
        raise ProvenanceError("{} must be a non-placeholder 64-character SHA-256 hex digest".format(name))


def validate_final_resources_lock(value: Mapping[str, Any]) -> None:
    """Validate the stricter lock semantics used only after Step 2A/2B closure."""

    from .storage import validate_resources_lock

    validate_resources_lock(value)
    for key in ("ontology_sha256", "source_registry_sha256", "scene_registry_sha256"):
        _require_sha(value.get(key), key)
    for key in ("source_split_version", "scene_split_version"):
        text = str(value.get(key, ""))
        if not text or text.startswith("PENDING") or text in {"fixture", "UNASSIGNED"}:
            raise ProvenanceError("{} is not a finalized split version".format(key))
    hrtf = value.get("hrtf")
    if not isinstance(hrtf, Mapping):
        raise ProvenanceError("hrtf must be an object")
    mode = hrtf.get("mode")
    if mode == "external_file":
        path_value = hrtf.get("path")
        if not isinstance(path_value, str) or not Path(path_value).is_file():
            raise ProvenanceError("external HRTF path must exist")
        _require_sha(hrtf.get("sha256"), "hrtf.sha256")
        if sha256_file(Path(path_value)) != hrtf.get("sha256"):
            raise ProvenanceError("hrtf.sha256 does not match the external HRTF file")
    elif mode == "embedded_or_not_exposed":
        if hrtf.get("path") is not None or hrtf.get("sha256") is not None:
            raise ProvenanceError("embedded/not-exposed HRTF must not claim a file hash")
        enclosing = hrtf.get("enclosing_implementation")
        if not isinstance(enclosing, Mapping):
            raise ProvenanceError("embedded/not-exposed HRTF requires enclosing implementation provenance")
        binary_path = enclosing.get("path")
        if not isinstance(binary_path, str) or not Path(binary_path).is_file():
            raise ProvenanceError("HRTF enclosing implementation path must exist")
        _require_sha(enclosing.get("sha256"), "hrtf.enclosing_implementation.sha256")
        if sha256_file(Path(binary_path)) != enclosing.get("sha256"):
            raise ProvenanceError("HRTF enclosing implementation SHA does not match its binary")
    else:
        raise ProvenanceError("hrtf.mode must be external_file or embedded_or_not_exposed")
