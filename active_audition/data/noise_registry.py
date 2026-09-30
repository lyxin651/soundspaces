"""MUSAN-noise provenance and technical-audit registry for A3."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from active_audition.data.speech_registry import sha256_file


NOISE_REGISTRY_SCHEMA_VERSION = "active-asr-a3-musan-noise-registry-v1"
NOISE_ROW_KEYS = (
    "schema_version",
    "corpus",
    "subset",
    "parent_recording_id",
    "relative_source_path",
    "source_file_sha256",
    "decoded_waveform_sha256",
    "sample_rate_hz",
    "dtype",
    "samples",
    "duration_sec",
    "legal_slice_ranges_sec",
    "technical_audit",
    "speech_leakage_audit",
    "strong_reverberation_audit",
    "excluded",
    "exclusion_reason",
    "recorded_noise_as_localized_source_approximation",
)


class NoiseRegistryError(ValueError):
    """Raised when MUSAN noise provenance is invalid."""


def decoded_noise_sha256(waveform: Any) -> str:
    value = np.ascontiguousarray(np.asarray(waveform, dtype="<f4"))
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise NoiseRegistryError("decoded noise must be a finite non-empty mono vector")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def validate_noise_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise NoiseRegistryError("noise registry row must be a mapping")
    unknown = sorted(set(row) - set(NOISE_ROW_KEYS))
    missing = [key for key in NOISE_ROW_KEYS if key not in row]
    if unknown or missing:
        raise NoiseRegistryError("invalid noise registry keys: unknown={} missing={}".format(unknown, missing))
    if row["schema_version"] != NOISE_REGISTRY_SCHEMA_VERSION or row["corpus"] != "MUSAN" or row["subset"] != "noise":
        raise NoiseRegistryError("invalid MUSAN noise identity")
    for key in ("parent_recording_id", "relative_source_path", "source_file_sha256", "decoded_waveform_sha256", "dtype", "exclusion_reason"):
        if not isinstance(row[key], str):
            raise NoiseRegistryError("{} must be a string".format(key))
    if not row["parent_recording_id"] or len(row["source_file_sha256"]) != 64 or len(row["decoded_waveform_sha256"]) != 64:
        raise NoiseRegistryError("invalid parent or SHA identity")
    if not isinstance(row["sample_rate_hz"], int) or row["sample_rate_hz"] <= 0 or row["dtype"] != "float32":
        raise NoiseRegistryError("invalid decoded noise format")
    if not isinstance(row["samples"], int) or row["samples"] <= 0 or float(row["duration_sec"]) <= 0.0:
        raise NoiseRegistryError("invalid decoded noise length")
    if not isinstance(row["legal_slice_ranges_sec"], list):
        raise NoiseRegistryError("legal_slice_ranges_sec must be a list")
    technical = row["technical_audit"]
    expected_technical = {"readable", "too_short", "extreme_silence", "rms"}
    if not isinstance(technical, Mapping) or set(technical) != expected_technical:
        raise NoiseRegistryError("technical_audit has invalid fields")
    for key in ("speech_leakage_audit", "strong_reverberation_audit"):
        audit = row[key]
        if not isinstance(audit, Mapping) or set(audit) != {"status", "reason"}:
            raise NoiseRegistryError("{} must contain status/reason".format(key))
        if audit["status"] not in ("NOT_AUDITED", "PASS", "FLAGGED"):
            raise NoiseRegistryError("{} has invalid status".format(key))
    if not isinstance(row["excluded"], bool) or row["recorded_noise_as_localized_source_approximation"] is not True:
        raise NoiseRegistryError("invalid exclusion/approximation flags")
    if not row["excluded"] and row["exclusion_reason"]:
        raise NoiseRegistryError("included row may not have an exclusion reason")
    return row


def build_noise_registry(
    musan_root: str,
    minimum_duration_sec: float = 4.0,
    extreme_silence_rms: float = 1.0e-5,
) -> List[Dict[str, Any]]:
    """Build only from the real MUSAN ``noise`` subset."""

    try:
        import soundfile as sf
    except ImportError as error:
        raise NoiseRegistryError("soundfile is required to build the noise registry") from error
    root = Path(musan_root).resolve()
    noise_root = root / "noise"
    if not noise_root.is_dir():
        raise NoiseRegistryError("MUSAN noise subset is missing: {}".format(noise_root))
    rows: List[Dict[str, Any]] = []
    for path in sorted(noise_root.rglob("*.wav")):
        relative = path.relative_to(root).as_posix()
        parent_id = Path(relative).with_suffix("").as_posix()
        reasons: List[str] = []
        try:
            waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
            waveform = np.asarray(waveform, dtype=np.float32)
            readable = bool(waveform.ndim == 1 and waveform.size > 0 and np.isfinite(waveform).all())
        except Exception as error:  # The exception text is evidence, not control flow identity.
            raise NoiseRegistryError("unreadable/corrupt MUSAN file {}: {}".format(path, error)) from error
        if not readable:
            raise NoiseRegistryError("unreadable/corrupt MUSAN waveform: {}".format(path))
        duration = float(waveform.size) / float(sample_rate)
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        too_short = duration < float(minimum_duration_sec)
        extreme_silence = rms <= float(extreme_silence_rms)
        if too_short:
            reasons.append("TOO_SHORT")
        if extreme_silence:
            reasons.append("EXTREME_SILENCE")
        row = {
            "schema_version": NOISE_REGISTRY_SCHEMA_VERSION,
            "corpus": "MUSAN",
            "subset": "noise",
            "parent_recording_id": parent_id,
            "relative_source_path": relative,
            "source_file_sha256": sha256_file(path),
            "decoded_waveform_sha256": decoded_noise_sha256(waveform),
            "sample_rate_hz": int(sample_rate),
            "dtype": "float32",
            "samples": int(waveform.size),
            "duration_sec": duration,
            "legal_slice_ranges_sec": [] if reasons else [[0.0, duration]],
            "technical_audit": {
                "readable": True,
                "too_short": too_short,
                "extreme_silence": extreme_silence,
                "rms": rms,
            },
            "speech_leakage_audit": {"status": "NOT_AUDITED", "reason": "manual_or_validated_audit_required"},
            "strong_reverberation_audit": {"status": "NOT_AUDITED", "reason": "manual_or_validated_audit_required"},
            "excluded": bool(reasons),
            "exclusion_reason": ";".join(reasons),
            "recorded_noise_as_localized_source_approximation": True,
        }
        validate_noise_row(row)
        rows.append(row)
    if not rows:
        raise NoiseRegistryError("MUSAN noise subset contains no WAV files")
    return rows


def parent_split_audit(assignments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_parent = defaultdict(set)
    for assignment in assignments:
        parent = str(assignment["parent_recording_id"])
        split = str(assignment["split"])
        by_parent[parent].add(split)
    leaked = sorted(parent for parent, splits in by_parent.items() if len(splits) > 1)
    return {
        "parent_recordings": len(by_parent),
        "leaked_parent_recording_ids": leaked,
        "parent_split_disjoint": not leaked,
        "status": "PASS" if not leaked else "FAIL",
    }


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [json.dumps(dict(row), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) for row in rows]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_noise_registry(path: str, rows: Sequence[Mapping[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = jsonl_bytes(rows)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_noise_registry(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise NoiseRegistryError("invalid JSON at line {}".format(line_number)) from error
        validate_noise_row(row)
        rows.append(dict(row))
    return rows
