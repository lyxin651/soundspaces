#!/usr/bin/env python3
"""Build the exact pre-ASR O1 manual listening package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

from active_audition.o1.noise_audit import (
    O1_NOISE_AUDIT_DECISION_SOURCE,
    O1_NOISE_AUDIT_FIELDS,
    O1_NOISE_AUDIT_POLICY_IDENTITY,
    O1_NOISE_AUDIT_SCHEMA_VERSION,
    O1NoiseParentAuditRecord,
)


ROOT = Path(__file__).resolve().parents[1]
PARENTS = (
    "noise/free-sound/noise-free-sound-0015",
    "noise/free-sound/noise-free-sound-0030",
    "noise/free-sound/noise-free-sound-0032",
    "noise/free-sound/noise-free-sound-0042",
)
FRACTIONS = (0.08, 0.24, 0.40, 0.56, 0.72, 0.88)
CLIP_SAMPLES = 15 * 16000


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded(path: Path) -> np.ndarray:
    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    value = np.ascontiguousarray(samples, dtype=np.dtype("<f4"))
    if int(rate) != 16000 or value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise RuntimeError("source must be finite mono native16: {}".format(path))
    return value


def _registry_rows() -> dict[str, dict]:
    result = {}
    for line in (ROOT / "registries/active_asr_a3/musan_noise.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["parent_recording_id"]] = row
    return result


def _preview_bounds(sample_count: int, fraction: float) -> tuple[int, int]:
    center = int(round(float(fraction) * (sample_count - 1)))
    start = center - CLIP_SAMPLES // 2
    start = max(0, min(start, sample_count - CLIP_SAMPLES))
    return start, start + CLIP_SAMPLES


def _write_readme(root: Path, rows: list[dict]) -> None:
    lines = [
        "# O1 MUSAN manual listening package",
        "",
        "This package is a pre-ASR gate. Codex has not assigned any manual decision.",
        "Before any O1 ASR, listen to candidate_01 through candidate_04 and fill the audit sheet.",
        "",
        "For every parent, judge only these three dimensions:",
        "",
        "1. **Speech leakage** — PASS means no obvious or suspicious human voice, broadcast, or intelligible sung language. FLAGGED means obvious or not safely excludable speech-like content.",
        "2. **Strong reverberation** — PASS means no strong room coloration, long echo, or reverberant tail in the recording itself. FLAGGED means clearly strong recorded room response.",
        "3. **Indoor localized-source compatibility** — PASS means the sound can reasonably be a relatively fixed indoor source such as a fan, appliance, HVAC, or local machine. FLAGGED means road/driving/train, outdoor diffuse ambience, crowd/public-space ambience, wind/rain field, a clearly moving source, or another sound incompatible with the indoor static localized-source approximation.",
        "",
        "Use UNCERTAIN when the decision cannot be made confidently. Record timestamps and inspect the corresponding full WAV path if needed. Do not infer a decision from the file name or category.",
        "",
        "No normalization, gain adjustment, fade, filtering, denoise, resampling, VAD, channel processing, feature extraction, ASR, WER, or Oracle was used. Each preview is an exact contiguous 15-second float32 sample slice from the decoded native-16-kHz mono parent.",
        "",
        "Full WAV paths (read-only source support):",
    ]
    for row in rows:
        lines.append("- candidate_{:02d}: {}".format(row["candidate_index"], row["full_source_absolute_path"]))
    lines += [
        "",
        "The selected scientific O1 parent rule is deterministic: after user audit, use the first two parents in candidate order for which all three dimensions are PASS. This package does not perform that selection automatically.",
        "",
        "| # | Parent | Speech leakage | Strong reverb | Indoor localized compatibility | Suspicious timestamp / note |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| {:02d} | {} | PENDING_USER | PENDING_USER | PENDING_USER | |".format(row["candidate_index"], row["parent_recording_id"]))
    lines += [
        "",
        "Codex must not fill PASS, FLAGGED, or UNCERTAIN. The user must provide the four final manual decisions before any O1 parent is scientific-use eligible.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/logs/o1_noise_manual_audit")
    args = parser.parse_args()
    root = args.output_root
    if root.exists():
        raise SystemExit("refusing to overwrite existing audit package: {}".format(root))
    rows_by_id = _registry_rows()
    manifest = json.loads((ROOT / "runs/active_asr_v1/o1_replica_apartment_2_c8c821eae922/o1_exploratory_manifest.json").read_text(encoding="utf-8"))
    selected = tuple(manifest["selection_policy"]["noise_rule"]["selected_parent_ids"])
    if selected != PARENTS:
        raise SystemExit("O1 manifest parent mapping does not match the four expected audit parents")
    root.mkdir(parents=True)
    audit_rows = []
    record_rows = []
    for index, parent_id in enumerate(PARENTS, 1):
        row = rows_by_id[parent_id]
        if row["speech_leakage_audit"]["status"] != "NOT_AUDITED" or row["strong_reverberation_audit"]["status"] != "NOT_AUDITED":
            raise SystemExit("A3 registry parent is no longer NOT_AUDITED: {}".format(parent_id))
        source = ROOT / "data/active_asr_a3/corpora/musan" / row["relative_source_path"]
        source_sha = _sha_file(source)
        if source_sha != row["source_file_sha256"]:
            raise SystemExit("source SHA mismatch: {}".format(parent_id))
        waveform = _decoded(source)
        decoded_sha = _sha_bytes(waveform.tobytes())
        if decoded_sha != row["decoded_waveform_sha256"] or waveform.size != int(row["samples"]):
            raise SystemExit("decoded provenance mismatch: {}".format(parent_id))
        parent_dir = root / "candidate_{:02d}_{}".format(index, parent_id.rsplit("-", 1)[-1])
        preview_dir = parent_dir / "previews"
        preview_dir.mkdir(parents=True)
        previews = []
        for clip_index, fraction in enumerate(FRACTIONS, 1):
            start, end = _preview_bounds(waveform.size, fraction)
            clip = np.ascontiguousarray(waveform[start:end], dtype=np.dtype("<f4"))
            filename = "clip_{:02d}_s{:08d}_e{:08d}.wav".format(clip_index, start, end)
            path = preview_dir / filename
            sf.write(str(path), clip, 16000, format="WAV", subtype="FLOAT")
            written = _decoded(path)
            expected_sha = _sha_bytes(clip.tobytes())
            written_sha = _sha_bytes(written.tobytes())
            if expected_sha != written_sha:
                raise SystemExit("preview sample verification failed: {}".format(path))
            previews.append({
                "relative_preview_path": str(path.relative_to(root)), "clip_index": clip_index,
                "center_fraction": fraction, "start_sample": start, "end_sample": end,
                "start_sec": start / 16000.0, "end_sec": end / 16000.0, "clip_sample_count": int(end - start),
                "expected_source_slice_float32_sha256": expected_sha,
                "written_clip_decoded_float32_sha256": written_sha, "verification_status": "SAMPLE_EXACT",
            })
        record = O1NoiseParentAuditRecord(
            schema_version=O1_NOISE_AUDIT_SCHEMA_VERSION,
            audit_policy_identity=O1_NOISE_AUDIT_POLICY_IDENTITY,
            decision_source=O1_NOISE_AUDIT_DECISION_SOURCE,
            parent_recording_id=parent_id, relative_source_path=row["relative_source_path"],
            source_file_sha256=source_sha, decoded_waveform_sha256=decoded_sha,
            sample_rate_hz=16000, sample_count=int(waveform.size), reviewed_preview_identities=tuple(previews),
            review_scope={"dimensions": list(O1_NOISE_AUDIT_FIELDS), "clip_duration_sec": 15.0, "candidate_index": index},
            manual_decision_provenance={"decision_source": O1_NOISE_AUDIT_DECISION_SOURCE, "status": "PENDING_USER"},
            speech_leakage="PENDING_USER", strong_reverberation="PENDING_USER",
            indoor_localized_source_compatibility="PENDING_USER", selected_for_o1_scientific_use=False,
            exclusion_reason="awaiting user manual audit", engineering_only=True,
        )
        record_rows.append(record.to_payload())
        audit_rows.append({
            "candidate_index": index, "parent_recording_id": parent_id,
            "relative_source_path": row["relative_source_path"],
            "full_source_absolute_path": str(source.resolve()), "duration_sec": waveform.size / 16000.0,
            "sample_rate_hz": 16000, "sample_count": int(waveform.size),
            "source_file_sha256": source_sha, "decoded_waveform_sha256": decoded_sha,
            "technical_eligibility": row["technical_audit"], "preview_clips": previews,
            "manual_audit": {field: "PENDING_USER" for field in O1_NOISE_AUDIT_FIELDS} | {"suspicious_timestamps": [], "notes": ""},
        })
    (root / "audit_sheet.json").write_text(json.dumps({"schema_version": "active-asr-o1-noise-manual-audit-sheet-v1", "candidate_count": 4, "candidates": audit_rows}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    (root / "o1_noise_audit_records.json").write_text(json.dumps({"schema_version": "active-asr-o1-noise-audit-batch-v1", "audit_policy_identity": O1_NOISE_AUDIT_POLICY_IDENTITY, "records": record_rows}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _write_readme(root, audit_rows)
    sums = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        sums.append("{}  {}".format(_sha_file(path), path.relative_to(root)))
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    tar_path = root.parent / "o1_noise_manual_audit.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    print(json.dumps({"output_root": str(root.resolve()), "tar_path": str(tar_path.resolve()), "tar_sha256": _sha_file(tar_path), "parents": audit_rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
