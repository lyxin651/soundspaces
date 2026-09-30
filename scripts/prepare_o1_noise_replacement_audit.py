#!/usr/bin/env python3
"""Freeze and package the next deterministic O1 noise replacement batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

from active_audition.o1.noise_audit import O1_NOISE_AUDIT_DECISION_SOURCE, O1_NOISE_AUDIT_FIELDS, O1_NOISE_AUDIT_POLICY_IDENTITY, O1_NOISE_AUDIT_SCHEMA_VERSION, O1NoiseParentAuditRecord
from active_audition.o1.replacement import O1ReplacementCandidateBatch


ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (0.08, 0.24, 0.40, 0.56, 0.72, 0.88)
CLIP_SAMPLES = 15 * 16000
A4_EXCLUDED = frozenset({
    "noise/free-sound/noise-free-sound-0002", "noise/free-sound/noise-free-sound-0020",
    "noise/free-sound/noise-free-sound-0048", "noise/free-sound/noise-free-sound-0270",
})


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded(path: Path) -> tuple[np.ndarray, str]:
    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    value = np.ascontiguousarray(samples, dtype=np.dtype("<f4"))
    if int(rate) != 16000 or value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise RuntimeError("replacement source is not finite mono native16: {}".format(path))
    return value, hashlib.sha256(value.tobytes()).hexdigest()


def _bounds(sample_count: int, fraction: float) -> tuple[int, int]:
    center = int(round(float(fraction) * (sample_count - 1)))
    start = max(0, min(center - CLIP_SAMPLES // 2, sample_count - CLIP_SAMPLES))
    return start, start + CLIP_SAMPLES


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _source_path(row: dict) -> Path:
    return ROOT / "data/active_asr_a3/corpora/musan" / row["relative_source_path"]


def _registry() -> tuple[list[dict], str]:
    path = ROOT / "registries/active_asr_a3/musan_noise.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], _sha_file(path)


def _write_readme(root: Path, records: list[dict], required: int) -> None:
    lines = [
        "# O1 noise replacement audit — batch 01", "",
        "This is a pre-ASR manual listening gate. The four candidates and their order were frozen by registry technical metadata before listening.",
        "", "No RIR, energy, DRR, component-SNR, ASR, WER, Oracle, or movement result was used.", "",
        "Listen in candidate order. For each candidate judge only:", "",
        "- **speech leakage** — PASS: no obvious or suspicious human voice, broadcast, or intelligible sung language; FLAGGED: obvious or not safely excludable speech-like content.",
        "- **strong reverberation** — PASS: no strong room coloration, long echo, or strong reverberant tail in the recorded parent; FLAGGED: clearly strong recorded room response.",
        "- **indoor localized-source compatibility** — PASS: reasonably explained by a relatively fixed indoor source; FLAGGED: road/driving/train, outdoor diffuse ambience, crowd/public-space ambience, wind/rain field, clear moving source, or another incompatible sound.",
        "", "Use UNCERTAIN when needed and record timestamps. Codex must not fill any decision.",
        "", "Each preview is an exact contiguous 15-second float32 slice from the native-16-kHz mono parent. No gain, normalization, fade, filtering, denoise, VAD, resampling, or channel processing was applied.",
        "", "Full source files:",
    ]
    for item in records:
        lines.append("- candidate_{:02d}: {}".format(item["candidate_index"], item["full_source_absolute_path"]))
    lines += ["", "The first two candidates in this frozen order with all three decisions PASS become replacement slot 1 and slot 2. Slot 1 replaces original parent 0015; slot 2 replaces original parent 0030. Do not swap slots.", "", "| # | Parent | Speech leakage | Strong reverb | Indoor localized compatibility | Suspicious timestamp / note |", "|---|---|---|---|---|---|"]
    for item in records:
        lines.append("| {:02d} | {} | PENDING_USER | PENDING_USER | PENDING_USER | |".format(item["candidate_index"], item["parent_recording_id"]))
    lines += ["", "The batch technical requirement is {:,} samples ({:.6f} sec) for the current four-episode O1 noise-plan maximum. If fewer than two candidates pass, the next batch continues after all batch-01 parents; no listening preference or result-dependent reordering is allowed.".format(required, required / 16000.0)]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/logs/o1_noise_replacement_audit_batch01")
    args = parser.parse_args()
    root = args.output_root
    if root.exists():
        raise SystemExit("refusing to overwrite existing replacement package: {}".format(root))
    manifest = json.loads((ROOT / "runs/active_asr_v1/o1_replica_apartment_2_c8c821eae922/o1_exploratory_manifest.json").read_text(encoding="utf-8"))
    current = set(manifest["selection_policy"]["noise_rule"]["selected_parent_ids"])
    required = max(int(block["noise_plan"]["required_parent_sample_count"]) for block in manifest["blocks"])
    rows, registry_sha = _registry()
    registry_excluded = {
        row["parent_recording_id"] for row in rows
        if row.get("excluded") is True
        or row.get("speech_leakage_audit", {}).get("status") in ("FLAGGED", "EXCLUDED")
        or row.get("strong_reverberation_audit", {}).get("status") in ("FLAGGED", "EXCLUDED")
    }
    excluded = A4_EXCLUDED | current | registry_excluded
    eligible = []
    for row in rows:
        technical = row.get("technical_audit", {})
        if not (row.get("corpus") == "MUSAN" and row.get("subset") == "noise" and row.get("excluded") is False and technical.get("readable") is True and technical.get("too_short") is False and technical.get("extreme_silence") is False and row.get("sample_rate_hz") == 16000 and int(row.get("samples", 0)) >= required and row["parent_recording_id"] not in excluded):
            continue
        eligible.append(row)
    eligible.sort(key=lambda row: row["parent_recording_id"])
    if len(eligible) < 4:
        raise SystemExit("fewer than four technical replacement candidates")
    selected = eligible[:4]
    candidates = []
    decoded_values = {}
    for index, row in enumerate(selected, 1):
        source = _source_path(row)
        if not source.is_file() or _sha_file(source) != row["source_file_sha256"]:
            raise SystemExit("selected replacement source provenance failed: {}".format(row["parent_recording_id"]))
        waveform, decoded_sha = _decoded(source)
        if decoded_sha != row["decoded_waveform_sha256"] or waveform.size != int(row["samples"]):
            raise SystemExit("selected replacement decoded provenance failed: {}".format(row["parent_recording_id"]))
        decoded_values[row["parent_recording_id"]] = (waveform, source)
        candidates.append({
            "candidate_index": index, "parent_recording_id": row["parent_recording_id"], "relative_source_path": row["relative_source_path"],
            "source_file_sha256": row["source_file_sha256"], "decoded_waveform_sha256": row["decoded_waveform_sha256"],
            "sample_rate_hz": 16000, "sample_count": int(waveform.size),
            "technical_audit": {
                "corpus": "MUSAN", "subset": "noise", "excluded": False, "readable": True, "too_short": False, "extreme_silence": False,
                "sample_rate_hz": 16000, "source_file_exists": True, "source_file_sha256_matches": True,
                "decoded_waveform_sha256_matches": True, "decoded_mono_finite_float32": True, "sample_count_sufficient": True,
            },
        })
    batch = O1ReplacementCandidateBatch(
        schema_version="active-asr-o1-noise-replacement-batch-v1", selection_policy_identity="active-asr-o1-noise-replacement-technical-lexical-v1",
        registry_relative_path="registries/active_asr_a3/musan_noise.jsonl", registry_sha256=registry_sha,
        required_parent_sample_count=required, required_parent_duration_sec=required / 16000.0,
        excluded_parent_recording_ids=tuple(sorted(excluded)), candidate_records=tuple(candidates), engineering_only=True,
    )
    root.mkdir(parents=True)
    (root / "candidate_selection.json").write_bytes(_canonical(batch.to_payload()) + b"\n")
    audit_records = []
    sheet_records = []
    for item in candidates:
        parent_id = item["parent_recording_id"]
        waveform, source = decoded_values[parent_id]
        parent_dir = root / "candidate_{:02d}_{}".format(item["candidate_index"], parent_id.rsplit("-", 1)[-1])
        preview_dir = parent_dir / "previews"
        preview_dir.mkdir(parents=True)
        previews = []
        for clip_index, fraction in enumerate(FRACTIONS, 1):
            start, end = _bounds(waveform.size, fraction)
            clip = np.ascontiguousarray(waveform[start:end], dtype=np.dtype("<f4"))
            path = preview_dir / "clip_{:02d}_s{:08d}_e{:08d}.wav".format(clip_index, start, end)
            sf.write(str(path), clip, 16000, format="WAV", subtype="FLOAT")
            written, written_sha = _decoded(path)
            expected_sha = hashlib.sha256(clip.tobytes()).hexdigest()
            if expected_sha != written_sha or written.size != clip.size:
                raise SystemExit("replacement preview sample verification failed: {}".format(path))
            previews.append({"relative_preview_path": str(path.relative_to(root)), "clip_index": clip_index, "center_fraction": fraction, "start_sample": start, "end_sample": end, "start_sec": start / 16000.0, "end_sec": end / 16000.0, "clip_sample_count": end - start, "expected_source_slice_float32_sha256": expected_sha, "written_clip_decoded_float32_sha256": written_sha, "verification_status": "SAMPLE_EXACT"})
        record = O1NoiseParentAuditRecord(
            schema_version=O1_NOISE_AUDIT_SCHEMA_VERSION, audit_policy_identity=O1_NOISE_AUDIT_POLICY_IDENTITY, decision_source=O1_NOISE_AUDIT_DECISION_SOURCE,
            parent_recording_id=parent_id, relative_source_path=item["relative_source_path"], source_file_sha256=item["source_file_sha256"], decoded_waveform_sha256=item["decoded_waveform_sha256"],
            sample_rate_hz=16000, sample_count=item["sample_count"], reviewed_preview_identities=tuple(previews), review_scope={"dimensions": list(O1_NOISE_AUDIT_FIELDS), "clip_duration_sec": 15.0, "candidate_index": item["candidate_index"], "replacement_batch_id": batch.batch_id},
            manual_decision_provenance={"decision_source": O1_NOISE_AUDIT_DECISION_SOURCE, "status": "PENDING_USER"}, speech_leakage="PENDING_USER", strong_reverberation="PENDING_USER", indoor_localized_source_compatibility="PENDING_USER", selected_for_o1_scientific_use=False, exclusion_reason="awaiting user manual audit", engineering_only=True,
        )
        audit_records.append(record.to_payload())
        sheet_records.append({"candidate_index": item["candidate_index"], "parent_recording_id": parent_id, "relative_source_path": item["relative_source_path"], "full_source_absolute_path": str(source.resolve()), "source_file_sha256": item["source_file_sha256"], "decoded_waveform_sha256": item["decoded_waveform_sha256"], "sample_rate_hz": 16000, "sample_count": item["sample_count"], "technical_eligibility": item["technical_audit"], "preview_clips": previews, "manual_audit": {field: "PENDING_USER" for field in O1_NOISE_AUDIT_FIELDS} | {"suspicious_timestamps": [], "notes": ""}})
    envelope = {"schema_version": "active-asr-o1-noise-replacement-audit-batch-v1", "replacement_batch_id": batch.batch_id, "replacement_batch_sha256": batch.batch_sha256, "records": audit_records}
    (root / "o1_noise_audit_records.json").write_bytes(_canonical(envelope) + b"\n")
    (root / "audit_sheet.json").write_bytes(_canonical({"schema_version": "active-asr-o1-noise-replacement-audit-sheet-v1", "status": "PENDING_USER", "replacement_batch_id": batch.batch_id, "candidates": sheet_records}) + b"\n")
    _write_readme(root, sheet_records, required)
    sums = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        sums.append("{}  {}".format(_sha_file(path), path.relative_to(root)))
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    tar_path = root.parent / "o1_noise_replacement_audit_batch01.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    print(json.dumps({"batch": batch.to_payload(), "output_root": str(root.resolve()), "tar_path": str(tar_path.resolve()), "tar_sha256": _sha_file(tar_path), "parents": sheet_records}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
