#!/usr/bin/env python3
"""Finalize the user decisions for replacement batch 01."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from active_audition.a4.identity import canonical_json_bytes
from active_audition.o1.noise_audit import (
    O1_NOISE_AUDIT_DECISION_SOURCE,
    O1_NOISE_AUDIT_POLICY_IDENTITY,
    O1NoiseParentAuditRecord,
)
from active_audition.o1.replacement import O1ReplacementCandidateBatch
from active_audition.o1.replacement_audit import O1FinalizedReplacementAuditBatch


ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "o1-replacement-batch-1d0f493be19458ee4928862dbd779c9bf62044eb7cb0f31e426794349436df95"
BATCH_SHA = "1d0f493be19458ee4928862dbd779c9bf62044eb7cb0f31e426794349436df95"
DECISIONS = {
    "noise/free-sound/noise-free-sound-0041": ("PASS", "PASS", "PASS", True, ""),
    "noise/free-sound/noise-free-sound-0064": ("NOT_EVALUATED", "NOT_EVALUATED", "FLAGGED", False, "OUTDOOR_MOVING_CAR_DRIVING_SOURCE"),
    "noise/free-sound/noise-free-sound-0066": ("NOT_EVALUATED", "NOT_EVALUATED", "FLAGGED", False, "MOVING_TRAIN_SOURCE"),
    "noise/free-sound/noise-free-sound-0073": ("PASS", "PASS", "PASS", True, ""),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/logs/o1_noise_replacement_audit_batch01")
    args = parser.parse_args()
    root = args.root
    batch = O1ReplacementCandidateBatch.from_payload(json.loads((root / "candidate_selection.json").read_text(encoding="utf-8")))
    if batch.batch_id != BATCH_ID or batch.batch_sha256 != BATCH_SHA:
        raise SystemExit("replacement candidate batch identity mismatch")
    source = json.loads((root / "o1_noise_audit_records.json").read_text(encoding="utf-8"))
    records = {item["parent_recording_id"]: O1NoiseParentAuditRecord.from_payload(item) for item in source["records"]}
    if tuple(sorted(records)) != tuple(sorted(DECISIONS)):
        raise SystemExit("replacement audit package does not match frozen batch 01")
    finalized = []
    for parent_id in sorted(DECISIONS):
        record = records[parent_id]
        speech, reverb, compatibility, selected, reason = DECISIONS[parent_id]
        provenance = dict(record.manual_decision_provenance)
        provenance.update({"decision_source": O1_NOISE_AUDIT_DECISION_SOURCE, "status": "FINALIZED_USER_DECISION"})
        payload = dict(record.to_payload())
        payload.update({
            "speech_leakage": speech,
            "strong_reverberation": reverb,
            "indoor_localized_source_compatibility": compatibility,
            "selected_for_o1_scientific_use": selected,
            "exclusion_reason": reason,
            "manual_decision_provenance": provenance,
            "audit_record_id": "",
            "audit_record_sha256": "",
        })
        finalized.append(O1NoiseParentAuditRecord.from_payload(payload).to_payload())
    envelope = O1FinalizedReplacementAuditBatch(
        schema_version="active-asr-o1-noise-replacement-audit-finalized-v1",
        audit_policy_identity=O1_NOISE_AUDIT_POLICY_IDENTITY,
        decision_source=O1_NOISE_AUDIT_DECISION_SOURCE,
        source_replacement_batch_id=BATCH_ID,
        source_replacement_batch_sha256=BATCH_SHA,
        records=tuple(finalized),
        selected_replacement_slots={
            "slot1": {"parent_recording_id": "noise/free-sound/noise-free-sound-0041", "replaces_parent_recording_id": "noise/free-sound/noise-free-sound-0015"},
            "slot2": {"parent_recording_id": "noise/free-sound/noise-free-sound-0073", "replaces_parent_recording_id": "noise/free-sound/noise-free-sound-0030"},
        },
    )
    (root / "o1_noise_audit_records_finalized.json").write_bytes(canonical_json_bytes(envelope.to_payload()) + b"\n")
    sheet_path = root / "audit_sheet.json"
    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    by_id = {item["parent_recording_id"]: item for item in finalized}
    for item in sheet["candidates"]:
        record = by_id[item["parent_recording_id"]]
        for field in ("speech_leakage", "strong_reverberation", "indoor_localized_source_compatibility"):
            item["manual_audit"][field] = record[field]
        item["manual_audit"]["notes"] = record["exclusion_reason"]
        item["finalized_audit_record_id"] = record["audit_record_id"]
        item["finalized_audit_record_sha256"] = record["audit_record_sha256"]
    sheet["schema_version"] = "active-asr-o1-noise-replacement-audit-sheet-v2"
    sheet["status"] = "FINALIZED_USER_MANUAL_AUDIT"
    sheet["selected_replacement_slots"] = envelope.selected_replacement_slots
    sheet_path.write_bytes(canonical_json_bytes(sheet) + b"\n")
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    text += "\n\nFinalized user decision: candidate 01 (0041) is replacement slot 1 for 0015; candidate 04 (0073) is replacement slot 2 for 0030. Candidates 0064 and 0066 are excluded.\n"
    readme.write_text(text, encoding="utf-8")
    sums = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        sums.append("{}  {}".format(_sha(path), path.relative_to(root)))
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    tar_path = root.parent / "o1_noise_replacement_audit_batch01_finalized.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    print(json.dumps({"batch_id": envelope.batch_id, "batch_sha256": envelope.batch_sha256, "tar_path": str(tar_path.resolve()), "tar_sha256": _sha(tar_path), "selected_replacement_slots": envelope.selected_replacement_slots}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
