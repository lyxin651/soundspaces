#!/usr/bin/env python3
"""Persist the four user-provided O1 manual noise decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from active_audition.a4.identity import identity_sha256, stable_id
from active_audition.o1.noise_audit import O1_NOISE_AUDIT_DECISION_SOURCE, O1_NOISE_AUDIT_POLICY_IDENTITY, O1NoiseParentAuditRecord


ROOT = Path(__file__).resolve().parents[1]
DECISIONS = {
    "noise/free-sound/noise-free-sound-0015": ("NOT_EVALUATED", "NOT_EVALUATED", "FLAGGED", False, "MOVING_ROAD_VEHICLE_SOURCE_WITH_FREQUENT_CAR_HORNS"),
    "noise/free-sound/noise-free-sound-0030": ("NOT_EVALUATED", "NOT_EVALUATED", "FLAGGED", False, "MOVING_CAR_DRIVING_SOURCE"),
    "noise/free-sound/noise-free-sound-0032": ("PASS", "PASS", "PASS", True, ""),
    "noise/free-sound/noise-free-sound-0042": ("PASS", "PASS", "PASS", True, ""),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/logs/o1_noise_manual_audit")
    args = parser.parse_args()
    root = args.root
    source = json.loads((root / "o1_noise_audit_records.json").read_text(encoding="utf-8"))
    records = {}
    for payload in source["records"]:
        record = O1NoiseParentAuditRecord.from_payload(payload)
        records[record.parent_recording_id] = record
    if tuple(sorted(records)) != tuple(sorted(DECISIONS)):
        raise SystemExit("manual package parents do not match the four frozen O1 parents")
    finalized = []
    for parent_id in sorted(DECISIONS):
        record = records[parent_id]
        speech, reverb, compatibility, selected, reason = DECISIONS[parent_id]
        provenance = dict(record.manual_decision_provenance)
        provenance.update({"decision_source": O1_NOISE_AUDIT_DECISION_SOURCE, "decision_status": "FINALIZED_USER_DECISION"})
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
    identity = {"schema_version": "active-asr-o1-noise-audit-finalized-batch-v1", "audit_policy_identity": O1_NOISE_AUDIT_POLICY_IDENTITY, "decision_source": O1_NOISE_AUDIT_DECISION_SOURCE, "records": finalized}
    output = dict(identity)
    output["batch_id"] = stable_id("o1-noise-audit-batch", identity)
    output["batch_sha256"] = identity_sha256(identity)
    (root / "o1_noise_audit_records_finalized.json").write_bytes(_canonical(output) + b"\n")
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
    sheet["schema_version"] = "active-asr-o1-noise-manual-audit-sheet-v2"
    sheet["status"] = "FINALIZED_USER_MANUAL_AUDIT"
    sheet_path.write_bytes(_canonical(sheet) + b"\n")
    readme_path = root / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    old = "The selected scientific O1 parent rule is deterministic: after user audit, use the first two parents in candidate order for which all three dimensions are PASS. This package does not perform that selection automatically."
    new = "The original four-block O1 set requires four distinct qualified parents. Parents 0032 and 0042 are retained as the two qualified original parents. Parents 0015 and 0030 are excluded. Only a separately frozen replacement batch may provide the two missing parents; its first and second passing candidates map to the original 0015 and 0030 block slots respectively."
    if old not in text:
        raise SystemExit("expected old README selection wording was not found")
    readme_path.write_text(text.replace(old, new), encoding="utf-8")
    sums = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        sums.append("{}  {}".format(_sha(path), path.relative_to(root)))
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    tar_path = root.parent / "o1_noise_manual_audit_finalized.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    print(json.dumps({"root": str(root.resolve()), "records": [{"parent_recording_id": item["parent_recording_id"], "audit_record_id": item["audit_record_id"], "audit_record_sha256": item["audit_record_sha256"], "selected_for_o1_scientific_use": item["selected_for_o1_scientific_use"]} for item in finalized], "tar_path": str(tar_path.resolve()), "tar_sha256": _sha(tar_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
