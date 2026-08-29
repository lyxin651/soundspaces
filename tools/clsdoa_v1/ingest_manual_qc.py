#!/usr/bin/env python3
"""Ingest completed pilot manual QC and audit the pre-canonical membership.

Blank TARGET decisions are accepted because the listening pass is complete.
Blank RESERVE decisions are never changed.  If a class is below the pilot
minimum, this tool emits a small incremental queue and does not prepare audio.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.clsdoa_v1.compress_manual_review_queue import (
    _identity_key,
    _selection_key,
)


USABLE_LICENSE_STATUSES = {
    "PER_RECORDING_METADATA",
    "DATASET_LEVEL_VERIFIED",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_qc_rows(paths, asset_root):
    by_audio = {}
    for path in paths:
        _, rows = _read_csv(path)
        for row in rows:
            raw = row.get("raw_relpath", "")
            if not raw:
                continue
            absolute = str(asset_root / raw)
            item = dict(row)
            item["identity_key"] = _identity_key(item)
            by_audio[absolute] = item
    return by_audio


def _group_values(rows, key):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            groups[value].append(row)
    return {key: value for key, value in groups.items() if len(value) > 1}


def _membership_row(row, qc):
    return {
        "review_order": row.get("review_order", ""),
        "canonical_class": row.get("canonical_class", ""),
        "source_dataset": row.get("source_dataset", ""),
        "source_label": row.get("source_label", ""),
        "original_id": row.get("original_id", ""),
        "base_clip_id": row.get("base_clip_id", ""),
        "identity_key": row.get("identity_key", ""),
        "audio_path": row.get("audio_path", ""),
        "manual_decision": row.get("manual_decision", ""),
        "manual_reason": row.get("manual_reason", ""),
        "manual_notes": row.get("manual_notes", ""),
        "auto_qc_status": row.get("auto_qc_status", ""),
        "auto_qc_reasons": row.get("auto_qc_reasons", ""),
        "mapping_type": row.get("mapping_type", ""),
        "license_status": row.get("license_status", ""),
        "license_raw": qc.get("license_raw", ""),
        "provenance_source": qc.get("provenance_source", ""),
        "pretrain_seen_status": qc.get("pretrain_seen_status", "unknown") or "unknown",
        "raw_sha256": qc.get("raw_sha256", ""),
        "resource_status": qc.get("resource_status", "PRESENT"),
    }


def _incremental_row(row, order):
    item = dict(row)
    item["incremental_review_order"] = str(order)
    item["incremental_review_required"] = "true"
    item["incremental_reason"] = "class_below_30_accept_identity_minimum"
    return item


def ingest(queue_path, qc_paths, asset_root, snapshot_path, stats_path,
           membership_path, incremental_path, target_minimum=30):
    fieldnames, original_rows = _read_csv(queue_path)
    shutil.copyfile(queue_path, snapshot_path)

    before = Counter(
        (row["pool_candidate_role"], row.get("manual_decision") or "<blank>")
        for row in original_rows
    )
    rows = [dict(row) for row in original_rows]
    filled = 0
    for row in rows:
        if row["pool_candidate_role"] == "TARGET" and not row.get("manual_decision"):
            row["manual_decision"] = "ACCEPT"
            filled += 1
    _write_csv(queue_path, fieldnames, rows)

    by_audio = _load_qc_rows(qc_paths, asset_root)
    accepted = [
        row for row in rows
        if row["pool_candidate_role"] == "TARGET" and row.get("manual_decision") == "ACCEPT"
    ]
    rejected = [
        row for row in rows
        if row["pool_candidate_role"] == "TARGET" and row.get("manual_decision") == "REJECT"
    ]
    accepted_membership = []
    missing_metadata = []
    for row in accepted:
        qc = by_audio.get(row["audio_path"])
        if qc is None:
            missing_metadata.append(row["audio_path"])
            qc = {}
        accepted_membership.append(_membership_row(row, qc))
    membership_fields = [
        "review_order", "canonical_class", "source_dataset", "source_label",
        "original_id", "base_clip_id", "identity_key", "audio_path",
        "manual_decision", "manual_reason", "manual_notes", "auto_qc_status",
        "auto_qc_reasons", "mapping_type", "license_status", "license_raw",
        "provenance_source", "pretrain_seen_status", "raw_sha256",
        "resource_status",
    ]
    _write_csv(membership_path, membership_fields, accepted_membership)

    identity_groups = _group_values(accepted_membership, "identity_key")
    base_groups = _group_values(accepted_membership, "base_clip_id")
    raw_sha_groups = _group_values(accepted_membership, "raw_sha256")
    raw_sha_conflicts = {
        key: value for key, value in raw_sha_groups.items()
        if len({row["canonical_class"] for row in value}) > 1
        or len({row["source_dataset"] for row in value}) > 1
    }
    exact_identity_groups = _group_values(accepted_membership, "audio_path")

    per_class = []
    accepted_by_class = defaultdict(list)
    for row in accepted_membership:
        accepted_by_class[row["canonical_class"]].append(row)
    all_classes = sorted({row["canonical_class"] for row in rows})
    for canonical_class in all_classes:
        members = accepted_by_class[canonical_class]
        identity_count = len({row["identity_key"] for row in members})
        per_class.append({
            "canonical_class": canonical_class,
            "accepted_rows": len(members),
            "accepted_independent_identities": identity_count,
            "hard_minimum": target_minimum,
            "status": "PASS" if identity_count >= target_minimum else "BELOW_MINIMUM",
        })

    deficient = {
        item["canonical_class"]: item["hard_minimum"] - item["accepted_independent_identities"]
        for item in per_class if item["status"] != "PASS"
    }
    reserve_candidates = []
    for row in rows:
        if (
            row["pool_candidate_role"] == "RESERVE"
            and not row.get("manual_decision")
            and row["canonical_class"] in deficient
        ):
            candidate = dict(row)
            candidate["identity_key"] = candidate.get("identity_key") or _identity_key(candidate)
            qc = by_audio.get(candidate["audio_path"], {})
            candidate.update({
                key: value for key, value in qc.items()
                if key not in candidate or not candidate.get(key)
            })
            reserve_candidates.append((candidate, row))
    reserve_candidates.sort(key=lambda item: (
        item[0]["canonical_class"],
        _selection_key(item[0]),
    ))
    incremental = []
    for canonical_class in sorted(deficient):
        need = deficient[canonical_class]
        candidates = [
            item for item in reserve_candidates
            if item[0]["canonical_class"] == canonical_class
        ]
        incremental.extend(
            _incremental_row(item[1], index + 1)
            for index, item in enumerate(candidates[:need])
        )
    if incremental:
        incremental_fields = list(fieldnames) + [
            "incremental_review_order", "incremental_review_required", "incremental_reason",
        ]
        _write_csv(incremental_path, incremental_fields, incremental)
    elif incremental_path.exists():
        incremental_path.unlink()

    license_counts = Counter(row["license_status"] or "<blank>" for row in accepted_membership)
    provenance_counts = Counter(
        "known" if row["provenance_source"] else "unknown"
        for row in accepted_membership
    )
    pretrain_counts = Counter(row["pretrain_seen_status"] for row in accepted_membership)
    stats = {
        "schema_version": "clsdoa_v1_manual_qc_ingest_v1",
        "queue_path": str(queue_path),
        "manual_snapshot_path": str(snapshot_path),
        "manual_snapshot_sha256": _sha256(snapshot_path),
        "queue_sha256_after_ingest": _sha256(queue_path),
        "before_decisions": {"{}/{}".format(key[0], key[1]): value for key, value in sorted(before.items())},
        "manual_ingest": {
            "target_blank_filled_accept": filled,
            "target_explicit_reject_preserved": len(rejected),
            "reserve_decisions_changed": 0,
            "formal_candidate_decisions_changed": 0,
        },
        "target_accept": len(accepted),
        "target_reject": len(rejected),
        "per_class": per_class,
        "accepted_identity_audit": {
            "accepted_rows": len(accepted_membership),
            "identity_count": len({row["identity_key"] for row in accepted_membership}),
            "identity_duplicate_groups": len(identity_groups),
            "base_clip_duplicate_groups": len(base_groups),
            "raw_sha_duplicate_groups": len(raw_sha_groups),
            "raw_sha_conflict_groups": len(raw_sha_conflicts),
            "exact_audio_path_duplicate_groups": len(exact_identity_groups),
            "missing_qc_metadata_rows": len(missing_metadata),
        },
        "raw_sha_duplicate_details": {
            key: [row["identity_key"] for row in value]
            for key, value in sorted(raw_sha_groups.items())
        },
        "license_status": dict(license_counts),
        "provenance_status": dict(provenance_counts),
        "pretrain_seen_status": dict(pretrain_counts),
        "license_provenance_gate": {
            "usable_license_rows": sum(row["license_status"] in USABLE_LICENSE_STATUSES for row in accepted_membership),
            "usable_provenance_rows": sum(bool(row["provenance_source"]) for row in accepted_membership),
            "unknown_pretrain_rows_allowed": sum(row["pretrain_seen_status"] == "unknown" for row in accepted_membership),
            "unknown_pretrain_is_not_rewritten_to_no": True,
        },
        "incremental_manual_review": {
            "required": bool(incremental),
            "path": str(incremental_path) if incremental else None,
            "rows": len(incremental),
            "needed_by_class": deficient,
            "reserve_auto_accepted": False,
        },
        "split": {
            "status": "NOT_RUN_BELOW_ACCEPT_MINIMUM" if deficient else "PENDING",
            "version": "clsdoa_source_split_v1",
        },
        "canonical_preparation": {
            "status": "NOT_EXECUTED",
            "reason": "manual incremental QC is required before the hard minimum gate",
        },
        "status": "STEP 2A INCREMENTAL MANUAL QC REQUIRED" if incremental else "READY_FOR_SPLIT_AND_PREPARATION",
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--qc", type=Path, nargs="+", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--incremental", type=Path, required=True)
    parser.add_argument("--target-minimum", type=int, default=30)
    args = parser.parse_args()
    stats = ingest(
        args.queue, args.qc, args.asset_root, args.snapshot, args.stats,
        args.membership, args.incremental, args.target_minimum,
    )
    print("target_accept={}".format(stats["target_accept"]))
    print("target_reject={}".format(stats["target_reject"]))
    print("status={}".format(stats["status"]))
    for item in stats["per_class"]:
        print("{canonical_class} accepted={accepted_independent_identities} status={status}".format(**item))
    print("incremental_rows={}".format(stats["incremental_manual_review"]["rows"]))


if __name__ == "__main__":
    main()
