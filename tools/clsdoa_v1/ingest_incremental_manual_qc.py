#!/usr/bin/env python3
"""Ingest the authorized dishes batch and audit the final pre-split pool."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.clsdoa_v1.compress_manual_review_queue import _identity_key, _selection_key
from tools.clsdoa_v1.ingest_manual_qc import (
    USABLE_LICENSE_STATUSES,
    _group_values,
    _load_qc_rows,
    _membership_row,
    _sha256,
    _write_csv,
)
from tools.clsdoa_v1.stratified_source_split import (
    DEFAULT_SALT,
    SPLIT_VERSION,
    audit_split,
    build_split,
)


MEMBERSHIP_FIELDS = [
    "review_order", "canonical_class", "source_dataset", "source_label",
    "original_id", "base_clip_id", "identity_key", "audio_path",
    "pool_candidate_role", "manual_decision", "manual_reason", "manual_notes",
    "auto_qc_status", "auto_qc_reasons", "mapping_type", "license_status",
    "license_raw", "provenance_source", "pretrain_seen_status", "raw_sha256",
    "resource_status",
]


def _copy_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_incremental(path, fieldnames, rows):
    fields = list(fieldnames)
    for field in ("incremental_review_order", "incremental_review_required", "incremental_reason"):
        if field not in fields:
            fields.append(field)
    _write_csv(path, fields, rows)


def ingest_incremental(queue_path, incremental_path, qc_paths, asset_root,
                       snapshot_path, membership_path, stats_path,
                       split_report_path, split_salt):
    queue_fields, queue_rows = None, None
    with queue_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        queue_fields = reader.fieldnames
        queue_rows = list(reader)
    incremental_rows = _copy_rows(incremental_path)
    if not incremental_rows:
        raise ValueError("incremental queue is empty")
    if any(row.get("manual_decision") not in ("", "ACCEPT") for row in incremental_rows):
        raise ValueError("incremental queue contains a non-ACCEPT decision")

    if not snapshot_path.exists():
        _write_csv(snapshot_path, queue_fields, queue_rows)

    authorized_ids = {row["identity_key"] for row in incremental_rows}
    if len(authorized_ids) != len(incremental_rows):
        raise ValueError("incremental queue contains duplicate identities")
    queue_by_identity = {row["identity_key"]: row for row in queue_rows}
    if authorized_ids - set(queue_by_identity):
        raise ValueError("incremental queue identity is absent from main queue")

    for row in incremental_rows:
        row["manual_decision"] = "ACCEPT"
        row["manual_notes"] = "human-authorized batch acceptance: dishes incremental review"
        row["incremental_review_required"] = "false"
        main = queue_by_identity[row["identity_key"]]
        if main["pool_candidate_role"] != "RESERVE":
            raise ValueError("incremental identity is not a RESERVE row")
        if main.get("manual_decision") == "REJECT":
            raise ValueError("would overwrite explicit RESERVE REJECT")
        main["manual_decision"] = "ACCEPT"
        main["manual_notes"] = "human-authorized batch acceptance: dishes incremental review"

    _write_csv(queue_path, queue_fields, queue_rows)
    _write_incremental(incremental_path, list(incremental_rows[0].keys()), incremental_rows)

    qc_by_audio = _load_qc_rows(qc_paths, asset_root)
    accepted = [row for row in queue_rows if row.get("manual_decision") == "ACCEPT"]
    accepted_membership = []
    missing_qc = []
    for row in accepted:
        qc = qc_by_audio.get(row["audio_path"], {})
        if not qc:
            missing_qc.append(row["audio_path"])
        accepted_membership.append(_membership_row(row, qc))
    _write_csv(membership_path, MEMBERSHIP_FIELDS, accepted_membership)

    identity_groups = _group_values(accepted_membership, "identity_key")
    base_groups = _group_values(accepted_membership, "base_clip_id")
    raw_sha_groups = _group_values(accepted_membership, "raw_sha256")
    audio_groups = _group_values(accepted_membership, "audio_path")
    raw_sha_conflicts = {
        key: rows for key, rows in raw_sha_groups.items()
        if len({row["canonical_class"] for row in rows}) > 1
        or len({row["source_dataset"] for row in rows}) > 1
    }

    by_class = defaultdict(list)
    for row in accepted_membership:
        by_class[row["canonical_class"]].append(row)
    # The old global threshold buckets are retired.  The dedicated v2 splitter
    # is run only after every class reaches the accepted identity minimum.
    split_report = {
        "schema_version": "clsdoa_v1_stratified_source_split_audit_v1",
        "split_version": SPLIT_VERSION,
        "split_salt": split_salt or DEFAULT_SALT,
        "status": "DEFERRED_BELOW_ACCEPT_MINIMUM",
        "reason": "run stratified_source_split.py after all classes reach 30 ACCEPT identities",
    }
    if all(len({row["identity_key"] for row in members}) >= 30 for members in by_class.values()):
        split_rows, quotas = build_split(accepted_membership, salt=split_salt or DEFAULT_SALT)
        split_report = audit_split(split_rows, quotas, salt=split_salt or DEFAULT_SALT)
    per_class = []
    for canonical_class in sorted(by_class):
        identities = {row["identity_key"] for row in by_class[canonical_class]}
        item = split_report.get("per_class", {}).get(canonical_class, {})
        per_class.append({
            "canonical_class": canonical_class,
            "accepted_independent_identities": len(identities),
            "train": item.get("train"), "val": item.get("val"), "test": item.get("test"),
            "identity_minimum_status": "PASS" if len(identities) >= 30 else "FAIL",
            "split_minimum_status": item.get("minimum_status", "DEFERRED"),
        })
    split_deficits = {
        key: {
            "train": max(0, 20 - value["train"]),
            "val": max(0, 4 - value["val"]),
            "test": max(0, 4 - value["test"]),
        }
        for key, value in split_report.get("per_class", {}).items()
        if value.get("minimum_status") != "PASS"
    }

    split_report["per_class"] = per_class
    split_report["deficits"] = split_deficits
    split_report_path.parent.mkdir(parents=True, exist_ok=True)
    split_report_path.write_text(json.dumps(split_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    license_counts = Counter(row["license_status"] or "<blank>" for row in accepted_membership)
    provenance_counts = Counter("known" if row["provenance_source"] else "unknown" for row in accepted_membership)
    pretrain_counts = Counter(row["pretrain_seen_status"] for row in accepted_membership)
    stats = {
        "schema_version": "clsdoa_v1_manual_qc_final_gate_v1",
        "manual_ingest": {
            "incremental_rows_authorized": len(incremental_rows),
            "incremental_class": "dishes",
            "incremental_decision": "ACCEPT",
            "target_accept_preserved": sum(
                row["pool_candidate_role"] == "TARGET" and row["manual_decision"] == "ACCEPT"
                for row in queue_rows
            ),
            "target_reject_preserved": sum(
                row["pool_candidate_role"] == "TARGET" and row["manual_decision"] == "REJECT"
                for row in queue_rows
            ),
            "other_reserve_or_formal_auto_accepted": False,
        },
        "accepted_total": len(accepted_membership),
        "accepted_by_role": dict(Counter(row["pool_candidate_role"] for row in accepted_membership)),
        "per_class": per_class,
        "accepted_identity_audit": {
            "identity_count": len({row["identity_key"] for row in accepted_membership}),
            "identity_duplicate_groups": len(identity_groups),
            "base_clip_duplicate_groups": len(base_groups),
            "raw_sha_duplicate_groups": len(raw_sha_groups),
            "raw_sha_conflict_groups": len(raw_sha_conflicts),
            "exact_audio_path_duplicate_groups": len(audio_groups),
            "missing_qc_metadata_rows": len(missing_qc),
        },
        "license_status": dict(license_counts),
        "provenance_status": dict(provenance_counts),
        "pretrain_seen_status": dict(pretrain_counts),
        "license_provenance_gate": {
            "usable_license_rows": sum(row["license_status"] in USABLE_LICENSE_STATUSES for row in accepted_membership),
            "usable_provenance_rows": sum(bool(row["provenance_source"]) for row in accepted_membership),
            "unknown_pretrain_rows_allowed": sum(row["pretrain_seen_status"] == "unknown" for row in accepted_membership),
            "unknown_pretrain_rewritten_to_no": False,
        },
        "split_gate": split_report,
        "canonical_preparation": "NOT_EXECUTED",
        "registry": "NOT_EXECUTED",
        "freeze": "NOT_EXECUTED",
        "queue_sha256_after_incremental": _sha256(queue_path),
        "incremental_queue_sha256_after_accept": _sha256(incremental_path),
        "snapshot_path": str(snapshot_path),
        "status": "STEP 2A SPLIT GATE REQUIRED" if split_deficits else "READY_FOR_CANONICAL_PREPARATION",
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--incremental", type=Path, required=True)
    parser.add_argument("--qc", type=Path, nargs="+", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--split-salt", default="clsdoa_v1_source_split_20260828")
    args = parser.parse_args()
    stats = ingest_incremental(
        args.queue, args.incremental, args.qc, args.asset_root, args.snapshot,
        args.membership, args.stats, args.split_report, args.split_salt,
    )
    print("accepted_total={}".format(stats["accepted_total"]))
    print("status={}".format(stats["status"]))
    for item in stats["per_class"]:
        print("{canonical_class} accepted={accepted_independent_identities} split={train}/{val}/{test}".format(**item))


if __name__ == "__main__":
    main()
