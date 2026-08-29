#!/usr/bin/env python3
"""Build the frozen deterministic per-class Pilot source split."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_VERSION = "clsdoa_source_split_v2_stratified"
DEFAULT_SALT = "clsdoa_v1_source_split_20260828"
FIELDS = [
    "canonical_class", "base_clip_id", "identity_key", "split",
    "split_sort_key", "split_version",
]


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _quota_counts(count, train_minimum=20, val_minimum=4, test_minimum=4):
    """Return integer quotas close to 70/15/15 with hard minima."""
    val = max(val_minimum, int(math.floor(count * 0.15 + 0.5)))
    test = max(test_minimum, int(math.floor(count * 0.15 + 0.5)))
    train = count - val - test
    while train < train_minimum:
        if val > val_minimum and val >= test:
            val -= 1
        elif test > test_minimum:
            test -= 1
        else:
            raise ValueError("count {} cannot satisfy split minima".format(count))
        train = count - val - test
    return train, val, test


def _sort_key(base_clip_id, salt):
    return hashlib.sha256(
        (salt + "|" + base_clip_id).encode("utf-8")
    ).hexdigest()


def build_split(membership_rows, salt=DEFAULT_SALT):
    accepted = []
    for row in membership_rows:
        if row.get("manual_decision") != "ACCEPT":
            raise ValueError("split input contains non-ACCEPT membership")
        if not row.get("canonical_class") or not row.get("base_clip_id"):
            raise ValueError("split input has missing class/base_clip_id")
        if not row.get("identity_key"):
            raise ValueError("split input has missing identity_key")
        if not row.get("license_status") or not row.get("provenance_source"):
            raise ValueError("split input has unusable license/provenance")
        accepted.append(row)

    if len({row["identity_key"] for row in accepted}) != len(accepted):
        raise ValueError("accepted membership contains duplicate identities")
    if len({row["base_clip_id"] for row in accepted}) != len(accepted):
        raise ValueError("accepted membership contains duplicate base_clip_id")

    by_class = defaultdict(list)
    for row in accepted:
        item = dict(row)
        item["split_sort_key"] = _sort_key(item["base_clip_id"], salt)
        by_class[item["canonical_class"]].append(item)

    output = []
    quotas = {}
    for canonical_class in sorted(by_class):
        members = sorted(
            by_class[canonical_class],
            key=lambda row: (row["split_sort_key"], row["base_clip_id"]),
        )
        train_count, val_count, test_count = _quota_counts(len(members))
        quotas[canonical_class] = {
            "total": len(members), "train": train_count,
            "val": val_count, "test": test_count,
        }
        for index, row in enumerate(members):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            output.append({
                "canonical_class": row["canonical_class"],
                "base_clip_id": row["base_clip_id"],
                "identity_key": row["identity_key"],
                "split": split,
                "split_sort_key": row["split_sort_key"],
                "split_version": SPLIT_VERSION,
            })
    output.sort(key=lambda row: (row["canonical_class"], row["split"], row["base_clip_id"]))
    return output, quotas


def audit_split(rows, quotas, salt=DEFAULT_SALT):
    by_class = defaultdict(list)
    by_base = defaultdict(set)
    for row in rows:
        by_class[row["canonical_class"]].append(row)
        by_base[row["base_clip_id"]].add(row["split"])
    overlap = {
        split_a + "_vs_" + split_b: sorted(
            {row["base_clip_id"] for row in rows if row["split"] == split_a}
            & {row["base_clip_id"] for row in rows if row["split"] == split_b}
        )
        for split_a, split_b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    per_class = {}
    for canonical_class in sorted(by_class):
        counts = Counter(row["split"] for row in by_class[canonical_class])
        per_class[canonical_class] = {
            "total": len(by_class[canonical_class]),
            "train": counts["train"], "val": counts["val"], "test": counts["test"],
            "quota": quotas[canonical_class],
            "minimum_status": "PASS" if (
                counts["train"] >= 20 and counts["val"] >= 4 and counts["test"] >= 4
            ) else "FAIL",
        }
    return {
        "schema_version": "clsdoa_v1_stratified_source_split_audit_v1",
        "split_version": SPLIT_VERSION,
        "split_salt": salt,
        "method": "per-class SHA256(salt + '|' + base_clip_id) lexicographic sort",
        "quota_policy": "nearest integer 70/15/15 with train>=20, val>=4, test>=4",
        "rows": len(rows),
        "unique_base_clip_id": len(by_base),
        "base_clip_multi_split_groups": sorted(
            key for key, values in by_base.items() if len(values) != 1
        ),
        "split_overlap": overlap,
        "per_class": per_class,
        "status": "PASS" if (
            len(rows) == len(by_base)
            and not any(overlap.values())
            and not any(item["minimum_status"] != "PASS" for item in per_class.values())
        ) else "FAIL",
    }


def run(membership_path, output_path, audit_path, salt=DEFAULT_SALT):
    rows = _read_csv(membership_path)
    split_rows, quotas = build_split(rows, salt=salt)
    audit = audit_split(split_rows, quotas, salt=salt)
    if audit["status"] != "PASS":
        raise ValueError("stratified split gate failed")
    _write_csv(output_path, split_rows)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args()
    audit = run(args.membership, args.output, args.audit, salt=args.salt)
    for name, item in audit["per_class"].items():
        print("{} {} / {} / {}".format(name, item["train"], item["val"], item["test"]))
    print("status={}".format(audit["status"]))


if __name__ == "__main__":
    main()
