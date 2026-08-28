#!/usr/bin/env python3
"""Build a deterministic, finite manual-review queue from automated QC CSVs.

The queue contains only resolvable waveforms.  Missing resources stay visible in
the candidate inventories and are never silently promoted to reviewable audio.
"""

import argparse
import csv
from pathlib import Path


QUEUE_FIELDS = [
    "review_order", "canonical_class", "source_dataset", "source_label",
    "original_id", "audio_path", "duration_sec", "active_duration_estimate_sec",
    "auto_qc_status", "auto_qc_reasons", "mapping_type", "pool_candidate_role",
    "manual_decision", "manual_reason", "manual_notes",
]


def _bool_true(value):
    return str(value).lower() == "true"


def _merge_rows(rows):
    """Collapse duplicate source-label rows for one listenable source identity."""
    merged = {}
    for row in rows:
        key = (row["source_dataset"], row["canonical_class"], row["original_id"])
        current = merged.get(key)
        if current is None:
            current = dict(row)
            merged[key] = current
            continue
        labels = {item for item in current.get("source_label", "").split("|") if item}
        labels.update(item for item in row.get("source_label", "").split("|") if item)
        current["source_label"] = "|".join(sorted(labels))
        if row.get("mapping_type") != "EXACT":
            current["mapping_type"] = "SEMANTIC_STRONG"
        reasons = {item for item in current.get("auto_qc_reason", "").split("|") if item}
        reasons.update(item for item in row.get("auto_qc_reason", "").split("|") if item)
        current["auto_qc_reason"] = "|".join(sorted(reasons))
    return list(merged.values())


def _sort_key(row):
    # Prefer exact, automatically clean, informative recordings without using
    # RMS as a proxy for semantic quality or choosing only the longest clips.
    mapping_rank = 0 if row.get("mapping_type") == "EXACT" else 1
    qc_rank = {"AUTO_PASS": 0, "AUTO_FLAG": 1, "AUTO_REJECT": 2}.get(
        row.get("auto_qc_status", ""), 3
    )
    try:
        active = -float(row.get("active_duration_estimate_sec", "0") or 0)
    except ValueError:
        active = 0.0
    return (
        mapping_rank, qc_rank, active, row.get("source_dataset", ""),
        row.get("canonical_class", ""), row.get("original_id", ""),
    )


def load_qc_rows(paths, asset_root):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("resource_status", "PRESENT") != "PRESENT":
                    continue
                raw_relpath = row.get("raw_relpath", "")
                audio = asset_root / raw_relpath
                if not audio.is_file():
                    continue
                row["audio_path"] = str(audio)
                rows.append(row)
    return _merge_rows(rows)


def build_queue(qc_paths, asset_root, output_path, target_per_class=40,
                reserve_per_class=12):
    rows = load_qc_rows(qc_paths, asset_root)
    by_class = {}
    for row in rows:
        key = (row.get("source_dataset", ""), row["canonical_class"])
        by_class.setdefault(key, []).append(row)

    queue = []
    for dataset, canonical_class in sorted(by_class):
        candidates = sorted(by_class[(dataset, canonical_class)], key=_sort_key)
        if dataset == "ESC-50":
            selected = [(row, "TARGET") for row in candidates]
            remaining = []
        else:
            target = candidates[:target_per_class]
            reserve = candidates[target_per_class:target_per_class + reserve_per_class]
            selected = ([(row, "TARGET") for row in target] +
                        [(row, "RESERVE") for row in reserve])
            selected_ids = {id(row) for row, _ in selected}
            remaining = [row for row in candidates if id(row) not in selected_ids]

        for row, role in selected:
            queue.append((row, role))
        for row in remaining:
            if (row.get("mapping_type") != "EXACT" or
                    row.get("auto_qc_status") in {"AUTO_FLAG", "AUTO_REJECT"}):
                queue.append((row, "SEMANTIC_REVIEW"))

    queue.sort(key=lambda item: (item[0].get("canonical_class", ""),
                                 item[0].get("source_dataset", ""),
                                 {"TARGET": 0, "RESERVE": 1,
                                  "SEMANTIC_REVIEW": 2}.get(item[1], 3),
                                 _sort_key(item[0])))
    output = []
    for order, (row, role) in enumerate(queue, start=1):
        output.append({
            "review_order": str(order),
            "canonical_class": row.get("canonical_class", ""),
            "source_dataset": row.get("source_dataset", ""),
            "source_label": row.get("source_label", ""),
            "original_id": row.get("original_id", ""),
            "audio_path": row["audio_path"],
            "duration_sec": row.get("duration_sec", row.get("original_duration_sec", "")),
            "active_duration_estimate_sec": row.get("active_duration_estimate_sec", ""),
            "auto_qc_status": row.get("auto_qc_status", ""),
            "auto_qc_reasons": row.get("auto_qc_reason", ""),
            "mapping_type": row.get("mapping_type", ""),
            "pool_candidate_role": role,
            "manual_decision": "",
            "manual_reason": "",
            "manual_notes": "",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc", type=Path, nargs="+", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-per-class", type=int, default=40)
    parser.add_argument("--reserve-per-class", type=int, default=12)
    args = parser.parse_args()
    rows = build_queue(args.qc, args.asset_root, args.output,
                       args.target_per_class, args.reserve_per_class)
    counts = {}
    for row in rows:
        key = (row["canonical_class"], row["pool_candidate_role"])
        counts[key] = counts.get(key, 0) + 1
    print("queue_rows={}".format(len(rows)))
    for key in sorted(counts):
        print("{}={}".format("/".join(key), counts[key]))


if __name__ == "__main__":
    main()
