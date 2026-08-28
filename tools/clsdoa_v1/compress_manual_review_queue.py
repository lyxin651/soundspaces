#!/usr/bin/env python3
"""Compress the broad Step 2A queue into a deterministic pilot shortlist.

This tool reads existing automated-QC outputs only.  It never edits candidate
inventories, QC outputs, raw audio, or canonical-preparation artifacts.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


QUEUE_FIELDS = [
    "review_order", "canonical_class", "source_dataset", "source_label",
    "original_id", "base_clip_id", "identity_key", "audio_path",
    "duration_sec", "active_duration_estimate_sec", "auto_qc_status",
    "auto_qc_reasons", "mapping_type", "license_status",
    "pool_candidate_role", "preview_start_sec", "preview_end_sec",
    "preview_audio_path", "manual_decision", "manual_reason",
    "manual_notes", "manual_crop_override_candidate",
]

SOURCE_PRIORITY = {
    "coughing": {"ESC-50": 0},
    "laughing": {"ESC-50": 0},
    "keyboard_typing": {"ESC-50": 0},
    "vacuum_cleaner": {
        "ESC-50": 0, "DESED isolated foreground": 1,
    },
    "clock_alarm": {
        "ESC-50": 0, "PSELD-selected FSD50K": 1,
        "DESED isolated foreground": 2,
    },
    "speech": {
        "DESED isolated foreground": 0, "PSELD-selected FSD50K": 1,
    },
    "running_water": {
        "DESED isolated foreground": 0, "PSELD-selected FSD50K": 1,
    },
    "frying": {
        "DESED isolated foreground": 0, "PSELD-selected FSD50K": 1,
    },
    "dishes": {
        "DESED isolated foreground": 0, "PSELD-selected FSD50K": 1,
    },
    "mechanical_fan": {"PSELD-selected FSD50K": 0},
    "microwave_oven": {"PSELD-selected FSD50K": 0},
    "printer": {"PSELD-selected FSD50K": 0},
}

PERSISTENT_CLASSES = {"vacuum_cleaner", "mechanical_fan", "microwave_oven", "printer"}


def _float(value, default=0.0):
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _identity_key(row):
    """Use one namespace per source and collapse DESED split duplicates."""
    dataset = row.get("source_dataset", "")
    original = row.get("original_id") or row.get("source_clip_id") or row.get("base_clip_id")
    if dataset == "DESED isolated foreground":
        return "DESED:{}".format(original)
    base = row.get("base_clip_id") or original
    return "{}:{}".format(dataset, base)


def _source_rank(row):
    return SOURCE_PRIORITY.get(row.get("canonical_class", {}), {}).get(
        row.get("source_dataset", ""), 99
    )


def _provenance_key(row):
    license_rank = {
        "PER_RECORDING_METADATA": 0,
        "DATASET_LEVEL_VERIFIED": 1,
    }.get(row.get("license_status", ""), 2)
    return (
        license_rank,
        0 if row.get("provenance_source") else 1,
        0 if row.get("raw_sha256") else 1,
    )


def _selection_key(row):
    mapping_rank = {"EXACT": 0, "SEMANTIC_STRONG": 1}.get(
        row.get("mapping_type", ""), 2
    )
    qc_rank = {"AUTO_PASS": 0, "AUTO_FLAG": 1, "AUTO_REJECT": 2}.get(
        row.get("auto_qc_status", ""), 3
    )
    activity = _float(row.get("active_duration_estimate_sec"))
    activity_rank = -activity if row.get("canonical_class") in PERSISTENT_CLASSES else 0.0
    return (
        mapping_rank,
        _source_rank(row),
        qc_rank,
        _provenance_key(row),
        activity_rank,
        row.get("identity_key", ""),
        row.get("source_clip_id", ""),
        row.get("raw_relpath", ""),
    )


def load_candidate_rows(qc_paths, asset_root):
    """Load PRESENT/listenable rows without changing the source QC files."""
    rows = []
    for path in qc_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("resource_status", "PRESENT") != "PRESENT":
                    continue
                audio = asset_root / row.get("raw_relpath", "")
                if not audio.is_file():
                    continue
                item = dict(row)
                item["audio_path"] = str(audio)
                item["identity_key"] = _identity_key(item)
                rows.append(item)
    return rows


def _deduplicate(rows):
    by_identity = {}
    for row in rows:
        current = by_identity.get(row["identity_key"])
        if current is None or _selection_key(row) < _selection_key(current):
            by_identity[row["identity_key"]] = row
    return list(by_identity.values())


def _preview_window(row):
    duration = _float(row.get("duration_sec") or row.get("original_duration_sec"))
    if duration > 5.0:
        start = (duration - 5.0) / 2.0
        end = start + 5.0
    else:
        start, end = 0.0, max(duration, 0.0)
    return "{:.6f}".format(start), "{:.6f}".format(end)


def _queue_row(row, role, order):
    start, end = _preview_window(row)
    return {
        "review_order": str(order),
        "canonical_class": row.get("canonical_class", ""),
        "source_dataset": row.get("source_dataset", ""),
        "source_label": row.get("source_label", ""),
        "original_id": row.get("original_id", ""),
        "base_clip_id": row.get("base_clip_id", ""),
        "identity_key": row["identity_key"],
        "audio_path": row["audio_path"],
        "duration_sec": row.get("duration_sec", row.get("original_duration_sec", "")),
        "active_duration_estimate_sec": row.get("active_duration_estimate_sec", ""),
        "auto_qc_status": row.get("auto_qc_status", ""),
        "auto_qc_reasons": row.get("auto_qc_reason", ""),
        "mapping_type": row.get("mapping_type", ""),
        "license_status": row.get("license_status", ""),
        "pool_candidate_role": role,
        "preview_start_sec": start,
        "preview_end_sec": end,
        "preview_audio_path": row["audio_path"],
        "manual_decision": "",
        "manual_reason": "",
        "manual_notes": "",
        "manual_crop_override_candidate": "false",
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_markdown(summary, path):
    lines = [
        "# ClassDOA V1 Step 2A.2 Queue Compression",
        "",
        "The prior queue carried every remaining semantic/flag candidate into `SEMANTIC_REVIEW`. "
        "This pilot queue keeps the full inventory untouched and selects one deterministic "
        "candidate per independent identity, with 40 TARGET and 12 RESERVE per class where available.",
        "",
        "## Result",
        "",
        "- Candidate rows read: `{}`; identities before/after dedup: `{}` / `{}`.".format(
            summary["candidate_rows_read"], summary["identity_count_before_dedup"],
            summary["identity_count_after_dedup"]),
        "- TARGET: `{}`; RESERVE: `{}`; immediate human review: `{}`.".format(
            summary["target_total"], summary["reserve_total"], summary["immediate_human_review_total"]),
        "- Unselected rows remain `FORMAL_CANDIDATE_UNREVIEWED`: `{}` rows (`{}` identities).".format(
            summary["unreviewed_candidate_count"], summary["unreviewed_identity_count"]),
        "- Canonical preparation: `NOT EXECUTED`; Step 2A status: `MANUAL_QC_REQUIRED`.",
        "",
        "| canonical class | TARGET | RESERVE | available identities | status |",
        "|---|---:|---:|---:|---|",
    ]
    for item in summary["per_class"]:
        lines.append("| {canonical_class} | {target} | {reserve} | {available_identities} | {status} |".format(**item))
    lines.extend(["", "## TARGET distributions", ""])
    for label, values in (
        ("AUTO QC", summary["target_auto_qc"]),
        ("Mapping", summary["target_mapping"]),
        ("Source dataset", summary["target_source_dataset"]),
    ):
        lines.append("- **{}**: {}".format(label, ", ".join("{}={}".format(k, values[k]) for k in sorted(values))))
    lines.extend([
        "",
        "All TARGET `AUTO_FLAG` reasons and `SEMANTIC_STRONG` mapping markers are retained "
        "and require 100% listening review. RESERVE rows are prepared but are not part of "
        "the immediate listening workload. Full audio remains at `audio_path`; the centered "
        "5-second window is described by `preview_start_sec`/`preview_end_sec` and "
        "`preview_audio_path`.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_compressed_queue(qc_paths, asset_root, output_path, summary_json=None,
                           summary_md=None, target_per_class=40,
                           reserve_per_class=12):
    rows = load_candidate_rows(qc_paths, asset_root)
    unique_rows = _deduplicate(rows)
    by_class = defaultdict(list)
    for row in unique_rows:
        by_class[row.get("canonical_class", "")].append(row)

    selected = []
    per_class = []
    for canonical_class in sorted(by_class):
        candidates = sorted(by_class[canonical_class], key=_selection_key)
        target = candidates[:target_per_class]
        reserve = candidates[target_per_class:target_per_class + reserve_per_class]
        selected.extend((row, "TARGET") for row in target)
        selected.extend((row, "RESERVE") for row in reserve)
        per_class.append({
            "canonical_class": canonical_class,
            "available_identities": len(candidates),
            "target": len(target),
            "reserve": len(reserve),
            "status": "OK" if len(target) >= target_per_class else "BELOW_TARGET_IDENTITY_AVAILABILITY",
        })

    selected.sort(key=lambda item: (
        item[0].get("canonical_class", ""),
        0 if item[1] == "TARGET" else 1,
        _selection_key(item[0]),
    ))
    output = [_queue_row(row, role, order) for order, (row, role) in enumerate(selected, 1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)

    targets = [row for row in output if row["pool_candidate_role"] == "TARGET"]
    summary = {
        "schema_version": "clsdoa_v1_queue_compression_v1",
        "status": "STEP 2A MANUAL_QC_REQUIRED",
        "candidate_rows_read": len(rows),
        # Before deduplication each listenable candidate row occupies one
        # identity entry; after deduplication one entry remains per identity.
        "identity_count_before_dedup": len(rows),
        "identity_count_after_dedup": len(unique_rows),
        "target_total": sum(row["pool_candidate_role"] == "TARGET" for row in output),
        "reserve_total": sum(row["pool_candidate_role"] == "RESERVE" for row in output),
        "immediate_human_review_total": len(targets),
        "unreviewed_candidate_count": len(rows) - len(output),
        "unreviewed_identity_count": len(unique_rows) - len(output),
        "unselected_status": "FORMAL_CANDIDATE_UNREVIEWED",
        "per_class": per_class,
        "target_auto_qc": dict(Counter(row["auto_qc_status"] for row in targets)),
        "target_mapping": dict(Counter(row["mapping_type"] for row in targets)),
        "target_source_dataset": dict(Counter(row["source_dataset"] for row in targets)),
        "target_semantic_or_flag_count": sum(
            row["mapping_type"] == "SEMANTIC_STRONG" or row["auto_qc_status"] == "AUTO_FLAG"
            for row in targets),
        "queue_path": str(output_path),
        "queue_sha256": _sha256(output_path),
        "input_qc_paths": [str(path) for path in qc_paths],
        "selection": {
            "target_per_class": target_per_class,
            "reserve_per_class": reserve_per_class,
            "identity_deduplication": "global source-namespaced original/base identity",
            "priority": [
                "EXACT over SEMANTIC_STRONG", "source-role priority within mapping tier",
                "AUTO_PASS over AUTO_FLAG", "provenance/license completeness",
                "persistent/repetitive effective activity", "stable source identity tie-break",
            ],
        },
        "constraints": {
            "raw_audio_modified": False,
            "candidate_inventory_modified": False,
            "automated_qc_modified": False,
            "canonical_preparation_executed": False,
        },
    }
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary_md:
        _write_markdown(summary, summary_md)
    return output, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc", type=Path, nargs="+", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--target-per-class", type=int, default=40)
    parser.add_argument("--reserve-per-class", type=int, default=12)
    args = parser.parse_args()
    _, summary = build_compressed_queue(
        args.qc, args.asset_root, args.output, args.summary_json, args.summary_md,
        args.target_per_class, args.reserve_per_class,
    )
    print("TARGET total={}".format(summary["target_total"]))
    print("RESERVE total={}".format(summary["reserve_total"]))
    print("immediate_human_review_total={}".format(summary["immediate_human_review_total"]))
    print("identity_before_after={}/{}".format(
        summary["identity_count_before_dedup"], summary["identity_count_after_dedup"]))
    for item in summary["per_class"]:
        print("{canonical_class} TARGET={target} RESERVE={reserve} available={available_identities}".format(**item))


if __name__ == "__main__":
    main()
