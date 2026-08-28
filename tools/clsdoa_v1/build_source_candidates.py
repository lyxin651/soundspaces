#!/usr/bin/env python3
"""Build a Step 2A candidate inventory from the Step 1A.1 mapping audit.

This tool intentionally performs no audio transformation.  Raw files remain
outside the repository and are referenced by source_asset_root_id/raw_relpath.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import soundfile as sf


FIELDS = [
    "canonical_class_id", "canonical_class", "temporal_type", "source_dataset",
    "source_label", "mapping_type", "original_id", "original_dataset_split",
    "raw_relpath", "raw_sha256", "original_sample_rate", "original_channels",
    "original_duration_sec", "license_raw", "license_status",
    "provenance_source", "pretrain_seen_status", "source_asset_root_id",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mapping(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["source_dataset"] == "ESC-50"
        and row["mapping_status"] == "EXACT"
        and row["primary_candidate"].lower() == "true"
    ]
    if not selected:
        raise ValueError("Step 1A.1 mapping contains no exact ESC-50 primary rows")
    return {row["source_label"]: row for row in selected}


def build_candidates(mapping_path, esc_metadata_path, esc_root, asset_root,
                     output_path, source_asset_root_id="clsdoa_v1"):
    mapping = load_mapping(mapping_path)
    with esc_metadata_path.open(newline="", encoding="utf-8") as handle:
        metadata = list(csv.DictReader(handle))

    rows = []
    for item in metadata:
        mapping_row = mapping.get(item["category"])
        if mapping_row is None:
            continue
        raw_path = esc_root / "audio" / item["filename"]
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        info = sf.info(str(raw_path))
        base_clip_id = "esc50:{}".format(item["src_file"])
        source_clip_id = "{}:clsdoa_v1_srcprep_v1:{}".format(
            base_clip_id, Path(item["filename"]).stem
        )
        raw_relpath = raw_path.relative_to(asset_root).as_posix()
        rows.append({
            "canonical_class_id": mapping_row["canonical_class_id"],
            "canonical_class": mapping_row["canonical_class"],
            "temporal_type": "persistent" if mapping_row["canonical_class"] in {
                "vacuum_cleaner"
            } else "transient",
            "source_dataset": "ESC-50",
            "source_label": item["category"],
            "mapping_type": mapping_row["mapping_type"],
            "original_id": item["src_file"],
            "original_dataset_split": "fold_{}".format(item["fold"]),
            "raw_relpath": raw_relpath,
            "raw_sha256": sha256_file(raw_path),
            "original_sample_rate": str(info.samplerate),
            "original_channels": str(info.channels),
            "original_duration_sec": "{:.9f}".format(info.frames / info.samplerate),
            "license_raw": "CC BY-NC 3.0; dataset-level license and attribution file",
            "license_status": "DATASET_LEVEL_VERIFIED",
            "provenance_source": (
                "https://github.com/karolpiczak/ESC-50; "
                "meta/esc50.csv; src_file={} take={}"
            ).format(item["src_file"], item["take"]),
            "pretrain_seen_status": "unknown",
            "source_asset_root_id": source_asset_root_id,
            "source_clip_id": source_clip_id,
            "base_clip_id": base_clip_id,
        })

    rows.sort(key=lambda row: (int(row["canonical_class_id"]), row["raw_relpath"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_clip_id", "base_clip_id"] + FIELDS
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--esc-metadata", type=Path, required=True)
    parser.add_argument("--esc-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_candidates(
        mapping_path=args.mapping,
        esc_metadata_path=args.esc_metadata,
        esc_root=args.esc_root,
        asset_root=args.asset_root,
        output_path=args.output,
    )
    print("candidate_rows={}".format(len(rows)))


if __name__ == "__main__":
    main()
