#!/usr/bin/env python3
"""Build selected PSELD-FSD50K candidates from the Step 1A.1 crosswalk."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf


FIELDS = [
    "source_clip_id", "base_clip_id", "canonical_class_id", "canonical_class",
    "temporal_type", "source_dataset", "source_label", "mapping_type",
    "audioset_mid", "pseld_class_id", "original_id", "original_dataset_split",
    "raw_relpath", "raw_sha256", "original_sample_rate", "original_channels",
    "original_duration_sec", "license_raw", "license_status", "provenance_source",
    "pretrain_seen_status", "source_asset_root_id", "resource_status",
    "missing_waveform", "metadata_mismatch", "duplicate_id", "inventory_issue",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crosswalk_entries(evidence_path):
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    crosswalk = evidence["official_and_local_evidence"]["PSELD-selected FSD50K"]
    entries = []
    for item in crosswalk["repaired_crosswalk"]["canonical_to_pseld_to_mid_to_fsd_crosswalk"]:
        for label in item["labels"]:
            tsv = label.get("source_tsv")
            if not tsv:
                continue
            entries.append({
                "canonical_class_id": next(
                    row["canonical_class_id"] for row in _mapping_rows(evidence_path)
                    if row["canonical_class"] == item["canonical_class"]
                ),
                "canonical_class": item["canonical_class"],
                "source_label": label["label"],
                "mapping_type": "EXACT" if label["role"] == "primary_mapping" and item["mapping_status"] == "EXACT" else "SEMANTIC_STRONG",
                "audioset_mid": label["mid"],
                "pseld_class_id": str(label["pseld_class_id"]),
                "tsv_path": tsv["path"],
                "tsv_sha256": tsv["sha256"],
                "role": label["role"],
            })
    return entries


def _mapping_rows(evidence_path):
    mapping_path = evidence_path.parent / "source_class_mapping_audit.csv"
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_info(metadata_root):
    info = {}
    for name in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
        path = metadata_root / name
        if path.is_file():
            info.update(json.loads(path.read_text(encoding="utf-8")))
    return info


def build_candidates(evidence_path, tsv_root, metadata_root, audio_root,
                     asset_root, output_path, source_asset_root_id="clsdoa_v1"):
    entries = crosswalk_entries(evidence_path)
    info = load_info(metadata_root)
    by_tsv = {}
    for entry in entries:
        by_tsv.setdefault(entry["tsv_path"], []).append(entry)

    audio_index = {}
    if audio_root.is_dir():
        for path in audio_root.rglob("*.wav"):
            audio_index.setdefault(path.stem, []).append(path)

    rows = []
    for tsv_path, tsv_entries in sorted(by_tsv.items()):
        path = tsv_root / tsv_path
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            clip_id, tsv_duration, _, split = fields[:4]
            for entry in tsv_entries:
                matches = audio_index.get(clip_id, [])
                audio_path = matches[0] if len(matches) == 1 else None
                missing_waveform = not matches
                duplicate_id = len(matches) > 1
                metadata_mismatch = False
                row = {
                    "source_clip_id": "fsd50k:{}:clsdoa_v1_srcprep_v1".format(clip_id),
                    "base_clip_id": "fsd50k:{}".format(clip_id),
                    "canonical_class_id": entry["canonical_class_id"],
                    "canonical_class": entry["canonical_class"],
                    "temporal_type": "persistent" if entry["canonical_class"] in {"mechanical_fan", "microwave_oven", "printer", "vacuum_cleaner"} else "transient",
                    "source_dataset": "PSELD-selected FSD50K",
                    "source_label": entry["source_label"],
                    "mapping_type": entry["mapping_type"],
                    "audioset_mid": entry["audioset_mid"],
                    "pseld_class_id": entry["pseld_class_id"],
                    "original_id": clip_id,
                    "original_dataset_split": split,
                    "raw_relpath": audio_path.relative_to(asset_root).as_posix() if audio_path else "",
                    "raw_sha256": sha256_file(audio_path) if audio_path else "",
                    "original_sample_rate": "",
                    "original_channels": "",
                    "original_duration_sec": tsv_duration,
                    "license_raw": info.get(clip_id, {}).get("license", ""),
                    "license_status": "DATASET_LEVEL_VERIFIED" if info.get(clip_id, {}).get("license") else "UNKNOWN",
                    "provenance_source": "PSELDNets 170-class crosswalk; SELD-Data-Generator {}; FSD50K clip {}".format(tsv_path, clip_id),
                    "pretrain_seen_status": "likely_yes",
                    "source_asset_root_id": source_asset_root_id,
                    "resource_status": "PRESENT" if audio_path else "MISSING_WAVEFORM",
                    "missing_waveform": str(missing_waveform).lower(),
                    "metadata_mismatch": "false",
                    "duplicate_id": str(duplicate_id).lower(),
                    "inventory_issue": "|".join(filter(None, [
                        "missing_waveform" if missing_waveform else "",
                        "duplicate_id" if duplicate_id else "",
                    ])),
                }
                if audio_path:
                    metadata = sf.info(str(audio_path))
                    row["original_sample_rate"] = str(metadata.samplerate)
                    row["original_channels"] = str(metadata.channels)
                    actual_duration = metadata.frames / metadata.samplerate
                    row["original_duration_sec"] = "{:.9f}".format(actual_duration)
                    try:
                        metadata_mismatch = abs(actual_duration - float(tsv_duration)) > 0.05
                    except ValueError:
                        metadata_mismatch = True
                    row["metadata_mismatch"] = str(metadata_mismatch).lower()
                    if metadata_mismatch:
                        row["inventory_issue"] = "metadata_mismatch"
                rows.append(row)

    rows.sort(key=lambda row: (int(row["canonical_class_id"]), row["canonical_class"], row["original_id"], row["source_label"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--tsv-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_candidates(
        args.evidence, args.tsv_root, args.metadata_root, args.audio_root,
        args.asset_root, args.output,
    )
    print("candidate_rows={}".format(len(rows)))
    print("resolved_waveforms={}".format(sum(row["resource_status"] == "PRESENT" for row in rows)))
    print("missing_waveforms={}".format(sum(row["resource_status"] == "MISSING_WAVEFORM" for row in rows)))


if __name__ == "__main__":
    main()
