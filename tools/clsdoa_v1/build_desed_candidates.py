#!/usr/bin/env python3
"""Build candidates from the official DESED isolated foreground soundbank."""

import argparse
import csv
import hashlib
from pathlib import Path

import soundfile as sf


CLASS_MAP = {
    "Speech": ("speech", "EXACT", "5"),
    "Running_water": ("running_water", "EXACT", "6"),
    "Frying": ("frying", "EXACT", "7"),
    "Dishes": ("dishes", "EXACT", "10"),
    "Vacuum_cleaner": ("vacuum_cleaner", "EXACT", "3"),
    "Alarm_bell_ringing": ("clock_alarm", "SEMANTIC_STRONG", "4"),
}

FIELDS = [
    "source_clip_id", "base_clip_id", "canonical_class_id", "canonical_class",
    "temporal_type", "source_dataset", "source_label", "mapping_type",
    "original_id", "original_dataset_split", "raw_relpath", "raw_sha256",
    "original_sample_rate", "original_channels", "original_duration_sec",
    "license_raw", "license_status", "provenance_source", "pretrain_seen_status",
    "source_asset_root_id", "resource_status",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_licenses(extracted_root):
    licenses = {}
    for split_name, filename in (("train", "license_training.tsv"),
                                 ("eval", "license_eval.tsv")):
        path = extracted_root / filename
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                license_path = row["filename"].replace("\\", "/")
                marker = "/training/" if split_name == "train" else "/"
                relative = license_path.split(marker, 1)[-1]
                if split_name == "train":
                    relative = "train/" + relative
                elif not relative.startswith("soundbank/"):
                    relative = "soundbank/" + relative
                if split_name == "eval":
                    relative = "eval/" + relative
                licenses[(split_name, relative)] = row
    return licenses


def _label_from_path(path):
    parts = path.parts
    try:
        index = parts.index("foreground")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def build_candidates(extracted_root, asset_root, output_path,
                     source_asset_root_id="clsdoa_v1"):
    licenses = load_licenses(extracted_root)
    audio_root = extracted_root / "audio"
    rows = []
    excluded_extraction_artifacts = 0
    for path in sorted(audio_root.rglob("*.wav")):
        if path.name.startswith("._"):
            excluded_extraction_artifacts += 1
            continue
        relative_audio = path.relative_to(extracted_root).as_posix()
        if "/soundbank/foreground/" not in "/" + relative_audio:
            continue
        label = _label_from_path(path)
        if label not in CLASS_MAP:
            continue
        canonical_class, mapping_type, canonical_id = CLASS_MAP[label]
        split = "train" if relative_audio.startswith("audio/train/") else "eval"
        license_key = (split, relative_audio[len("audio/"):])
        license_row = licenses.get(license_key, {})
        info = sf.info(str(path))
        freescape_id = license_row.get("id", path.stem.rsplit("_", 1)[0])
        rows.append({
            "source_clip_id": "desed:{}:{}:clsdoa_v1_srcprep_v1".format(split, path.stem),
            "base_clip_id": "desed:{}:{}".format(split, freescape_id),
            "canonical_class_id": canonical_id,
            "canonical_class": canonical_class,
            "temporal_type": "persistent" if canonical_class == "vacuum_cleaner" else "transient",
            "source_dataset": "DESED isolated foreground",
            "source_label": label,
            "mapping_type": mapping_type,
            "original_id": freescape_id,
            "original_dataset_split": split,
            "raw_relpath": path.relative_to(asset_root).as_posix(),
            "raw_sha256": sha256_file(path),
            "original_sample_rate": str(info.samplerate),
            "original_channels": str(info.channels),
            "original_duration_sec": "{:.9f}".format(info.frames / info.samplerate),
            "license_raw": license_row.get("license", ""),
            "license_status": "PER_RECORDING_METADATA" if license_row.get("license") else "UNKNOWN",
            "provenance_source": "DESED_synth_soundbank.tar.gz; {} ; official {} license TSV".format(
                relative_audio, split
            ),
            "pretrain_seen_status": "unknown",
            "source_asset_root_id": source_asset_root_id,
            "resource_status": "PRESENT",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, excluded_extraction_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, excluded = build_candidates(args.extracted_root, args.asset_root, args.output)
    print("candidate_rows={}".format(len(rows)))
    print("independent_ids={}".format(len({row['base_clip_id'] for row in rows})))
    print("excluded_extraction_artifacts={}".format(excluded))


if __name__ == "__main__":
    main()
