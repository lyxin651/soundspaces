#!/usr/bin/env python3
"""Prepare, validate, and freeze the ClassDOA V1 Pilot source pool."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.io import wavfile
from scipy.signal import resample_poly


POOL_ID = "source_pool_pilot_001"
ALLOWED_LICENSE_STATUSES = {
    "PER_RECORDING_METADATA",
    "DATASET_LEVEL_VERIFIED",
}
REGISTRY_FIELDS = [
    "source_clip_id", "canonical_class", "source_dataset", "source_label",
    "original_id", "base_clip_id", "identity_key", "audio_path",
    "canonical_path", "canonical_relpath", "canonical_wav_sha256",
    "canonical_pcm_sha256", "raw_sha256", "split", "split_version",
    "manual_decision", "mapping_type", "auto_qc_status", "auto_qc_reasons",
    "license_status", "license_raw", "provenance_source",
    "pretrain_seen_status", "resource_status", "canonical_sample_rate_hz",
    "canonical_channels", "canonical_dtype", "canonical_duration_sec",
    "crop_policy", "crop_start_sec", "crop_end_sec", "source_offset_sec",
    "normalization_method", "active_rms_before_dbfs", "active_rms_after_dbfs",
    "peak_before", "peak_after", "gain_db", "peak_guard_limited",
    "pilot_eligible",
]


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_pcm(waveform):
    return hashlib.sha256(np.asarray(waveform, dtype="<f4").tobytes()).hexdigest()


def _dbfs(value):
    return -float("inf") if value <= 0 else 20.0 * math.log10(value)


def load_source_prep_config(config_path):
    """Load the source-prep YAML and reject drift from the frozen contract."""
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    canonical = config.get("canonical", {})
    normalization = config.get("normalization", {})
    split = config.get("split", {})
    expected = {
        "sample_rate": 24000,
        "channels": 1,
        "dtype": "float32",
        "max_duration_sec": 5.0,
        "normalization_method": "active_rms_v1",
        "target_active_rms_dbfs": -24.0,
        "peak_guard": 0.50,
        "split_version": "clsdoa_source_split_v2_stratified",
    }
    actual = {
        "sample_rate": canonical.get("sample_rate"),
        "channels": canonical.get("channels"),
        "dtype": canonical.get("dtype"),
        "max_duration_sec": canonical.get("max_duration_sec"),
        "normalization_method": normalization.get("method"),
        "target_active_rms_dbfs": normalization.get("target_active_rms_dbfs"),
        "peak_guard": normalization.get("peak_guard"),
        "split_version": split.get("version"),
    }
    if actual != expected:
        raise ValueError("source-prep YAML disagrees with the frozen contract: {}".format(actual))
    return config


def source_offset_contract(canonical_duration_sec, config):
    """Return the frozen offset rule; Step 3 supplies random offsets for shorts."""
    max_duration = float(config["canonical"]["max_duration_sec"])
    duration = float(canonical_duration_sec)
    if duration <= 0 or duration > max_duration:
        raise ValueError("canonical duration outside source-offset contract")
    if math.isclose(duration, max_duration, rel_tol=0.0, abs_tol=1e-9):
        return "fixed_zero", 0.0
    return config["canonical"]["short_source_offset_policy"], None


def _resample(mono, source_rate, target_rate):
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate == target_rate:
        return mono.astype(np.float32, copy=True)
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        mono, up=target_rate // divisor, down=source_rate // divisor,
        window=("kaiser", 5.0), padtype="constant",
    ).astype(np.float32)


def _normalize(waveform, target_active_rms, peak_guard):
    peak_before = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    if not np.isfinite(waveform).all() or peak_before <= 1e-8:
        raise ValueError("canonical waveform is empty, nonfinite, or silent")
    threshold = max(1e-4, peak_before * 0.10)
    active = waveform[np.abs(waveform) >= threshold]
    if len(active) == 0:
        raise ValueError("canonical waveform has no active samples")
    active_rms_before = float(np.sqrt(np.mean(np.square(active))))
    gain = float(target_active_rms) / active_rms_before
    limited = False
    if peak_before * gain > float(peak_guard):
        gain = float(peak_guard) / peak_before
        limited = True
    output = (waveform * np.float32(gain)).astype(np.float32)
    peak_after = float(np.max(np.abs(output)))
    active_after = output[np.abs(output) >= max(1e-4, peak_after * 0.10)]
    active_rms_after = float(np.sqrt(np.mean(np.square(active_after))))
    return output, {
        "active_rms_before_dbfs": _dbfs(active_rms_before),
        "active_rms_after_dbfs": _dbfs(active_rms_after),
        "peak_before": peak_before,
        "peak_after": peak_after,
        "gain_db": 20.0 * math.log10(gain),
        "peak_guard_limited": "true" if limited else "false",
    }


def _safe_name(value):
    return value.replace("/", "__").replace(":", "__").replace("\\", "__")


def _split_map(split_path):
    rows = _read_csv(split_path)
    result = {}
    for row in rows:
        base = row["base_clip_id"]
        if base in result:
            raise ValueError("duplicate split base_clip_id: {}".format(base))
        result[base] = row
    return result


def _ontology_classes(ontology_path):
    with Path(ontology_path).open(encoding="utf-8") as handle:
        ontology = yaml.safe_load(handle)["ontology"]["classes"]
    return {item["name"] for item in ontology}


def validate_registry_metadata(records, config, ontology_classes):
    """Validate non-audio hard gates before a pool can be finalized."""
    classes = {row.get("canonical_class") for row in records}
    if len(classes) != 12 or classes != set(ontology_classes):
        raise ValueError("canonical class set is not exactly the frozen 12-class ontology")
    if any(row.get("manual_decision") != "ACCEPT" for row in records):
        raise ValueError("pilot registry contains a non-ACCEPT source")
    if any(row.get("license_status") not in ALLOWED_LICENSE_STATUSES for row in records):
        raise ValueError("pilot registry contains an unusable license status")
    if any(not row.get("provenance_source") for row in records):
        raise ValueError("pilot registry contains a source without provenance")
    if any(row.get("pilot_eligible") != "true" for row in records):
        raise ValueError("pilot registry contains a non-eligible source")
    peak_guard = float(config["normalization"]["peak_guard"])
    if any(
        not np.isfinite(float(row.get("peak_after", "nan")))
        or float(row["peak_after"]) > peak_guard + 1e-7
        for row in records
    ):
        raise ValueError("canonical peak exceeds configured peak guard")


def validate_canonical_files(records, config):
    """Validate physical WAV properties; soundfile dtype conversion is insufficient."""
    canonical = config["canonical"]
    sample_rate = int(canonical["sample_rate"])
    max_duration = float(canonical["max_duration_sec"])
    peak_guard = float(config["normalization"]["peak_guard"])
    for row in records:
        path = Path(row["canonical_path"])
        info = sf.info(str(path))
        if info.subtype != "FLOAT":
            raise ValueError("canonical WAV is not FLOAT subtype: {}".format(path))
        if info.samplerate != sample_rate or info.channels != int(canonical["channels"]):
            raise ValueError("canonical WAV format validation failed: {}".format(path))
        if info.frames <= 0 or info.frames / float(info.samplerate) > max_duration:
            raise ValueError("canonical duration validation failed: {}".format(path))
        data, _ = sf.read(str(path), dtype="float32")
        if data.ndim != 1 or not np.isfinite(data).all() or len(data) == 0:
            raise ValueError("canonical finite/non-empty validation failed: {}".format(path))
        if float(np.max(np.abs(data))) > peak_guard + 1e-7:
            raise ValueError("physical canonical peak exceeds configured guard: {}".format(path))
        if _sha256_file(path) != row["canonical_wav_sha256"]:
            raise ValueError("canonical WAV SHA validation failed: {}".format(path))


def validate_split(records):
    by_split = defaultdict(set)
    by_class = defaultdict(Counter)
    for row in records:
        by_split[row["split"]].add(row["base_clip_id"])
        by_class[row["canonical_class"]][row["split"]] += 1
    overlap = {
        left + "_vs_" + right: sorted(by_split[left] & by_split[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if {row["split"] for row in records} != {"train", "val", "test"}:
        raise ValueError("split set is not exactly train/val/test")
    if any(overlap.values()):
        raise ValueError("train/val/test base_clip_id overlap detected")
    for canonical_class, counts in by_class.items():
        if counts["train"] < 20 or counts["val"] < 4 or counts["test"] < 4:
            raise ValueError("split minimum failed for {}".format(canonical_class))
    return overlap, {
        key: {split: counts[split] for split in ("train", "val", "test")}
        for key, counts in sorted(by_class.items())
    }


def validate_final_records(records, config, ontology_path):
    """Run the complete future-finalize hard gate without mutating any asset."""
    validate_registry_metadata(records, config, _ontology_classes(ontology_path))
    validate_split(records)
    validate_canonical_files(records, config)


def prepare_once(membership_path, split_path, output_root, config):
    canonical_config = config["canonical"]
    normalization_config = config["normalization"]
    sample_rate = int(canonical_config["sample_rate"])
    max_duration = float(canonical_config["max_duration_sec"])
    target_active_rms = 10.0 ** (float(normalization_config["target_active_rms_dbfs"]) / 20.0)
    peak_guard = float(normalization_config["peak_guard"])
    membership = _read_csv(membership_path)
    split_by_base = _split_map(split_path)
    if len(membership) != len(split_by_base):
        raise ValueError("membership/split row count mismatch")
    records = []
    raw_groups = defaultdict(list)
    canonical_groups = defaultdict(list)
    for row in membership:
        if row.get("manual_decision") != "ACCEPT":
            raise ValueError("non-ACCEPT row reached canonical preparation")
        if not row.get("license_status") or not row.get("provenance_source"):
            raise ValueError("unusable license/provenance for {}".format(row["base_clip_id"]))
        split = split_by_base.get(row["base_clip_id"])
        if split is None or split.get("split_version") != config["split"]["version"]:
            raise ValueError("missing v2 split for {}".format(row["base_clip_id"]))
        path = Path(row["audio_path"])
        if not path.is_file():
            raise ValueError("missing raw audio: {}".format(path))
        if row.get("raw_sha256") and _sha256_file(path) != row["raw_sha256"]:
            raise ValueError("raw SHA mismatch: {}".format(path))
        data, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if not np.isfinite(data).all() or len(data) == 0:
            raise ValueError("raw audio is empty/nonfinite: {}".format(path))
        mono = data.mean(axis=1, dtype=np.float32)
        duration = len(mono) / float(source_rate)
        if duration <= max_duration:
            crop_start = 0
            crop_policy = "preserve_full_short_source"
        else:
            crop_count = int(round(max_duration * source_rate))
            crop_start = (len(mono) - crop_count) // 2
            mono = mono[crop_start:crop_start + crop_count]
            crop_policy = "deterministic_center"
        canonical = _resample(mono, source_rate, sample_rate)
        if len(canonical) == 0 or len(canonical) > int(max_duration * sample_rate):
            raise ValueError("canonical duration outside contract: {}".format(path))
        canonical, norm = _normalize(canonical, target_active_rms, peak_guard)
        relpath = Path("wav") / row["canonical_class"] / (
            _safe_name(row["base_clip_id"]) + ".wav"
        )
        canonical_path = output_root / relpath
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(canonical_path), sample_rate, canonical.astype("<f4"))
        written, written_rate = sf.read(str(canonical_path), dtype="float32")
        if written_rate != SAMPLE_RATE or written.ndim != 1 or not np.isfinite(written).all():
            raise ValueError("written canonical WAV validation failed: {}".format(canonical_path))
        canonical_wav_sha = _sha256_file(canonical_path)
        canonical_pcm_sha = _sha256_pcm(written)
        crop_start_sec = crop_start / float(source_rate)
        crop_end_sec = crop_start_sec + len(mono) / float(source_rate)
        record = {
            "source_clip_id": "{}:{}".format(row["source_dataset"], row["original_id"]),
            "canonical_class": row["canonical_class"],
            "source_dataset": row["source_dataset"],
            "source_label": row.get("source_label", ""),
            "original_id": row["original_id"],
            "base_clip_id": row["base_clip_id"],
            "identity_key": row["identity_key"],
            "audio_path": str(path),
            "canonical_path": str(canonical_path),
            "canonical_relpath": str(relpath),
            "canonical_wav_sha256": canonical_wav_sha,
            "canonical_pcm_sha256": canonical_pcm_sha,
            "raw_sha256": row.get("raw_sha256", ""),
            "split": split["split"],
            "split_version": split["split_version"],
            "manual_decision": row["manual_decision"],
            "mapping_type": row.get("mapping_type", ""),
            "auto_qc_status": row.get("auto_qc_status", ""),
            "auto_qc_reasons": row.get("auto_qc_reasons", ""),
            "license_status": row["license_status"],
            "license_raw": row.get("license_raw", ""),
            "provenance_source": row["provenance_source"],
            "pretrain_seen_status": row.get("pretrain_seen_status", "unknown") or "unknown",
            "resource_status": row.get("resource_status", "PRESENT"),
            "canonical_sample_rate_hz": str(sample_rate),
            "canonical_channels": "1", "canonical_dtype": "float32",
            "canonical_duration_sec": "{:.9f}".format(len(canonical) / sample_rate),
            "crop_policy": crop_policy,
            "crop_start_sec": "{:.9f}".format(crop_start_sec),
            "crop_end_sec": "{:.9f}".format(crop_end_sec),
            "source_offset_sec": "",
            "normalization_method": "active_rms_v1",
            **{key: "{:.9f}".format(value) if isinstance(value, float) else value
               for key, value in norm.items()},
            "pilot_eligible": "true",
        }
        records.append(record)
        raw_groups[row.get("raw_sha256", "")].append(record)
        canonical_groups[canonical_pcm_sha].append(record)

    _write_csv(output_root / "source_registry.csv", records, REGISTRY_FIELDS)
    duplicate_audit = {
        "raw_sha_duplicate_groups": {
            key: [item["base_clip_id"] for item in values]
            for key, values in sorted(raw_groups.items()) if key and len(values) > 1
        },
        "canonical_pcm_duplicate_groups": {
            key: [item["base_clip_id"] for item in values]
            for key, values in sorted(canonical_groups.items()) if len(values) > 1
        },
        "canonical_duplicate_conflicts": [],
    }
    duplicate_audit["canonical_duplicate_conflicts"] = [
        {"canonical_pcm_sha256": key, "base_clip_ids": value}
        for key, value in duplicate_audit["canonical_pcm_duplicate_groups"].items()
        if len(value) > 1
    ]
    if duplicate_audit["canonical_duplicate_conflicts"]:
        raise ValueError("canonical PCM duplicate conflict")
    return records, duplicate_audit


def _compare_records(first, second):
    first_map = {row["base_clip_id"]: row for row in first}
    second_map = {row["base_clip_id"]: row for row in second}
    if first_map.keys() != second_map.keys():
        return False
    fields = ["canonical_pcm_sha256", "canonical_wav_sha256", "canonical_relpath",
              "split", "canonical_duration_sec", "gain_db", "peak_after"]
    return all(first_map[key][field] == second_map[key][field]
               for key in first_map for field in fields)


def finalize(membership_path, split_path, config_path, output_root, repo_registry_path,
             report_path):
    config = load_source_prep_config(config_path)
    ontology_path = Path(__file__).resolve().parents[2] / "registries/ontology.yaml"
    output_root = Path(output_root)
    if (output_root / "_SUCCESS").exists():
        raise ValueError("refusing to overwrite an already frozen pool")
    output_root.mkdir(parents=True, exist_ok=True)
    records, duplicate_audit = prepare_once(membership_path, split_path, output_root, config)
    with tempfile.TemporaryDirectory(prefix="clsdoa_pilot_prep_") as temp:
        rerun_records, _ = prepare_once(membership_path, split_path, Path(temp), config)
        determinism = _compare_records(records, rerun_records)
    if not determinism:
        raise ValueError("canonical preparation determinism failed")

    validate_registry_metadata(records, config, _ontology_classes(ontology_path))
    split_overlap, per_class = validate_split(records)
    validate_canonical_files(records, config)

    repo_registry_path = Path(repo_registry_path)
    _write_csv(repo_registry_path, records, REGISTRY_FIELDS)
    shutil.copyfile(output_root / "source_registry.csv", output_root / "source_registry_snapshot.csv")
    raw_lock = {
        "schema_version": "clsdoa_v1_raw_resources_lock_v1",
        "pool_id": POOL_ID,
        "raw_resources": [
            {"base_clip_id": row["base_clip_id"], "audio_path": row["audio_path"],
             "raw_sha256": row["raw_sha256"], "license_status": row["license_status"],
             "provenance_source": row["provenance_source"]}
            for row in sorted(records, key=lambda item: item["base_clip_id"])
        ],
    }
    (output_root / "raw_resources.lock.json").write_text(
        json.dumps(raw_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_sha = _sha256_file(Path(config_path))
    registry_sha = _sha256_file(output_root / "source_registry_snapshot.csv")
    pool_identity = {
        "schema_version": "clsdoa_v1_pool_identity_v1",
        "pool_id": POOL_ID,
        "status": "FINALIZED",
        "split_version": config["split"]["version"],
        "accepted_identity_count": len(records),
        "canonical_preparation": "mono_{}_{}_{}_deterministic_crop".format(
            config["canonical"]["sample_rate"], config["canonical"]["dtype"],
            config["canonical"]["channels"]
        ),
        "normalization": {"method": config["normalization"]["method"],
                          "target_active_rms_dbfs": config["normalization"]["target_active_rms_dbfs"],
                          "peak_guard": config["normalization"]["peak_guard"],
                          "source_level_only": True},
        "config_sha256": config_sha,
        "source_registry_snapshot_sha256": registry_sha,
        "determinism": "PASS",
    }
    (output_root / "pool_identity.json").write_text(
        json.dumps(pool_identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "clsdoa_v1_source_pool_final_audit_v1",
        "pool_id": POOL_ID,
        "status": "PASS",
        "accepted_total": len(records),
        "per_class_split": per_class,
        "split_overlap": split_overlap,
        "duplicate_audit": duplicate_audit,
        "license_provenance": {
            "usable_rows": sum(bool(row["license_status"] and row["provenance_source"]) for row in records),
            "pretrain_seen_status": dict(Counter(row["pretrain_seen_status"] for row in records)),
        },
        "normalization": {
            "method": config["normalization"]["method"],
            "target_active_rms_dbfs": config["normalization"]["target_active_rms_dbfs"],
            "peak_guard": config["normalization"]["peak_guard"],
            "peak_guard_limited_rows": sum(row["peak_guard_limited"] == "true" for row in records),
            "max_peak_after": max(float(row["peak_after"]) for row in records),
            "min_peak_after": min(float(row["peak_after"]) for row in records),
        },
        "determinism": "PASS",
        "hard_validation": "PASS",
        "registry_path": str(output_root / "source_registry_snapshot.csv"),
        "registry_sha256": registry_sha,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "_SUCCESS").write_text(
        "ClassDOA V1 source pool finalized\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-registry", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(
        membership_path=args.membership,
        split_path=args.split,
        config_path=args.config,
        output_root=args.output_root,
        repo_registry_path=args.repo_registry,
        report_path=args.report,
    )
    print("pool={} accepted={} status={}".format(
        report["pool_id"], report["accepted_total"], report["status"]
    ))
    for canonical_class, counts in report["per_class_split"].items():
        print("{} {} / {} / {}".format(canonical_class, counts["train"], counts["val"], counts["test"]))


if __name__ == "__main__":
    main()
