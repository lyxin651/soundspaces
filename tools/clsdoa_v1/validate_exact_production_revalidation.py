"""Validate the exact-production Step 2B acoustic evidence contract."""

import argparse
import json
from pathlib import Path

import yaml


class EvidenceValidationError(ValueError):
    """Raised when authoritative evidence violates a frozen contract."""


def _require(condition, message):
    if not condition:
        raise EvidenceValidationError(message)


def validate_rows(rows, registry, historical_marker, tolerance=1.0e-5):
    scenes = registry["scenes"]
    expected = {key for key, value in scenes.items() if value.get("admitted") == "PASS"}
    failed = {key for key, value in scenes.items() if value.get("admitted") != "PASS"}
    actual = {row["scene_id"] for row in rows}
    _require(len(rows) == 206 and len(actual) == 103, "row/probe count mismatch")
    _require(actual == expected and not actual & failed, "scene set mismatch or geometry FAIL leaked into evidence")
    _require(registry.get("split_version") == "clsdoa_v1_scene_split_v2", "split version mismatch")
    _require(all(value.get("unit_scale_status") == "PASS" for value in scenes.values()), "unit-scale not PASS")
    split_counts = {}
    for scene_id, value in scenes.items():
        if value.get("admitted") == "PASS":
            split_counts.setdefault(value["scene_family"], {}).setdefault(value["split"], 0)
            split_counts[value["scene_family"]][value["split"]] += 1
    _require(split_counts == {"Replica": {"train": 12, "val": 3, "test": 3}, "MP3D": {"train": 59, "val": 13, "test": 13}}, "split quota mismatch")
    _require(all(row["sample_rate_hz"] == 24000 for row in rows), "wrong sample rate")
    _require(all(not row["materials_enabled"] for row in rows), "Materials enabled")
    _require(all((row["indirect_ray_count"], row["source_ray_count"]) == (5000, 200) for row in rows), "ray config mismatch")
    _require(all(row["normalization_applied"] is False for row in rows), "normalization enabled")
    _require(all(row["receiver_error_m"] <= tolerance for row in rows), "receiver error above tolerance")
    _require(all(row["foa_receiver_error_m"] <= tolerance for row in rows), "FOA receiver error above tolerance")
    _require(all(row["binaural"]["channel_count"] == 2 for row in rows), "wrong Binaural channels")
    _require(all(row["binaural"]["finite"] and row["binaural"]["nonzero"] for row in rows), "Binaural finite/nonzero gate failed")
    _require(all(row["foa"]["channel_count"] == 4 for row in rows), "wrong FOA channels")
    _require(all(row["foa"]["finite"] and row["foa"]["nonzero"] for row in rows), "FOA finite/nonzero gate failed")
    keys = [(row["scene_id"], row["probe_id"]) for row in rows]
    _require(len(keys) == len(set(keys)), "duplicate scene+probe key")
    _require(all(sorted(row["probe_id"] for row in rows if row["scene_id"] == scene) == ["probe_0", "probe_1"] for scene in actual), "missing probe_0/probe_1 coverage")
    _require(historical_marker == "SUPERSEDED_BY_CORRECT_RECEIVER_REVALIDATION", "historical superseded marker mismatch")
    return {"rows": len(rows), "scenes": len(actual), "status": "PASS"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--registry", required=True)
    args = parser.parse_args()
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    rows = evidence["rows"] if isinstance(evidence, dict) else evidence
    marker = evidence.get("historical_acoustic_evidence", "SUPERSEDED_BY_CORRECT_RECEIVER_REVALIDATION") if isinstance(evidence, dict) else "SUPERSEDED_BY_CORRECT_RECEIVER_REVALIDATION"
    result = validate_rows(rows, registry, marker)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
