"""Validate the exact-production Step 2B acoustic evidence contract."""

import argparse
import json
from pathlib import Path

import yaml


def validate_rows(rows, registry, historical_marker, tolerance=1.0e-5):
    scenes = registry["scenes"]
    expected = {key for key, value in scenes.items() if value.get("admitted") == "PASS"}
    failed = {key for key, value in scenes.items() if value.get("admitted") != "PASS"}
    actual = {row["scene_id"] for row in rows}
    assert len(rows) == 206 and len(actual) == 103
    assert actual == expected and not actual & failed
    assert registry.get("split_version") == "clsdoa_v1_scene_split_v2"
    assert all(value.get("unit_scale_status") == "PASS" for value in scenes.values())
    split_counts = {}
    for scene_id, value in scenes.items():
        if value.get("admitted") == "PASS":
            split_counts.setdefault(value["scene_family"], {}).setdefault(value["split"], 0)
            split_counts[value["scene_family"]][value["split"]] += 1
    assert split_counts == {"Replica": {"train": 12, "val": 3, "test": 3}, "MP3D": {"train": 59, "val": 13, "test": 13}}
    assert all(row["sample_rate_hz"] == 24000 and not row["materials_enabled"] for row in rows)
    assert all((row["indirect_ray_count"], row["source_ray_count"]) == (5000, 200) for row in rows)
    assert all(row["normalization_applied"] is False for row in rows)
    assert all(row["receiver_error_m"] <= tolerance and row["foa_receiver_error_m"] <= tolerance for row in rows)
    assert all(row["binaural"]["channel_count"] == 2 and row["binaural"]["finite"] and row["binaural"]["nonzero"] for row in rows)
    assert all(row["foa"]["channel_count"] == 4 and row["foa"]["finite"] and row["foa"]["nonzero"] for row in rows)
    keys = [(row["scene_id"], row["probe_id"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert all(sorted(row["probe_id"] for row in rows if row["scene_id"] == scene) == ["probe_0", "probe_1"] for scene in actual)
    assert historical_marker == "SUPERSEDED_BY_CORRECT_RECEIVER_REVALIDATION"
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
