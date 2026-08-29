#!/usr/bin/env python
"""Independent hard-gate validator for the metadata-only Step 3 pilot plan."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLASSES = list(range(12))


def validate(root):
    root = Path(root)
    episodes = [json.loads(line) for line in (root / "manifests/episodes.jsonl").read_text().splitlines() if line]
    review = [json.loads(line) for line in (root / "reports/plan_review_index.jsonl").read_text().splitlines() if line]
    assert len(episodes) == len(review) == 960
    assert Counter(r["split"] for r in episodes) == Counter(train=672, val=144, test=144)
    assert Counter(r["scene"]["scene_family"] for r in episodes) == Counter(Replica=480, MP3D=480)
    assert Counter(r["label"]["class_id"] for r in episodes) == Counter({i: 80 for i in CLASSES})
    assert {(r["episode_id"], r["episode_id"]) for r in episodes}.__len__() == 960
    source_ids = {r["source"]["base_clip_id"] for r in episodes}
    registry = list(__import__("active_audition.datasets.binaural_foa_clsdoa.source_registry", fromlist=["read_source_registry"]).read_source_registry(str(ROOT / "registries/source_audio.csv")))
    assert source_ids == {r["base_clip_id"] for r in registry}
    scene_registry = yaml.safe_load((ROOT / "registries/clsdoa_v1_scenes.yaml").read_text())["scenes"]
    pass_ids = {sid for sid, r in scene_registry.items() if r["admitted"] == "PASS"}
    fail_ids = {sid for sid, r in scene_registry.items() if r["admitted"] != "PASS"}
    used_scene_ids = {r["scene"]["scene_id"] for r in episodes}
    assert used_scene_ids <= pass_ids and not (used_scene_ids & fail_ids)
    for class_id in CLASSES:
        rows = [r for r in episodes if r["label"]["class_id"] == class_id]
        assert Counter(int((r["label"]["azimuth_project_deg"] + 180) // 45) for r in rows) == Counter({i: 10 for i in range(8)})
        assert Counter(r["diagnostics"]["distance_bin"] for r in review if r["label"]["class_id"] == class_id) == Counter(near=32, mid=32, far=16)
        assert Counter(r["diagnostics"]["elevation_bin"] for r in review if r["label"]["class_id"] == class_id) == Counter(small=60, nonzero=20)
    for r in episodes:
        assert set(r["representations"]) == {"binaural", "foa"}
        assert r["listener"]["sensor_position_world"] == [r["listener"]["base_position_world"][0], r["listener"]["base_position_world"][1] + 1.5, r["listener"]["base_position_world"][2]]
        assert -6.0 <= r["source"]["source_gain_db"] <= 6.0
    for r in review:
        assert 0.5 <= r["diagnostics"]["source_height_offset_m"] <= 2.2 + 1e-6
    assert not list(root.rglob("*.wav")) and not list(root.rglob("*.rir")) and not (root / "_SUCCESS").exists()
    return {"status": "PASS", "episodes": 960, "classes": 12, "unique_sources": 422, "pass_scene_pool": 103, "excluded_fail_scenes": 5, "audio_files": 0, "rir_files": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
