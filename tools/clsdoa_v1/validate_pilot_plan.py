#!/usr/bin/env python
"""Independent hard-gate validator for the metadata-only Step 3 pilot plan."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import yaml
import numpy as np
from scipy.io import wavfile
from tools.clsdoa_v1.scheduler import distance_schedule, elevation_schedule, azimuth_schedule, gain_schedule
from tools.clsdoa_v1.integrity import verify_plan_integrity

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLASSES = list(range(12))


class PilotPlanValidationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise PilotPlanValidationError(message)


def validate_episode_membership(episodes, source_registry, scene_registry):
    sources = {row["base_clip_id"]: row for row in source_registry}
    scenes = scene_registry.get("scenes", scene_registry)
    for row in episodes:
        source = sources.get(row["source"]["base_clip_id"])
        scene = scenes.get(row["scene"]["scene_id"])
        _require(source is not None, "episode source is absent from registry")
        _require(scene is not None, "episode scene is absent from registry")
        _require(source["split"] == row["split"], "source split mismatch")
        _require(scene["split"] == row["split"], "scene split mismatch")
        _require(scene["scene_family"] == row["scene"]["scene_family"], "scene family mismatch")


def validate(root):
    root = Path(root)
    episodes = [json.loads(line) for line in (root / "manifests/episodes.jsonl").read_text().splitlines() if line]
    review = [json.loads(line) for line in (root / "reports/plan_review_index.jsonl").read_text().splitlines() if line]
    _require(len(episodes) == len(review) == 960, "episode/review count must be 960")
    _require(len({r["episode_id"] for r in episodes}) == 960, "episode IDs must be unique")
    _require(Counter(r["split"] for r in episodes) == Counter(train=672, val=144, test=144), "split quota mismatch")
    _require(Counter(r["scene"]["scene_family"] for r in episodes) == Counter(Replica=480, MP3D=480), "family quota mismatch")
    _require(Counter(r["label"]["class_id"] for r in episodes) == Counter({i: 80 for i in CLASSES}), "class quota mismatch")
    registry = list(__import__("active_audition.datasets.binaural_foa_clsdoa.source_registry", fromlist=["read_source_registry"]).read_source_registry(str(ROOT / "registries/source_audio.csv")))
    _require({r["source"]["base_clip_id"] for r in episodes} == {r["base_clip_id"] for r in registry}, "source pool mismatch")
    scene_registry = yaml.safe_load((ROOT / "registries/clsdoa_v1_scenes.yaml").read_text())["scenes"]
    validate_episode_membership(episodes, registry, scene_registry)
    pass_ids = {sid for sid, r in scene_registry.items() if r["admitted"] == "PASS"}
    fail_ids = {sid for sid, r in scene_registry.items() if r["admitted"] != "PASS"}
    used_scene_ids = {r["scene"]["scene_id"] for r in episodes}
    _require(used_scene_ids == pass_ids, "scene pool must equal exact PASS scene set")
    _require(not (used_scene_ids & fail_ids), "scene pool contains non-PASS scene")
    for class_id in CLASSES:
        rows = [r for r in episodes if r["label"]["class_id"] == class_id]
        _require(len({r["label"]["azimuth_project_deg"] for r in rows}) > 0, "empty azimuth schedule")
        _require(Counter(r["diagnostics"]["distance_bin"] for r in review if r["label"]["class_id"] == class_id) == Counter(near=32, mid=32, far=16), "distance quota mismatch")
        _require(Counter(r["diagnostics"]["elevation_bin"] for r in review if r["label"]["class_id"] == class_id) == Counter(small=60, nonzero=20), "elevation quota mismatch")
        for split in ("train", "val", "test"):
            counts = Counter(r["source"]["base_clip_id"] for r in episodes if r["label"]["class_id"] == class_id and r["split"] == split)
            _require(max(counts.values()) - min(counts.values()) <= 1, "source reuse imbalance")
            split_rows = [r for r in review if r["label"]["class_id"] == class_id and r["split"] == split]
            _require(set(r["diagnostics"]["azimuth_bin"] for r in split_rows) == set(range(8)), "split azimuth coverage mismatch")
            _require(set(r["diagnostics"]["gain_bin"] for r in split_rows) == set(range(8)), "split gain coverage mismatch")
            for family in ("Replica", "MP3D"):
                block = [r for r in split_rows if r["scene"]["scene_family"] == family]
                _require([r["diagnostics"]["distance_bin"] for r in block] and Counter(r["diagnostics"]["distance_bin"] for r in block) == Counter(distance_schedule(split, class_id, family)), "family distance micro-pattern mismatch")
                _require(Counter(r["diagnostics"]["elevation_bin"] for r in block) == Counter(elevation_schedule(split, class_id, family)), "family elevation micro-pattern mismatch")
                _require(Counter(r["diagnostics"]["azimuth_bin"] for r in block) == Counter(azimuth_schedule(split, class_id, family)), "family azimuth micro-pattern mismatch")
                _require(Counter(r["diagnostics"]["gain_bin"] for r in block) == Counter(gain_schedule(split, class_id, family)), "family gain micro-pattern mismatch")
    for r in episodes:
        _require(set(r["representations"]) == {"binaural", "foa"}, "representation mismatch")
        _require(r["listener"]["sensor_position_world"] == [r["listener"]["base_position_world"][0], r["listener"]["base_position_world"][1] + 1.5, r["listener"]["base_position_world"][2]], "receiver invariant mismatch")
        _require(-6.0 <= r["source"]["source_gain_db"] <= 6.0, "gain out of range")
    for r in review:
        if r["diagnostics"].get("candidate_origin") == "STEP2B_FIXED_PROBE":
            _require(r["diagnostics"].get("source_height_offset_m") is None and r["diagnostics"].get("geodesic_distance_m") is None, "fixed probe contains fabricated geometry diagnostics")
        else:
            _require(0.5 <= r["diagnostics"]["source_height_offset_m"] <= 2.2 + 1e-6, "source height out of range")
    _require(not list(root.rglob("*.wav")) and not list(root.rglob("*.rir")) and not list((root / "cache/rir").rglob("*")) and not (root / "_SUCCESS").exists(), "plan contains render payload")
    _require((root / "manifests/plan.lock.json").is_file(), "plan lock missing")
    verify_plan_integrity(root, ROOT)
    return {"status": "PASS", "episodes": 960, "classes": 12, "unique_sources": 422, "pass_scene_pool": 103, "excluded_fail_scenes": 5, "audio_files": 0, "rir_files": 0}


def validate_render_payload(root):
    """Payload gate used by finalize; plan-only roots must fail explicitly."""
    root = Path(root)
    _require((root / "manifests/episodes.jsonl").is_file(), "episodes manifest missing")
    renders = root / "manifests/renders.jsonl"
    _require(renders.is_file(), "render manifest missing")
    rows = [json.loads(line) for line in renders.read_text().splitlines() if line]
    verify_plan_integrity(root, ROOT)
    episodes = [json.loads(line) for line in (root / "manifests/episodes.jsonl").read_text().splitlines() if line]
    _require(len(episodes) == 960 and len({row["episode_id"] for row in episodes}) == 960, "payload requires all 960 unique episodes")
    expected = {(row["episode_id"], representation) for row in episodes for representation in ("binaural", "foa")}
    complete = [row for row in rows if row.get("render_status") == "complete"]
    keys = {(row.get("episode_id"), row.get("representation")) for row in complete}
    _require(keys == expected and len(complete) == 1920, "payload must contain exactly one complete binaural and foa record per episode")
    for row in complete:
        validate_render_record_payload(root, row)
    _require(not (root / "_SUCCESS").exists(), "dataset already finalized")
    return True


def validate_render_record_payload(root, row):
    root = Path(root)
    _require(row["sample_rate_hz"] == 24000 and row["num_samples"] == 120000 and row["dtype"] == "float32", "payload audio contract mismatch")
    _require(row["num_channels"] == (2 if row["representation"] == "binaural" else 4), "payload channel contract mismatch")
    _require(Path(root / row["audio_path"]).is_file(), "render WAV missing")
    _require(row.get("rir_path") and Path(root / row["rir_path"]).is_file(), "render RIR missing")
    sample_rate, audio = wavfile.read(str(root / row["audio_path"]))
    audio = np.asarray(audio)
    _require(int(sample_rate) == 24000 and audio.shape == (120000, row["num_channels"]), "WAV payload shape/rate mismatch")
    _require(audio.dtype == np.dtype("float32"), "WAV dtype mismatch")
    _require(np.isfinite(audio).all() and np.any(np.abs(audio)), "WAV must be finite and non-zero")
    rir = np.asarray(np.load(str(root / row["rir_path"]), allow_pickle=False))
    _require(rir.ndim == 2 and ((row["representation"] == "binaural" and rir.shape[1] == 2) or (row["representation"] == "foa" and rir.shape[0] == 4)), "RIR channel payload mismatch")
    _require(np.isfinite(rir).all() and np.any(np.abs(rir)), "RIR must be finite and non-zero")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
