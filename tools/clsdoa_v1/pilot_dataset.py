#!/usr/bin/env python
"""Deterministic metadata-only ClassDOA V1 Pilot planner."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import quaternion
import yaml
import habitat_sim

from active_audition.datasets.binaural_foa_clsdoa.geometry import project_geometry, validate_geometry_independently
from active_audition.datasets.binaural_foa_clsdoa.recipe import make_episode_recipe
from active_audition.datasets.binaural_foa_clsdoa.scene_registry import resolve_generation_scene_resources
from active_audition.datasets.binaural_foa_clsdoa.source_registry import read_source_registry


LOCK = ROOT / "configs/active_audition/clsdoa_v1_resources.lock.json"
CONFIG = ROOT / "configs/active_audition/clsdoa_v1_pilot_001.yaml"
SCENES = ROOT / "registries/clsdoa_v1_scenes.yaml"
CLASSES = ["coughing", "laughing", "keyboard_typing", "vacuum_cleaner", "clock_alarm", "speech", "running_water", "frying", "mechanical_fan", "microwave_oven", "dishes", "printer"]


def stable_int(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def _scene_rows():
    raw = yaml.safe_load(SCENES.read_text(encoding="utf-8"))
    result = []
    for scene_id, row in raw["scenes"].items():
        item = dict(row, scene_id=scene_id)
        if item["admitted"] == "PASS":
            for key in ("scene_asset", "navmesh", "semantic_info"):
                item[key] = str((ROOT / item[key]).resolve())
            result.append(resolve_generation_scene_resources(item))
    return sorted(result, key=lambda item: item["scene_id"])


def _clear(sim, point, radius):
    directions = (np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([0, 0, -1]))
    for direction in directions:
        hits = sim.cast_ray(habitat_sim.geo.Ray(np.asarray(point, dtype=np.float32), direction), max_distance=radius)
        if hits.has_hits() and float(hits.hits[0].ray_distance) < radius:
            return False
    return True


def _collect_candidates(entry, config):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = entry["scene_asset"]
    backend.load_semantic_mesh = True
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, []))
    try:
        if not sim.pathfinder.is_loaded:
            sim.pathfinder.load_nav_mesh(entry["navmesh"])
        rng = np.random.default_rng(stable_int(config["global_seed"], entry["scene_id"], "geometry"))
        candidates = []
        for attempt in range(int(config["geometry"]["max_attempts_per_scene"])):
            listener = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
            source_base = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
            if not np.isfinite(listener).all() or not np.isfinite(source_base).all() or np.allclose(listener, source_base):
                continue
            sensor = listener + np.array([0.0, config["scene"]["sensor_height_m"], 0.0])
            if not _clear(sim, sensor, config["scene"]["listener_clearance_radius_m"]):
                continue
            source_height = float(rng.uniform(config["scene"]["source_height_min_m"], config["scene"]["source_height_max_m"]))
            source = source_base + np.array([0.0, source_height, 0.0])
            if not _clear(sim, source, config["scene"]["source_clearance_radius_m"]):
                continue
            path = habitat_sim.ShortestPath()
            path.requested_start = listener.astype(np.float32)
            path.requested_end = source_base.astype(np.float32)
            if not sim.pathfinder.find_path(path) or not np.isfinite(path.geodesic_distance):
                continue
            distance = float(np.linalg.norm(source - sensor))
            if 1.0 <= distance <= 6.0:
                elevation = math.degrees(math.atan2(float(source[1] - sensor[1]), float(np.linalg.norm(source[[0, 2]] - sensor[[0, 2]]))))
                candidates.append({"listener": listener.tolist(), "sensor": sensor.tolist(), "source": source.tolist(), "distance": distance, "elevation": elevation, "geodesic": float(path.geodesic_distance), "attempt": attempt})
        return candidates
    finally:
        sim.close()


def _load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _source_rows():
    return list(read_source_registry(str(ROOT / "registries/source_audio.csv")))


def _pick_candidate(pool, distance_bin, elevation_bin, slot):
    def match(item):
        d = item["distance"]
        dist = (1.0 <= d < 2.0) if distance_bin == "near" else (2.0 <= d < 4.0) if distance_bin == "mid" else (4.0 <= d <= 6.0)
        elev = abs(item["elevation"]) < 5.0 if elevation_bin == "small" else abs(item["elevation"]) >= 5.0
        return dist and elev
    options = [item for item in pool if match(item)]
    if not options:
        raise RuntimeError("no geometry candidate for {} / {}".format(distance_bin, elevation_bin))
    return options[slot % len(options)]


def build_plan():
    config = _load_config()
    sources = _source_rows()
    scenes = _scene_rows()
    by_class_split = defaultdict(list)
    for row in sources:
        by_class_split[(row["canonical_class"], row["split"])].append(row)
    for rows in by_class_split.values():
        rows.sort(key=lambda row: hashlib.sha256(("source|" + row["base_clip_id"]).encode()).hexdigest())
    scene_pools = {}
    for entry in scenes:
        scene_pools[entry["scene_id"]] = _collect_candidates(entry, config)
    all_candidates = {(entry["scene_family"], entry["split"]): [candidate for entry in scenes if entry["scene_family"] == family and entry["split"] == split for candidate in scene_pools[entry["scene_id"]]] for family in ("Replica", "MP3D") for split in ("train", "val", "test")}
    recipes = []
    review = []
    slots = 0
    for class_id, class_name in enumerate(CLASSES):
        for split, split_count in (("train", 56), ("val", 12), ("test", 12)):
            for family in ("Replica", "MP3D"):
                count = split_count // 2
                for local in range(count):
                    distance_bin = "near" if local < round(count * 0.4) else "mid" if local < round(count * 0.8) else "far"
                    elevation_bin = "small" if local < round(count * 0.75) else "nonzero"
                    key = (family, split)
                    candidate = _pick_candidate(all_candidates[key], distance_bin, elevation_bin, stable_int(config["global_seed"], class_id, split, family, local))
                    scene_id = next(entry["scene_id"] for entry in scenes if entry["scene_family"] == family and entry["split"] == split and candidate in scene_pools[entry["scene_id"]])
                    source = by_class_split[(class_name, split)][local % len(by_class_split[(class_name, split)])]
                    duration = float(source["canonical_duration_sec"])
                    offset = 0.0 if duration >= 5.0 else float(np.random.default_rng(stable_int(config["global_seed"], source["source_clip_id"], slots, "offset")).uniform(0.0, 5.0 - duration))
                    gain = -6.0 + 12.0 * ((local % 10) + 0.5) / 10.0
                    yaw = float(np.random.default_rng(stable_int(config["global_seed"], class_id, split, family, local, "yaw")).uniform(-180.0, 180.0))
                    episode_id = "clsdoa_v1_pilot_001_ep_{:06d}".format(slots + 1)
                    recipe = make_episode_recipe(episode_id=episode_id, split=split, scene_id=scene_id, scene_family=family, source_clip_id=source["source_clip_id"], base_clip_id=source["base_clip_id"], source_dataset=source["source_dataset"], class_id=class_id, source_position_world=candidate["source"], source_gain_db=gain, source_offset_sec=offset, listener_base_position_world=candidate["listener"], listener_sensor_position_world=candidate["sensor"], listener_yaw_deg=yaw)
                    validate_geometry_independently(recipe.source_position_world, recipe.listener_sensor_position_world, recipe.listener_yaw_deg, {"distance_m": recipe.distance_m, "azimuth_project_deg": recipe.azimuth_project_deg, "elevation_project_deg": recipe.elevation_project_deg, "doa_unit_project": recipe.doa_unit_project})
                    recipes.append(recipe.to_dict())
                    review.append(dict(recipe.to_dict(), diagnostics={"distance_bin": distance_bin, "elevation_bin": elevation_bin, "geodesic_distance_m": candidate["geodesic"], "sampling_attempt": candidate["attempt"], "clearance": "PASS", "reachable": "PASS"}, source_duration_sec=duration, pretrain_seen_status=source["pretrain_seen_status"]))
                    slots += 1
    return recipes, review, scenes, sources


def write_plan(root):
    recipes, review, scenes, sources = build_plan()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    episodes = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in recipes)
    review_text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in review)
    (root / "manifests/episodes.jsonl").write_text(episodes, encoding="utf-8")
    (root / "reports/plan_review_index.jsonl").write_text(review_text, encoding="utf-8")
    summary = {"dataset_id": "clsdoa_v1_pilot_001", "plan_version": "clsdoa_v1_pilot_plan_v1", "episode_count": len(recipes), "source_rows": len(sources), "scene_pass_pool": len(scenes), "audio_files": 0, "rir_files": 0, "success_marker": False, "render_started": False, "quota": {"split": dict(Counter(row["split"] for row in recipes)), "family": dict(Counter(row["scene"]["scene_family"] for row in recipes)), "class": dict(Counter(row["source"]["class_id"] for row in recipes))}}
    (root / "reports/plan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "identity.json").write_text(json.dumps({"dataset_id": "clsdoa_v1_pilot_001", "schema_version": "clsdoa_v1.0", "generation_code_commit": "PENDING_STEP3_CODE_COMMIT"}, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", choices=["plan"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=str(ROOT / "datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_001"))
    args = parser.parse_args()
    if Path(args.root).exists() and any(Path(args.root).iterdir()):
        raise SystemExit("refusing to overwrite non-empty dataset root")
    print(json.dumps(write_plan(Path(args.root)), sort_keys=True))


if __name__ == "__main__":
    main()
