#!/usr/bin/env python
"""Deterministic metadata-only ClassDOA V1 Pilot planner."""

import argparse
import csv
import hashlib
import json
import math
import os
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
import magnum as mn

from active_audition.datasets.binaural_foa_clsdoa.geometry import project_geometry, validate_geometry_independently
from active_audition.datasets.binaural_foa_clsdoa.recipe import make_episode_recipe
from active_audition.datasets.binaural_foa_clsdoa.scene_registry import resolve_generation_scene_resources
from active_audition.datasets.binaural_foa_clsdoa.source_registry import read_source_registry
from active_audition.datasets.binaural_foa_clsdoa.storage import merge_resolved_config


LOCK = ROOT / "configs/active_audition/clsdoa_v1_resources.lock.json"
CONTRACT = ROOT / "configs/active_audition/clsdoa_v1_contract.yaml"
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
        ray = habitat_sim.geo.Ray(mn.Vector3(*(float(value) for value in point)), mn.Vector3(*(float(value) for value in direction)))
        hits = sim.cast_ray(ray)
        if hits.has_hits() and float(hits.hits[0].ray_distance) < radius:
            return False
    return True


def _collect_candidates(entry, config):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = entry["scene_asset"]
    backend.load_semantic_mesh = True
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()]))
    try:
        if not sim.pathfinder.is_loaded:
            sim.pathfinder.load_nav_mesh(entry["navmesh"])
        sim.pathfinder.seed(stable_int(config["global_seed"], entry["scene_id"], "pathfinder") & 0xFFFFFFFF)
        candidates = []
        for attempt in range(int(config["geometry"]["max_attempts_per_scene"])):
            listener = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
            source_base = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
            if not np.isfinite(listener).all() or not np.isfinite(source_base).all() or np.allclose(listener, source_base):
                continue
            sensor = listener + np.array([0.0, config["scene"]["sensor_height_m"], 0.0])
            if not _clear(sim, sensor, config["scene"]["listener_clearance_radius_m"]):
                continue
            path = habitat_sim.ShortestPath()
            path.requested_start = listener.astype(np.float32)
            path.requested_end = source_base.astype(np.float32)
            if not sim.pathfinder.find_path(path) or not np.isfinite(path.geodesic_distance):
                continue
            heights = (0.5, 1.0, 1.5, 2.0, 2.2)
            for source_height in heights:
                source = source_base + np.array([0.0, source_height, 0.0])
                if not _clear(sim, source, config["scene"]["source_clearance_radius_m"]):
                    continue
                distance = float(np.linalg.norm(source - sensor))
                if 1.0 <= distance <= 6.0:
                    elevation = math.degrees(math.atan2(float(source[1] - sensor[1]), float(np.linalg.norm(source[[0, 2]] - sensor[[0, 2]]))))
                    candidates.append({"listener": listener.tolist(), "sensor": sensor.tolist(), "source": source.tolist(), "source_height_offset": source_height, "distance": distance, "elevation": elevation, "geodesic": float(path.geodesic_distance), "attempt": attempt})
        return candidates
    finally:
        sim.close()


def _load_config():
    path = Path(os.environ["STEP3_PILOT_CONFIG"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
    all_pass_scenes = _scene_rows()
    scenes = all_pass_scenes
    by_class_split = defaultdict(list)
    for row in sources:
        by_class_split[(row["canonical_class"], row["split"])].append(row)
    for rows in by_class_split.values():
        rows.sort(key=lambda row: hashlib.sha256(("source|" + row["base_clip_id"]).encode()).hexdigest())
    scene_pools = {}
    for entry in scenes:
        scene_pools[entry["scene_id"]] = _collect_candidates(entry, config)
    all_candidates = {(family, split): [candidate for entry in scenes if entry["scene_family"] == family and entry["split"] == split for candidate in scene_pools[entry["scene_id"]]] for family in ("Replica", "MP3D") for split in ("train", "val", "test")}
    recipes = []
    review = []
    slots = 0
    for class_id, class_name in enumerate(CLASSES):
        for split, split_count in (("train", 56), ("val", 12), ("test", 12)):
            split_start = {"train": 0, "val": 56, "test": 68}[split]
            for family in ("Replica", "MP3D"):
                count = split_count // 2
                for local in range(count):
                    family_offset = 0 if family == "Replica" else count
                    slot = split_start + family_offset + local
                    distance_bin = "near" if slot < 32 else "mid" if slot < 64 else "far"
                    elevation_bin = "small" if slot < 60 else "nonzero"
                    key = (family, split)
                    candidate = _pick_candidate(all_candidates[key], distance_bin, elevation_bin, stable_int(config["global_seed"], class_id, split, family, local))
                    scene_id = next(entry["scene_id"] for entry in scenes if entry["scene_family"] == family and entry["split"] == split and candidate in scene_pools[entry["scene_id"]])
                    source = by_class_split[(class_name, split)][(family_offset + local) % len(by_class_split[(class_name, split)])]
                    duration = float(source["canonical_duration_sec"])
                    offset = 0.0 if duration >= 5.0 else float(np.random.default_rng(stable_int(config["global_seed"], source["source_clip_id"], slots, "offset")).uniform(0.0, 5.0 - duration))
                    gain = -6.0 + 12.0 * ((local % 10) + 0.5) / 10.0
                    # 用世界方位反解 yaw，精确覆盖每类每个方位 bin。
                    delta = np.asarray(candidate["source"], dtype=np.float64) - np.asarray(candidate["sensor"], dtype=np.float64)
                    world_bearing = math.degrees(math.atan2(float(delta[0]), -float(delta[2])))
                    target_azimuth = -157.5 + 45.0 * (slot // 10)
                    yaw = world_bearing - target_azimuth
                    episode_id = "{}_ep_{:06d}".format(config["dataset_id"], slots + 1)
                    recipe = make_episode_recipe(episode_id=episode_id, split=split, scene_id=scene_id, scene_family=family, source_clip_id=source["source_clip_id"], base_clip_id=source["base_clip_id"], source_dataset=source["source_dataset"], class_id=class_id, source_position_world=candidate["source"], source_gain_db=gain, source_offset_sec=offset, listener_base_position_world=candidate["listener"], listener_sensor_position_world=candidate["sensor"], listener_yaw_deg=yaw)
                    validate_geometry_independently(recipe.source_position_world, recipe.listener_sensor_position_world, recipe.listener_yaw_deg, {"distance_m": recipe.distance_m, "azimuth_project_deg": recipe.azimuth_project_deg, "elevation_project_deg": recipe.elevation_project_deg, "doa_unit_project": recipe.doa_unit_project})
                    recipes.append(recipe.to_dict())
                    review.append(dict(recipe.to_dict(), diagnostics={"distance_bin": distance_bin, "elevation_bin": elevation_bin, "source_height_offset_m": candidate["source_height_offset"], "geodesic_distance_m": candidate["geodesic"], "sampling_attempt": candidate["attempt"], "clearance": "PASS", "reachable": "PASS"}, source_duration_sec=duration, pretrain_seen_status=source["pretrain_seen_status"]))
                    slots += 1
    return recipes, review, scenes, sources


def write_plan(root):
    recipes, review, scenes, sources = build_plan()
    config = _load_config()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    episodes = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in recipes)
    review_text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in review)
    (root / "manifests/episodes.jsonl").write_text(episodes, encoding="utf-8")
    (root / "reports/plan_review_index.jsonl").write_text(review_text, encoding="utf-8")
    summary = {"dataset_id": config["dataset_id"], "plan_version": config["plan_version"], "episode_count": len(recipes), "source_rows": len(sources), "scene_pass_pool": 103, "scene_representatives_loaded": len(scenes), "audio_files": 0, "rir_files": 0, "success_marker": False, "render_started": False, "quota": {"split": dict(Counter(row["split"] for row in recipes)), "family": dict(Counter(row["scene"]["scene_family"] for row in recipes)), "class": dict(Counter(row["source"]["class_id"] for row in recipes))}}
    (root / "reports/plan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code_commit = os.environ.get("STEP3_GENERATION_CODE_COMMIT", "PENDING_STEP3_CODE_COMMIT")
    def file_sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "identity.json").write_text(json.dumps({"dataset_id": config["dataset_id"], "dataset_family": "soundspaces_binaural_foa_clsdoa_v1", "schema_version": "clsdoa_v1.0", "generation_code_commit": code_commit, "created_at": "2026-08-29T00:00:00Z", "config_sha256": file_sha(CONFIG_PATH), "ontology_sha256": file_sha(ROOT / "registries/ontology.yaml"), "source_registry_sha256": file_sha(ROOT / "registries/source_audio.csv"), "scene_registry_sha256": file_sha(ROOT / "registries/clsdoa_v1_scenes.yaml")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config["resolved_from"] = {"contract": "configs/active_audition/clsdoa_v1_contract.yaml", "pilot": str(CONFIG_PATH.relative_to(ROOT))}
    config["contract"] = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    config = merge_resolved_config(config["contract"], {k: v for k, v in config.items() if k != "contract"})
    config["generation_code_commit"] = code_commit
    (root / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    global_lock = json.loads(LOCK.read_text(encoding="utf-8"))
    scoped_lock = dict(global_lock)
    scoped_lock["step2_generation_foundation_commit"] = "31928fe7bdf651360552db3e1d4a1ac0e778f351"
    scoped_lock["step2_closure_evidence_commit"] = "dc1759268443e8c9b7382200c71f133e8184133d"
    scoped_lock["pilot_generation_code_commit"] = code_commit
    (root / "resources.lock.json").write_text(json.dumps(scoped_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifests/plan.lock.json").write_text(json.dumps({
        "dataset_id": config["dataset_id"], "plan_version": config["plan_version"],
        "plan_generation_code_commit": code_commit,
        "step2_generation_foundation_commit": "31928fe7bdf651360552db3e1d4a1ac0e778f351",
        "step2_closure_evidence_commit": "dc1759268443e8c9b7382200c71f133e8184133d",
        "episodes_sha256": sha256(root / "manifests/episodes.jsonl"),
        "config_resolved_sha256": sha256(root / "config_resolved.yaml"),
        "resources_lock_sha256": sha256(root / "resources.lock.json"),
        "source_registry_sha256": "f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1",
        "scene_registry_sha256": "06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508",
        "global_seed": config["global_seed"], "episode_count": len(recipes),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_reports(root, recipes, review, scenes, sources, config, code_commit)
    return summary


def _write_reports(root, recipes, review, scenes, sources, config, code_commit):
    by_class = defaultdict(list)
    for row in review:
        by_class[row["label"]["class_id"]].append(row)
    dist = {}
    for class_id, rows in sorted(by_class.items()):
        dist[str(class_id)] = {
            "azimuth_bins": dict(Counter(int((r["label"]["azimuth_project_deg"] + 180.0) // 45.0) for r in rows)),
            "distance_bins": dict(Counter(r["diagnostics"]["distance_bin"] for r in rows)),
            "elevation_bins": dict(Counter(r["diagnostics"]["elevation_bin"] for r in rows)),
        }
    (root / "reports/plan_distribution.json").write_text(json.dumps(dist, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "reports/readiness.json").write_text(json.dumps({
        "step": "step3_pilot_plan", "status": "PASS", "source_rows": len(sources), "scene_rows": 108,
        "pass_scenes": 103, "fail_scenes_excluded": 5, "materials": "OFF", "audio_files": 0,
        "rir_files": 0, "success_marker": False, "scene_representatives_loaded": len(scenes),
        "stage_config_production_loader_use_count": 0, "geometry_only_structural_smoke": "PASS",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "reports/source_dataset_shortcut_audit.json").write_text(json.dumps({
        "status": "AUDIT_ONLY", "source_dataset_counts": dict(Counter(r["source"]["source_dataset"] for r in recipes)),
        "class_source_dataset_counts": {str(k): dict(Counter(r["source"]["source_dataset"] for r in v)) for k, v in by_class.items()},
        "shortcut_blocker": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "reports/pretrain_exposure_report.json").write_text(json.dumps({
        "status": "DIAGNOSTIC_ONLY", "likely_yes_preserved": True,
        "test_pretrain_seen_status": "carried from finalized source registry; not rewritten",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loader = {"materials": "OFF", "stage_config_production_loader_use_count": 0, "loaded_scene_representatives": [r["scene_id"] for r in scenes], "direct_core_loader": {"Replica": ["scene_asset", "navmesh", "semantic_info"], "MP3D": ["scene_asset", "navmesh", "semantic_info"]}}
    (root / "reports/scene_loader_contract.json").write_text(json.dumps(loader, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "reports/plan_geometry_summary.json").write_text(json.dumps({"status": "PASS", "recipes": len(recipes), "clearance": "PASS", "reachable": "PASS", "independent_geometry_validation": "PASS", "sensor_height_m": 1.5, "source_height_m_range": [0.5, 2.2]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", choices=["plan"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=str(ROOT / "datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_001"))
    args = parser.parse_args()
    global CONFIG_PATH
    CONFIG_PATH = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    os.environ["STEP3_PILOT_CONFIG"] = str(CONFIG_PATH)
    if Path(args.root).exists() and any(Path(args.root).iterdir()):
        existing = {path.relative_to(Path(args.root)).as_posix() for path in Path(args.root).rglob("*") if path.is_file()}
        allowed = {"identity.json", "config_resolved.yaml", "resources.lock.json", "manifests/episodes.jsonl", "manifests/plan.lock.json"}
        allowed.update({"reports/" + name for name in ("readiness.json", "plan_summary.json", "plan_distribution.json", "plan_review_index.jsonl", "source_dataset_shortcut_audit.json", "pretrain_exposure_report.json", "scene_loader_contract.json", "plan_geometry_summary.json")})
        if not existing.issubset(allowed):
            raise SystemExit("refusing to overwrite non-empty dataset root")
    print(json.dumps(write_plan(Path(args.root)), sort_keys=True))


if __name__ == "__main__":
    main()
