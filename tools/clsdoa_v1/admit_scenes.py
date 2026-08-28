"""Run crash-isolated Step 2B scene admission probes."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# 允许 controller 以脚本路径直接启动 worker 时复用仓库内 FOA converter。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.foa_adapter import native_foa_to_canonical


REASONS = (
    "FAIL_SCENE_LOAD", "FAIL_NATIVE_CRASH", "FAIL_NAVMESH_LOAD",
    "FAIL_NAVIGABLE_SPACE", "FAIL_REACHABILITY", "FAIL_GEOMETRY_CLEARANCE",
    "FAIL_BINAURAL_RENDER", "FAIL_FOA_RENDER", "FAIL_ACOUSTIC_HARD",
    "FAIL_UNIT_SCALE",
)
RADIUS_DIRECTIONS = np.asarray(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.float32,
)


def stable_seed(version: str, family: str, scene_id: str) -> int:
    digest = hashlib.sha256((version + "|" + family + "|" + scene_id).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def inclusive_distance(value: float, lower: float, upper: float) -> bool:
    return lower <= float(value) <= upper


def stable_split(scene_family: str, scene_id: str, version: str = "clsdoa_v1_scene_split_v1", train: float = 0.70, val: float = 0.85) -> str:
    digest = hashlib.sha256((version + "|" + scene_family + "|" + scene_id).encode()).digest()
    u = int.from_bytes(digest[:8], "big") / float(2**64)
    if u < train:
        return "train"
    if u < val:
        return "val"
    return "test"


def validate_rir(values: Any, channels: int) -> Dict[str, Any]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] != channels or array.shape[1] <= 0:
        return {"finite": False, "nonzero": False, "shape": list(array.shape), "reason": "shape"}
    finite = bool(np.isfinite(array).all())
    energy = float(np.sum(np.square(array, dtype=np.float64))) if finite else 0.0
    peak = float(np.max(np.abs(array))) if finite else 0.0
    return {"finite": finite, "nonzero": bool(energy > 0.0 and peak > 0.0), "shape": list(array.shape), "energy": energy, "peak_abs": peak}


def paired_geometry_equal(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    keys = ("listener_base_position_world", "listener_sensor_position_world", "listener_yaw_deg", "source_position_world")
    return all(first.get(key) == second.get(key) for key in keys)


def clearance_acceptance(hit_distances: Iterable[Optional[float]], radius: float) -> bool:
    return all(distance is None or float(distance) >= float(radius) for distance in hit_distances)


def _scene_paths(entry: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    return Path(entry["scene_asset_path"]), Path(entry["navmesh_path"]), Path(entry["semantic_info_path"])


def _make_sim(scene_asset: Path, channel_layout: Any, channel_count: int, config: Dict[str, Any]):
    import quaternion  # noqa: F401, must precede habitat_sim
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_asset)
    backend.enable_physics = True
    backend.load_semantic_mesh = True
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()]))
    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = False
    spec.channelLayout.type = channel_layout
    spec.channelLayout.channelCount = channel_count
    spec.position = [0.0, 0.0, 0.0]
    spec.acousticsConfig.sampleRate = int(config["audio"]["sample_rate_hz"])
    spec.acousticsConfig.direct = True
    spec.acousticsConfig.indirect = True
    spec.acousticsConfig.diffraction = True
    spec.acousticsConfig.transmission = True
    spec.acousticsConfig.indirectRayCount = int(config["audio"]["indirect_ray_count"])
    spec.acousticsConfig.sourceRayCount = int(config["audio"]["source_ray_count"])
    simulator.add_sensor(spec)
    return simulator


def _set_pose(simulator: Any, listener_base: Sequence[float], yaw_deg: float, source: Sequence[float]) -> None:
    import quaternion

    sensor = simulator.get_agent(0)._sensors["audio_sensor"]
    sensor.reset()
    sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
    state = simulator.get_agent(0).get_state()
    state.position = np.asarray(listener_base, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, math.radians(yaw_deg), 0.0]))
    state.sensor_states = {}
    simulator.get_agent(0).set_state(state, True)


def _ray_clear(simulator: Any, point: Sequence[float], radius: float) -> Tuple[bool, int]:
    import habitat_sim

    hits = 0
    for direction in RADIUS_DIRECTIONS:
        results = simulator.cast_ray(habitat_sim.geo.Ray(np.asarray(point, dtype=np.float32), direction), max_distance=float(radius))
        if results.has_hits():
            hit = float(results.hits[0].ray_distance)
            if hit < float(radius):
                hits += 1
    return hits == 0, hits


def _sample_probes(simulator: Any, entry: Dict[str, Any], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    import habitat_sim

    rng = np.random.default_rng(stable_seed(config["admission_version"], entry["scene_family"], entry["scene_id"]))
    pathfinder = simulator.pathfinder
    diagnostics = {"total_sampling_attempts": 0, "listener_clearance_rejects": 0, "source_clearance_rejects": 0, "distance_rejects": 0, "reachability_rejects": 0}
    probes: List[Dict[str, Any]] = []
    target = int(config["geometry_probe"]["required_valid_pairs_per_scene"])
    for _ in range(int(config["geometry_probe"]["max_sampling_attempts"])):
        if len(probes) >= target:
            break
        diagnostics["total_sampling_attempts"] += 1
        listener_base = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
        source_base = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
        if not np.isfinite(listener_base).all() or not np.isfinite(source_base).all():
            diagnostics["reachability_rejects"] += 1
            continue
        listener_sensor = listener_base + np.asarray([0.0, float(config["listener"]["sensor_height_m"]), 0.0])
        listener_ok, listener_hits = _ray_clear(simulator, listener_sensor, float(config["listener"]["local_clearance_radius_m"]))
        if not listener_ok:
            diagnostics["listener_clearance_rejects"] += listener_hits
            continue
        if not pathfinder.is_navigable(listener_base) or not pathfinder.is_navigable(source_base):
            diagnostics["reachability_rejects"] += 1
            continue
        shortest = habitat_sim.ShortestPath()
        shortest.requested_start = listener_base.astype(np.float32)
        shortest.requested_end = source_base.astype(np.float32)
        if not pathfinder.find_path(shortest) or not np.isfinite(shortest.geodesic_distance):
            diagnostics["reachability_rejects"] += 1
            continue
        source_height = float(rng.uniform(float(config["source"]["height_min_m"]), float(config["source"]["height_max_m"])))
        source = source_base + np.asarray([0.0, source_height, 0.0])
        source_ok, source_hits = _ray_clear(simulator, source, float(config["source"]["local_clearance_radius_m"]))
        if not source_ok:
            diagnostics["source_clearance_rejects"] += source_hits
            continue
        distance = float(np.linalg.norm(source - listener_sensor))
        if not inclusive_distance(distance, float(config["geometry_probe"]["pair_distance_min_m"]), float(config["geometry_probe"]["pair_distance_max_m"])):
            diagnostics["distance_rejects"] += 1
            continue
        probes.append({
            "listener_base_position_world": listener_base.tolist(),
            "listener_sensor_position_world": listener_sensor.tolist(),
            "listener_yaw_deg": float(rng.uniform(-180.0, 180.0)),
            "source_position_world": source.tolist(),
            "distance_m": distance,
            "geodesic_distance_m": float(shortest.geodesic_distance),
        })
    return probes, diagnostics


def run_worker(entry: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    import quaternion  # noqa: F401, must precede habitat_sim
    import habitat_sim

    started = time.perf_counter()
    scene_asset, navmesh, semantic_info = _scene_paths(entry)
    result: Dict[str, Any] = {
        "scene_id": entry["scene_id"], "scene_family": entry["scene_family"], "readiness_status": entry["readiness_status"],
        "scene_load_status": "FAIL", "navmesh_status": "NOT_RUN", "listener_clearance_status": "NOT_RUN",
        "source_clearance_status": "NOT_RUN", "valid_probe_pair_count": 0, "probe_recipes": [],
        "binaural_status": "NOT_RUN", "binaural_probe_metrics": [], "foa_status": "NOT_RUN", "foa_probe_metrics": [],
        "paired_geometry_match": False, "unit_scale_status": "TO_VERIFY_IN_STEP_2B", "acoustic_hard_status": "NOT_RUN",
        "acoustic_soft_flags": [], "admitted_status": "FAIL", "exclude_reason": None, "split": "UNASSIGNED",
        "resource_fingerprint": entry.get("resource_fingerprint", {}), "runtime_sec": None, "worker_return_code": 0,
    }
    simulator = None
    try:
        simulator = _make_sim(scene_asset, habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural, 2, config)
        result["scene_load_status"] = "PASS"
        result["navmesh_status"] = "PASS" if simulator.pathfinder.is_loaded or simulator.pathfinder.load_nav_mesh(str(navmesh)) else "FAIL"
        if result["navmesh_status"] != "PASS" or not simulator.pathfinder.is_loaded:
            result["exclude_reason"] = "FAIL_NAVMESH_LOAD"
            return result
        bounds = simulator.pathfinder.get_bounds()
        result["navigable_point_count"] = 5
        samples = [simulator.pathfinder.get_random_navigable_point() for _ in range(5)]
        if not all(np.isfinite(np.asarray(point)).all() for point in samples) or not np.isfinite(np.asarray(bounds)).all():
            result["exclude_reason"] = "FAIL_NAVIGABLE_SPACE"
            return result
        probes, diagnostics = _sample_probes(simulator, entry, config)
        result["sampling_diagnostics"] = diagnostics
        result["valid_probe_pair_count"] = len(probes)
        result["probe_recipes"] = probes
        result["listener_clearance_status"] = "PASS" if probes else "FAIL"
        result["source_clearance_status"] = "PASS" if probes else "FAIL"
        if len(probes) < int(config["geometry_probe"]["required_valid_pairs_per_scene"]):
            result["exclude_reason"] = "FAIL_GEOMETRY_CLEARANCE"
            return result
        binaural_rows = []
        for probe in probes:
            _set_pose(simulator, probe["listener_base_position_world"], probe["listener_yaw_deg"], probe["source_position_world"])
            value = np.asarray(simulator.get_sensor_observations()["audio_sensor"])
            if value.ndim == 2 and value.shape[0] != 2:
                value = value.T
            check = validate_rir(value, 2)
            binaural_rows.append(check)
        result["binaural_probe_metrics"] = binaural_rows
        if not all(row["finite"] and row["nonzero"] for row in binaural_rows):
            result["binaural_status"] = "FAIL"
            result["exclude_reason"] = "FAIL_BINAURAL_RENDER"
            return result
        result["binaural_status"] = "PASS"
        simulator.close()
        simulator = _make_sim(scene_asset, habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics, 4, config)
        foa_rows = []
        for probe in probes:
            _set_pose(simulator, probe["listener_base_position_world"], probe["listener_yaw_deg"], probe["source_position_world"])
            value = np.asarray(simulator.get_sensor_observations()["audio_sensor"])
            if value.ndim == 2 and value.shape[0] != 4:
                value = value.T
            canonical = native_foa_to_canonical(value, probe["listener_yaw_deg"])
            check = validate_rir(canonical, 4)
            check["w_nonzero"] = bool(np.any(canonical[0]))
            check["directional_nonzero"] = bool(np.any(canonical[1:4]))
            foa_rows.append(check)
        result["foa_probe_metrics"] = foa_rows
        if not all(row["finite"] and row["nonzero"] and row["w_nonzero"] and row["directional_nonzero"] for row in foa_rows):
            result["foa_status"] = "FAIL"
            result["exclude_reason"] = "FAIL_FOA_RENDER"
            return result
        result["foa_status"] = "PASS"
        result["paired_geometry_match"] = True
        result["acoustic_hard_status"] = "PASS"
        result["admitted_status"] = "PASS"
        result["exclude_reason"] = None
        result["split"] = "UNASSIGNED"
        return result
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
        result["exclude_reason"] = "FAIL_SCENE_LOAD" if result["scene_load_status"] != "PASS" else "FAIL_ACOUSTIC_HARD"
        return result
    finally:
        if simulator is not None:
            simulator.close()
        result["runtime_sec"] = time.perf_counter() - started


def _load_candidates(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_controller(candidates_path: Path, config_path: Path, output_dir: Path, python: str) -> Dict[str, Any]:
    candidates = sorted(_load_candidates(candidates_path), key=lambda item: item["scene_id"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    events = output_dir / "admission_events.jsonl"
    results = []
    with events.open("w", encoding="utf-8") as handle:
        for entry in candidates:
            if entry.get("readiness_status") != "PRESENT_COMPLETE":
                result = {"scene_id": entry["scene_id"], "scene_family": entry["scene_family"], "admitted_status": "FAIL", "exclude_reason": "FAIL_SCENE_LOAD", "split": "UNASSIGNED", "worker_return_code": 0}
            else:
                command = [python, str(Path(__file__).resolve()), "--worker-json", json.dumps(entry), "--config", str(config_path)]
                started = time.perf_counter()
                completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                result = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.stdout.strip() else {"scene_id": entry["scene_id"], "admitted_status": "FAIL", "exclude_reason": "FAIL_NATIVE_CRASH"}
                result["worker_return_code"] = completed.returncode
                result["worker_stderr_tail"] = completed.stderr[-1000:]
                result["controller_runtime_sec"] = time.perf_counter() - started
                if completed.returncode < 0:
                    result["admitted_status"] = "FAIL"
                    result["exclude_reason"] = "FAIL_NATIVE_CRASH"
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            results.append(result)
    (output_dir / "scene_admission_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"count": len(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--worker-json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.worker_json:
        print(json.dumps(run_worker(json.loads(args.worker_json), config), sort_keys=True))
        return
    if not args.candidates or not args.output_dir:
        parser.error("controller requires --candidates and --output-dir")
    result = run_controller(Path(args.candidates), Path(args.config), Path(args.output_dir), sys.executable)
    print(json.dumps({"count": result["count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
