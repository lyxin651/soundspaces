# ClassDOA V1 Step 2B Code/Evidence Review Bundle

This is a read-only review export generated at review base `a96c244c03ce4ce1e18af1076aa5e84f34726cd4`. It does not replace or regenerate admission evidence. No scene, result, split, registry, unit-scale, or acoustic result was modified while producing this bundle.
## 1. Git Provenance

The working tree was clean before export. The current branch is `dataset-v1/step2b-scene-admission`; current HEAD is `a96c244c03ce4ce1e18af1076aa5e84f34726cd4`.

### Log from Step 1B base to HEAD

```text
a96c244 (HEAD -> dataset-v1/step2b-scene-admission, personal/dataset-v1/step2b-scene-admission) Clarify ClassDOA V1 Step 2B provenance
dd628cd Correct Step 2B registry report path
5c45b70 Finalize ClassDOA V1 scene admission
efec692 Fix direct Step 2B split execution
53ecc6d Seed Step 2B geometry sampling deterministically
25ccebe Support Step 1B candidate schema in admission worker
cbcfa0c Fix Step 2B worker module path
08bed9f Implement ClassDOA V1 scene admission
6184ffc (personal/dataset-v1/step1b-final-review, dataset-v1/step1b-final-review) Integrate ClassDOA V1 scene readiness
```

### Commit roles and changed files

`53ecc6d` is the code commit used for the formal run. Its complete changed-file list is:

```text
tools/clsdoa_v1/admit_scenes.py
```

`5c45b70bb1d7a1f95efd411fc25e274665af44ac` is the first result commit. Its complete changed-file list is:

```text
docs/audits/clsdoa_v1/scene_admission/acoustic_outliers.json
docs/audits/clsdoa_v1/scene_admission/mp3d_unit_scale_report.json
docs/audits/clsdoa_v1/scene_admission/resource_recovery_provenance.json
docs/audits/clsdoa_v1/scene_admission/scene_admission_metrics.csv
docs/audits/clsdoa_v1/scene_admission/scene_admission_report.md
docs/audits/clsdoa_v1/scene_admission/scene_admission_summary.json
docs/audits/clsdoa_v1/scene_admission/structural_smoke.json
registries/clsdoa_v1_scenes.yaml
```

`dd628cd06b6e603b8eed3e93b455f57802a2242f` is the provenance/hardening commit; it changed only `docs/audits/clsdoa_v1/scene_admission/scene_admission_summary.json`, correcting the registry path. `a96c244c03ce4ce1e18af1076aa5e84f34726cd4` is the provenance closure commit; it added this role clarification to the audit documents. Current HEAD is `a96c244c03ce4ce1e18af1076aa5e84f34726cd4`.
## 2. Admission Core Code

The controller/worker, resolver, deterministic seed, clearance ray checks, probe sampling, Binaural/FOA lifecycle, RIR hard validation, and admitted-status logic are all in `admit_scenes.py`. The FOA converter directly called by the worker is included because it affects model-facing channel conversion.

### `tools/clsdoa_v1/admit_scenes.py`

```python
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
    return (
        Path(entry.get("scene_asset_path") or entry["scene_asset"]),
        Path(entry.get("navmesh_path") or entry["navmesh"]),
        Path(entry.get("semantic_info_path") or entry["semantic_info"]),
    )


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
    if hasattr(pathfinder, "seed"):
        pathfinder.seed(int(stable_seed(config["admission_version"], entry["scene_family"], entry["scene_id"]) % (2**31 - 1)))
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
```

### `examples/foa_adapter.py`

```python
"""Small, explicit SoundSpaces FOA to Dataset V1 converter."""

import numpy as np


NATIVE_ORDER = ("W", "Y", "Z", "X")
CANONICAL_ORDER = ("W", "Y", "Z", "X")


def _validate_foa(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 4 or array.shape[1] == 0:
        raise ValueError("expected non-empty FOA with channel-first shape (4, N)")
    if not np.isfinite(array).all():
        raise ValueError("FOA values must be finite")
    return array


def find_shared_direct_sample(values):
    """Find one arrival sample shared by every FOA channel."""
    array = _validate_foa(values)
    return int(np.argmax(np.sum(array * array, axis=0)))


def measure_shared_direct_coefficients(values, window_radius=8):
    """Measure signed FOA coefficients around one shared direct arrival."""
    array = _validate_foa(values)
    if window_radius < 0:
        raise ValueError("window_radius must be non-negative")
    sample = find_shared_direct_sample(array)
    start = max(0, sample - window_radius)
    stop = min(array.shape[1], sample + window_radius + 1)
    return {
        "direct_sample": sample,
        "signed_coefficients": array[:, sample].copy(),
        "window_energy": np.sum(array[:, start:stop] ** 2, axis=1),
    }


def native_foa_to_ambix(values):
    """Convert N3D [W,Y,Z,X] to AmbiX ACN/SN3D [W,Y,Z,X]."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] != 4:
        raise ValueError("expected FOA with channel-first shape (4, ...)")
    output = array.copy()
    output[1:4] *= np.float32(1.0 / np.sqrt(3.0))
    return output


def native_foa_to_canonical(values, listener_yaw_deg=0.0):
    """Convert world-fixed native N3D FOA to model-facing AmbiX ACN/SN3D.

    Native channels are [W,Y_RLR,Z_RLR,X_RLR], where the Habitat/RLR
    listener-local axes are +X=right, +Y=up, +Z=back. The directional
    vector is first rotated from world axes into listener-local axes. The
    DCASE/STARSS model axes are +X=front, +Y=left, +Z=up, so the output
    channels are [W,Y_DCASE,Z_DCASE,X_DCASE] = [W,-X_RLR,+Y_RLR,-Z_RLR].
    """
    native = _validate_foa(values)
    yaw = np.deg2rad(float(listener_yaw_deg))
    x_world = native[3]
    y_world = native[1]
    z_world = native[2]
    x_local = np.cos(yaw) * x_world - np.sin(yaw) * z_world
    z_local = np.sin(yaw) * x_world + np.cos(yaw) * z_world
    output = np.empty_like(native)
    output[0] = native[0]
    output[1] = -x_local
    output[2] = y_world
    output[3] = -z_local
    output[1:4] *= np.float32(1.0 / np.sqrt(3.0))
    return output


def project_to_dcase_azimuth(azimuth_deg):
    """Project convention is right-positive; DCASE convention is left-positive."""
    return -float(azimuth_deg)
```

### `configs/active_audition/clsdoa_v1_scene_admission.yaml`

```yaml
admission_version: clsdoa_v1_scene_admission_v1
global_seed: 20260828
materials_mode: off

listener:
  sensor_height_m: 1.5
  local_clearance_radius_m: 0.10
source:
  height_min_m: 0.5
  height_max_m: 2.2
  local_clearance_radius_m: 0.15
geometry_probe:
  required_valid_pairs_per_scene: 2
  pair_distance_min_m: 1.0
  pair_distance_max_m: 4.0
  max_sampling_attempts: 100
audio:
  sample_rate_hz: 24000
  indirect_ray_count: 5000
  source_ray_count: 200
  paired_mode: sequential_sensor_lifecycle
  materials_enabled: false
split:
  version: clsdoa_v1_scene_split_v1
  train_threshold: 0.70
  val_threshold: 0.85
```

### `tests/tools/test_clsdoa_v1_scene_admission.py`

```python
import unittest

from tools.clsdoa_v1.admit_scenes import (
    inclusive_distance,
    paired_geometry_equal,
    stable_seed,
    stable_split,
    validate_rir,
)
from tools.clsdoa_v1.split_admitted_scenes import split_results


class SceneAdmissionLogicTests(unittest.TestCase):
    def test_seed_and_split_are_stable(self):
        self.assertEqual(stable_seed("v1", "MP3D", "mp3d.a"), stable_seed("v1", "MP3D", "mp3d.a"))
        self.assertEqual(stable_split("Replica", "replica.office_0"), stable_split("Replica", "replica.office_0"))

    def test_distance_boundaries_are_inclusive(self):
        self.assertTrue(inclusive_distance(1.0, 1.0, 4.0))
        self.assertTrue(inclusive_distance(4.0, 1.0, 4.0))
        self.assertFalse(inclusive_distance(0.999, 1.0, 4.0))
        self.assertFalse(inclusive_distance(4.001, 1.0, 4.0))

    def test_rir_hard_validation(self):
        self.assertTrue(validate_rir([[1.0, 0.0], [0.0, 1.0]], 2)["nonzero"])
        self.assertFalse(validate_rir([[0.0], [0.0]], 2)["nonzero"])
        self.assertFalse(validate_rir([[float("nan")]], 2)["finite"])

    def test_paired_geometry_requires_all_pose_fields(self):
        row = {"listener_base_position_world": [0, 0, 0], "listener_sensor_position_world": [0, 1.5, 0], "listener_yaw_deg": 10, "source_position_world": [1, 1, 0]}
        self.assertTrue(paired_geometry_equal(row, dict(row)))
        altered = dict(row)
        altered["listener_yaw_deg"] = 11
        self.assertFalse(paired_geometry_equal(row, altered))

    def test_split_excludes_failures_and_keeps_unassigned(self):
        rows = [
            {"scene_id": "replica.a", "scene_family": "Replica", "admitted_status": "PASS"},
            {"scene_id": "mp3d.b", "scene_family": "MP3D", "admitted_status": "FAIL", "exclude_reason": "FAIL_FOA_RENDER"},
        ]
        result = split_results(rows, "v1")
        self.assertEqual(len(result["scenes"]), 1)
        self.assertEqual(result["excluded"][0].get("split"), None)


if __name__ == "__main__":
    unittest.main()
```
## 3. Split Core Code

### `tools/clsdoa_v1/split_admitted_scenes.py`

```python
"""Create a stable family-wise split from PASS-only admission results."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# 允许以脚本路径直接执行 split helper。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.clsdoa_v1.admit_scenes import stable_split


def split_results(results: List[Dict[str, Any]], version: str) -> Dict[str, Any]:
    if any(item.get("admitted_status") == "REVIEW_REQUIRED" for item in results):
        raise ValueError("cannot split while REVIEW_REQUIRED results remain")
    admitted = []
    for item in results:
        copy = dict(item)
        if item.get("admitted_status") == "PASS":
            copy["split"] = stable_split(item["scene_family"], item["scene_id"], version)
            admitted.append(copy)
        else:
            copy["split"] = "UNASSIGNED"
    return {"split_version": version, "scenes": admitted, "excluded": [item for item in results if item.get("admitted_status") != "PASS"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--version", default="clsdoa_v1_scene_split_v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = split_results(json.loads(Path(args.results).read_text(encoding="utf-8")), args.version)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"admitted": len(result["scenes"]), "excluded": len(result["excluded"])}, sort_keys=True))


if __name__ == "__main__":
    main()
```
The split algorithm hashes `clsdoa_v1_scene_split_v1|scene_family|scene_id` with SHA-256, maps the first 64-bit integer to `[0,1)`, assigns `<0.70` to train, `<0.85` to val, and the remainder to test. Only `admitted_status == PASS` rows receive a split; all other rows remain `UNASSIGNED`. The helper rejects any `REVIEW_REQUIRED` row before splitting. The exported split was checked for duplicate scene IDs: `0` overlap/duplicates. The existing deterministic tests cover stable split output, PASS-only filtering, and failure exclusion; there is no separate overlap assertion in the unit test file, so the zero-overlap check here is an export-time validation.

### Complete PASS split assignment

The source split report is `data/logs/clsdoa_v1/scene_admission_rerun_002/scene_split_report.json`. Every PASS assignment is reproduced below; excluded rows are the five FAIL records in Section 5 and remain `UNASSIGNED`.

```text
Replica/train (14): replica.apartment_2, replica.frl_apartment_0, replica.frl_apartment_1, replica.frl_apartment_2, replica.frl_apartment_3, replica.frl_apartment_4, replica.frl_apartment_5, replica.hotel_0, replica.office_0, replica.office_1, replica.office_4, replica.room_0, replica.room_1, replica.room_2
Replica/val (3): replica.apartment_1, replica.office_2, replica.office_3
Replica/test (1): replica.apartment_0
MP3D/train (54): mp3d.1LXtFkjw3qL, mp3d.1pXnuDYAj8r, mp3d.2t7WUuJeko7, mp3d.5q7pvUzZiYa, mp3d.8194nk5LbLH, mp3d.82sE5b5pLXE, mp3d.8WUmhLawc2A, mp3d.B6ByNegPMKs, mp3d.D7N2EKCX4Sj, mp3d.EDJbREhghzL, mp3d.EU6Fwq7SyZv, mp3d.GdvgFV5R1Z5, mp3d.HxpKQynjfin, mp3d.JeFG25nYj2p, mp3d.JmbYfDe2QKZ, mp3d.PX4nDJXEHrG, mp3d.S9hNv5qa7GM, mp3d.SN83YJsR3w2, mp3d.UwV83HsGsw3, mp3d.Uxmj2M2itWa, mp3d.V2XKFyX4ASd, mp3d.VFuaQ6m2Qom, mp3d.VLzqgDo317F, mp3d.VVfe2KiqLaN, mp3d.XcA2TqTSSAj, mp3d.YFuZgdQ5vWj, mp3d.YVUC4YcDtcY, mp3d.Z6MFQCViBuw, mp3d.ZMojNkEp431, mp3d.aayBHfsNo7d, mp3d.ac26ZMwG7aT, mp3d.b8cTxDM8gDG, mp3d.gxdoqLR6rwA, mp3d.i5noydFURQK, mp3d.jh4fc5c5qoQ, mp3d.jtcxE69GiFV, mp3d.mJXqzFtmKg4, mp3d.oLBMNvg9in8, mp3d.p5wJjkQkbXX, mp3d.pLe4wQe7qrG, mp3d.pRbA3pwrgk9, mp3d.pa4otMbVnkk, mp3d.q9vSo1VnCiC, mp3d.r1Q1Z4BcV1o, mp3d.r47D5H71a5s, mp3d.rPc6DW4iMge, mp3d.rqfALeAoiTq, mp3d.sKLMLpTHeUy, mp3d.sT4fr6TAbpF, mp3d.uNb9QFRL6hY, mp3d.vyrNrziPKCB, mp3d.wc2JMjhGNzB, mp3d.x8F5xyUWy9e, mp3d.zsNo4HB9uLZ
MP3D/val (16): mp3d.17DRP5sb8fy, mp3d.2n8kARJN3HM, mp3d.5LpN3gDmAk7, mp3d.7y3sRwLe3Va, mp3d.D7G3Y4RVNrH, mp3d.JF19kD82Mey, mp3d.Pm6F8kyY3z2, mp3d.QUCTc6BB5sX, mp3d.TbHJrupSAjP, mp3d.Vt2qJdWjCF2, mp3d.X7HyMhZNoso, mp3d.cV4RVeZvu5T, mp3d.gTV8FGcVJC9, mp3d.qoiz87JEwZ2, mp3d.s8pcmisQ38h, mp3d.yqstnuAEVhm
MP3D/test (15): mp3d.29hnd4uzFmX, mp3d.2azQ1b91cZZ, mp3d.5ZKStnWn8Zo, mp3d.759xd9YjKW5, mp3d.ARNzJeq3xxb, mp3d.PuKPg4mmafe, mp3d.RPmz2sHmrrY, mp3d.ULsKaCPVFJR, mp3d.Vvot9Ly1tCj, mp3d.WYY7iVyf5p8, mp3d.YmJkqBEsHnH, mp3d.e9zR4mvMWw7, mp3d.gYvKGZ5eRqb, mp3d.gZ6f7yhEvPG, mp3d.ur6pFq6Qu1A
```

## 4. Registry

The complete V1 registry is reproduced below. It contains only PASS entries.

```yaml
dataset: clsdoa_v1
admission_version: clsdoa_v1_scene_admission_v1
split_version: clsdoa_v1_scene_split_v1
materials_mode: 'off'
scenes:
  mp3d.17DRP5sb8fy:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb
    navmesh: data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh
    semantic_info: data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.1LXtFkjw3qL:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb
    navmesh: data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.navmesh
    semantic_info: data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.1pXnuDYAj8r:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb
    navmesh: data/scene_datasets/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.navmesh
    semantic_info: data/scene_datasets/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.29hnd4uzFmX:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/29hnd4uzFmX/29hnd4uzFmX.glb
    navmesh: data/scene_datasets/mp3d/29hnd4uzFmX/29hnd4uzFmX.navmesh
    semantic_info: data/scene_datasets/mp3d/29hnd4uzFmX/29hnd4uzFmX.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.2azQ1b91cZZ:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb
    navmesh: data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.navmesh
    semantic_info: data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.2n8kARJN3HM:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/2n8kARJN3HM/2n8kARJN3HM.glb
    navmesh: data/scene_datasets/mp3d/2n8kARJN3HM/2n8kARJN3HM.navmesh
    semantic_info: data/scene_datasets/mp3d/2n8kARJN3HM/2n8kARJN3HM.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.2t7WUuJeko7:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/2t7WUuJeko7/2t7WUuJeko7.glb
    navmesh: data/scene_datasets/mp3d/2t7WUuJeko7/2t7WUuJeko7.navmesh
    semantic_info: data/scene_datasets/mp3d/2t7WUuJeko7/2t7WUuJeko7.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.5LpN3gDmAk7:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.glb
    navmesh: data/scene_datasets/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.navmesh
    semantic_info: data/scene_datasets/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.5ZKStnWn8Zo:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/5ZKStnWn8Zo/5ZKStnWn8Zo.glb
    navmesh: data/scene_datasets/mp3d/5ZKStnWn8Zo/5ZKStnWn8Zo.navmesh
    semantic_info: data/scene_datasets/mp3d/5ZKStnWn8Zo/5ZKStnWn8Zo.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.5q7pvUzZiYa:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/5q7pvUzZiYa/5q7pvUzZiYa.glb
    navmesh: data/scene_datasets/mp3d/5q7pvUzZiYa/5q7pvUzZiYa.navmesh
    semantic_info: data/scene_datasets/mp3d/5q7pvUzZiYa/5q7pvUzZiYa.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.759xd9YjKW5:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/759xd9YjKW5/759xd9YjKW5.glb
    navmesh: data/scene_datasets/mp3d/759xd9YjKW5/759xd9YjKW5.navmesh
    semantic_info: data/scene_datasets/mp3d/759xd9YjKW5/759xd9YjKW5.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.7y3sRwLe3Va:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb
    navmesh: data/scene_datasets/mp3d/7y3sRwLe3Va/7y3sRwLe3Va.navmesh
    semantic_info: data/scene_datasets/mp3d/7y3sRwLe3Va/7y3sRwLe3Va.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.8194nk5LbLH:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/8194nk5LbLH/8194nk5LbLH.glb
    navmesh: data/scene_datasets/mp3d/8194nk5LbLH/8194nk5LbLH.navmesh
    semantic_info: data/scene_datasets/mp3d/8194nk5LbLH/8194nk5LbLH.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.82sE5b5pLXE:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/82sE5b5pLXE/82sE5b5pLXE.glb
    navmesh: data/scene_datasets/mp3d/82sE5b5pLXE/82sE5b5pLXE.navmesh
    semantic_info: data/scene_datasets/mp3d/82sE5b5pLXE/82sE5b5pLXE.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.8WUmhLawc2A:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb
    navmesh: data/scene_datasets/mp3d/8WUmhLawc2A/8WUmhLawc2A.navmesh
    semantic_info: data/scene_datasets/mp3d/8WUmhLawc2A/8WUmhLawc2A.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.ARNzJeq3xxb:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/ARNzJeq3xxb/ARNzJeq3xxb.glb
    navmesh: data/scene_datasets/mp3d/ARNzJeq3xxb/ARNzJeq3xxb.navmesh
    semantic_info: data/scene_datasets/mp3d/ARNzJeq3xxb/ARNzJeq3xxb.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.B6ByNegPMKs:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/B6ByNegPMKs/B6ByNegPMKs.glb
    navmesh: data/scene_datasets/mp3d/B6ByNegPMKs/B6ByNegPMKs.navmesh
    semantic_info: data/scene_datasets/mp3d/B6ByNegPMKs/B6ByNegPMKs.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.D7G3Y4RVNrH:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/D7G3Y4RVNrH/D7G3Y4RVNrH.glb
    navmesh: data/scene_datasets/mp3d/D7G3Y4RVNrH/D7G3Y4RVNrH.navmesh
    semantic_info: data/scene_datasets/mp3d/D7G3Y4RVNrH/D7G3Y4RVNrH.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.D7N2EKCX4Sj:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/D7N2EKCX4Sj/D7N2EKCX4Sj.glb
    navmesh: data/scene_datasets/mp3d/D7N2EKCX4Sj/D7N2EKCX4Sj.navmesh
    semantic_info: data/scene_datasets/mp3d/D7N2EKCX4Sj/D7N2EKCX4Sj.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.EDJbREhghzL:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/EDJbREhghzL/EDJbREhghzL.glb
    navmesh: data/scene_datasets/mp3d/EDJbREhghzL/EDJbREhghzL.navmesh
    semantic_info: data/scene_datasets/mp3d/EDJbREhghzL/EDJbREhghzL.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.EU6Fwq7SyZv:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb
    navmesh: data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.navmesh
    semantic_info: data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.GdvgFV5R1Z5:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/GdvgFV5R1Z5/GdvgFV5R1Z5.glb
    navmesh: data/scene_datasets/mp3d/GdvgFV5R1Z5/GdvgFV5R1Z5.navmesh
    semantic_info: data/scene_datasets/mp3d/GdvgFV5R1Z5/GdvgFV5R1Z5.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.HxpKQynjfin:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/HxpKQynjfin/HxpKQynjfin.glb
    navmesh: data/scene_datasets/mp3d/HxpKQynjfin/HxpKQynjfin.navmesh
    semantic_info: data/scene_datasets/mp3d/HxpKQynjfin/HxpKQynjfin.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.JF19kD82Mey:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/JF19kD82Mey/JF19kD82Mey.glb
    navmesh: data/scene_datasets/mp3d/JF19kD82Mey/JF19kD82Mey.navmesh
    semantic_info: data/scene_datasets/mp3d/JF19kD82Mey/JF19kD82Mey.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.JeFG25nYj2p:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/JeFG25nYj2p/JeFG25nYj2p.glb
    navmesh: data/scene_datasets/mp3d/JeFG25nYj2p/JeFG25nYj2p.navmesh
    semantic_info: data/scene_datasets/mp3d/JeFG25nYj2p/JeFG25nYj2p.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.JmbYfDe2QKZ:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/JmbYfDe2QKZ/JmbYfDe2QKZ.glb
    navmesh: data/scene_datasets/mp3d/JmbYfDe2QKZ/JmbYfDe2QKZ.navmesh
    semantic_info: data/scene_datasets/mp3d/JmbYfDe2QKZ/JmbYfDe2QKZ.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.PX4nDJXEHrG:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/PX4nDJXEHrG/PX4nDJXEHrG.glb
    navmesh: data/scene_datasets/mp3d/PX4nDJXEHrG/PX4nDJXEHrG.navmesh
    semantic_info: data/scene_datasets/mp3d/PX4nDJXEHrG/PX4nDJXEHrG.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.Pm6F8kyY3z2:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/Pm6F8kyY3z2/Pm6F8kyY3z2.glb
    navmesh: data/scene_datasets/mp3d/Pm6F8kyY3z2/Pm6F8kyY3z2.navmesh
    semantic_info: data/scene_datasets/mp3d/Pm6F8kyY3z2/Pm6F8kyY3z2.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.PuKPg4mmafe:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/PuKPg4mmafe/PuKPg4mmafe.glb
    navmesh: data/scene_datasets/mp3d/PuKPg4mmafe/PuKPg4mmafe.navmesh
    semantic_info: data/scene_datasets/mp3d/PuKPg4mmafe/PuKPg4mmafe.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.QUCTc6BB5sX:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb
    navmesh: data/scene_datasets/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.navmesh
    semantic_info: data/scene_datasets/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.RPmz2sHmrrY:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/RPmz2sHmrrY/RPmz2sHmrrY.glb
    navmesh: data/scene_datasets/mp3d/RPmz2sHmrrY/RPmz2sHmrrY.navmesh
    semantic_info: data/scene_datasets/mp3d/RPmz2sHmrrY/RPmz2sHmrrY.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.S9hNv5qa7GM:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/S9hNv5qa7GM/S9hNv5qa7GM.glb
    navmesh: data/scene_datasets/mp3d/S9hNv5qa7GM/S9hNv5qa7GM.navmesh
    semantic_info: data/scene_datasets/mp3d/S9hNv5qa7GM/S9hNv5qa7GM.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.SN83YJsR3w2:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/SN83YJsR3w2/SN83YJsR3w2.glb
    navmesh: data/scene_datasets/mp3d/SN83YJsR3w2/SN83YJsR3w2.navmesh
    semantic_info: data/scene_datasets/mp3d/SN83YJsR3w2/SN83YJsR3w2.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.TbHJrupSAjP:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb
    navmesh: data/scene_datasets/mp3d/TbHJrupSAjP/TbHJrupSAjP.navmesh
    semantic_info: data/scene_datasets/mp3d/TbHJrupSAjP/TbHJrupSAjP.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.ULsKaCPVFJR:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/ULsKaCPVFJR/ULsKaCPVFJR.glb
    navmesh: data/scene_datasets/mp3d/ULsKaCPVFJR/ULsKaCPVFJR.navmesh
    semantic_info: data/scene_datasets/mp3d/ULsKaCPVFJR/ULsKaCPVFJR.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.UwV83HsGsw3:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/UwV83HsGsw3/UwV83HsGsw3.glb
    navmesh: data/scene_datasets/mp3d/UwV83HsGsw3/UwV83HsGsw3.navmesh
    semantic_info: data/scene_datasets/mp3d/UwV83HsGsw3/UwV83HsGsw3.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.Uxmj2M2itWa:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/Uxmj2M2itWa/Uxmj2M2itWa.glb
    navmesh: data/scene_datasets/mp3d/Uxmj2M2itWa/Uxmj2M2itWa.navmesh
    semantic_info: data/scene_datasets/mp3d/Uxmj2M2itWa/Uxmj2M2itWa.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.V2XKFyX4ASd:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/V2XKFyX4ASd/V2XKFyX4ASd.glb
    navmesh: data/scene_datasets/mp3d/V2XKFyX4ASd/V2XKFyX4ASd.navmesh
    semantic_info: data/scene_datasets/mp3d/V2XKFyX4ASd/V2XKFyX4ASd.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.VFuaQ6m2Qom:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/VFuaQ6m2Qom/VFuaQ6m2Qom.glb
    navmesh: data/scene_datasets/mp3d/VFuaQ6m2Qom/VFuaQ6m2Qom.navmesh
    semantic_info: data/scene_datasets/mp3d/VFuaQ6m2Qom/VFuaQ6m2Qom.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.VLzqgDo317F:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/VLzqgDo317F/VLzqgDo317F.glb
    navmesh: data/scene_datasets/mp3d/VLzqgDo317F/VLzqgDo317F.navmesh
    semantic_info: data/scene_datasets/mp3d/VLzqgDo317F/VLzqgDo317F.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.VVfe2KiqLaN:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/VVfe2KiqLaN/VVfe2KiqLaN.glb
    navmesh: data/scene_datasets/mp3d/VVfe2KiqLaN/VVfe2KiqLaN.navmesh
    semantic_info: data/scene_datasets/mp3d/VVfe2KiqLaN/VVfe2KiqLaN.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.Vt2qJdWjCF2:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/Vt2qJdWjCF2/Vt2qJdWjCF2.glb
    navmesh: data/scene_datasets/mp3d/Vt2qJdWjCF2/Vt2qJdWjCF2.navmesh
    semantic_info: data/scene_datasets/mp3d/Vt2qJdWjCF2/Vt2qJdWjCF2.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.Vvot9Ly1tCj:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/Vvot9Ly1tCj/Vvot9Ly1tCj.glb
    navmesh: data/scene_datasets/mp3d/Vvot9Ly1tCj/Vvot9Ly1tCj.navmesh
    semantic_info: data/scene_datasets/mp3d/Vvot9Ly1tCj/Vvot9Ly1tCj.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.WYY7iVyf5p8:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/WYY7iVyf5p8/WYY7iVyf5p8.glb
    navmesh: data/scene_datasets/mp3d/WYY7iVyf5p8/WYY7iVyf5p8.navmesh
    semantic_info: data/scene_datasets/mp3d/WYY7iVyf5p8/WYY7iVyf5p8.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.X7HyMhZNoso:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb
    navmesh: data/scene_datasets/mp3d/X7HyMhZNoso/X7HyMhZNoso.navmesh
    semantic_info: data/scene_datasets/mp3d/X7HyMhZNoso/X7HyMhZNoso.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.XcA2TqTSSAj:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/XcA2TqTSSAj/XcA2TqTSSAj.glb
    navmesh: data/scene_datasets/mp3d/XcA2TqTSSAj/XcA2TqTSSAj.navmesh
    semantic_info: data/scene_datasets/mp3d/XcA2TqTSSAj/XcA2TqTSSAj.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.YFuZgdQ5vWj:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/YFuZgdQ5vWj/YFuZgdQ5vWj.glb
    navmesh: data/scene_datasets/mp3d/YFuZgdQ5vWj/YFuZgdQ5vWj.navmesh
    semantic_info: data/scene_datasets/mp3d/YFuZgdQ5vWj/YFuZgdQ5vWj.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.YVUC4YcDtcY:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/YVUC4YcDtcY/YVUC4YcDtcY.glb
    navmesh: data/scene_datasets/mp3d/YVUC4YcDtcY/YVUC4YcDtcY.navmesh
    semantic_info: data/scene_datasets/mp3d/YVUC4YcDtcY/YVUC4YcDtcY.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.YmJkqBEsHnH:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/YmJkqBEsHnH/YmJkqBEsHnH.glb
    navmesh: data/scene_datasets/mp3d/YmJkqBEsHnH/YmJkqBEsHnH.navmesh
    semantic_info: data/scene_datasets/mp3d/YmJkqBEsHnH/YmJkqBEsHnH.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.Z6MFQCViBuw:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb
    navmesh: data/scene_datasets/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.navmesh
    semantic_info: data/scene_datasets/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.ZMojNkEp431:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/ZMojNkEp431/ZMojNkEp431.glb
    navmesh: data/scene_datasets/mp3d/ZMojNkEp431/ZMojNkEp431.navmesh
    semantic_info: data/scene_datasets/mp3d/ZMojNkEp431/ZMojNkEp431.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.aayBHfsNo7d:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/aayBHfsNo7d/aayBHfsNo7d.glb
    navmesh: data/scene_datasets/mp3d/aayBHfsNo7d/aayBHfsNo7d.navmesh
    semantic_info: data/scene_datasets/mp3d/aayBHfsNo7d/aayBHfsNo7d.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.ac26ZMwG7aT:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/ac26ZMwG7aT/ac26ZMwG7aT.glb
    navmesh: data/scene_datasets/mp3d/ac26ZMwG7aT/ac26ZMwG7aT.navmesh
    semantic_info: data/scene_datasets/mp3d/ac26ZMwG7aT/ac26ZMwG7aT.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.b8cTxDM8gDG:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/b8cTxDM8gDG/b8cTxDM8gDG.glb
    navmesh: data/scene_datasets/mp3d/b8cTxDM8gDG/b8cTxDM8gDG.navmesh
    semantic_info: data/scene_datasets/mp3d/b8cTxDM8gDG/b8cTxDM8gDG.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.cV4RVeZvu5T:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/cV4RVeZvu5T/cV4RVeZvu5T.glb
    navmesh: data/scene_datasets/mp3d/cV4RVeZvu5T/cV4RVeZvu5T.navmesh
    semantic_info: data/scene_datasets/mp3d/cV4RVeZvu5T/cV4RVeZvu5T.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.e9zR4mvMWw7:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/e9zR4mvMWw7/e9zR4mvMWw7.glb
    navmesh: data/scene_datasets/mp3d/e9zR4mvMWw7/e9zR4mvMWw7.navmesh
    semantic_info: data/scene_datasets/mp3d/e9zR4mvMWw7/e9zR4mvMWw7.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.gTV8FGcVJC9:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/gTV8FGcVJC9/gTV8FGcVJC9.glb
    navmesh: data/scene_datasets/mp3d/gTV8FGcVJC9/gTV8FGcVJC9.navmesh
    semantic_info: data/scene_datasets/mp3d/gTV8FGcVJC9/gTV8FGcVJC9.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.gYvKGZ5eRqb:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/gYvKGZ5eRqb/gYvKGZ5eRqb.glb
    navmesh: data/scene_datasets/mp3d/gYvKGZ5eRqb/gYvKGZ5eRqb.navmesh
    semantic_info: data/scene_datasets/mp3d/gYvKGZ5eRqb/gYvKGZ5eRqb.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.gZ6f7yhEvPG:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb
    navmesh: data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.navmesh
    semantic_info: data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.gxdoqLR6rwA:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/gxdoqLR6rwA/gxdoqLR6rwA.glb
    navmesh: data/scene_datasets/mp3d/gxdoqLR6rwA/gxdoqLR6rwA.navmesh
    semantic_info: data/scene_datasets/mp3d/gxdoqLR6rwA/gxdoqLR6rwA.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.i5noydFURQK:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/i5noydFURQK/i5noydFURQK.glb
    navmesh: data/scene_datasets/mp3d/i5noydFURQK/i5noydFURQK.navmesh
    semantic_info: data/scene_datasets/mp3d/i5noydFURQK/i5noydFURQK.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.jh4fc5c5qoQ:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/jh4fc5c5qoQ/jh4fc5c5qoQ.glb
    navmesh: data/scene_datasets/mp3d/jh4fc5c5qoQ/jh4fc5c5qoQ.navmesh
    semantic_info: data/scene_datasets/mp3d/jh4fc5c5qoQ/jh4fc5c5qoQ.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.jtcxE69GiFV:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/jtcxE69GiFV/jtcxE69GiFV.glb
    navmesh: data/scene_datasets/mp3d/jtcxE69GiFV/jtcxE69GiFV.navmesh
    semantic_info: data/scene_datasets/mp3d/jtcxE69GiFV/jtcxE69GiFV.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.mJXqzFtmKg4:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/mJXqzFtmKg4/mJXqzFtmKg4.glb
    navmesh: data/scene_datasets/mp3d/mJXqzFtmKg4/mJXqzFtmKg4.navmesh
    semantic_info: data/scene_datasets/mp3d/mJXqzFtmKg4/mJXqzFtmKg4.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.oLBMNvg9in8:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/oLBMNvg9in8/oLBMNvg9in8.glb
    navmesh: data/scene_datasets/mp3d/oLBMNvg9in8/oLBMNvg9in8.navmesh
    semantic_info: data/scene_datasets/mp3d/oLBMNvg9in8/oLBMNvg9in8.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.p5wJjkQkbXX:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/p5wJjkQkbXX/p5wJjkQkbXX.glb
    navmesh: data/scene_datasets/mp3d/p5wJjkQkbXX/p5wJjkQkbXX.navmesh
    semantic_info: data/scene_datasets/mp3d/p5wJjkQkbXX/p5wJjkQkbXX.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.pLe4wQe7qrG:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb
    navmesh: data/scene_datasets/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.navmesh
    semantic_info: data/scene_datasets/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.pRbA3pwrgk9:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/pRbA3pwrgk9/pRbA3pwrgk9.glb
    navmesh: data/scene_datasets/mp3d/pRbA3pwrgk9/pRbA3pwrgk9.navmesh
    semantic_info: data/scene_datasets/mp3d/pRbA3pwrgk9/pRbA3pwrgk9.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.pa4otMbVnkk:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/pa4otMbVnkk/pa4otMbVnkk.glb
    navmesh: data/scene_datasets/mp3d/pa4otMbVnkk/pa4otMbVnkk.navmesh
    semantic_info: data/scene_datasets/mp3d/pa4otMbVnkk/pa4otMbVnkk.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.q9vSo1VnCiC:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/q9vSo1VnCiC/q9vSo1VnCiC.glb
    navmesh: data/scene_datasets/mp3d/q9vSo1VnCiC/q9vSo1VnCiC.navmesh
    semantic_info: data/scene_datasets/mp3d/q9vSo1VnCiC/q9vSo1VnCiC.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.qoiz87JEwZ2:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/qoiz87JEwZ2/qoiz87JEwZ2.glb
    navmesh: data/scene_datasets/mp3d/qoiz87JEwZ2/qoiz87JEwZ2.navmesh
    semantic_info: data/scene_datasets/mp3d/qoiz87JEwZ2/qoiz87JEwZ2.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.r1Q1Z4BcV1o:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/r1Q1Z4BcV1o/r1Q1Z4BcV1o.glb
    navmesh: data/scene_datasets/mp3d/r1Q1Z4BcV1o/r1Q1Z4BcV1o.navmesh
    semantic_info: data/scene_datasets/mp3d/r1Q1Z4BcV1o/r1Q1Z4BcV1o.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.r47D5H71a5s:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/r47D5H71a5s/r47D5H71a5s.glb
    navmesh: data/scene_datasets/mp3d/r47D5H71a5s/r47D5H71a5s.navmesh
    semantic_info: data/scene_datasets/mp3d/r47D5H71a5s/r47D5H71a5s.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.rPc6DW4iMge:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/rPc6DW4iMge/rPc6DW4iMge.glb
    navmesh: data/scene_datasets/mp3d/rPc6DW4iMge/rPc6DW4iMge.navmesh
    semantic_info: data/scene_datasets/mp3d/rPc6DW4iMge/rPc6DW4iMge.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.rqfALeAoiTq:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/rqfALeAoiTq/rqfALeAoiTq.glb
    navmesh: data/scene_datasets/mp3d/rqfALeAoiTq/rqfALeAoiTq.navmesh
    semantic_info: data/scene_datasets/mp3d/rqfALeAoiTq/rqfALeAoiTq.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.s8pcmisQ38h:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/s8pcmisQ38h/s8pcmisQ38h.glb
    navmesh: data/scene_datasets/mp3d/s8pcmisQ38h/s8pcmisQ38h.navmesh
    semantic_info: data/scene_datasets/mp3d/s8pcmisQ38h/s8pcmisQ38h.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.sKLMLpTHeUy:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/sKLMLpTHeUy/sKLMLpTHeUy.glb
    navmesh: data/scene_datasets/mp3d/sKLMLpTHeUy/sKLMLpTHeUy.navmesh
    semantic_info: data/scene_datasets/mp3d/sKLMLpTHeUy/sKLMLpTHeUy.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.sT4fr6TAbpF:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/sT4fr6TAbpF/sT4fr6TAbpF.glb
    navmesh: data/scene_datasets/mp3d/sT4fr6TAbpF/sT4fr6TAbpF.navmesh
    semantic_info: data/scene_datasets/mp3d/sT4fr6TAbpF/sT4fr6TAbpF.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.uNb9QFRL6hY:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/uNb9QFRL6hY/uNb9QFRL6hY.glb
    navmesh: data/scene_datasets/mp3d/uNb9QFRL6hY/uNb9QFRL6hY.navmesh
    semantic_info: data/scene_datasets/mp3d/uNb9QFRL6hY/uNb9QFRL6hY.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.ur6pFq6Qu1A:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/ur6pFq6Qu1A/ur6pFq6Qu1A.glb
    navmesh: data/scene_datasets/mp3d/ur6pFq6Qu1A/ur6pFq6Qu1A.navmesh
    semantic_info: data/scene_datasets/mp3d/ur6pFq6Qu1A/ur6pFq6Qu1A.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.vyrNrziPKCB:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/vyrNrziPKCB/vyrNrziPKCB.glb
    navmesh: data/scene_datasets/mp3d/vyrNrziPKCB/vyrNrziPKCB.navmesh
    semantic_info: data/scene_datasets/mp3d/vyrNrziPKCB/vyrNrziPKCB.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.wc2JMjhGNzB:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/wc2JMjhGNzB/wc2JMjhGNzB.glb
    navmesh: data/scene_datasets/mp3d/wc2JMjhGNzB/wc2JMjhGNzB.navmesh
    semantic_info: data/scene_datasets/mp3d/wc2JMjhGNzB/wc2JMjhGNzB.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.x8F5xyUWy9e:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb
    navmesh: data/scene_datasets/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.navmesh
    semantic_info: data/scene_datasets/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.yqstnuAEVhm:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/yqstnuAEVhm/yqstnuAEVhm.glb
    navmesh: data/scene_datasets/mp3d/yqstnuAEVhm/yqstnuAEVhm.navmesh
    semantic_info: data/scene_datasets/mp3d/yqstnuAEVhm/yqstnuAEVhm.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  mp3d.zsNo4HB9uLZ:
    dataset: mp3d
    scene_family: MP3D
    scene_asset: data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb
    navmesh: data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.navmesh
    semantic_info: data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.house
    stage_config: null
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.apartment_0:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/apartment_0/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/apartment_0/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/apartment_0/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/apartment_0/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: test
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.apartment_1:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/apartment_1/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/apartment_1/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/apartment_1/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/apartment_1/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.apartment_2:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/apartment_2/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/apartment_2/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/apartment_2/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/apartment_2/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.frl_apartment_0:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/frl_apartment_0/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/frl_apartment_0/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/frl_apartment_0/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/frl_apartment_0/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.frl_apartment_1:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/frl_apartment_1/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/frl_apartment_1/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/frl_apartment_1/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/frl_apartment_1/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.frl_apartment_2:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/frl_apartment_2/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/frl_apartment_2/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/frl_apartment_2/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/frl_apartment_2/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.frl_apartment_3:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/frl_apartment_3/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/frl_apartment_3/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/frl_apartment_3/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/frl_apartment_3/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.frl_apartment_4:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/frl_apartment_4/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/frl_apartment_4/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/frl_apartment_4/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/frl_apartment_4/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.frl_apartment_5:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/frl_apartment_5/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/frl_apartment_5/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/frl_apartment_5/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/frl_apartment_5/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.hotel_0:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/hotel_0/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/hotel_0/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/hotel_0/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/hotel_0/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.office_0:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/office_0/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/office_0/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/office_0/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/office_0/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.office_1:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/office_1/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/office_1/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/office_1/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.office_2:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/office_2/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/office_2/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/office_2/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/office_2/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.office_3:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/office_3/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/office_3/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/office_3/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/office_3/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: val
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.office_4:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/office_4/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/office_4/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/office_4/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/office_4/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.room_0:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/room_0/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/room_0/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/room_0/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/room_0/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.room_1:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/room_1/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/room_1/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/room_1/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/room_1/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
  replica.room_2:
    dataset: replica
    scene_family: Replica
    scene_asset: data/scene_datasets/replica/room_2/habitat/mesh_semantic.ply
    navmesh: data/scene_datasets/replica/room_2/habitat/mesh_semantic.navmesh
    semantic_info: data/scene_datasets/replica/room_2/habitat/info_semantic.json
    stage_config: data/scene_datasets/replica/room_2/habitat/replica_stage.stage_config.json
    materials_mode: 'off'
    admitted: PASS
    split: train
    unit_scale_status: PASS_WITH_REVIEW
    admission_commit: 53ecc6d
```

Registry statistics: total entries = **103**, Replica = **18**, MP3D = **85**. Split counts are Replica train/val/test = **14/3/1** and MP3D train/val/test = **54/16/15**. Unique `admitted` values = `['PASS']`; unique `materials_mode` values = `['off']`.
## 5. Admission Result Evidence

The source result file is `data/logs/clsdoa_v1/scene_admission_rerun_002/scene_admission_results.json`; its schema is represented by the complete worker result fields in Section 2 and the summary below. Ordinary PASS rows are summarized: total 108, PASS 103, FAIL 5, REVIEW_REQUIRED 0; Replica 18 PASS, MP3D 85 PASS. The PASS-only split source is `data/logs/clsdoa_v1/scene_admission_rerun_002/scene_split_report.json`; the exported split counts are listed above and all five exclusions remain `UNASSIGNED`.

### All FAIL records

```json
[
  {
    "admitted_status": "FAIL",
    "binaural_status": "NOT_RUN",
    "exclude_reason": "FAIL_GEOMETRY_CLEARANCE",
    "foa_status": "NOT_RUN",
    "listener_clearance_status": "FAIL",
    "navmesh_status": "PASS",
    "sampling_diagnostics": {
      "distance_rejects": 96,
      "listener_clearance_rejects": 0,
      "reachability_rejects": 4,
      "source_clearance_rejects": 0,
      "total_sampling_attempts": 100
    },
    "scene_family": "MP3D",
    "scene_id": "mp3d.E9uDoFAP3SH",
    "scene_load_status": "PASS",
    "source_clearance_status": "FAIL",
    "valid_probe_pair_count": 0
  },
  {
    "admitted_status": "FAIL",
    "binaural_status": "NOT_RUN",
    "exclude_reason": "FAIL_GEOMETRY_CLEARANCE",
    "foa_status": "NOT_RUN",
    "listener_clearance_status": "FAIL",
    "navmesh_status": "PASS",
    "sampling_diagnostics": {
      "distance_rejects": 83,
      "listener_clearance_rejects": 0,
      "reachability_rejects": 17,
      "source_clearance_rejects": 0,
      "total_sampling_attempts": 100
    },
    "scene_family": "MP3D",
    "scene_id": "mp3d.VzqfbhrpDEA",
    "scene_load_status": "PASS",
    "source_clearance_status": "FAIL",
    "valid_probe_pair_count": 0
  },
  {
    "admitted_status": "FAIL",
    "binaural_status": "NOT_RUN",
    "exclude_reason": "FAIL_GEOMETRY_CLEARANCE",
    "foa_status": "NOT_RUN",
    "listener_clearance_status": "PASS",
    "navmesh_status": "PASS",
    "sampling_diagnostics": {
      "distance_rejects": 83,
      "listener_clearance_rejects": 0,
      "reachability_rejects": 16,
      "source_clearance_rejects": 0,
      "total_sampling_attempts": 100
    },
    "scene_family": "MP3D",
    "scene_id": "mp3d.dhjEzFoUFzH",
    "scene_load_status": "PASS",
    "source_clearance_status": "PASS",
    "valid_probe_pair_count": 1
  },
  {
    "admitted_status": "FAIL",
    "binaural_status": "NOT_RUN",
    "exclude_reason": "FAIL_GEOMETRY_CLEARANCE",
    "foa_status": "NOT_RUN",
    "listener_clearance_status": "PASS",
    "navmesh_status": "PASS",
    "sampling_diagnostics": {
      "distance_rejects": 67,
      "listener_clearance_rejects": 0,
      "reachability_rejects": 32,
      "source_clearance_rejects": 0,
      "total_sampling_attempts": 100
    },
    "scene_family": "MP3D",
    "scene_id": "mp3d.fzynW3qQPVF",
    "scene_load_status": "PASS",
    "source_clearance_status": "PASS",
    "valid_probe_pair_count": 1
  },
  {
    "admitted_status": "FAIL",
    "binaural_status": "NOT_RUN",
    "exclude_reason": "FAIL_GEOMETRY_CLEARANCE",
    "foa_status": "NOT_RUN",
    "listener_clearance_status": "PASS",
    "navmesh_status": "PASS",
    "sampling_diagnostics": {
      "distance_rejects": 63,
      "listener_clearance_rejects": 0,
      "reachability_rejects": 36,
      "source_clearance_rejects": 0,
      "total_sampling_attempts": 100
    },
    "scene_family": "MP3D",
    "scene_id": "mp3d.kEZ7cmS4wCh",
    "scene_load_status": "PASS",
    "source_clearance_status": "PASS",
    "valid_probe_pair_count": 1
  }
]
```

### All soft acoustic outlier records

```json
{
  "hard_exclusion_applied": false,
  "method": "3xIQR on PASS Binaural probe energy",
  "outlier_count": 4,
  "outliers": [
    {
      "lower": 0,
      "metric": "binaural_energy",
      "probe_index": 1,
      "scene_id": "mp3d.Vt2qJdWjCF2",
      "upper": 5.713634693480594,
      "value": 9.126357193270204
    },
    {
      "lower": 0,
      "metric": "binaural_energy",
      "probe_index": 0,
      "scene_id": "replica.frl_apartment_2",
      "upper": 5.713634693480594,
      "value": 6.348157963826868
    },
    {
      "lower": 0,
      "metric": "binaural_energy",
      "probe_index": 0,
      "scene_id": "replica.office_0",
      "upper": 5.713634693480594,
      "value": 6.14222098683292
    },
    {
      "lower": 0,
      "metric": "binaural_energy",
      "probe_index": 0,
      "scene_id": "replica.office_1",
      "upper": 5.713634693480594,
      "value": 6.4354470015730225
    }
  ]
}
```

The four outlier scenes are `mp3d.Vt2qJdWjCF2`, `replica.frl_apartment_2`, `replica.office_0`, and `replica.office_1`. Each triggered `binaural_energy` using the 3xIQR rule: lower threshold `0`, upper threshold `5.713634693480594`; values were respectively `9.126357193270204`, `6.348157963826868`, `6.14222098683292`, and `6.4354470015730225` at the recorded probe indices. All four had final `admitted_status=PASS`; outliers were soft-only and did not alter admission.

### Unit-scale evidence

Source report: `docs/audits/clsdoa_v1/scene_admission/mp3d_unit_scale_report.json`. Its raw official evidence is `SoundSpaces2.md`, line 41: `| unitScale | float | 1.f | Unit scale for the scene. Mesh and positions are multiplied by this factor |`. The report records `status=PASS_WITH_REVIEW`, 85 admitted MP3D scenes, finite probe coordinates, probe distances from `1.047123344730889` to `3.978875597911416`, and no observed tenfold anomaly. There is no explicit `meter`/`meters` word in the cited `unitScale` annotation; the admission config uses `_m` field names and the probe bounds are treated as engineering meters, but that is not independent official meter-unit proof. Human confirmation remains required.
## 6. Key Audit Answers

* The five FAIL IDs are the five records above; every one stopped before Binaural and FOA because `valid_probe_pair_count < 2`.
* The four soft outlier IDs and exact metric/value/threshold are listed above; all four remained PASS.
* Replica `14/3/1` and MP3D `54/16/15` come from the SHA-256 split algorithm and thresholds `0.70`/`0.85` in the committed config, applied independently by family and scene ID.
* No manual scene override was applied.
* No admission result, split assignment, registry content, unit-scale conclusion, acoustic outlier conclusion, or scene asset was changed by this export.

## 7. Read-only Verification

The required full unittest discovery completed with `75 passed, 0 failed`. `git diff --check` passed before commit. The bundle itself is the only intended new tracked file.
