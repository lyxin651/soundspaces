"""Determine whether native FOA coefficients are world-fixed or listener-local."""

import json
import math
from pathlib import Path

import quaternion  # 必须先于 habitat_sim 导入
import habitat_sim  # noqa: F401
import numpy as np

from check_foa_direct_only_repair import make_sim, render, set_pose
from foa_adapter import measure_shared_direct_coefficients


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data/logs/seld_dataset_v1_preflight"
SCENE = ROOT / "data/scene_datasets/replica/office_0/habitat/mesh_semantic.ply"
SOURCE = np.asarray([0.1810176075, 0.5311309695, 2.44338727], dtype=np.float32)
LISTENER = np.asarray([1.6240532398, 0.5311309695, -0.5125486851], dtype=np.float32)
YAWS = (0.0, 90.0, 180.0, -90.0)


def direction_from_coefficients(coefficients):
    w, y, z, x = np.asarray(coefficients, dtype=np.float64)
    vector = np.asarray([x, y, z]) / max(abs(w) * math.sqrt(3.0), 1e-12)
    return vector / max(np.linalg.norm(vector), 1e-12)


def angle_deg(left, right):
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right), -1.0, 1.0))))


def world_to_local(vector, yaw_deg):
    yaw = math.radians(yaw_deg)
    return np.asarray([math.cos(yaw) * vector[0] - math.sin(yaw) * vector[2], vector[1], math.sin(yaw) * vector[0] + math.cos(yaw) * vector[2]])


def main():
    sim = make_sim(SCENE, 24000, indirect=False)
    world_vector = SOURCE.astype(np.float64) - LISTENER.astype(np.float64)
    world_vector /= np.linalg.norm(world_vector)
    rows = []
    try:
        for yaw in YAWS:
            set_pose(sim, SOURCE, LISTENER, yaw)
            coefficients = measure_shared_direct_coefficients(render(sim))["signed_coefficients"]
            actual = direction_from_coefficients(coefficients)
            world_error = angle_deg(actual, world_vector)
            listener_error = angle_deg(actual, world_to_local(world_vector, yaw))
            rows.append({"yaw_deg": yaw, "shared_direct_coefficients": coefficients.tolist(), "world_frame_error_deg": world_error, "listener_frame_error_deg": listener_error})
    finally:
        sim.close()
    world_error = float(np.mean([row["world_frame_error_deg"] for row in rows]))
    listener_error = float(np.mean([row["listener_frame_error_deg"] for row in rows]))
    result = {"source_world": SOURCE.tolist(), "listener_world": LISTENER.tolist(), "rows": rows, "world_frame_error_deg": world_error, "listener_frame_error_deg": listener_error, "winner": "world_fixed" if world_error < listener_error else "listener_local", "margin_deg": abs(listener_error - world_error), "verdict": "PASS" if world_error < listener_error else "INCONCLUSIVE"}
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "foa_frame_behavior.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("world_frame_error_deg", "listener_frame_error_deg", "winner", "margin_deg", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
