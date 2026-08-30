"""Independent DCASE-style intensity-vector check for model-facing FOA."""

import json
import math
from pathlib import Path

import numpy as np

from foa_adapter import native_foa_to_canonical


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data/logs/seld_dataset_v1_preflight"
FIXTURE = ROOT / "data/seld_dataset_v1_preflight/fixtures/foa_direct_only_fixture.yaml"
IDENTIFICATION = LOG_ROOT / "foa_direct_only_identification.json"


def world_to_local(vector, yaw_deg):
    yaw = math.radians(yaw_deg)
    return np.asarray([math.cos(yaw) * vector[0] + math.sin(yaw) * vector[2], vector[1], -math.sin(yaw) * vector[0] + math.cos(yaw) * vector[2]], dtype=np.float64)


def angular_error_deg(actual, expected):
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    return float(np.degrees(np.arccos(np.clip(np.dot(actual, expected), -1.0, 1.0))))


def off_axis_yawed_fixture():
    """Return a synthetic world-fixed native FOA sample for an oblique yaw."""
    source = np.asarray([1.0, 0.0, 0.25], dtype=np.float64)
    listener = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    yaw_deg = 45.0
    world_direction = (source - listener) / np.linalg.norm(source - listener)
    # Native RLR channels are [W,Y,Z,X] and use N3D directional coefficients.
    signed_peak = np.asarray(
        [1.0, 0.0, np.sqrt(3.0) * world_direction[2], np.sqrt(3.0) * world_direction[0]],
        dtype=np.float32,
    )
    return {
        "id": "off_axis_yawed",
        "position_world": source.tolist(),
        "listener_position_world": listener.tolist(),
        "listener_yaw_deg": yaw_deg,
        "measurement": {"signed_peak": signed_peak.tolist()},
    }


def main():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    measured = json.loads(IDENTIFICATION.read_text(encoding="utf-8"))
    rows = []
    for source_spec, measurement_row in zip(fixture["sources"], measured["directions"]):
        source = np.asarray(source_spec["position_world"], dtype=np.float64)
        listener = np.asarray(source_spec.get("listener_position_world", fixture["listener_position_world"]), dtype=np.float64)
        world_direction = source - listener
        local_direction = world_to_local(world_direction, source_spec["listener_yaw_deg"])
        # DCASE/STARSS Cartesian axes are [front, left, up].
        expected_dcase = np.asarray([-local_direction[2], -local_direction[0], local_direction[1]])
        native = np.asarray(measurement_row["measurement"]["signed_peak"], dtype=np.float32)[:, None]
        canonical = native_foa_to_canonical(native, source_spec["listener_yaw_deg"])
        # Independent reference, following local DCASE SELD feature code:
        # external/seld_models/DCASE2025_seld_baseline/extract_features.py.
        # I = Re(conj(W) * [X,Y,Z]); only this minimal formula is used here.
        w = canonical[0, 0]
        intensity = np.asarray([canonical[3, 0], canonical[1, 0], canonical[2, 0]], dtype=np.float64) * float(w)
        error = angular_error_deg(intensity, expected_dcase)
        rows.append({"fixture": source_spec["id"], "angular_error_deg": error, "pass": error < 1.0})
    source_spec = off_axis_yawed_fixture()
    source = np.asarray(source_spec["position_world"], dtype=np.float64)
    listener = np.asarray(source_spec["listener_position_world"], dtype=np.float64)
    local_direction = world_to_local(source - listener, source_spec["listener_yaw_deg"])
    expected_dcase = np.asarray([-local_direction[2], -local_direction[0], local_direction[1]])
    native = np.asarray(source_spec["measurement"]["signed_peak"], dtype=np.float32)[:, None]
    canonical = native_foa_to_canonical(native, source_spec["listener_yaw_deg"])
    w = canonical[0, 0]
    intensity = np.asarray([canonical[3, 0], canonical[1, 0], canonical[2, 0]], dtype=np.float64) * float(w)
    error = angular_error_deg(intensity, expected_dcase)
    rows.append({"fixture": source_spec["id"], "angular_error_deg": error, "pass": error < 1.0})
    result = {"reference": "DCASE-style FOA intensity I=Re(conj(W)*directional channels), channel order [W,Y,Z,X]", "rows": rows, "max_angular_error_deg": max(row["angular_error_deg"] for row in rows), "verdict": "PASS" if all(row["pass"] for row in rows) else "FAIL"}
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "foa_model_facing_golden.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"max_angular_error_deg": result["max_angular_error_deg"], "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
