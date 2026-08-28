"""Project-coordinate geometry generation and an independent validator path."""

import math
from typing import Any, Mapping, Tuple

import numpy as np


class GeometryError(ValueError):
    """Raised for invalid or inconsistent project geometry."""


def _finite_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise GeometryError("{} must be a finite length-3 vector".format(name))
    return result


def project_geometry(
    source_position_world: Any,
    listener_sensor_position_world: Any,
    listener_yaw_deg: float,
) -> Mapping[str, Any]:
    """Generate project [x=right, y=up, z=back] DOA using Habitat/RLR axes."""

    source = _finite_vector(source_position_world, "source_position_world")
    sensor = _finite_vector(listener_sensor_position_world, "listener_sensor_position_world")
    yaw = float(listener_yaw_deg)
    if not math.isfinite(yaw):
        raise GeometryError("listener_yaw_deg must be finite")
    delta = source - sensor
    distance = float(np.linalg.norm(delta))
    if distance <= 0.0:
        raise GeometryError("source and listener sensor must not coincide")
    radians = math.radians(yaw)
    right_world = np.array([math.cos(radians), 0.0, math.sin(radians)], dtype=np.float64)
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    back_world = np.array([-math.sin(radians), 0.0, math.cos(radians)], dtype=np.float64)
    project_vector = np.array(
        [np.dot(delta, right_world), np.dot(delta, up_world), np.dot(delta, back_world)],
        dtype=np.float64,
    )
    horizontal = float(np.linalg.norm([project_vector[0], project_vector[2]]))
    azimuth = math.degrees(math.atan2(project_vector[0], -project_vector[2]))
    azimuth = ((azimuth + 180.0) % 360.0) - 180.0
    elevation = math.degrees(math.atan2(project_vector[1], horizontal))
    return {
        "distance_m": distance,
        "doa_unit_project": tuple((project_vector / distance).tolist()),
        "azimuth_project_deg": azimuth,
        "elevation_project_deg": elevation,
    }


def validate_geometry_independently(
    source_position_world: Any,
    listener_sensor_position_world: Any,
    listener_yaw_deg: float,
    label: Mapping[str, Any],
    tolerance: float = 1.0e-6,
) -> None:
    """Recompute labels without calling project_geometry."""

    source = _finite_vector(source_position_world, "source_position_world")
    sensor = _finite_vector(listener_sensor_position_world, "listener_sensor_position_world")
    delta = source - sensor
    distance = float(np.sqrt(np.sum(delta * delta)))
    if distance <= 0.0:
        raise GeometryError("source and listener sensor must not coincide")
    radians = math.radians(float(listener_yaw_deg))
    local_right = delta[0] * math.cos(radians) + delta[2] * math.sin(radians)
    local_up = delta[1]
    local_back = -delta[0] * math.sin(radians) + delta[2] * math.cos(radians)
    horizontal = math.sqrt(local_right * local_right + local_back * local_back)
    azimuth = math.degrees(math.atan2(local_right, -local_back))
    azimuth = ((azimuth + 180.0) % 360.0) - 180.0
    elevation = math.degrees(math.atan2(local_up, horizontal))
    expected_unit = np.array([local_right, local_up, local_back], dtype=np.float64) / distance
    actual_unit = _finite_vector(label.get("doa_unit_project"), "doa_unit_project")
    if not math.isclose(float(label.get("distance_m")), distance, abs_tol=tolerance):
        raise GeometryError("distance label mismatch")
    if not math.isclose(float(label.get("azimuth_project_deg")), azimuth, abs_tol=tolerance):
        raise GeometryError("azimuth label mismatch")
    if not math.isclose(float(label.get("elevation_project_deg")), elevation, abs_tol=tolerance):
        raise GeometryError("elevation label mismatch")
    if not np.allclose(actual_unit, expected_unit, atol=tolerance, rtol=0.0):
        raise GeometryError("DOA unit vector label mismatch")
