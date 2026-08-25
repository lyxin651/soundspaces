"""The single coordinate and yaw implementation for V0."""

import math
from typing import Iterable, Tuple, Union

import numpy as np
import quaternion


WorldVector = Union[Tuple[float, float, float], Iterable[float], np.ndarray]
SENSOR_OFFSET_M = np.array([0.0, 1.5, 0.0], dtype=np.float64)


def _array(value: WorldVector) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError("world vector must have shape (3,), got {}".format(result.shape))
    if not np.isfinite(result).all():
        raise ValueError("world vector must be finite")
    return result


def normalize_yaw_deg(yaw_deg: float) -> float:
    value = float(yaw_deg)
    if not math.isfinite(value):
        raise ValueError("yaw must be finite")
    return ((value + 180.0) % 360.0) - 180.0


def yaw_to_quaternion(yaw_deg: float):
    yaw = normalize_yaw_deg(yaw_deg)
    return quaternion.from_rotation_vector(np.array([0.0, math.radians(yaw), 0.0]))


def _rotation_matrix(yaw_deg: float) -> np.ndarray:
    return quaternion.as_rotation_matrix(yaw_to_quaternion(yaw_deg))


def yaw_to_forward_world(yaw_deg: float) -> Tuple[float, float, float]:
    return tuple(_rotation_matrix(yaw_deg).dot(np.array([0.0, 0.0, -1.0])))


def yaw_to_right_world(yaw_deg: float) -> Tuple[float, float, float]:
    return tuple(_rotation_matrix(yaw_deg).dot(np.array([1.0, 0.0, 0.0])))


def agent_local_direction_to_world(
    direction: str, yaw_deg: float, distance_m: float = 1.0
) -> Tuple[float, float, float]:
    directions = {
        "forward": np.array([0.0, 0.0, -1.0]),
        "backward": np.array([0.0, 0.0, 1.0]),
        "left": np.array([-1.0, 0.0, 0.0]),
        "right": np.array([1.0, 0.0, 0.0]),
    }
    if direction not in directions:
        raise ValueError("unsupported local direction: {}".format(direction))
    return tuple(_rotation_matrix(yaw_deg).dot(directions[direction] * float(distance_m)))


def world_direction_to_agent(direction_world: WorldVector, yaw_deg: float) -> Tuple[float, float, float]:
    return tuple(_rotation_matrix(yaw_deg).T.dot(_array(direction_world)))


def rotate_left_yaw(yaw_deg: float, angle_deg: float = 45.0) -> float:
    return normalize_yaw_deg(float(yaw_deg) + float(angle_deg))


def rotate_right_yaw(yaw_deg: float, angle_deg: float = 45.0) -> float:
    return normalize_yaw_deg(float(yaw_deg) - float(angle_deg))


def relative_azimuth_deg(
    source_position_world: WorldVector,
    listener_sensor_position_world: WorldVector,
    listener_yaw_deg: float,
) -> float:
    delta_world = _array(source_position_world) - _array(listener_sensor_position_world)
    delta_agent = np.asarray(world_direction_to_agent(delta_world, listener_yaw_deg))
    angle = math.degrees(math.atan2(float(delta_agent[0]), float(-delta_agent[2])))
    return normalize_yaw_deg(angle)


def listener_sensor_position(base_position_world: WorldVector) -> Tuple[float, float, float]:
    return tuple(_array(base_position_world) + SENSOR_OFFSET_M)
