"""Deterministic six-candidate M1 action generation."""

from typing import Any, Dict, Iterable, List

import numpy as np

from active_audition.navigation.pathfinder import PathFinderAdapter
from active_audition.scene.pose import (
    agent_local_direction_to_world,
    listener_sensor_position,
    rotate_left_yaw,
    rotate_right_yaw,
)
from active_audition.types import Candidate, ListenerPose


class CandidateError(ValueError):
    """Raised when candidate generation violates the frozen action contract."""


EPSILON_M = 1e-8


def _point(value: Iterable[float]):
    return tuple(float(item) for item in np.asarray(value, dtype=np.float64))


def _distance(first: Iterable[float], second: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)))


def _translation_candidate(
    episode_id: str,
    listener: ListenerPose,
    pathfinder: PathFinderAdapter,
    direction: str,
    distance_m: float,
    thresholds_enabled: bool,
) -> Candidate:
    requested = np.asarray(listener.base_position_world) + np.asarray(
        agent_local_direction_to_world(direction, listener.yaw_deg, distance_m)
    )
    requested_tuple = _point(requested)
    snapped = pathfinder.snap_point(requested_tuple)
    if snapped is None:
        return Candidate(
            episode_id, "trans_" + direction + "_r100", "translation", direction,
            float(distance_m), 0.0, requested_tuple, None, None, float(listener.yaw_deg),
            None, 0.0, None, False, False, None, False, "snap_failed",
        )
    snap_error = _distance(requested_tuple, snapped)
    move_euclidean = _distance(listener.base_position_world, snapped)
    snapped_is_navigable = pathfinder.is_navigable(snapped)
    if not snapped_is_navigable:
        return Candidate(
            episode_id, "trans_" + direction + "_r100", "translation", direction,
            float(distance_m), 0.0, requested_tuple, snapped, None, float(listener.yaw_deg),
            snap_error, move_euclidean, None, False, False, None, False, "outside_navmesh",
        )
    if move_euclidean <= EPSILON_M:
        return Candidate(
            episode_id, "trans_" + direction + "_r100", "translation", direction,
            float(distance_m), 0.0, requested_tuple, snapped, _point(listener_sensor_position(snapped)),
            float(listener.yaw_deg), snap_error, move_euclidean, None, True, False, None,
            False, "actual_move_too_small",
        )
    path = pathfinder.shortest_path(listener.base_position_world, snapped)
    if not path.found or path.points_world is None or path.geodesic_distance_m is None:
        return Candidate(
            episode_id, "trans_" + direction + "_r100", "translation", direction,
            float(distance_m), 0.0, requested_tuple, snapped, _point(listener_sensor_position(snapped)),
            float(listener.yaw_deg), snap_error, move_euclidean, None, True, False, None,
            False, "no_path",
        )
    valid = True
    reason = None
    if thresholds_enabled:
        raise CandidateError("numeric navigation thresholds are not frozen for M1")
    return Candidate(
        episode_id,
        "trans_" + direction + "_r100",
        "translation",
        direction,
        float(distance_m),
        0.0,
        requested_tuple,
        snapped,
        _point(listener_sensor_position(snapped)),
        float(listener.yaw_deg),
        snap_error,
        move_euclidean,
        path.geodesic_distance_m,
        snapped_is_navigable,
        path.found,
        path.points_world,
        valid,
        reason,
    )


def _rotation_candidate(
    episode_id: str, listener: ListenerPose, direction: str, angle_deg: float
) -> Candidate:
    yaw = rotate_left_yaw(listener.yaw_deg, angle_deg) if direction == "left" else rotate_right_yaw(
        listener.yaw_deg, angle_deg
    )
    candidate_id = "rot_{}_{}".format(direction, int(angle_deg))
    return Candidate(
        episode_id,
        candidate_id,
        "rotation",
        None,
        0.0,
        float(angle_deg if direction == "left" else -angle_deg),
        listener.base_position_world,
        listener.base_position_world,
        listener.sensor_position_world,
        float(yaw),
        0.0,
        0.0,
        0.0,
        True,
        True,
        None,
        True,
        None,
    )


def generate_candidates(
    episode_id: str,
    listener: ListenerPose,
    pathfinder: PathFinderAdapter,
    config: Dict[str, Any],
) -> List[Candidate]:
    """Generate candidates from listener pose only; source GT is not an input."""

    navigation = config["navigation"]
    translations = config["candidate"]["translation"]
    rotations = config["candidate"]["rotation"]
    result = [
        _translation_candidate(
            episode_id,
            listener,
            pathfinder,
            direction,
            float(translations["distance_m"]),
            bool(navigation["thresholds_enabled"]),
        )
        for direction in translations["directions"]
    ]
    result.extend(
        _rotation_candidate(episode_id, listener, direction, float(rotations["angle_deg"]))
        for direction in rotations["directions"]
    )
    return result
