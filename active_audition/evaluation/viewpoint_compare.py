"""Initial-versus-candidate derived comparisons for M3 diagnostics."""

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from active_audition.scene.pose import relative_azimuth_deg, normalize_yaw_deg


COMPARISON_FIELDS = (
    "rms_mean_dbfs",
    "ild_db",
    "peak_max",
    "interaural_correlation",
    "interaural_lag_samples",
)


def _source_distance(source, sensor) -> float:
    return float(np.linalg.norm(np.asarray(source, dtype=np.float64) - np.asarray(sensor, dtype=np.float64)))


def build_viewpoint_comparisons(
    episodes: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    viewpoints: Sequence[Mapping[str, object]],
    metrics_by_key: Mapping[tuple, Mapping[str, object]],
):
    episode_map = {row["episode_id"]: row for row in episodes}
    candidate_map = {(row["episode_id"], row["candidate_id"]): row for row in candidates}
    viewpoint_map = {(row["episode_id"], row["viewpoint_id"]): row for row in viewpoints}
    result = []
    for episode_id in sorted(episode_map):
        episode = episode_map[episode_id]
        initial_viewpoint = viewpoint_map.get((episode_id, "initial"))
        initial_metrics = metrics_by_key.get((episode_id, "initial"))
        if initial_viewpoint is None or initial_metrics is None:
            raise ValueError("missing initial viewpoint for comparison: {}".format(episode_id))
        initial_azimuth = relative_azimuth_deg(
            episode["source"]["position_world"],
            initial_viewpoint["sensor_position_world"],
            initial_viewpoint["yaw_deg"],
        )
        initial_distance = _source_distance(
            episode["source"]["position_world"], initial_viewpoint["sensor_position_world"]
        )
        episode_candidates = sorted(
            (row for row in candidates if row["episode_id"] == episode_id and row.get("valid")),
            key=lambda row: str(row["candidate_id"]),
        )
        for candidate in episode_candidates:
            viewpoint = viewpoint_map.get((episode_id, candidate["candidate_id"]))
            candidate_metrics = metrics_by_key.get((episode_id, candidate["candidate_id"]))
            if viewpoint is None or candidate_metrics is None:
                continue
            candidate_azimuth = relative_azimuth_deg(
                episode["source"]["position_world"],
                viewpoint["sensor_position_world"],
                viewpoint["yaw_deg"],
            )
            candidate_distance = _source_distance(
                episode["source"]["position_world"], viewpoint["sensor_position_world"]
            )
            row = {
                "episode_id": episode_id,
                "viewpoint_id": viewpoint["viewpoint_id"],
                "candidate_id": candidate["candidate_id"],
                "action_type": candidate["action_type"],
                "initial_source_distance_m": initial_distance,
                "candidate_source_distance_m": candidate_distance,
                "delta_source_distance_m": candidate_distance - initial_distance,
                "initial_relative_azimuth_deg": initial_azimuth,
                "candidate_relative_azimuth_deg": candidate_azimuth,
                "delta_relative_azimuth_deg": normalize_yaw_deg(candidate_azimuth - initial_azimuth),
                "movement_euclidean_m": float(candidate.get("move_euclidean_m", 0.0)),
                "movement_geodesic_m": candidate.get("move_geodesic_m"),
            }
            for field in COMPARISON_FIELDS:
                initial_value = float(initial_metrics[field])
                candidate_value = float(candidate_metrics[field])
                row["initial_" + field] = initial_value
                row["candidate_" + field] = candidate_value
                row["delta_" + field] = candidate_value - initial_value
            result.append(row)
    return result
