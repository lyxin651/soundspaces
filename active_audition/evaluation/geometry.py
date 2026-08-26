"""Deterministic geometry and candidate validity summaries for M3."""

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np


def summarize(values: Iterable[float]) -> Mapping[str, object]:
    finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    percentiles = np.percentile(finite, [5, 25, 50, 75, 95])
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "max": float(np.max(finite)),
    }


def _distance(first, second) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)))


def build_geometry_report(
    episodes: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    viewpoints: Sequence[Mapping[str, object]],
):
    translations = [row for row in candidates if row.get("action_type") == "translation" and row.get("valid")]
    pairwise = []
    by_episode = {}
    for candidate in translations:
        by_episode.setdefault(candidate["episode_id"], []).append(candidate)
    for rows in by_episode.values():
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                pairwise.append(_distance(left["snapped_base_position_world"], right["snapped_base_position_world"]))
    detours = []
    for row in translations:
        euclidean = float(row.get("move_euclidean_m", 0.0))
        geodesic = row.get("move_geodesic_m")
        if euclidean > 1.0e-8 and geodesic is not None and np.isfinite(geodesic):
            detours.append(float(geodesic) / euclidean)
    invalid = [row for row in candidates if not row.get("valid")]
    valid_count = sum(bool(row.get("valid")) for row in candidates)
    attempted_viewpoints = len(viewpoints)
    return {
        "episode_source_listener_euclidean_m": summarize(row.get("source_listener_euclidean_m") for row in episodes),
        "episode_source_listener_geodesic_m": summarize(row.get("source_listener_geodesic_m") for row in episodes),
        "translation_snap_error_m": summarize(row.get("snap_error_m") for row in translations),
        "translation_actual_euclidean_m": summarize(row.get("move_euclidean_m") for row in translations),
        "translation_geodesic_m": summarize(row.get("move_geodesic_m") for row in translations),
        "translation_geodesic_detour_ratio": summarize(detours),
        "translation_pairwise_distance_m": summarize(pairwise),
        "candidate_counts": {
            "total": len(candidates),
            "valid": valid_count,
            "invalid": len(invalid),
            "valid_rate": float(valid_count / len(candidates)) if candidates else 0.0,
            "invalid_rate": float(len(invalid) / len(candidates)) if candidates else 0.0,
            "invalid_reason_counts": dict(sorted(Counter(row.get("invalid_reason") for row in invalid).items())),
        },
        "viewpoint_counts": {
            "attempted": attempted_viewpoints,
            "rendered": attempted_viewpoints,
            "render_failure_count": 0,
        },
    }
