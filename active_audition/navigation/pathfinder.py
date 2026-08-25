"""Thin, Habitat-object-free PathFinder adapter for M1."""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import quaternion  # Must be imported before habitat_sim.
import habitat_sim


WorldXYZ = Tuple[float, float, float]


class PathFinderError(ValueError):
    """Raised when a PathFinder result is malformed."""


def _point(value: Iterable[float]) -> WorldXYZ:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise PathFinderError("point must be a finite xyz vector")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class PathResult:
    requested_start: WorldXYZ
    requested_end: WorldXYZ
    geodesic_distance_m: Optional[float]
    points_world: Optional[Tuple[WorldXYZ, ...]]
    found: bool


class PathFinderAdapter:
    """Expose only project-level navigation primitives."""

    def __init__(self, pathfinder: object):
        self._pathfinder = pathfinder

    @property
    def is_loaded(self) -> bool:
        return bool(self._pathfinder.is_loaded)

    def snap_point(self, point: Iterable[float]) -> Optional[WorldXYZ]:
        requested = _point(point)
        snapped = np.asarray(self._pathfinder.snap_point(np.asarray(requested)), dtype=np.float64)
        if snapped.shape != (3,) or not np.isfinite(snapped).all():
            return None
        return _point(snapped)

    def is_navigable(self, point: Iterable[float]) -> bool:
        return bool(self._pathfinder.is_navigable(np.asarray(_point(point))))

    def shortest_path(self, start: Iterable[float], end: Iterable[float]) -> PathResult:
        requested_start = _point(start)
        requested_end = _point(end)

        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(requested_start, dtype=np.float32)
        path.requested_end = np.asarray(requested_end, dtype=np.float32)
        found = bool(self._pathfinder.find_path(path))
        distance = float(path.geodesic_distance)
        if not found or not np.isfinite(distance):
            return PathResult(requested_start, requested_end, None, None, False)
        points = tuple(_point(item) for item in path.points)
        if not points:
            return PathResult(requested_start, requested_end, distance, None, False)
        return PathResult(requested_start, requested_end, distance, points, found)

    def geodesic_distance(
        self, start: Iterable[float], end: Iterable[float]
    ) -> Optional[float]:
        return self.shortest_path(start, end).geodesic_distance_m

    def sample_navigable_point(self, rng: Optional[np.random.Generator] = None) -> WorldXYZ:
        if rng is not None and hasattr(self._pathfinder, "seed"):
            seed = int(rng.integers(0, 2**31 - 1))
            self._pathfinder.seed(seed)
        return _point(self._pathfinder.get_random_navigable_point())
