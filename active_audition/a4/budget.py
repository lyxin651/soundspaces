"""Pure A4-1 motion-budget semantics.

This module intentionally has no Habitat/SoundSpaces imports.  A pathfinder
adapter supplies the authoritative geodesic distance and polyline; this
module only validates and computes the deterministic cost breakdown.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Optional, Tuple

from active_audition.a4.identity import identity_sha256, stable_id


MOTION_PARAMETERS_SCHEMA_VERSION = "active-asr-a4-motion-parameters-v1"
MOTION_COST_SCHEMA_VERSION = "active-asr-a4-motion-cost-v1"
MOTION_COST_ALGORITHM_IDENTITY = "active-asr-a4-motion-cost-v1"
MOTION_EXECUTION_IDENTITY = "active-asr-a4-polyline-turn-execution-v1"
HORIZONTAL_EPSILON_M = 1.0e-9
YAW_EPSILON_DEG = 1.0e-12


class MotionCostError(ValueError):
    """Raised when a motion path or cost contract is malformed."""


WorldXYZ = Tuple[float, float, float]


def _distance(first: WorldXYZ, second: WorldXYZ) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionCostError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise MotionCostError("{} must be finite".format(path))
    return result


def _point(value: Iterable[float], path: str) -> WorldXYZ:
    try:
        values = tuple(_finite(item, "{}[{}]".format(path, index)) for index, item in enumerate(value))
    except TypeError as exc:
        raise MotionCostError("{} must be an iterable xyz vector".format(path)) from exc
    if len(values) != 3:
        raise MotionCostError("{} must contain three coordinates".format(path))
    return values  # type: ignore[return-value]


def normalize_yaw_deg(yaw_deg: float) -> float:
    value = _finite(yaw_deg, "yaw_deg")
    return ((value + 180.0) % 360.0) - 180.0


def shortest_yaw_delta_deg(from_yaw_deg: float, to_yaw_deg: float) -> float:
    return normalize_yaw_deg(float(to_yaw_deg) - float(from_yaw_deg))


def _horizontal_segment_heading_deg(start: WorldXYZ, end: WorldXYZ) -> Optional[float]:
    dx = float(end[0] - start[0])
    dz = float(end[2] - start[2])
    if math.hypot(dx, dz) <= HORIZONTAL_EPSILON_M:
        return None
    # A0 convention: world forward is -Z and positive yaw turns left.
    return normalize_yaw_deg(math.degrees(math.atan2(-dx, -dz)))


def _polyline(value: Iterable[Iterable[float]]) -> Tuple[WorldXYZ, ...]:
    points = tuple(_point(point, "path_polyline") for point in value)
    if not points:
        raise MotionCostError("path_polyline must not be empty")
    return points


def _polyline_length(points: Tuple[WorldXYZ, ...]) -> float:
    return sum(
        math.sqrt(sum((float(end[index]) - float(start[index])) ** 2 for index in range(3)))
        for start, end in zip(points, points[1:])
    )


def _turn_degrees(headings: Tuple[float, ...], initial_yaw_deg: float, final_yaw_deg: float) -> Tuple[float, float, float]:
    if not headings:
        initial_turn = 0.0
        internal_turn = 0.0
        final_turn = abs(shortest_yaw_delta_deg(initial_yaw_deg, final_yaw_deg))
        return initial_turn, internal_turn, final_turn
    initial_turn = abs(shortest_yaw_delta_deg(initial_yaw_deg, headings[0]))
    internal_turn = sum(
        abs(shortest_yaw_delta_deg(previous, current))
        for previous, current in zip(headings, headings[1:])
    )
    final_turn = abs(shortest_yaw_delta_deg(headings[-1], final_yaw_deg))
    return initial_turn, internal_turn, final_turn


@dataclass(frozen=True)
class MotionParameters:
    """Versioned parameterization supplied by CandidateContract/config."""

    schema_version: str
    algorithm_identity: str
    execution_identity: str
    translation_speed_mps: float
    rotation_speed_dps: float
    settling_sec: float
    budget_sec: float

    def __post_init__(self) -> None:
        if self.schema_version != MOTION_PARAMETERS_SCHEMA_VERSION:
            raise MotionCostError("motion parameter schema version is invalid")
        if self.algorithm_identity != MOTION_COST_ALGORITHM_IDENTITY:
            raise MotionCostError("motion.algorithm_identity is unsupported")
        if self.execution_identity != MOTION_EXECUTION_IDENTITY:
            raise MotionCostError("motion.execution_identity is unsupported")
        translation_speed = _finite(self.translation_speed_mps, "translation_speed_mps")
        rotation_speed = _finite(self.rotation_speed_dps, "rotation_speed_dps")
        settling = _finite(self.settling_sec, "settling_sec")
        budget = _finite(self.budget_sec, "budget_sec")
        if translation_speed <= 0.0:
            raise MotionCostError("translation_speed_mps must be positive")
        if rotation_speed <= 0.0:
            raise MotionCostError("rotation_speed_dps must be positive")
        if settling < 0.0:
            raise MotionCostError("settling_sec must not be negative")
        if budget < 0.0:
            raise MotionCostError("budget_sec must not be negative")
        object.__setattr__(self, "translation_speed_mps", translation_speed)
        object.__setattr__(self, "rotation_speed_dps", rotation_speed)
        object.__setattr__(self, "settling_sec", settling)
        object.__setattr__(self, "budget_sec", budget)

    def identity_payload(self):
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "execution_identity": self.execution_identity,
            "translation_speed_mps": self.translation_speed_mps,
            "rotation_speed_dps": self.rotation_speed_dps,
            "settling_sec": self.settling_sec,
            "budget_sec": self.budget_sec,
        }

    @property
    def motion_contract_identity(self) -> str:
        return stable_id("motion-contract", self.identity_payload())

    def to_payload(self):
        return dict(self.identity_payload(), motion_contract_identity=self.motion_contract_identity)


@dataclass(frozen=True)
class MotionCost:
    """Recomputable cost result using PathFinder geodesic authority."""

    schema_version: str
    motion_contract_identity: str
    geodesic_path_length_m: float
    polyline_length_m: float
    initial_to_path_turn_deg: float
    internal_path_turn_deg: float
    final_turn_deg: float
    translation_speed_mps: float
    rotation_speed_dps: float
    settling_sec: float
    budget_sec: float
    translation_sec: float
    rotation_sec: float
    total_cost_sec: float
    budget_feasible: bool

    def __post_init__(self) -> None:
        if self.schema_version != MOTION_COST_SCHEMA_VERSION:
            raise MotionCostError("motion cost schema version is invalid")
        if not isinstance(self.motion_contract_identity, str) or not self.motion_contract_identity:
            raise MotionCostError("motion_contract_identity must be non-empty")
        for field in (
            "geodesic_path_length_m", "polyline_length_m", "initial_to_path_turn_deg",
            "internal_path_turn_deg", "final_turn_deg", "translation_speed_mps",
            "rotation_speed_dps", "settling_sec", "budget_sec", "translation_sec",
            "rotation_sec", "total_cost_sec",
        ):
            if _finite(getattr(self, field), "motion." + field) < 0.0:
                raise MotionCostError("motion.{} must not be negative".format(field))
        if not isinstance(self.budget_feasible, bool):
            raise MotionCostError("motion.budget_feasible must be boolean")
        expected_total = self.translation_sec + self.rotation_sec + self.settling_sec
        if not math.isclose(self.total_cost_sec, expected_total, rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise MotionCostError("motion total cost does not match its breakdown")
        if self.budget_feasible != (self.total_cost_sec <= self.budget_sec + 1.0e-12):
            raise MotionCostError("motion budget_feasible does not match total cost and budget")

    def identity_payload(self):
        return {
            "schema_version": self.schema_version,
            "motion_contract_identity": self.motion_contract_identity,
            "geodesic_path_length_m": self.geodesic_path_length_m,
            "polyline_length_m": self.polyline_length_m,
            "initial_to_path_turn_deg": self.initial_to_path_turn_deg,
            "internal_path_turn_deg": self.internal_path_turn_deg,
            "final_turn_deg": self.final_turn_deg,
            "translation_speed_mps": self.translation_speed_mps,
            "rotation_speed_dps": self.rotation_speed_dps,
            "settling_sec": self.settling_sec,
            "budget_sec": self.budget_sec,
            "translation_sec": self.translation_sec,
            "rotation_sec": self.rotation_sec,
            "total_cost_sec": self.total_cost_sec,
            "budget_feasible": self.budget_feasible,
        }

    def to_payload(self):
        return self.identity_payload()

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


def compute_motion_cost(
    initial_position_world: Iterable[float],
    initial_yaw_deg: float,
    path_polyline: Iterable[Iterable[float]],
    final_yaw_deg: float,
    geodesic_path_length_m: float,
    parameters: MotionParameters,
) -> MotionCost:
    """Compute cost without reading source geometry or acoustic results."""

    initial = _point(initial_position_world, "initial_position_world")
    points = _polyline(path_polyline)
    if _distance(points[0], initial) > HORIZONTAL_EPSILON_M:
        raise MotionCostError("path_polyline must start at initial_position_world")
    geodesic = _finite(geodesic_path_length_m, "geodesic_path_length_m")
    if geodesic < 0.0:
        raise MotionCostError("geodesic_path_length_m must not be negative")
    headings = tuple(
        heading
        for start, end in zip(points, points[1:])
        for heading in (_horizontal_segment_heading_deg(start, end),)
        if heading is not None
    )
    polyline_length = _polyline_length(points)
    horizontal_polyline_length = sum(
        math.hypot(end[0] - start[0], end[2] - start[2])
        for start, end in zip(points, points[1:])
    )
    same_position = math.sqrt(sum((points[-1][index] - initial[index]) ** 2 for index in range(3))) <= HORIZONTAL_EPSILON_M
    if geodesic <= HORIZONTAL_EPSILON_M and horizontal_polyline_length > HORIZONTAL_EPSILON_M:
        raise MotionCostError("zero geodesic path cannot contain horizontal translation")
    if geodesic <= HORIZONTAL_EPSILON_M and not headings:
        rotation = abs(shortest_yaw_delta_deg(initial_yaw_deg, final_yaw_deg))
        if rotation <= YAW_EPSILON_DEG:
            translation_sec = 0.0
            rotation_sec = 0.0
            settling = 0.0
            total = 0.0
        else:
            translation_sec = 0.0
            rotation_sec = rotation / parameters.rotation_speed_dps
            settling = parameters.settling_sec
            total = rotation_sec + settling
        initial_turn = 0.0
        internal_turn = 0.0
        final_turn = rotation
    else:
        if not headings and geodesic > HORIZONTAL_EPSILON_M:
            raise MotionCostError("nonzero geodesic path has no horizontal path heading")
        initial_turn, internal_turn, final_turn = _turn_degrees(
            headings, initial_yaw_deg, final_yaw_deg
        )
        translation_sec = geodesic / parameters.translation_speed_mps
        rotation_sec = (initial_turn + internal_turn + final_turn) / parameters.rotation_speed_dps
        settling = parameters.settling_sec
        total = translation_sec + rotation_sec + settling
        if same_position and geodesic > HORIZONTAL_EPSILON_M:
            raise MotionCostError("geodesic path endpoint is inconsistent with initial position")
    return MotionCost(
        schema_version=MOTION_COST_SCHEMA_VERSION,
        motion_contract_identity=parameters.motion_contract_identity,
        geodesic_path_length_m=geodesic,
        polyline_length_m=polyline_length,
        initial_to_path_turn_deg=initial_turn,
        internal_path_turn_deg=internal_turn,
        final_turn_deg=final_turn,
        translation_speed_mps=parameters.translation_speed_mps,
        rotation_speed_dps=parameters.rotation_speed_dps,
        settling_sec=settling,
        budget_sec=parameters.budget_sec,
        translation_sec=translation_sec,
        rotation_sec=rotation_sec,
        total_cost_sec=total,
        budget_feasible=total <= parameters.budget_sec + 1.0e-12,
    )


__all__ = [
    "HORIZONTAL_EPSILON_M",
    "MOTION_COST_ALGORITHM_IDENTITY",
    "MOTION_COST_SCHEMA_VERSION",
    "MOTION_EXECUTION_IDENTITY",
    "MOTION_PARAMETERS_SCHEMA_VERSION",
    "MotionCost",
    "MotionCostError",
    "MotionParameters",
    "compute_motion_cost",
    "normalize_yaw_deg",
    "shortest_yaw_delta_deg",
]
