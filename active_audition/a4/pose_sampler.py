"""Pure A4-1 source-free sampling and post-sampling legality.

The sampler accepts a deliberately source-free :class:`SamplerContext` and a
duck-typed pathfinder.  Target/noise positions are accepted only by
``annotate_geometry_legality`` after the source-free result has been frozen.
This module imports no Habitat, quaternion, NumPy, renderer, or ASR code.
"""

from dataclasses import dataclass, replace
import math
from collections import Counter
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from active_audition.a4.budget import (
    MOTION_COST_ALGORITHM_IDENTITY,
    MOTION_EXECUTION_IDENTITY,
    MotionCost,
    MotionParameters,
    compute_motion_cost,
    normalize_yaw_deg,
)
from active_audition.a4.identity import (
    A4IdentityError,
    canonical_json_bytes,
    identity_sha256,
    stable_id,
    validate_sha256,
    validate_stable_id,
)
from active_audition.a4.records import (
    GEOMETRY_SCHEMA_VERSION,
    POSE_SCHEMA_VERSION,
    GeometryRecord,
    PoseRecord,
    RecordError,
    pose_identity_payload,
)


CANDIDATE_CONTRACT_SCHEMA_VERSION = "active-asr-a4-candidate-contract-v1"
SAMPLER_CONTEXT_SCHEMA_VERSION = "active-asr-a4-sampler-context-v1"
PROBE_ATTEMPT_SCHEMA_VERSION = "active-asr-a4-probe-attempt-v1"
POSITION_PLAN_SCHEMA_VERSION = "active-asr-a4-position-plan-v1"
YAW_PLAN_SCHEMA_VERSION = "active-asr-a4-yaw-plan-v1"
SAMPLER_OUTPUT_SCHEMA_VERSION = "active-asr-a4-sampler-output-v1"
SAMPLER_ALGORITHM_IDENTITY = "active-asr-a4-local-polar-sampler-v1"
COVERAGE_SELECTION_IDENTITY = "active-asr-a4-farthest-coverage-v1"
COORDINATE_CONVENTION_IDENTITY = "active-asr-a0-positive-left-forward-minus-z-v1"
TIE_BREAK_IDENTITY = "active-asr-a4-probe-id-lexical-tie-break-v1"
TECHNICAL_FILTER_ORDER = (
    "requested_point",
    "snap_point",
    "finite_check",
    "navigability",
    "snap_error",
    "shortest_path",
    "geodesic_radius",
    "duplicate_filter",
)
TECHNICAL_REASONS = (
    "SNAP_FAILED",
    "NONFINITE_SNAPPED_POINT",
    "NON_NAVIGABLE",
    "SNAP_TOO_FAR",
    "NO_PATH",
    "OUTSIDE_GEODESIC_RADIUS",
    "DUPLICATE_SNAPPED_POSITION",
)
SOURCE_CLEARANCE_REASONS = (
    "TARGET_SOURCE_CLEARANCE",
    "NOISE_SOURCE_CLEARANCE",
    "TARGET_AND_NOISE_SOURCE_CLEARANCE",
)
POINT_EPSILON_M = 1.0e-9


class SamplerError(ValueError):
    """Raised when a source-free sampler contract or result is malformed."""


class PathFinderLike(Protocol):
    """Minimal injected pathfinder surface required by the pure sampler."""

    def snap_point(self, point: Iterable[float]) -> Optional[Iterable[float]]: ...

    def is_navigable(self, point: Iterable[float]) -> bool: ...

    def shortest_path(self, start: Iterable[float], end: Iterable[float]) -> Any: ...


WorldXYZ = Tuple[float, float, float]


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamplerError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise SamplerError("{} must be finite".format(path))
    return result


def _point(value: Iterable[float], path: str) -> WorldXYZ:
    try:
        values = tuple(_finite(item, "{}[{}]".format(path, index)) for index, item in enumerate(value))
    except TypeError as exc:
        raise SamplerError("{} must be an iterable xyz vector".format(path)) from exc
    if len(values) != 3:
        raise SamplerError("{} must contain three coordinates".format(path))
    return values  # type: ignore[return-value]


def _distance(first: WorldXYZ, second: WorldXYZ) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _ids(values: Iterable[float], path: str, allow_empty: bool = False) -> Tuple[float, ...]:
    result = tuple(_finite(value, "{}[{}]".format(path, index)) for index, value in enumerate(values))
    if not result and not allow_empty:
        raise SamplerError("{} must not be empty".format(path))
    if len(set(result)) != len(result):
        raise SamplerError("{} must not contain duplicates".format(path))
    return result


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SamplerError("{} must be a non-empty string".format(path))
    return value


def _sha_map(value: Mapping, path: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise SamplerError("{} must be a non-empty mapping".format(path))
    result = {}
    for key, item in value.items():
        key = _string(key, path + ".key")
        try:
            validate_sha256(item, path + "." + key)
        except A4IdentityError as exc:
            raise SamplerError(str(exc)) from exc
        result[key] = item
    return result


def _strict_payload(payload: Mapping, expected: Sequence[str], path: str) -> None:
    unknown = sorted(set(payload) - set(expected))
    missing = sorted(set(expected) - set(payload))
    if unknown or missing:
        raise SamplerError("{} fields invalid: unknown={}, missing={}".format(path, unknown, missing))


def _stable_id(value: Any, namespace: str, path: str) -> str:
    try:
        validate_stable_id(value, namespace, path)
    except A4IdentityError as exc:
        raise SamplerError(str(exc)) from exc
    return value


@dataclass(frozen=True)
class CandidateContract:
    """Versioned per-experiment candidate and budget parameterization.

    The values are intentionally supplied by the experiment/config layer and
    are not embedded in the infrastructure contract as permanent constants.
    """

    schema_version: str
    sampler_algorithm_identity: str
    coordinate_convention_identity: str
    coverage_selection_identity: str
    tie_break_identity: str
    radii_m: Tuple[float, ...]
    azimuth_offsets_deg: Tuple[float, ...]
    max_geodesic_radius_m: float
    max_snap_error_m: float
    duplicate_position_tolerance_m: float
    max_noninitial_positions: int
    yaw_offsets_deg: Tuple[float, ...]
    source_clearance_m: float
    translation_speed_mps: float
    rotation_speed_dps: float
    settling_sec: float
    budget_sec: float

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_CONTRACT_SCHEMA_VERSION:
            raise SamplerError("candidate contract schema version is invalid")
        expected_identities = {
            "sampler_algorithm_identity": SAMPLER_ALGORITHM_IDENTITY,
            "coordinate_convention_identity": COORDINATE_CONVENTION_IDENTITY,
            "coverage_selection_identity": COVERAGE_SELECTION_IDENTITY,
            "tie_break_identity": TIE_BREAK_IDENTITY,
        }
        for field, expected in expected_identities.items():
            if _string(getattr(self, field), "candidate." + field) != expected:
                raise SamplerError("candidate.{} is unsupported".format(field))
        radii = _ids(self.radii_m, "candidate.radii_m")
        if any(value <= 0.0 for value in radii) or tuple(sorted(radii)) != radii:
            raise SamplerError("candidate.radii_m must be strictly increasing and positive")
        azimuths = _ids(self.azimuth_offsets_deg, "candidate.azimuth_offsets_deg")
        if any(value < 0.0 or value >= 360.0 for value in azimuths):
            raise SamplerError("candidate.azimuth_offsets_deg must be in [0, 360)")
        yaw_offsets = _ids(self.yaw_offsets_deg, "candidate.yaw_offsets_deg")
        if any(value < 0.0 or value >= 360.0 for value in yaw_offsets):
            raise SamplerError("candidate.yaw_offsets_deg must be in [0, 360)")
        normalized_nonnegative = {}
        for field in (
            "max_geodesic_radius_m", "max_snap_error_m", "duplicate_position_tolerance_m",
            "source_clearance_m", "settling_sec", "budget_sec",
        ):
            normalized = _finite(getattr(self, field), "candidate." + field)
            normalized_nonnegative[field] = normalized
            if normalized < 0.0:
                raise SamplerError("candidate.{} must not be negative".format(field))
        if not isinstance(self.max_noninitial_positions, int) or isinstance(self.max_noninitial_positions, bool):
            raise SamplerError("candidate.max_noninitial_positions must be an integer")
        if self.max_noninitial_positions < 0:
            raise SamplerError("candidate.max_noninitial_positions must not be negative")
        translation_speed = _finite(self.translation_speed_mps, "candidate.translation_speed_mps")
        rotation_speed = _finite(self.rotation_speed_dps, "candidate.rotation_speed_dps")
        if translation_speed <= 0.0:
            raise SamplerError("candidate.translation_speed_mps must be positive")
        if rotation_speed <= 0.0:
            raise SamplerError("candidate.rotation_speed_dps must be positive")
        object.__setattr__(self, "max_geodesic_radius_m", normalized_nonnegative["max_geodesic_radius_m"])
        object.__setattr__(self, "max_snap_error_m", normalized_nonnegative["max_snap_error_m"])
        object.__setattr__(self, "duplicate_position_tolerance_m", normalized_nonnegative["duplicate_position_tolerance_m"])
        object.__setattr__(self, "source_clearance_m", normalized_nonnegative["source_clearance_m"])
        object.__setattr__(self, "settling_sec", normalized_nonnegative["settling_sec"])
        object.__setattr__(self, "budget_sec", normalized_nonnegative["budget_sec"])
        object.__setattr__(self, "translation_speed_mps", translation_speed)
        object.__setattr__(self, "rotation_speed_dps", rotation_speed)
        object.__setattr__(self, "radii_m", radii)
        object.__setattr__(self, "azimuth_offsets_deg", azimuths)
        object.__setattr__(self, "yaw_offsets_deg", yaw_offsets)

    def identity_payload(self):
        return {
            "schema_version": self.schema_version,
            "sampler_algorithm_identity": self.sampler_algorithm_identity,
            "coordinate_convention_identity": self.coordinate_convention_identity,
            "coverage_selection_identity": self.coverage_selection_identity,
            "tie_break_identity": self.tie_break_identity,
            "radii_m": list(self.radii_m),
            "azimuth_offsets_deg": list(self.azimuth_offsets_deg),
            "max_geodesic_radius_m": self.max_geodesic_radius_m,
            "max_snap_error_m": self.max_snap_error_m,
            "duplicate_position_tolerance_m": self.duplicate_position_tolerance_m,
            "max_noninitial_positions": self.max_noninitial_positions,
            "yaw_offsets_deg": list(self.yaw_offsets_deg),
            "source_clearance_m": self.source_clearance_m,
            "translation_speed_mps": self.translation_speed_mps,
            "rotation_speed_dps": self.rotation_speed_dps,
            "settling_sec": self.settling_sec,
            "budget_sec": self.budget_sec,
        }

    @property
    def yaw_grid_identity(self) -> str:
        return stable_id("yaw-grid", {"schema_version": self.schema_version, "yaw_offsets_deg": list(self.yaw_offsets_deg)})

    @property
    def candidate_contract_identity(self) -> str:
        return stable_id("candidate-contract", self.identity_payload())

    @property
    def motion_parameters(self) -> MotionParameters:
        return MotionParameters(
            schema_version="active-asr-a4-motion-parameters-v1",
            algorithm_identity=MOTION_COST_ALGORITHM_IDENTITY,
            execution_identity=MOTION_EXECUTION_IDENTITY,
            translation_speed_mps=self.translation_speed_mps,
            rotation_speed_dps=self.rotation_speed_dps,
            settling_sec=self.settling_sec,
            budget_sec=self.budget_sec,
        )

    def to_payload(self):
        # Only normalized constructor fields belong in the reconstructable
        # contract payload.  Candidate identity and derived grid/motion
        # identities are properties, not constructor inputs.
        return _plain(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: Mapping) -> "CandidateContract":
        expected = tuple(field.name for field in cls.__dataclass_fields__.values())
        if not isinstance(payload, Mapping):
            raise SamplerError("candidate contract payload must be a mapping")
        _strict_payload(payload, expected, "candidate")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise SamplerError("candidate contract payload is invalid: {}".format(exc)) from exc


@dataclass(frozen=True)
class SamplerContext:
    """The only context accepted by the source-free sampler."""

    schema_version: str
    sampler_context_id: str
    scene_id: str
    scene_resource_identities: Mapping[str, str]
    initial_listener_requested_base_xyz: WorldXYZ
    actual_listener_base_xyz: WorldXYZ
    sensor_transform_identity: str
    sensor_xyz: WorldXYZ
    initial_yaw_deg: float
    candidate_contract_identity: str
    sampler_algorithm_identity: str

    def __post_init__(self) -> None:
        if self.schema_version != SAMPLER_CONTEXT_SCHEMA_VERSION:
            raise SamplerError("sampler context schema version is invalid")
        _stable_id(self.sampler_context_id, "sampler-context", "sampler_context_id")
        scene_id = _string(self.scene_id, "scene_id")
        resources = _sha_map(self.scene_resource_identities, "scene_resource_identities")
        requested = _point(self.initial_listener_requested_base_xyz, "initial_listener_requested_base_xyz")
        actual = _point(self.actual_listener_base_xyz, "actual_listener_base_xyz")
        sensor_transform = _string(self.sensor_transform_identity, "sensor_transform_identity")
        sensor_xyz = _point(self.sensor_xyz, "sensor_xyz")
        initial_yaw = normalize_yaw_deg(self.initial_yaw_deg)
        _stable_id(self.candidate_contract_identity, "candidate-contract", "candidate_contract_identity")
        if _string(self.sampler_algorithm_identity, "sampler_algorithm_identity") != SAMPLER_ALGORITHM_IDENTITY:
            raise SamplerError("sampler_algorithm_identity is unsupported")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "scene_resource_identities", _freeze(resources))
        object.__setattr__(self, "initial_listener_requested_base_xyz", requested)
        object.__setattr__(self, "actual_listener_base_xyz", actual)
        object.__setattr__(self, "sensor_transform_identity", sensor_transform)
        object.__setattr__(self, "sensor_xyz", sensor_xyz)
        object.__setattr__(self, "initial_yaw_deg", initial_yaw)
        if self.sampler_context_id != stable_id("sampler-context", self.identity_payload()):
            raise SamplerError("sampler_context_id does not match source-free payload")

    def identity_payload(self):
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "scene_resource_identities": _plain(self.scene_resource_identities),
            "initial_listener_requested_base_xyz": list(self.initial_listener_requested_base_xyz),
            "actual_listener_base_xyz": list(self.actual_listener_base_xyz),
            "sensor_transform_identity": self.sensor_transform_identity,
            "sensor_xyz": list(self.sensor_xyz),
            "initial_yaw_deg": self.initial_yaw_deg,
            "candidate_contract_identity": self.candidate_contract_identity,
            "sampler_algorithm_identity": self.sampler_algorithm_identity,
        }

    def to_payload(self):
        return dict(self.identity_payload(), sampler_context_id=self.sampler_context_id)


def make_sampler_context(
    scene_id: str,
    scene_resource_identities: Mapping[str, str],
    initial_listener_requested_base_xyz: Iterable[float],
    actual_listener_base_xyz: Iterable[float],
    sensor_transform_identity: str,
    sensor_xyz: Iterable[float],
    initial_yaw_deg: float,
    candidate_contract: CandidateContract,
) -> SamplerContext:
    scene_resources = _sha_map(scene_resource_identities, "scene_resource_identities")
    requested = _point(initial_listener_requested_base_xyz, "initial_listener_requested_base_xyz")
    actual = _point(actual_listener_base_xyz, "actual_listener_base_xyz")
    sensor = _point(sensor_xyz, "sensor_xyz")
    scene_name = _string(scene_id, "scene_id")
    sensor_identity = _string(sensor_transform_identity, "sensor_transform_identity")
    yaw = normalize_yaw_deg(initial_yaw_deg)
    payload = {
        "schema_version": SAMPLER_CONTEXT_SCHEMA_VERSION,
        "scene_id": scene_name,
        "scene_resource_identities": _plain(scene_resources),
        "initial_listener_requested_base_xyz": list(requested),
        "actual_listener_base_xyz": list(actual),
        "sensor_transform_identity": sensor_identity,
        "sensor_xyz": list(sensor),
        "initial_yaw_deg": yaw,
        "candidate_contract_identity": candidate_contract.candidate_contract_identity,
        "sampler_algorithm_identity": candidate_contract.sampler_algorithm_identity,
    }
    return SamplerContext(sampler_context_id=stable_id("sampler-context", payload), **payload)


@dataclass(frozen=True)
class ProbeAttemptRecord:
    schema_version: str
    probe_id: str
    sampler_context_id: str
    radius_m: float
    azimuth_deg: float
    requested_base_xyz: WorldXYZ
    snapped_base_xyz: Optional[WorldXYZ]
    snap_error_m: Optional[float]
    path_polyline: Optional[Tuple[WorldXYZ, ...]]
    geodesic_path_length_m: Optional[float]
    status: str
    reason: Optional[str]

    def __post_init__(self) -> None:
        if self.schema_version != PROBE_ATTEMPT_SCHEMA_VERSION:
            raise SamplerError("probe attempt schema version is invalid")
        _stable_id(self.probe_id, "probe", "probe_id")
        _stable_id(self.sampler_context_id, "sampler-context", "sampler_context_id")
        radius = _finite(self.radius_m, "probe.radius_m")
        azimuth = _finite(self.azimuth_deg, "probe.azimuth_deg")
        requested = _point(self.requested_base_xyz, "probe.requested_base_xyz")
        snapped = None if self.snapped_base_xyz is None else _point(self.snapped_base_xyz, "probe.snapped_base_xyz")
        snap_error = None if self.snap_error_m is None else _finite(self.snap_error_m, "probe.snap_error_m")
        path = None if self.path_polyline is None else tuple(_point(point, "probe.path_polyline") for point in self.path_polyline)
        geodesic = None if self.geodesic_path_length_m is None else _finite(self.geodesic_path_length_m, "probe.geodesic_path_length_m")
        if radius <= 0.0:
            raise SamplerError("probe.radius_m must be positive")
        if snap_error is not None and snap_error < 0.0:
            raise SamplerError("probe.snap_error_m must not be negative")
        if geodesic is not None and geodesic < 0.0:
            raise SamplerError("probe.geodesic_path_length_m must not be negative")
        if self.status not in ("REJECTED", "TECHNICAL_VALID", "SELECTED"):
            raise SamplerError("probe.status is invalid")
        if self.status == "REJECTED":
            if self.reason not in TECHNICAL_REASONS:
                raise SamplerError("probe rejected reason is invalid")
        elif self.reason is not None:
            raise SamplerError("accepted probe cannot have a rejection reason")
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "azimuth_deg", azimuth)
        object.__setattr__(self, "requested_base_xyz", requested)
        object.__setattr__(self, "snapped_base_xyz", snapped)
        object.__setattr__(self, "snap_error_m", snap_error)
        object.__setattr__(self, "path_polyline", path)
        object.__setattr__(self, "geodesic_path_length_m", geodesic)
        if self.probe_id != stable_id("probe", self.identity_payload()):
            raise SamplerError("probe_id does not match deterministic probe payload")

    def identity_payload(self):
        return {
            "schema_version": self.schema_version,
            "sampler_context_id": self.sampler_context_id,
            "radius_m": self.radius_m,
            "azimuth_deg": self.azimuth_deg,
            "requested_base_xyz": list(self.requested_base_xyz),
        }

    def to_payload(self):
        return {
            "schema_version": self.schema_version,
            "probe_id": self.probe_id,
            "sampler_context_id": self.sampler_context_id,
            "radius_m": self.radius_m,
            "azimuth_deg": self.azimuth_deg,
            "requested_base_xyz": list(self.requested_base_xyz),
            "snapped_base_xyz": None if self.snapped_base_xyz is None else list(self.snapped_base_xyz),
            "snap_error_m": self.snap_error_m,
            "path_polyline": None if self.path_polyline is None else [list(point) for point in self.path_polyline],
            "geodesic_path_length_m": self.geodesic_path_length_m,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SampledPositionPlan:
    schema_version: str
    position_id: str
    sampler_context_id: str
    source_probe_id: str
    requested_base_xyz: WorldXYZ
    actual_snapped_base_xyz: WorldXYZ
    sensor_transform_identity: str
    sensor_xyz: WorldXYZ
    snap_error_m: float
    path_polyline: Tuple[WorldXYZ, ...]
    geodesic_path_length_m: float
    is_initial: bool

    def __post_init__(self) -> None:
        if self.schema_version != POSITION_PLAN_SCHEMA_VERSION:
            raise SamplerError("position plan schema version is invalid")
        _stable_id(self.position_id, "position", "position_id")
        _stable_id(self.sampler_context_id, "sampler-context", "sampler_context_id")
        _string(self.source_probe_id, "source_probe_id")
        for field in ("requested_base_xyz", "actual_snapped_base_xyz", "sensor_xyz"):
            object.__setattr__(self, field, _point(getattr(self, field), "position." + field))
        object.__setattr__(self, "sensor_transform_identity", _string(self.sensor_transform_identity, "sensor_transform_identity"))
        snap_error = _finite(self.snap_error_m, "position.snap_error_m")
        geodesic = _finite(self.geodesic_path_length_m, "position.geodesic_path_length_m")
        if snap_error < 0.0 or geodesic < 0.0:
            raise SamplerError("position distances must not be negative")
        path = tuple(_point(point, "position.path_polyline") for point in self.path_polyline)
        if not path:
            raise SamplerError("position.path_polyline must not be empty")
        if not isinstance(self.is_initial, bool):
            raise SamplerError("position.is_initial must be boolean")
        object.__setattr__(self, "snap_error_m", snap_error)
        object.__setattr__(self, "geodesic_path_length_m", geodesic)
        object.__setattr__(self, "path_polyline", path)
        if self.position_id != stable_id("position", self.identity_payload()):
            raise SamplerError("position_id does not match source-free payload")

    def identity_payload(self):
        return {
            "schema_version": self.schema_version,
            "sampler_context_id": self.sampler_context_id,
            "source_probe_id": self.source_probe_id,
            "requested_base_xyz": list(self.requested_base_xyz),
            "actual_snapped_base_xyz": list(self.actual_snapped_base_xyz),
            "sensor_transform_identity": self.sensor_transform_identity,
            "sensor_xyz": list(self.sensor_xyz),
            "snap_error_m": self.snap_error_m,
            "path_polyline": [list(point) for point in self.path_polyline],
            "geodesic_path_length_m": self.geodesic_path_length_m,
            "is_initial": self.is_initial,
        }

    def to_payload(self):
        return dict(self.identity_payload(), position_id=self.position_id)


@dataclass(frozen=True)
class SampledYawPlan:
    schema_version: str
    yaw_id: str
    position_id: str
    sampler_context_id: str
    yaw_grid_identity: str
    yaw_offset_deg: float
    yaw_deg: float
    motion_cost: MotionCost

    def __post_init__(self) -> None:
        if self.schema_version != YAW_PLAN_SCHEMA_VERSION:
            raise SamplerError("yaw plan schema version is invalid")
        _stable_id(self.yaw_id, "yaw", "yaw_id")
        _stable_id(self.position_id, "position", "position_id")
        _stable_id(self.sampler_context_id, "sampler-context", "sampler_context_id")
        _stable_id(self.yaw_grid_identity, "yaw-grid", "yaw_grid_identity")
        if not isinstance(self.motion_cost, MotionCost):
            raise SamplerError("yaw plan motion_cost must be MotionCost")
        object.__setattr__(self, "yaw_offset_deg", _finite(self.yaw_offset_deg, "yaw_offset_deg"))
        object.__setattr__(self, "yaw_deg", _finite(self.yaw_deg, "yaw_deg"))
        if self.yaw_id != stable_id("yaw", self.identity_payload()):
            raise SamplerError("yaw_id does not match source-free payload")

    def identity_payload(self):
        return {
            "schema_version": self.schema_version,
            "position_id": self.position_id,
            "sampler_context_id": self.sampler_context_id,
            "yaw_grid_identity": self.yaw_grid_identity,
            "yaw_offset_deg": self.yaw_offset_deg,
            "yaw_deg": self.yaw_deg,
        }

    def to_payload(self):
        return dict(self.identity_payload(), yaw_id=self.yaw_id, motion_cost=self.motion_cost.to_payload())


@dataclass(frozen=True)
class SamplerOutput:
    schema_version: str
    sampler_context_id: str
    candidate_contract_identity: str
    probe_attempts: Tuple[ProbeAttemptRecord, ...]
    selected_positions: Tuple[SampledPositionPlan, ...]
    yaw_plans: Tuple[SampledYawPlan, ...]
    opportunity_constrained: bool
    opportunity_reason: Optional[str]

    def __post_init__(self) -> None:
        if self.schema_version != SAMPLER_OUTPUT_SCHEMA_VERSION:
            raise SamplerError("sampler output schema version is invalid")
        _stable_id(self.sampler_context_id, "sampler-context", "sampler_context_id")
        _stable_id(self.candidate_contract_identity, "candidate-contract", "candidate_contract_identity")
        if not self.probe_attempts:
            raise SamplerError("sampler output must retain raw probe attempts")
        if not self.selected_positions or not self.selected_positions[0].is_initial:
            raise SamplerError("initial position must be first and selected")
        if sum(position.is_initial for position in self.selected_positions) != 1:
            raise SamplerError("sampler output must contain exactly one initial position")
        if any(position.sampler_context_id != self.sampler_context_id for position in self.selected_positions):
            raise SamplerError("selected position context identity mismatch")
        if any(plan.sampler_context_id != self.sampler_context_id for plan in self.yaw_plans):
            raise SamplerError("yaw plan context identity mismatch")
        if any(attempt.sampler_context_id != self.sampler_context_id for attempt in self.probe_attempts):
            raise SamplerError("probe attempt context identity mismatch")
        if any(plan.position_id not in {position.position_id for position in self.selected_positions} for plan in self.yaw_plans):
            raise SamplerError("yaw plan references an unselected position")
        if not isinstance(self.opportunity_constrained, bool):
            raise SamplerError("opportunity_constrained must be boolean")
        if self.opportunity_constrained and self.opportunity_reason != "INSUFFICIENT_LOCAL_POSITIONS":
            raise SamplerError("opportunity_constrained reason is invalid")
        if not self.opportunity_constrained and self.opportunity_reason is not None:
            raise SamplerError("unconstrained output cannot have an opportunity reason")

    def to_payload(self):
        return {
            "schema_version": self.schema_version,
            "sampler_context_id": self.sampler_context_id,
            "candidate_contract_identity": self.candidate_contract_identity,
            "probe_attempts": [attempt.to_payload() for attempt in self.probe_attempts],
            "selected_positions": [position.to_payload() for position in self.selected_positions],
            "yaw_plans": [plan.to_payload() for plan in self.yaw_plans],
            "opportunity_constrained": self.opportunity_constrained,
            "opportunity_reason": self.opportunity_reason,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    def audit_counts(self):
        rejected = Counter(
            attempt.reason for attempt in self.probe_attempts if attempt.status == "REJECTED"
        )
        return {
            "raw_probe_count": len(self.probe_attempts),
            "technical_rejection_counts": dict(sorted(rejected.items())),
            "technical_valid_pool_count": sum(
                attempt.status in ("TECHNICAL_VALID", "SELECTED") for attempt in self.probe_attempts
            ),
            "selected_position_count": len(self.selected_positions),
            "yaw_expanded_count": len(self.yaw_plans),
            "opportunity_constrained": self.opportunity_constrained,
            "opportunity_reason": self.opportunity_reason,
        }


def _path_parts(result: Any) -> Tuple[bool, Optional[float], Optional[Tuple[WorldXYZ, ...]]]:
    if result is None:
        return False, None, None
    found = bool(getattr(result, "found", False))
    distance = getattr(result, "geodesic_distance_m", None)
    points = getattr(result, "points_world", None)
    if points is None:
        points = getattr(result, "points", None)
    if distance is not None:
        distance = _finite(distance, "path.geodesic_distance_m")
        if distance < 0.0:
            return False, distance, None
    if points is not None:
        try:
            points = tuple(_point(point, "path.points_world") for point in points)
        except (TypeError, SamplerError):
            return False, None, None
    if not found or distance is None or points is None or not points:
        return False, distance, points
    return True, distance, points


def _requested_probe(initial: WorldXYZ, initial_yaw_deg: float, radius_m: float, azimuth_deg: float) -> WorldXYZ:
    heading = normalize_yaw_deg(initial_yaw_deg + azimuth_deg)
    radians = math.radians(heading)
    # Shared A0 convention: forward=-Z, positive yaw is left.
    values = (
        initial[0] - math.sin(radians) * radius_m,
        initial[1],
        initial[2] - math.cos(radians) * radius_m,
    )
    # Normalize trigonometric round-off at cardinal directions before the
    # point becomes part of any source-free identity payload.
    return tuple(0.0 if abs(value) < POINT_EPSILON_M else value for value in values)  # type: ignore[return-value]


def _make_probe_id(
    context: SamplerContext,
    contract: CandidateContract,
    radius_m: float,
    azimuth_deg: float,
    requested: WorldXYZ,
) -> str:
    return stable_id(
        "probe",
        {
            "schema_version": PROBE_ATTEMPT_SCHEMA_VERSION,
            "sampler_context_id": context.sampler_context_id,
            "radius_m": radius_m,
            "azimuth_deg": azimuth_deg,
            "requested_base_xyz": list(requested),
        },
    )


def _make_position(context: SamplerContext, source_probe_id: str, requested: WorldXYZ, snapped: WorldXYZ, snap_error: float, path: Tuple[WorldXYZ, ...], geodesic: float, is_initial: bool) -> SampledPositionPlan:
    offset = tuple(context.sensor_xyz[index] - context.actual_listener_base_xyz[index] for index in range(3))
    sensor = tuple(snapped[index] + offset[index] for index in range(3))
    payload = {
        "schema_version": POSITION_PLAN_SCHEMA_VERSION,
        "sampler_context_id": context.sampler_context_id,
        "source_probe_id": source_probe_id,
        "requested_base_xyz": list(requested),
        "actual_snapped_base_xyz": list(snapped),
        "sensor_transform_identity": context.sensor_transform_identity,
        "sensor_xyz": list(sensor),
        "snap_error_m": snap_error,
        "path_polyline": [list(point) for point in path],
        "geodesic_path_length_m": geodesic,
        "is_initial": is_initial,
    }
    return SampledPositionPlan(position_id=stable_id("position", payload), **payload)


def _select_positions(
    context: SamplerContext,
    contract: CandidateContract,
    pathfinder: PathFinderLike,
    attempts: Sequence[ProbeAttemptRecord],
) -> Tuple[Tuple[ProbeAttemptRecord, ...], Tuple[SampledPositionPlan, ...], bool, Optional[str]]:
    initial = _make_position(
        context,
        "initial-listener-position",
        context.initial_listener_requested_base_xyz,
        context.actual_listener_base_xyz,
        _distance(context.initial_listener_requested_base_xyz, context.actual_listener_base_xyz),
        (context.actual_listener_base_xyz,),
        0.0,
        True,
    )
    pool = [attempt for attempt in attempts if attempt.status == "TECHNICAL_VALID"]
    selected_attempt_ids = set()
    selected_positions = [initial]
    selected_snapped = [initial.actual_snapped_base_xyz]
    remaining = list(pool)
    for index in range(contract.max_noninitial_positions):
        if not remaining:
            break
        if index == 0:
            ordered = sorted(remaining, key=lambda item: (-float(item.geodesic_path_length_m), item.probe_id))
        else:
            scored = []
            for attempt in remaining:
                distances = []
                candidate_point = attempt.snapped_base_xyz
                for selected_point in selected_snapped:
                    try:
                        found, distance, _ = _path_parts(pathfinder.shortest_path(selected_point, candidate_point))
                    except Exception:
                        found, distance = False, None
                    distances.append(float(distance) if found and distance is not None else -1.0)
                scored.append((min(distances) if distances else -1.0, attempt))
            ordered = [item for _, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].probe_id))]
        chosen = ordered[0]
        remaining.remove(chosen)
        selected_attempt_ids.add(chosen.probe_id)
        selected_snapped.append(chosen.snapped_base_xyz)
        selected_positions.append(
            _make_position(
                context,
                chosen.probe_id,
                chosen.requested_base_xyz,
                chosen.snapped_base_xyz,
                chosen.snap_error_m,
                chosen.path_polyline,
                chosen.geodesic_path_length_m,
                False,
            )
        )
    updated_attempts = tuple(
        replace(attempt, status="SELECTED") if attempt.probe_id in selected_attempt_ids else attempt
        for attempt in attempts
    )
    constrained = len(pool) < contract.max_noninitial_positions
    return updated_attempts, tuple(selected_positions), constrained, "INSUFFICIENT_LOCAL_POSITIONS" if constrained else None


def sample_source_free(
    context: SamplerContext,
    candidate_contract: CandidateContract,
    pathfinder: PathFinderLike,
) -> SamplerOutput:
    """Generate deterministic positions/yaws without accepting GeometryRecord."""

    if not isinstance(context, SamplerContext):
        raise SamplerError("source-free sampler requires SamplerContext, not GeometryRecord or another record")
    if not isinstance(candidate_contract, CandidateContract):
        raise SamplerError("source-free sampler requires CandidateContract")
    if context.candidate_contract_identity != candidate_contract.candidate_contract_identity:
        raise SamplerError("sampler context and candidate contract identities disagree")
    if context.sampler_algorithm_identity != SAMPLER_ALGORITHM_IDENTITY:
        raise SamplerError("unsupported sampler algorithm identity")
    for name in ("snap_point", "is_navigable", "shortest_path"):
        if not callable(getattr(pathfinder, name, None)):
            raise SamplerError("pathfinder lacks required operation {}".format(name))

    attempts = []
    initial = context.actual_listener_base_xyz
    for radius in candidate_contract.radii_m:
        for azimuth in candidate_contract.azimuth_offsets_deg:
            requested = _requested_probe(initial, context.initial_yaw_deg, radius, azimuth)
            probe_id = _make_probe_id(context, candidate_contract, radius, azimuth, requested)
            try:
                snapped_raw = pathfinder.snap_point(requested)
            except Exception:
                attempts.append(ProbeAttemptRecord(
                    PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                    radius, azimuth, requested, None, None, None, None, "REJECTED", "SNAP_FAILED"
                ))
                continue
            if snapped_raw is None:
                attempts.append(ProbeAttemptRecord(
                    PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                    radius, azimuth, requested, None, None, None, None, "REJECTED", "SNAP_FAILED"
                ))
                continue
            try:
                snapped = _point(snapped_raw, "probe.snapped_base_xyz")
            except SamplerError:
                attempts.append(ProbeAttemptRecord(
                    PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                    radius, azimuth, requested, None, None, None, None, "REJECTED", "NONFINITE_SNAPPED_POINT"
                ))
                continue
            snap_error = _distance(requested, snapped)
            try:
                navigable = bool(pathfinder.is_navigable(snapped))
            except Exception:
                navigable = False
            if not navigable:
                reason = "NON_NAVIGABLE"
                attempts.append(ProbeAttemptRecord(
                    PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                    radius, azimuth, requested, snapped, snap_error, None, None, "REJECTED", reason
                ))
                continue
            if snap_error > candidate_contract.max_snap_error_m:
                attempts.append(ProbeAttemptRecord(
                    PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                    radius, azimuth, requested, snapped, snap_error, None, None, "REJECTED", "SNAP_TOO_FAR"
                ))
                continue
            try:
                found, distance, points = _path_parts(pathfinder.shortest_path(initial, snapped))
            except Exception:
                found, distance, points = False, None, None
            if not found or distance is None or points is None:
                attempts.append(ProbeAttemptRecord(
                    PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                    radius, azimuth, requested, snapped, snap_error, points, distance, "REJECTED", "NO_PATH"
                ))
                continue
            if distance > candidate_contract.max_geodesic_radius_m:
                attempts.append(ProbeAttemptRecord(
                    PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                    radius, azimuth, requested, snapped, snap_error, points, distance, "REJECTED", "OUTSIDE_GEODESIC_RADIUS"
                ))
                continue
            duplicate = _distance(snapped, initial) <= candidate_contract.duplicate_position_tolerance_m
            if not duplicate:
                duplicate = any(
                    existing.snapped_base_xyz is not None
                    and _distance(snapped, existing.snapped_base_xyz) <= candidate_contract.duplicate_position_tolerance_m
                    for existing in attempts if existing.status == "TECHNICAL_VALID"
                )
            if duplicate:
                attempts.append(ProbeAttemptRecord(
                    PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                    radius, azimuth, requested, snapped, snap_error, points, distance, "REJECTED", "DUPLICATE_SNAPPED_POSITION"
                ))
                continue
            attempts.append(ProbeAttemptRecord(
                PROBE_ATTEMPT_SCHEMA_VERSION, probe_id, context.sampler_context_id,
                radius, azimuth, requested, snapped, snap_error, points, distance, "TECHNICAL_VALID", None
            ))

    attempts, positions, constrained, constrained_reason = _select_positions(context, candidate_contract, pathfinder, attempts)
    yaw_plans = []
    for position in positions:
        for offset in candidate_contract.yaw_offsets_deg:
            yaw = normalize_yaw_deg(context.initial_yaw_deg + offset)
            cost = compute_motion_cost(
                context.actual_listener_base_xyz,
                context.initial_yaw_deg,
                position.path_polyline,
                yaw,
                position.geodesic_path_length_m,
                candidate_contract.motion_parameters,
            )
            payload = {
                "schema_version": YAW_PLAN_SCHEMA_VERSION,
                "position_id": position.position_id,
                "sampler_context_id": context.sampler_context_id,
                "yaw_grid_identity": candidate_contract.yaw_grid_identity,
                "yaw_offset_deg": offset,
                "yaw_deg": yaw,
            }
            yaw_plans.append(SampledYawPlan(
                schema_version=YAW_PLAN_SCHEMA_VERSION,
                yaw_id=stable_id("yaw", payload),
                position_id=position.position_id,
                sampler_context_id=context.sampler_context_id,
                yaw_grid_identity=candidate_contract.yaw_grid_identity,
                yaw_offset_deg=offset,
                yaw_deg=yaw,
                motion_cost=cost,
            ))
    return SamplerOutput(
        schema_version=SAMPLER_OUTPUT_SCHEMA_VERSION,
        sampler_context_id=context.sampler_context_id,
        candidate_contract_identity=candidate_contract.candidate_contract_identity,
        probe_attempts=attempts,
        selected_positions=positions,
        yaw_plans=tuple(yaw_plans),
        opportunity_constrained=constrained,
        opportunity_reason=constrained_reason,
    )


def _source_position(pose: Mapping, path: str) -> WorldXYZ:
    if not isinstance(pose, Mapping):
        raise SamplerError("{} must be a mapping".format(path))
    for key in ("position_xyz", "source_position_world", "position_world"):
        if key in pose:
            return _point(pose[key], path + "." + key)
    raise SamplerError("{} has no acoustic source position".format(path))


def annotate_geometry_legality(
    sampler_output: SamplerOutput,
    context: SamplerContext,
    candidate_contract: CandidateContract,
    geometry: GeometryRecord,
) -> Tuple[PoseRecord, ...]:
    """Bind source clearance only after source-free sampling is complete."""

    if geometry.candidate_contract_identity != candidate_contract.candidate_contract_identity:
        raise SamplerError("geometry and candidate contract identities disagree")
    if sampler_output.sampler_context_id != context.sampler_context_id:
        raise SamplerError("sampler output and context identities disagree")
    target = _source_position(geometry.target_world_pose, "geometry.target_world_pose")
    noise = _source_position(geometry.noise_world_pose, "geometry.noise_world_pose")
    position_by_id = {position.position_id: position for position in sampler_output.selected_positions}
    records = []
    for yaw_plan in sampler_output.yaw_plans:
        position = position_by_id[yaw_plan.position_id]
        target_close = _distance(position.sensor_xyz, target) < candidate_contract.source_clearance_m
        noise_close = _distance(position.sensor_xyz, noise) < candidate_contract.source_clearance_m
        if target_close and noise_close:
            reason = "TARGET_AND_NOISE_SOURCE_CLEARANCE"
        elif target_close:
            reason = "TARGET_SOURCE_CLEARANCE"
        elif noise_close:
            reason = "NOISE_SOURCE_CLEARANCE"
        else:
            reason = None
        payload = {
            "schema_version": POSE_SCHEMA_VERSION,
            "geometry_id": geometry.geometry_id,
            "position_id": position.position_id,
            "yaw_id": yaw_plan.yaw_id,
            "requested_base_xyz": list(position.requested_base_xyz),
            "actual_snapped_base_xyz": list(position.actual_snapped_base_xyz),
            "sensor_transform_identity": position.sensor_transform_identity,
            "sensor_xyz": list(position.sensor_xyz),
            "yaw_deg": yaw_plan.yaw_deg,
            "snap_error_m": position.snap_error_m,
            "path_polyline": [list(point) for point in position.path_polyline],
            "geodesic_path_length_m": position.geodesic_path_length_m,
            "initial_to_path_turn_deg": yaw_plan.motion_cost.initial_to_path_turn_deg,
            "internal_path_turn_deg": yaw_plan.motion_cost.internal_path_turn_deg,
            "final_turn_deg": yaw_plan.motion_cost.final_turn_deg,
            "settling_sec": yaw_plan.motion_cost.settling_sec,
            "total_cost_sec": yaw_plan.motion_cost.total_cost_sec,
            "motion_contract_identity": yaw_plan.motion_cost.motion_contract_identity,
            "path_polyline_length_m": yaw_plan.motion_cost.polyline_length_m,
            "translation_speed_mps": yaw_plan.motion_cost.translation_speed_mps,
            "rotation_speed_dps": yaw_plan.motion_cost.rotation_speed_dps,
            "translation_sec": yaw_plan.motion_cost.translation_sec,
            "rotation_sec": yaw_plan.motion_cost.rotation_sec,
            "budget_sec": yaw_plan.motion_cost.budget_sec,
            "budget_feasible": yaw_plan.motion_cost.budget_feasible,
            "geometry_legality": "ILLEGAL" if reason else "LEGAL",
            "invalid_reason": reason,
        }
        records.append(PoseRecord(pose_id=stable_id("pose", pose_identity_payload(payload)), **payload))
    return tuple(records)


__all__ = [
    "CANDIDATE_CONTRACT_SCHEMA_VERSION",
    "COORDINATE_CONVENTION_IDENTITY",
    "COVERAGE_SELECTION_IDENTITY",
    "CandidateContract",
    "PathFinderLike",
    "ProbeAttemptRecord",
    "SamplerContext",
    "SamplerError",
    "SamplerOutput",
    "SampledPositionPlan",
    "SampledYawPlan",
    "SAMPLER_ALGORITHM_IDENTITY",
    "annotate_geometry_legality",
    "make_sampler_context",
    "sample_source_free",
]
