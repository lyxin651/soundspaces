"""Immutable A4-0 typed records and their stable identity dependencies.

These records describe an experiment plan and its provenance.  They do not
execute sampling, calibration, rendering, mixing, or decoding.  In
particular, a :class:`BlockRecord` is never mutated when calibration produces
an alpha; calibration is represented by a separate immutable artifact.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import math
from types import MappingProxyType
from typing import Any, Dict, Iterable, Optional, Tuple

from active_audition.a4.identity import (
    A4IdentityError,
    identity_sha256,
    stable_id,
    validate_sha256,
    validate_stable_id,
)
from active_audition.a4.noise_segments import NoiseSegmentError, NoiseSegmentRecord


GEOMETRY_SCHEMA_VERSION = "active-asr-a4-geometry-v1"
POSE_SCHEMA_VERSION = "active-asr-a4-pose-v2"
BLOCK_SCHEMA_VERSION = "active-asr-a4-block-v1"
EPISODE_SCHEMA_VERSION = "active-asr-a4-episode-v1"
CALIBRATION_SCHEMA_VERSION = "active-asr-a4-calibration-v1"
RESULT_DEPENDENT_KEYS = frozenset(
    {
        "wer",
        "cer",
        "hypothesis",
        "score",
        "sdiw",
        "run_timestamp",
        "timestamp",
        "created_at",
        "updated_at",
        "raw_decoder_metadata",
        "asr_result",
    }
)


class RecordError(ValueError):
    """Raised when an A4 record is incomplete or identity-inconsistent."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _error(message: str) -> None:
    raise RecordError(message)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _error("{} must be a non-empty string".format(path))
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _error("{} must be boolean".format(path))
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error("{} must be numeric".format(path))
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        _error("{} must be finite".format(path))
    return result


def _normalize_yaw_deg(value: Any, path: str) -> float:
    result = _finite(value, path)
    return ((result + 180.0) % 360.0) - 180.0


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _error("{} must be a positive integer".format(path))
    return value


def _xyz(value: Any, path: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        _error("{} must contain three coordinates".format(path))
    return tuple(_finite(item, "{}[{}]".format(path, index)) for index, item in enumerate(value))


def _polyline_length(value: Any, path: str) -> float:
    if not isinstance(value, (list, tuple)) or not value:
        _error("{} must not be empty".format(path))
    points = tuple(_xyz(point, "{}[{}]".format(path, index)) for index, point in enumerate(value))
    return sum(
        math.sqrt(sum((end[index] - start[index]) ** 2 for index in range(3)))
        for start, end in zip(points, points[1:])
    )


def _mapping(value: Any, path: str, nonempty: bool = True) -> Mapping:
    if not isinstance(value, Mapping) or (nonempty and not value):
        _error("{} must be a non-empty mapping".format(path))
    _validate_nested(value, path)
    return value


def _validate_nested(value: Any, path: str) -> None:
    """Validate semantic maps without imposing an algorithmic schema here."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                _error("{} contains a non-string key".format(path))
            if key.lower() in RESULT_DEPENDENT_KEYS:
                _error("{} contains result-dependent field {!r}".format(path, key))
            if key.endswith("sha256") or key == "sha256":
                try:
                    validate_sha256(item, "{}.{}".format(path, key))
                except A4IdentityError as exc:
                    _error(str(exc))
            else:
                _validate_nested(item, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_nested(item, "{}[{}]".format(path, index))
    elif isinstance(value, float):
        _finite(value, path)
    elif value is not None and not isinstance(value, (str, int, bool)):
        _error("{} contains unsupported value type {}".format(path, type(value).__name__))


def _identity_string(value: Any, path: str) -> str:
    """Validate a non-empty semantic identity token.

    Contract identities may be human-readable names while content identities
    are normally full SHA-derived IDs.  Explicit ``*_sha256`` fields are
    validated recursively by :func:`_validate_nested`.
    """

    return _string(value, path)


def _ids(value: Any, path: str, count: Optional[int] = None) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        _error("{} must be a list".format(path))
    if count is not None and len(value) != count:
        _error("{} must contain {} values".format(path, count))
    result = tuple(_string(item, "{}[{}]".format(path, index)) for index, item in enumerate(value))
    if len(set(result)) != len(result):
        _error("{} must not contain duplicate IDs".format(path))
    return result


def _validate_top_level(payload: Mapping, expected: Iterable[str], path: str) -> None:
    if not isinstance(payload, Mapping):
        _error("{} must be a mapping".format(path))
    expected_set = set(expected)
    unknown = sorted(set(payload) - expected_set)
    missing = sorted(expected_set - set(payload))
    if unknown or missing:
        _error("{} fields invalid: unknown={}, missing={}".format(path, unknown, missing))


def _record_init(value: Any, expected: str, path: str) -> str:
    return _string(value, path) if value == expected else _error("{} must be {!r}".format(path, expected))


def _check_id(actual: str, namespace: str, payload: Mapping, path: str) -> None:
    expected = stable_id(namespace, payload)
    if actual != expected:
        _error("{} does not match canonical semantic payload".format(path))


def _plain_record(record: Any) -> Dict[str, Any]:
    return {field: _plain(getattr(record, field)) for field in record.__dataclass_fields__}


def _pose_identity_payload(record: Any) -> Dict[str, Any]:
    """Return only the geometry-bound pose-plan semantics.

    Motion breakdown values and legality annotations are derived/provenance
    fields.  They remain serialized and validated, but changing one of them
    cannot silently create a second plan identity.  The motion contract and
    geometry identities bind the semantics that produced those values.
    """

    def value(key: str) -> Any:
        if isinstance(record, Mapping):
            return record[key]
        return getattr(record, key)

    return {
        "schema_version": value("schema_version"),
        "geometry_id": value("geometry_id"),
        "position_id": value("position_id"),
        "yaw_id": value("yaw_id"),
        "requested_base_xyz": _plain(value("requested_base_xyz")),
        "actual_snapped_base_xyz": _plain(value("actual_snapped_base_xyz")),
        "sensor_transform_identity": value("sensor_transform_identity"),
        "sensor_xyz": _plain(value("sensor_xyz")),
        "yaw_deg": value("yaw_deg"),
        "snap_error_m": value("snap_error_m"),
        "path_polyline": _plain(value("path_polyline")),
        "geodesic_path_length_m": value("geodesic_path_length_m"),
        "motion_contract_identity": value("motion_contract_identity"),
    }


def pose_identity_payload(value: Any) -> Dict[str, Any]:
    """Public identity projection for record builders and audit tests."""

    return _pose_identity_payload(value)


def _from_payload(cls: Any, payload: Mapping) -> Any:
    if not isinstance(payload, Mapping):
        _error("record payload must be a mapping")
    try:
        return cls(**dict(payload))
    except TypeError as exc:
        raise RecordError("record payload fields are invalid: {}".format(exc)) from exc


@dataclass(frozen=True)
class GeometryRecord:
    schema_version: str
    geometry_id: str
    scene_id: str
    scene_resource_identities: Mapping[str, str]
    initial_listener_requested_base_xyz: Tuple[float, float, float]
    actual_listener_base_xyz: Tuple[float, float, float]
    sensor_xyz: Tuple[float, float, float]
    sensor_transform_identity: str
    initial_yaw_deg: float
    target_world_pose: Mapping[str, Any]
    noise_world_pose: Mapping[str, Any]
    production_acoustic_policy_identity: str
    candidate_contract_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_resource_identities", _freeze(self.scene_resource_identities))
        object.__setattr__(self, "initial_listener_requested_base_xyz", _freeze(self.initial_listener_requested_base_xyz))
        object.__setattr__(self, "actual_listener_base_xyz", _freeze(self.actual_listener_base_xyz))
        object.__setattr__(self, "sensor_xyz", _freeze(self.sensor_xyz))
        object.__setattr__(self, "target_world_pose", _freeze(self.target_world_pose))
        object.__setattr__(self, "noise_world_pose", _freeze(self.noise_world_pose))
        validate_geometry_record(self)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "scene_resource_identities": _plain(self.scene_resource_identities),
            "initial_listener_requested_base_xyz": _plain(self.initial_listener_requested_base_xyz),
            "actual_listener_base_xyz": _plain(self.actual_listener_base_xyz),
            "sensor_xyz": _plain(self.sensor_xyz),
            "sensor_transform_identity": self.sensor_transform_identity,
            "initial_yaw_deg": self.initial_yaw_deg,
            "target_world_pose": _plain(self.target_world_pose),
            "noise_world_pose": _plain(self.noise_world_pose),
            "production_acoustic_policy_identity": self.production_acoustic_policy_identity,
            "candidate_contract_identity": self.candidate_contract_identity,
        }

    def to_payload(self) -> Dict[str, Any]:
        return _plain_record(self)

    as_payload = to_payload
    to_dict = to_payload

    @classmethod
    def from_payload(cls, payload: Mapping) -> "GeometryRecord":
        return _from_payload(cls, payload)

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


def validate_geometry_record(record: GeometryRecord) -> GeometryRecord:
    _record_init(record.schema_version, GEOMETRY_SCHEMA_VERSION, "geometry.schema_version")
    _string(record.geometry_id, "geometry.geometry_id")
    try:
        validate_stable_id(record.geometry_id, "geometry", "geometry.geometry_id")
    except A4IdentityError as exc:
        _error(str(exc))
    _string(record.scene_id, "geometry.scene_id")
    resources = _mapping(record.scene_resource_identities, "geometry.scene_resource_identities")
    if not resources:
        _error("geometry.scene_resource_identities must not be empty")
    for key, value in resources.items():
        _string(key, "geometry.scene_resource_identities key")
        try:
            validate_sha256(value, "geometry.scene_resource_identities.{}".format(key))
        except A4IdentityError as exc:
            _error(str(exc))
    _xyz(record.initial_listener_requested_base_xyz, "geometry.initial_listener_requested_base_xyz")
    _xyz(record.actual_listener_base_xyz, "geometry.actual_listener_base_xyz")
    _xyz(record.sensor_xyz, "geometry.sensor_xyz")
    _identity_string(record.sensor_transform_identity, "geometry.sensor_transform_identity")
    _finite(record.initial_yaw_deg, "geometry.initial_yaw_deg")
    _mapping(record.target_world_pose, "geometry.target_world_pose")
    _mapping(record.noise_world_pose, "geometry.noise_world_pose")
    _identity_string(record.production_acoustic_policy_identity, "geometry.production_acoustic_policy_identity")
    _identity_string(record.candidate_contract_identity, "geometry.candidate_contract_identity")
    _check_id(record.geometry_id, "geometry", record.identity_payload(), "geometry.geometry_id")
    return record


@dataclass(frozen=True)
class PoseRecord:
    schema_version: str
    geometry_id: str
    pose_id: str
    position_id: str
    yaw_id: str
    requested_base_xyz: Tuple[float, float, float]
    actual_snapped_base_xyz: Tuple[float, float, float]
    sensor_transform_identity: str
    sensor_xyz: Tuple[float, float, float]
    yaw_deg: float
    snap_error_m: float
    path_polyline: Tuple[Tuple[float, float, float], ...]
    geodesic_path_length_m: float
    initial_to_path_turn_deg: float
    internal_path_turn_deg: float
    final_turn_deg: float
    settling_sec: float
    total_cost_sec: float
    motion_contract_identity: str
    path_polyline_length_m: float
    translation_speed_mps: float
    rotation_speed_dps: float
    translation_sec: float
    rotation_sec: float
    budget_sec: float
    budget_feasible: bool
    geometry_legality: str
    invalid_reason: Optional[str]

    def __post_init__(self) -> None:
        for field in ("requested_base_xyz", "actual_snapped_base_xyz", "sensor_xyz"):
            object.__setattr__(self, field, _freeze(_xyz(getattr(self, field), "pose." + field)))
        object.__setattr__(self, "path_polyline", _freeze(tuple(
            _xyz(point, "pose.path_polyline") for point in self.path_polyline
        )))
        object.__setattr__(self, "yaw_deg", _normalize_yaw_deg(self.yaw_deg, "pose.yaw_deg"))
        for field in (
            "snap_error_m", "geodesic_path_length_m", "initial_to_path_turn_deg",
            "internal_path_turn_deg", "final_turn_deg", "settling_sec", "total_cost_sec",
            "path_polyline_length_m", "translation_speed_mps", "rotation_speed_dps",
            "translation_sec", "rotation_sec", "budget_sec",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), "pose." + field))
        validate_pose_record(self)

    def identity_payload(self) -> Dict[str, Any]:
        return _pose_identity_payload(self)

    def to_payload(self) -> Dict[str, Any]:
        return _plain_record(self)

    as_payload = to_payload
    to_dict = to_payload

    @classmethod
    def from_payload(cls, payload: Mapping) -> "PoseRecord":
        return _from_payload(cls, payload)

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


def validate_pose_record(record: PoseRecord) -> PoseRecord:
    _record_init(record.schema_version, POSE_SCHEMA_VERSION, "pose.schema_version")
    try:
        validate_stable_id(record.geometry_id, "geometry", "pose.geometry_id")
        validate_stable_id(record.pose_id, "pose", "pose.pose_id")
    except A4IdentityError as exc:
        _error(str(exc))
    _string(record.position_id, "pose.position_id")
    _string(record.yaw_id, "pose.yaw_id")
    _xyz(record.requested_base_xyz, "pose.requested_base_xyz")
    _xyz(record.actual_snapped_base_xyz, "pose.actual_snapped_base_xyz")
    _identity_string(record.sensor_transform_identity, "pose.sensor_transform_identity")
    _xyz(record.sensor_xyz, "pose.sensor_xyz")
    expected_snap_error = math.sqrt(sum(
        (record.actual_snapped_base_xyz[index] - record.requested_base_xyz[index]) ** 2
        for index in range(3)
    ))
    if not math.isclose(expected_snap_error, record.snap_error_m, rel_tol=1.0e-9, abs_tol=1.0e-9):
        _error("pose.snap_error_m does not match requested and snapped coordinates")
    try:
        validate_stable_id(record.motion_contract_identity, "motion-contract", "pose.motion_contract_identity")
    except A4IdentityError as exc:
        _error(str(exc))
    for field in ("yaw_deg", "snap_error_m", "geodesic_path_length_m", "initial_to_path_turn_deg", "internal_path_turn_deg", "final_turn_deg", "settling_sec", "total_cost_sec", "path_polyline_length_m", "translation_speed_mps", "rotation_speed_dps", "translation_sec", "rotation_sec", "budget_sec"):
        if _finite(getattr(record, field), "pose." + field) < 0.0 and field not in ("yaw_deg",):
            _error("pose.{} must not be negative".format(field))
    if _finite(record.translation_speed_mps, "pose.translation_speed_mps") <= 0.0:
        _error("pose.translation_speed_mps must be positive")
    if _finite(record.rotation_speed_dps, "pose.rotation_speed_dps") <= 0.0:
        _error("pose.rotation_speed_dps must be positive")
    if not isinstance(record.path_polyline, (list, tuple)) or len(record.path_polyline) == 0:
        _error("pose.path_polyline must not be empty")
    for index, point in enumerate(record.path_polyline):
        _xyz(point, "pose.path_polyline[{}]".format(index))
    polyline_length = _polyline_length(record.path_polyline, "pose.path_polyline")
    if not math.isclose(polyline_length, record.path_polyline_length_m, rel_tol=1.0e-9, abs_tol=1.0e-9):
        _error("pose.path_polyline_length_m does not match path_polyline")
    if not math.isclose(record.translation_sec, record.geodesic_path_length_m / record.translation_speed_mps, rel_tol=1.0e-9, abs_tol=1.0e-9):
        _error("pose.translation_sec does not match geodesic distance authority")
    expected_rotation = (record.initial_to_path_turn_deg + record.internal_path_turn_deg + record.final_turn_deg) / record.rotation_speed_dps
    if not math.isclose(record.rotation_sec, expected_rotation, rel_tol=1.0e-9, abs_tol=1.0e-9):
        _error("pose.rotation_sec does not match turn breakdown")
    if math.isclose(record.geodesic_path_length_m, 0.0, abs_tol=1.0e-9) and math.isclose(expected_rotation, 0.0, abs_tol=1.0e-9):
        if record.settling_sec != 0.0 or record.total_cost_sec != 0.0:
            _error("stay pose must have exact zero settling and total cost")
    if not math.isclose(record.total_cost_sec, record.translation_sec + record.rotation_sec + record.settling_sec, rel_tol=1.0e-9, abs_tol=1.0e-9):
        _error("pose.total_cost_sec does not match cost breakdown")
    _bool(record.budget_feasible, "pose.budget_feasible")
    if record.budget_feasible != (record.total_cost_sec <= record.budget_sec + 1.0e-12):
        _error("pose.budget_feasible does not match total cost and budget")
    _string(record.geometry_legality, "pose.geometry_legality")
    if record.geometry_legality not in ("LEGAL", "ILLEGAL", "NOT_EVALUATED"):
        _error("pose.geometry_legality is invalid")
    if record.invalid_reason is not None:
        _string(record.invalid_reason, "pose.invalid_reason")
    if record.geometry_legality == "LEGAL" and record.invalid_reason is not None:
        _error("legal pose cannot have invalid_reason")
    if record.geometry_legality != "LEGAL" and record.invalid_reason is None:
        _error("illegal pose must have invalid_reason")
    expected = stable_id("pose", record.identity_payload())
    if record.pose_id != expected:
        _error("pose.pose_id does not match canonical semantic payload")
    return record


@dataclass(frozen=True)
class BlockRecord:
    schema_version: str
    block_id: str
    geometry_id: str
    speaker_id: str
    noise_parent_id: str
    nominal_initial_snr_db: float
    global_gain_identity: str
    selection_utterance_ids: Tuple[str, str]
    evaluation_utterance_ids: Tuple[str, str]
    noise_segment_plan_identity: str

    def __post_init__(self) -> None:
        for field in ("selection_utterance_ids", "evaluation_utterance_ids"):
            object.__setattr__(self, field, _freeze(getattr(self, field)))
        validate_block_record(self)

    def identity_payload(self) -> Dict[str, Any]:
        # The complete block plan identity.  Episode links and calibration
        # outputs belong to separate records/artifacts, never to this plan.
        return {
            "geometry_id": self.geometry_id,
            "speaker_id": self.speaker_id,
            "noise_parent_id": self.noise_parent_id,
            "nominal_initial_snr_db": self.nominal_initial_snr_db,
            "selection_utterance_ids": _plain(self.selection_utterance_ids),
            "evaluation_utterance_ids": _plain(self.evaluation_utterance_ids),
            "noise_segment_plan_identity": self.noise_segment_plan_identity,
            "global_gain_identity": self.global_gain_identity,
        }

    def to_payload(self) -> Dict[str, Any]:
        return _plain_record(self)

    as_payload = to_payload
    to_dict = to_payload

    @classmethod
    def from_payload(cls, payload: Mapping) -> "BlockRecord":
        return _from_payload(cls, payload)

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


def validate_block_record(record: BlockRecord) -> BlockRecord:
    _record_init(record.schema_version, BLOCK_SCHEMA_VERSION, "block.schema_version")
    try:
        validate_stable_id(record.block_id, "block", "block.block_id")
        validate_stable_id(record.geometry_id, "geometry", "block.geometry_id")
    except A4IdentityError as exc:
        _error(str(exc))
    _identity_string(record.speaker_id, "block.speaker_id")
    _identity_string(record.global_gain_identity, "block.global_gain_identity")
    try:
        validate_stable_id(record.noise_parent_id, "noise-parent", "block.noise_parent_id")
        validate_stable_id(record.noise_segment_plan_identity, "noise-segment-plan", "block.noise_segment_plan_identity")
    except A4IdentityError as exc:
        _error(str(exc))
    _finite(record.nominal_initial_snr_db, "block.nominal_initial_snr_db")
    _ids(record.selection_utterance_ids, "block.selection_utterance_ids", 2)
    _ids(record.evaluation_utterance_ids, "block.evaluation_utterance_ids", 2)
    if set(record.selection_utterance_ids) & set(record.evaluation_utterance_ids):
        _error("selection and evaluation utterance IDs must be disjoint")
    _check_id(record.block_id, "block", record.identity_payload(), "block.block_id")
    return record


@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: str
    episode_id: str
    block_id: str
    role: str
    utterance_identity: Mapping[str, Any]
    reference_identity: Mapping[str, Any]
    fixed_dry_noise_segment_identity: Mapping[str, Any]
    target_source_duration_sec: float
    noise_source_time_start_sec: float
    noise_source_time_end_sec: float
    noise_segment_duration_sec: float

    def __post_init__(self) -> None:
        for field in ("utterance_identity", "reference_identity", "fixed_dry_noise_segment_identity"):
            object.__setattr__(self, field, _freeze(getattr(self, field)))
        validate_episode_record(self)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "role": self.role,
            "utterance_identity": _plain(self.utterance_identity),
            "reference_identity": _plain(self.reference_identity),
            "fixed_dry_noise_segment_identity": _plain(self.fixed_dry_noise_segment_identity),
            "target_source_duration_sec": self.target_source_duration_sec,
            "noise_source_time_start_sec": self.noise_source_time_start_sec,
            "noise_source_time_end_sec": self.noise_source_time_end_sec,
            "noise_segment_duration_sec": self.noise_segment_duration_sec,
        }

    def to_payload(self) -> Dict[str, Any]:
        return _plain_record(self)

    as_payload = to_payload
    to_dict = to_payload

    @classmethod
    def from_payload(cls, payload: Mapping) -> "EpisodeRecord":
        return _from_payload(cls, payload)

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


def validate_episode_record(record: EpisodeRecord) -> EpisodeRecord:
    _record_init(record.schema_version, EPISODE_SCHEMA_VERSION, "episode.schema_version")
    try:
        validate_stable_id(record.episode_id, "episode", "episode.episode_id")
        validate_stable_id(record.block_id, "block", "episode.block_id")
    except A4IdentityError as exc:
        _error(str(exc))
    if record.role not in ("selection", "evaluation"):
        _error("episode.role must be selection or evaluation")
    _mapping(record.utterance_identity, "episode.utterance_identity")
    _mapping(record.reference_identity, "episode.reference_identity")
    _mapping(record.fixed_dry_noise_segment_identity, "episode.fixed_dry_noise_segment_identity")
    try:
        segment = NoiseSegmentRecord.from_payload(_plain(record.fixed_dry_noise_segment_identity))
    except (NoiseSegmentError, TypeError) as exc:
        _error("episode.fixed_dry_noise_segment_identity is not a valid A4 noise segment: {}".format(exc))
    if segment.role != record.role:
        _error("episode.role does not match fixed noise segment role")
    utterance_id = record.utterance_identity.get("utterance_id")
    if utterance_id != segment.utterance_identity:
        _error("episode utterance_id does not match fixed noise segment utterance identity")
    target_duration = _finite(record.target_source_duration_sec, "episode.target_source_duration_sec")
    start = _finite(record.noise_source_time_start_sec, "episode.noise_source_time_start_sec")
    end = _finite(record.noise_source_time_end_sec, "episode.noise_source_time_end_sec")
    duration = _finite(record.noise_segment_duration_sec, "episode.noise_segment_duration_sec")
    if (
        target_duration <= 0.0
        or end <= start
        or duration <= 0.0
        or abs((end - start) - duration) > 1.0e-9
        or not math.isclose(target_duration, segment.target_duration_sec, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(start, segment.source_time_start_sec, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(end, segment.source_time_end_sec, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(duration, segment.source_time_end_sec - segment.source_time_start_sec, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        _error("episode noise source-time interval is inconsistent")
    _check_id(record.episode_id, "episode", record.identity_payload(), "episode.episode_id")
    return record


@dataclass(frozen=True)
class CalibrationArtifact:
    """Immutable result interface for future selection-only calibration.

    A4-0 validates the artifact shape and identity only.  It does not compute
    ``alpha`` or measured SNR and does not accept waveforms.
    """

    schema_version: str
    calibration_artifact_id: str
    block_id: str
    calibration_contract_identity: str
    selection_episode_ids: Tuple[str, str]
    input_component_identities: Mapping[str, Any]
    ps: float
    pn: float
    active_sample_count: int
    alpha: float
    nominal_snr_db: float
    measured_snr_db: float
    status: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field in ("selection_episode_ids", "input_component_identities", "provenance"):
            object.__setattr__(self, field, _freeze(getattr(self, field)))
        validate_calibration_artifact(self)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "calibration_contract_identity": self.calibration_contract_identity,
            "selection_episode_ids": _plain(self.selection_episode_ids),
            "input_component_identities": _plain(self.input_component_identities),
            "ps": self.ps,
            "pn": self.pn,
            "active_sample_count": self.active_sample_count,
            "alpha": self.alpha,
            "nominal_snr_db": self.nominal_snr_db,
            "measured_snr_db": self.measured_snr_db,
            "status": self.status,
            "provenance": _plain(self.provenance),
        }

    def to_payload(self) -> Dict[str, Any]:
        return _plain_record(self)

    as_payload = to_payload
    to_dict = to_payload

    @classmethod
    def from_payload(cls, payload: Mapping) -> "CalibrationArtifact":
        return _from_payload(cls, payload)

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


def validate_calibration_artifact(record: CalibrationArtifact) -> CalibrationArtifact:
    _record_init(record.schema_version, CALIBRATION_SCHEMA_VERSION, "calibration.schema_version")
    try:
        validate_stable_id(record.calibration_artifact_id, "calibration", "calibration.calibration_artifact_id")
        validate_stable_id(record.block_id, "block", "calibration.block_id")
    except A4IdentityError as exc:
        _error(str(exc))
    _identity_string(record.calibration_contract_identity, "calibration.calibration_contract_identity")
    _ids(record.selection_episode_ids, "calibration.selection_episode_ids", 2)
    _mapping(record.input_component_identities, "calibration.input_component_identities")
    for field in ("ps", "pn", "alpha"):
        if _finite(getattr(record, field), "calibration." + field) <= 0.0:
            _error("calibration.{} must be positive".format(field))
    _positive_int(record.active_sample_count, "calibration.active_sample_count")
    _finite(record.nominal_snr_db, "calibration.nominal_snr_db")
    _finite(record.measured_snr_db, "calibration.measured_snr_db")
    if record.status not in ("CALIBRATED", "FAILED", "INVALID"):
        _error("calibration.status is invalid")
    _mapping(record.provenance, "calibration.provenance")
    _check_id(record.calibration_artifact_id, "calibration", record.identity_payload(), "calibration.calibration_artifact_id")
    return record


__all__ = [
    "BLOCK_SCHEMA_VERSION",
    "BlockRecord",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationArtifact",
    "EPISODE_SCHEMA_VERSION",
    "EpisodeRecord",
    "GEOMETRY_SCHEMA_VERSION",
    "GeometryRecord",
    "POSE_SCHEMA_VERSION",
    "PoseRecord",
    "RecordError",
    "validate_block_record",
    "validate_calibration_artifact",
    "validate_episode_record",
    "validate_geometry_record",
    "validate_pose_record",
    "pose_identity_payload",
]
