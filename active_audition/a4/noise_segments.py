"""Pure deterministic planning for fixed source-time noise segments.

This module plans metadata and sample ranges only.  It deliberately does not
decode audio, concatenate payloads, loop a parent, render RIRs, or consume any
SNR/ASR/Oracle result.
"""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.identity import (
    identity_sha256,
    stable_id,
    validate_sha256,
    validate_stable_id,
)


NOISE_PARENT_SCHEMA_VERSION = "active-asr-a4-noise-parent-v1"
NOISE_SEGMENT_SCHEMA_VERSION = "active-asr-a4-noise-segment-v2"
NOISE_SEGMENT_PLAN_SCHEMA_VERSION = "active-asr-a4-noise-segment-plan-v2"
NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY = "active-asr-a4-noise-segment-planner-v2"
SOURCE_TIME_CONVENTION_IDENTITY = "active-asr-a4-target-dry-onset-zero-v1"
SAMPLE_INDEX_CONVENTION_IDENTITY = "active-asr-a4-half-open-sample-range-v1"
EPISODE_ORDER_IDENTITY = "active-asr-a4-stable-episode-order-v1"
SAMPLE_ROUNDING_IDENTITY = "active-asr-a4-seconds-to-samples-nearest-half-up-v1"
A3_NOISE_PARENT_ADAPTER_IDENTITY = "active-asr-a4-a3-parent-provenance-adapter-v1"
EPISODE_SLOT_ORDER = (("selection", 0), ("selection", 1), ("evaluation", 0), ("evaluation", 1))


class NoiseSegmentError(ValueError):
    """Raised when a fixed parent cannot satisfy the segment plan."""


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise NoiseSegmentError("{} must be a non-empty string".format(path))
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NoiseSegmentError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise NoiseSegmentError("{} must be finite".format(path))
    return result


def _nonnegative(value: Any, path: str) -> float:
    result = _finite(value, path)
    if result < 0.0:
        raise NoiseSegmentError("{} must be non-negative".format(path))
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NoiseSegmentError("{} must be a positive integer".format(path))
    return int(value)


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NoiseSegmentError("{} must be a non-negative integer".format(path))
    return int(value)


def _exact_fields(payload: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(payload, Mapping):
        raise NoiseSegmentError("{} must be a mapping".format(path))
    unknown = sorted(set(payload) - set(expected))
    missing = sorted(set(expected) - set(payload))
    if unknown or missing:
        raise NoiseSegmentError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _seconds_to_samples(value: float, sample_rate_hz: int, path: str) -> int:
    """Convert seconds using the versioned nearest-integer half-up rule."""

    scaled = value * sample_rate_hz
    result = int(math.floor(scaled + 0.5))
    if result < 0:
        raise NoiseSegmentError("{} converts to a negative sample count".format(path))
    return result


@dataclass(frozen=True)
class NoiseSegmentPlannerParameters:
    """Versioned per-plan values; not permanent Infrastructure constants."""

    sample_rate_hz: int
    pre_roll_sec: float
    post_roll_sec: float
    guard_interval_sec: float = 0.0
    schema_version: str = NOISE_SEGMENT_PLAN_SCHEMA_VERSION
    algorithm_identity: str = NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY
    episode_order_identity: str = EPISODE_ORDER_IDENTITY
    sample_rounding_identity: str = SAMPLE_ROUNDING_IDENTITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_rate_hz", _positive_int(self.sample_rate_hz, "sample_rate_hz"))
        for field in ("pre_roll_sec", "post_roll_sec", "guard_interval_sec"):
            object.__setattr__(self, field, _nonnegative(getattr(self, field), field))
        object.__setattr__(self, "schema_version", _string(self.schema_version, "schema_version"))
        object.__setattr__(self, "algorithm_identity", _string(self.algorithm_identity, "algorithm_identity"))
        object.__setattr__(self, "episode_order_identity", _string(self.episode_order_identity, "episode_order_identity"))
        object.__setattr__(self, "sample_rounding_identity", _string(self.sample_rounding_identity, "sample_rounding_identity"))
        if self.schema_version != NOISE_SEGMENT_PLAN_SCHEMA_VERSION:
            raise NoiseSegmentError("schema_version is not the A4-2A noise plan schema")
        if self.algorithm_identity != NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY:
            raise NoiseSegmentError("algorithm_identity is not the A4-2A planner")
        if self.episode_order_identity != EPISODE_ORDER_IDENTITY:
            raise NoiseSegmentError("episode_order_identity is not the fixed A4 slot order")
        if self.sample_rounding_identity != SAMPLE_ROUNDING_IDENTITY:
            raise NoiseSegmentError("sample_rounding_identity is not the versioned half-up rule")

    def to_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "sample_rate_hz": self.sample_rate_hz,
            "pre_roll_sec": self.pre_roll_sec,
            "post_roll_sec": self.post_roll_sec,
            "guard_interval_sec": self.guard_interval_sec,
            "episode_order_identity": self.episode_order_identity,
            "sample_rounding_identity": self.sample_rounding_identity,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoiseSegmentPlannerParameters":
        expected = ("schema_version", "algorithm_identity", "sample_rate_hz", "pre_roll_sec", "post_roll_sec", "guard_interval_sec", "episode_order_identity", "sample_rounding_identity")
        _exact_fields(payload, expected, "noise_segment_planner_parameters")
        return cls(**dict(payload))

    @property
    def identity(self) -> str:
        return stable_id("noise-planner-contract", self.to_payload())


@dataclass(frozen=True)
class NoiseParentMetadata:
    """Identity and decoded/resampled extent of one fixed noise parent."""

    noise_parent_id: str
    decoded_resampled_payload_sha256: str
    sample_rate_hz: int
    sample_count: int
    schema_version: str = NOISE_PARENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "noise_parent_id", _string(self.noise_parent_id, "noise_parent_id"))
        try:
            validate_stable_id(self.noise_parent_id, "noise-parent", "noise_parent_id")
            validate_sha256(self.decoded_resampled_payload_sha256, "decoded_resampled_payload_sha256")
        except ValueError as exc:
            raise NoiseSegmentError(str(exc)) from exc
        object.__setattr__(self, "sample_rate_hz", _positive_int(self.sample_rate_hz, "sample_rate_hz"))
        object.__setattr__(self, "sample_count", _nonnegative_int(self.sample_count, "sample_count"))
        object.__setattr__(self, "schema_version", _string(self.schema_version, "schema_version"))
        if self.schema_version != NOISE_PARENT_SCHEMA_VERSION:
            raise NoiseSegmentError("schema_version is not the A4-2A noise parent schema")

    def to_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "noise_parent_id": self.noise_parent_id,
            "decoded_resampled_payload_sha256": self.decoded_resampled_payload_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoiseParentMetadata":
        expected = ("schema_version", "noise_parent_id", "decoded_resampled_payload_sha256", "sample_rate_hz", "sample_count")
        _exact_fields(payload, expected, "noise_parent")
        return cls(**dict(payload))

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.to_payload())


def noise_parent_from_a3_provenance(
    parent_recording_id: str,
    original_payload_sha256: str,
    decoded_payload_sha256: str,
    resampled_payload_sha256: str,
    sample_rate_hz: int,
    sample_count: int,
) -> NoiseParentMetadata:
    """Adapt a frozen A3 registry row without treating its path as an A4 ID.

    This synthetic/provenance-only adapter is intentionally separate from A3
    registry selection.  The resulting A4 parent ID binds the A3 recording
    token and all decoded/resampled provenance, while the A4 record stores the
    resampled payload hash as its payload authority.
    """

    parent_recording_id = _string(parent_recording_id, "parent_recording_id")
    for value, path in (
        (original_payload_sha256, "original_payload_sha256"),
        (decoded_payload_sha256, "decoded_payload_sha256"),
        (resampled_payload_sha256, "resampled_payload_sha256"),
    ):
        try:
            validate_sha256(value, path)
        except ValueError as exc:
            raise NoiseSegmentError(str(exc)) from exc
    sample_rate_hz = _positive_int(sample_rate_hz, "sample_rate_hz")
    sample_count = _nonnegative_int(sample_count, "sample_count")
    identity_payload = {
        "adapter_algorithm_identity": A3_NOISE_PARENT_ADAPTER_IDENTITY,
        "parent_recording_id": parent_recording_id,
        "original_payload_sha256": original_payload_sha256,
        "decoded_payload_sha256": decoded_payload_sha256,
        "resampled_payload_sha256": resampled_payload_sha256,
        "sample_rate_hz": sample_rate_hz,
        "sample_count": sample_count,
    }
    return NoiseParentMetadata(
        noise_parent_id=stable_id("noise-parent", identity_payload),
        decoded_resampled_payload_sha256=resampled_payload_sha256,
        sample_rate_hz=sample_rate_hz,
        sample_count=sample_count,
    )


@dataclass(frozen=True)
class EpisodeNoiseRequest:
    """One fixed block episode slot, independent of final EpisodeRecord IDs."""

    role: str
    role_index: int
    utterance_identity: str
    target_sample_count: int

    def __post_init__(self) -> None:
        for field in ("utterance_identity",):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        object.__setattr__(self, "role", _string(self.role, "role"))
        if self.role not in ("selection", "evaluation"):
            raise NoiseSegmentError("role must be selection or evaluation")
        object.__setattr__(self, "role_index", _nonnegative_int(self.role_index, "role_index"))
        if self.role_index not in (0, 1):
            raise NoiseSegmentError("role_index must be 0 or 1")
        object.__setattr__(self, "target_sample_count", _positive_int(self.target_sample_count, "target_sample_count"))

    def to_payload(self) -> dict:
        return {
            "role": self.role,
            "role_index": self.role_index,
            "utterance_identity": self.utterance_identity,
            "target_sample_count": self.target_sample_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeNoiseRequest":
        expected = ("role", "role_index", "utterance_identity", "target_sample_count")
        _exact_fields(payload, expected, "episode_noise_request")
        return cls(**dict(payload))

    @property
    def slot(self) -> Tuple[str, int]:
        return self.role, self.role_index


@dataclass(frozen=True)
class NoiseSegmentRecord:
    """One fixed half-open parent range and its source-time interpretation."""

    noise_parent_id: str
    decoded_resampled_payload_sha256: str
    sample_rate_hz: int
    role: str
    role_index: int
    utterance_identity: str
    segment_index: int
    target_sample_count: int
    source_time_start_sec: float
    source_time_end_sec: float
    pre_roll_sec: float
    post_roll_sec: float
    guard_interval_sec: float
    guard_after_samples: int
    start_sample: int
    end_sample: int
    planner_contract_identity: str
    planner_algorithm_identity: str = NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY
    schema_version: str = NOISE_SEGMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            validate_stable_id(self.noise_parent_id, "noise-parent", "noise_parent_id")
            validate_sha256(self.decoded_resampled_payload_sha256, "decoded_resampled_payload_sha256")
        except ValueError as exc:
            raise NoiseSegmentError(str(exc)) from exc
        object.__setattr__(self, "sample_rate_hz", _positive_int(self.sample_rate_hz, "sample_rate_hz"))
        for field in ("utterance_identity", "planner_contract_identity", "planner_algorithm_identity", "schema_version"):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        object.__setattr__(self, "role", _string(self.role, "role"))
        if self.role not in ("selection", "evaluation"):
            raise NoiseSegmentError("role must be selection or evaluation")
        object.__setattr__(self, "role_index", _nonnegative_int(self.role_index, "role_index"))
        if (self.role, self.role_index) not in EPISODE_SLOT_ORDER:
            raise NoiseSegmentError("role/role_index is not a fixed A4 episode slot")
        object.__setattr__(self, "segment_index", _nonnegative_int(self.segment_index, "segment_index"))
        if self.segment_index != EPISODE_SLOT_ORDER.index((self.role, self.role_index)):
            raise NoiseSegmentError("segment_index does not match fixed role slot")
        object.__setattr__(self, "target_sample_count", _positive_int(self.target_sample_count, "target_sample_count"))
        for field in ("pre_roll_sec", "post_roll_sec", "guard_interval_sec"):
            object.__setattr__(self, field, _nonnegative(getattr(self, field), field))
        object.__setattr__(self, "source_time_start_sec", _finite(self.source_time_start_sec, "source_time_start_sec"))
        object.__setattr__(self, "source_time_end_sec", _finite(self.source_time_end_sec, "source_time_end_sec"))
        if not self.source_time_end_sec > self.source_time_start_sec:
            raise NoiseSegmentError("source-time interval must have positive duration")
        pre_roll_samples = _seconds_to_samples(self.pre_roll_sec, self.sample_rate_hz, "pre_roll_sec")
        post_roll_samples = _seconds_to_samples(self.post_roll_sec, self.sample_rate_hz, "post_roll_sec")
        expected_source_start = -pre_roll_samples / float(self.sample_rate_hz)
        expected_source_end = (self.target_sample_count + post_roll_samples) / float(self.sample_rate_hz)
        if not math.isclose(self.source_time_start_sec, expected_source_start, rel_tol=0.0, abs_tol=1.0e-12):
            raise NoiseSegmentError("source_time_start_sec must equal negative realized pre-roll samples")
        if not math.isclose(self.source_time_end_sec, expected_source_end, rel_tol=0.0, abs_tol=1.0e-12):
            raise NoiseSegmentError("source_time_end_sec must equal target samples plus realized post-roll samples")
        object.__setattr__(self, "guard_after_samples", _nonnegative_int(self.guard_after_samples, "guard_after_samples"))
        object.__setattr__(self, "start_sample", _nonnegative_int(self.start_sample, "start_sample"))
        object.__setattr__(self, "end_sample", _nonnegative_int(self.end_sample, "end_sample"))
        if self.end_sample <= self.start_sample:
            raise NoiseSegmentError("noise segment sample range must be non-empty")
        expected_length = (
            _seconds_to_samples(self.pre_roll_sec, self.sample_rate_hz, "pre_roll_sec")
            + self.target_sample_count
            + _seconds_to_samples(self.post_roll_sec, self.sample_rate_hz, "post_roll_sec")
        )
        if self.end_sample - self.start_sample != expected_length:
            raise NoiseSegmentError("sample range does not match pre/target/post source-time duration")
        expected_guard = _seconds_to_samples(self.guard_interval_sec, self.sample_rate_hz, "guard_interval_sec")
        if self.guard_after_samples not in (0, expected_guard):
            raise NoiseSegmentError("guard_after_samples does not match the planner contract")
        if self.planner_algorithm_identity != NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY:
            raise NoiseSegmentError("planner_algorithm_identity is not the A4-2A planner")
        try:
            validate_stable_id(self.planner_contract_identity, "noise-planner-contract", "planner_contract_identity")
        except ValueError as exc:
            raise NoiseSegmentError(str(exc)) from exc
        if self.schema_version != NOISE_SEGMENT_SCHEMA_VERSION:
            raise NoiseSegmentError("schema_version is not the A4-2A segment schema")

    @property
    def target_duration_sec(self) -> float:
        return self.target_sample_count / float(self.sample_rate_hz)

    def identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "planner_algorithm_identity": self.planner_algorithm_identity,
            "planner_contract_identity": self.planner_contract_identity,
            "noise_parent_id": self.noise_parent_id,
            "decoded_resampled_payload_sha256": self.decoded_resampled_payload_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "role": self.role,
            "role_index": self.role_index,
            "utterance_identity": self.utterance_identity,
            "segment_index": self.segment_index,
            "target_sample_count": self.target_sample_count,
            "source_time_start_sec": self.source_time_start_sec,
            "source_time_end_sec": self.source_time_end_sec,
            "pre_roll_sec": self.pre_roll_sec,
            "post_roll_sec": self.post_roll_sec,
            "guard_interval_sec": self.guard_interval_sec,
            "guard_after_samples": self.guard_after_samples,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
        }

    @property
    def segment_id(self) -> str:
        return stable_id("noise-segment", self.identity_payload())

    def to_payload(self) -> dict:
        payload = dict(self.identity_payload())
        payload["target_duration_sec"] = self.target_duration_sec
        payload["source_time_start_sec"] = self.source_time_start_sec
        payload["source_time_end_sec"] = self.source_time_end_sec
        payload["segment_id"] = self.segment_id
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoiseSegmentRecord":
        expected = tuple(list(NoiseSegmentRecord.__dataclass_fields__) + ["target_duration_sec", "segment_id"])
        _exact_fields(payload, expected, "noise_segment")
        values = dict(payload)
        target_duration_sec = values.pop("target_duration_sec")
        actual = values.pop("segment_id")
        record = cls(**values)
        target_duration_sec = _finite(target_duration_sec, "noise_segment.target_duration_sec")
        if not math.isclose(target_duration_sec, record.target_duration_sec, rel_tol=0.0, abs_tol=1.0e-12):
            raise NoiseSegmentError("noise_segment.target_duration_sec is not derived from target_sample_count")
        if actual != record.segment_id:
            raise NoiseSegmentError("noise_segment.segment_id does not match semantic payload")
        return record


@dataclass(frozen=True)
class NoiseSegmentPlan:
    """A complete four-episode allocation from one immutable parent."""

    noise_parent_id: str
    decoded_resampled_payload_sha256: str
    sample_rate_hz: int
    planner_contract_identity: str
    required_parent_sample_count: int
    segments: Tuple[NoiseSegmentRecord, ...]
    planner_algorithm_identity: str = NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY
    schema_version: str = NOISE_SEGMENT_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            validate_stable_id(self.noise_parent_id, "noise-parent", "noise_parent_id")
            validate_sha256(self.decoded_resampled_payload_sha256, "decoded_resampled_payload_sha256")
        except ValueError as exc:
            raise NoiseSegmentError(str(exc)) from exc
        object.__setattr__(self, "sample_rate_hz", _positive_int(self.sample_rate_hz, "sample_rate_hz"))
        object.__setattr__(self, "planner_contract_identity", _string(self.planner_contract_identity, "planner_contract_identity"))
        object.__setattr__(self, "required_parent_sample_count", _nonnegative_int(self.required_parent_sample_count, "required_parent_sample_count"))
        object.__setattr__(self, "segments", tuple(self.segments))
        if len(self.segments) != 4:
            raise NoiseSegmentError("A4-2A noise plans require exactly four episodes")
        if any(not isinstance(segment, NoiseSegmentRecord) for segment in self.segments):
            raise NoiseSegmentError("noise plan segments must be NoiseSegmentRecord values")
        if self.planner_algorithm_identity != NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY:
            raise NoiseSegmentError("planner_algorithm_identity is not the A4-2A planner")
        try:
            validate_stable_id(self.planner_contract_identity, "noise-planner-contract", "planner_contract_identity")
        except ValueError as exc:
            raise NoiseSegmentError(str(exc)) from exc
        if self.schema_version != NOISE_SEGMENT_PLAN_SCHEMA_VERSION:
            raise NoiseSegmentError("schema_version is not the A4-2A plan schema")
        ranges = [(segment.start_sample, segment.end_sample) for segment in self.segments]
        if any(end <= start for start, end in ranges):
            raise NoiseSegmentError("noise segment ranges must be non-empty")
        if ranges[0][0] != 0:
            raise NoiseSegmentError("noise segment allocation must begin at parent sample zero")
        if any(first[1] > second[0] for first, second in zip(ranges, ranges[1:])):
            raise NoiseSegmentError("noise segment ranges overlap")
        if ranges[-1][1] != self.required_parent_sample_count:
            raise NoiseSegmentError("required parent sample count must end at the final segment boundary")
        if tuple((segment.role, segment.role_index) for segment in self.segments) != EPISODE_SLOT_ORDER:
            raise NoiseSegmentError("segments must use selection[0], selection[1], evaluation[0], evaluation[1]")
        if tuple(segment.segment_index for segment in self.segments) != tuple(range(4)):
            raise NoiseSegmentError("segment indices must be the deterministic range 0..3")
        if self.segments[-1].guard_after_samples != 0:
            raise NoiseSegmentError("final segment must not reserve a guard interval")
        for previous, current in zip(self.segments, self.segments[1:]):
            if current.start_sample != previous.end_sample + previous.guard_after_samples:
                raise NoiseSegmentError("segment ranges do not follow exact guard accounting")
            expected_guard = _seconds_to_samples(
                previous.guard_interval_sec, self.sample_rate_hz, "guard_interval_sec"
            )
            if previous.guard_after_samples != expected_guard:
                raise NoiseSegmentError("intermediate segment guard does not match contract")
        if any(
            segment.noise_parent_id != self.noise_parent_id
            or segment.decoded_resampled_payload_sha256 != self.decoded_resampled_payload_sha256
            or segment.sample_rate_hz != self.sample_rate_hz
            or segment.planner_contract_identity != self.planner_contract_identity
            for segment in self.segments
        ):
            raise NoiseSegmentError("all segments must bind the same parent and planner contract")

    def identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "planner_algorithm_identity": self.planner_algorithm_identity,
            "planner_contract_identity": self.planner_contract_identity,
            "noise_parent_id": self.noise_parent_id,
            "decoded_resampled_payload_sha256": self.decoded_resampled_payload_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "required_parent_sample_count": self.required_parent_sample_count,
            "segments": [segment.identity_payload() for segment in self.segments],
        }

    @property
    def plan_id(self) -> str:
        return stable_id("noise-segment-plan", self.identity_payload())

    def to_payload(self) -> dict:
        payload = dict(self.identity_payload())
        payload["segments"] = [segment.to_payload() for segment in self.segments]
        payload["plan_id"] = self.plan_id
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoiseSegmentPlan":
        expected = tuple(list(NoiseSegmentPlan.__dataclass_fields__) + ["plan_id"])
        _exact_fields(payload, expected, "noise_segment_plan")
        values = dict(payload)
        actual = values.pop("plan_id")
        if not isinstance(values["segments"], (list, tuple)):
            raise NoiseSegmentError("noise_segment_plan.segments must be a sequence")
        values["segments"] = tuple(NoiseSegmentRecord.from_payload(item) for item in values["segments"])
        plan = cls(**values)
        if actual != plan.plan_id:
            raise NoiseSegmentError("noise_segment_plan.plan_id does not match semantic payload")
        return plan


def _ordered_episodes(episodes: Sequence[EpisodeNoiseRequest]) -> Tuple[EpisodeNoiseRequest, ...]:
    if not isinstance(episodes, (list, tuple)) or len(episodes) != 4:
        raise NoiseSegmentError("A4-2A noise planning requires exactly four episodes")
    requests = tuple(episodes)
    if any(not isinstance(item, EpisodeNoiseRequest) for item in requests):
        raise NoiseSegmentError("episodes must contain EpisodeNoiseRequest values")
    if {item.slot for item in requests} != set(EPISODE_SLOT_ORDER):
        raise NoiseSegmentError("the four episodes must exactly fill the fixed A4 episode slots")
    return tuple(sorted(requests, key=lambda item: EPISODE_SLOT_ORDER.index(item.slot)))


def plan_noise_segments(
    noise_parent: NoiseParentMetadata,
    episodes: Sequence[EpisodeNoiseRequest],
    planner_contract: NoiseSegmentPlannerParameters,
) -> NoiseSegmentPlan:
    """Allocate four non-overlapping source-time ranges from one parent."""

    if not isinstance(noise_parent, NoiseParentMetadata):
        raise NoiseSegmentError("noise_parent must be NoiseParentMetadata")
    if not isinstance(planner_contract, NoiseSegmentPlannerParameters):
        raise NoiseSegmentError("planner_contract must be NoiseSegmentPlannerParameters")
    if noise_parent.sample_rate_hz != planner_contract.sample_rate_hz:
        raise NoiseSegmentError("noise parent and planner sample rates differ")
    ordered = _ordered_episodes(episodes)
    sample_rate = planner_contract.sample_rate_hz
    pre_samples = _seconds_to_samples(planner_contract.pre_roll_sec, sample_rate, "pre_roll_sec")
    post_samples = _seconds_to_samples(planner_contract.post_roll_sec, sample_rate, "post_roll_sec")
    guard_samples = _seconds_to_samples(planner_contract.guard_interval_sec, sample_rate, "guard_interval_sec")
    lengths = []
    for item in ordered:
        target_samples = item.target_sample_count
        lengths.append((item, target_samples, pre_samples + target_samples + post_samples))
    required = sum(length for _, _, length in lengths) + guard_samples * (len(lengths) - 1)
    if noise_parent.sample_count < required:
        raise NoiseSegmentError(
            "INSUFFICIENT_NOISE_PARENT_LENGTH: need {}, have {}".format(required, noise_parent.sample_count)
        )
    segments = []
    cursor = 0
    for index, (item, target_samples, length) in enumerate(lengths):
        start = cursor
        end = start + length
        guard_after = guard_samples if index < len(lengths) - 1 else 0
        segment = NoiseSegmentRecord(
            noise_parent_id=noise_parent.noise_parent_id,
            decoded_resampled_payload_sha256=noise_parent.decoded_resampled_payload_sha256,
            sample_rate_hz=sample_rate,
            role=item.role,
            role_index=item.role_index,
            utterance_identity=item.utterance_identity,
            segment_index=index,
            target_sample_count=target_samples,
            source_time_start_sec=-pre_samples / float(sample_rate),
            source_time_end_sec=(target_samples + post_samples) / float(sample_rate),
            pre_roll_sec=planner_contract.pre_roll_sec,
            post_roll_sec=planner_contract.post_roll_sec,
            guard_interval_sec=planner_contract.guard_interval_sec,
            guard_after_samples=guard_after,
            start_sample=start,
            end_sample=end,
            planner_contract_identity=planner_contract.identity,
        )
        segments.append(segment)
        cursor = end + guard_after
    return NoiseSegmentPlan(
        noise_parent_id=noise_parent.noise_parent_id,
        decoded_resampled_payload_sha256=noise_parent.decoded_resampled_payload_sha256,
        sample_rate_hz=sample_rate,
        planner_contract_identity=planner_contract.identity,
        required_parent_sample_count=required,
        segments=tuple(segments),
    )


__all__ = [
    "A3_NOISE_PARENT_ADAPTER_IDENTITY",
    "EPISODE_ORDER_IDENTITY",
    "EPISODE_SLOT_ORDER",
    "EpisodeNoiseRequest",
    "NOISE_PARENT_SCHEMA_VERSION",
    "NOISE_SEGMENT_PLAN_SCHEMA_VERSION",
    "NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY",
    "NOISE_SEGMENT_SCHEMA_VERSION",
    "NoiseParentMetadata",
    "NoiseSegmentError",
    "NoiseSegmentPlan",
    "NoiseSegmentPlannerParameters",
    "NoiseSegmentRecord",
    "SAMPLE_INDEX_CONVENTION_IDENTITY",
    "SAMPLE_ROUNDING_IDENTITY",
    "SOURCE_TIME_CONVENTION_IDENTITY",
    "noise_parent_from_a3_provenance",
    "plan_noise_segments",
]
