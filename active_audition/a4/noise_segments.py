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
NOISE_SEGMENT_SCHEMA_VERSION = "active-asr-a4-noise-segment-v1"
NOISE_SEGMENT_PLAN_SCHEMA_VERSION = "active-asr-a4-noise-segment-plan-v1"
NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY = "active-asr-a4-noise-segment-planner-v1"
SOURCE_TIME_CONVENTION_IDENTITY = "active-asr-a4-target-dry-onset-zero-v1"
SAMPLE_INDEX_CONVENTION_IDENTITY = "active-asr-a4-half-open-sample-range-v1"
EPISODE_ORDER_IDENTITY = "active-asr-a4-stable-episode-order-v1"


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_rate_hz", _positive_int(self.sample_rate_hz, "sample_rate_hz"))
        for field in ("pre_roll_sec", "post_roll_sec", "guard_interval_sec"):
            object.__setattr__(self, field, _nonnegative(getattr(self, field), field))
        object.__setattr__(self, "schema_version", _string(self.schema_version, "schema_version"))
        object.__setattr__(self, "algorithm_identity", _string(self.algorithm_identity, "algorithm_identity"))
        if self.schema_version != NOISE_SEGMENT_PLAN_SCHEMA_VERSION:
            raise NoiseSegmentError("schema_version is not the A4-2A noise plan schema")
        if self.algorithm_identity != NOISE_SEGMENT_PLANNER_ALGORITHM_IDENTITY:
            raise NoiseSegmentError("algorithm_identity is not the A4-2A planner")

    def to_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "sample_rate_hz": self.sample_rate_hz,
            "pre_roll_sec": self.pre_roll_sec,
            "post_roll_sec": self.post_roll_sec,
            "guard_interval_sec": self.guard_interval_sec,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoiseSegmentPlannerParameters":
        expected = ("schema_version", "algorithm_identity", "sample_rate_hz", "pre_roll_sec", "post_roll_sec", "guard_interval_sec")
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


@dataclass(frozen=True)
class EpisodeNoiseRequest:
    """Stable episode/utterance binding used by the planner."""

    episode_id: str
    utterance_identity: str
    role: str
    stable_order_key: str
    target_duration_sec: float

    def __post_init__(self) -> None:
        for field in ("episode_id", "utterance_identity", "stable_order_key"):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        object.__setattr__(self, "role", _string(self.role, "role"))
        if self.role not in ("selection", "evaluation"):
            raise NoiseSegmentError("role must be selection or evaluation")
        object.__setattr__(self, "target_duration_sec", _finite(self.target_duration_sec, "target_duration_sec"))
        if self.target_duration_sec <= 0.0:
            raise NoiseSegmentError("target_duration_sec must be positive")

    def to_payload(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "utterance_identity": self.utterance_identity,
            "role": self.role,
            "stable_order_key": self.stable_order_key,
            "target_duration_sec": self.target_duration_sec,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeNoiseRequest":
        expected = ("episode_id", "utterance_identity", "role", "stable_order_key", "target_duration_sec")
        _exact_fields(payload, expected, "episode_noise_request")
        return cls(**dict(payload))


@dataclass(frozen=True)
class NoiseSegmentRecord:
    """One fixed half-open parent range and its source-time interpretation."""

    noise_parent_id: str
    decoded_resampled_payload_sha256: str
    sample_rate_hz: int
    episode_id: str
    utterance_identity: str
    role: str
    segment_index: int
    target_duration_sec: float
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
        for field in ("episode_id", "utterance_identity", "planner_contract_identity", "planner_algorithm_identity", "schema_version"):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        object.__setattr__(self, "role", _string(self.role, "role"))
        if self.role not in ("selection", "evaluation"):
            raise NoiseSegmentError("role must be selection or evaluation")
        object.__setattr__(self, "segment_index", _nonnegative_int(self.segment_index, "segment_index"))
        for field in ("target_duration_sec", "pre_roll_sec", "post_roll_sec", "guard_interval_sec"):
            object.__setattr__(self, field, _nonnegative(getattr(self, field), field))
        if self.target_duration_sec <= 0.0:
            raise NoiseSegmentError("target_duration_sec must be positive")
        object.__setattr__(self, "source_time_start_sec", _finite(self.source_time_start_sec, "source_time_start_sec"))
        object.__setattr__(self, "source_time_end_sec", _finite(self.source_time_end_sec, "source_time_end_sec"))
        if not self.source_time_end_sec > self.source_time_start_sec:
            raise NoiseSegmentError("source-time interval must have positive duration")
        if not math.isclose(self.source_time_start_sec, -self.pre_roll_sec, rel_tol=0.0, abs_tol=1.0e-12):
            raise NoiseSegmentError("source_time_start_sec must equal negative pre_roll_sec")
        if not math.isclose(self.source_time_end_sec, self.target_duration_sec + self.post_roll_sec, rel_tol=0.0, abs_tol=1.0e-12):
            raise NoiseSegmentError("source_time_end_sec must equal target duration plus post_roll_sec")
        object.__setattr__(self, "guard_after_samples", _nonnegative_int(self.guard_after_samples, "guard_after_samples"))
        object.__setattr__(self, "start_sample", _nonnegative_int(self.start_sample, "start_sample"))
        object.__setattr__(self, "end_sample", _nonnegative_int(self.end_sample, "end_sample"))
        if self.end_sample <= self.start_sample:
            raise NoiseSegmentError("noise segment sample range must be non-empty")
        expected_length = _seconds_to_samples(
            self.pre_roll_sec + self.target_duration_sec + self.post_roll_sec,
            self.sample_rate_hz,
            "noise segment duration",
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

    def identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "planner_algorithm_identity": self.planner_algorithm_identity,
            "planner_contract_identity": self.planner_contract_identity,
            "noise_parent_id": self.noise_parent_id,
            "decoded_resampled_payload_sha256": self.decoded_resampled_payload_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "episode_id": self.episode_id,
            "utterance_identity": self.utterance_identity,
            "role": self.role,
            "segment_index": self.segment_index,
            "target_duration_sec": self.target_duration_sec,
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
        payload["segment_id"] = self.segment_id
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoiseSegmentRecord":
        expected = tuple(list(NoiseSegmentRecord.__dataclass_fields__) + ["segment_id"])
        _exact_fields(payload, expected, "noise_segment")
        values = dict(payload)
        actual = values.pop("segment_id")
        record = cls(**values)
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
        if any(first[1] > second[0] for first, second in zip(ranges, ranges[1:])):
            raise NoiseSegmentError("noise segment ranges overlap")
        if ranges[-1][1] != self.required_parent_sample_count:
            raise NoiseSegmentError("required parent sample count must end at the final segment boundary")
        if tuple(segment.segment_index for segment in self.segments) != tuple(range(4)):
            raise NoiseSegmentError("segment indices must be the deterministic range 0..3")
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
    if len({item.episode_id for item in requests}) != len(requests):
        raise NoiseSegmentError("episode IDs must be unique")
    if len({item.stable_order_key for item in requests}) != len(requests):
        raise NoiseSegmentError("stable episode order keys must be unique")
    if sum(item.role == "selection" for item in requests) != 2 or sum(item.role == "evaluation" for item in requests) != 2:
        raise NoiseSegmentError("the four episodes must contain two selection and two evaluation episodes")
    return tuple(sorted(requests, key=lambda item: (0 if item.role == "selection" else 1, item.stable_order_key, item.episode_id)))


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
        target_samples = _seconds_to_samples(item.target_duration_sec, sample_rate, "target_duration_sec")
        if target_samples <= 0:
            raise NoiseSegmentError("target_duration_sec is too short to produce one sample")
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
            episode_id=item.episode_id,
            utterance_identity=item.utterance_identity,
            role=item.role,
            segment_index=index,
            target_duration_sec=item.target_duration_sec,
            source_time_start_sec=-planner_contract.pre_roll_sec,
            source_time_end_sec=item.target_duration_sec + planner_contract.post_roll_sec,
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
    "EPISODE_ORDER_IDENTITY",
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
    "SOURCE_TIME_CONVENTION_IDENTITY",
    "plan_noise_segments",
]
