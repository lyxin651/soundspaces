"""Pure common receiver-time alignment for A4-2B1.

This module places independently propagated target and noise components on one
integer-sample receiver timeline.  It does not mix, normalize, render RIRs,
decode ASR, or inspect any recognition result.
"""

import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from active_audition.a4.identity import stable_id, validate_stable_id


TIMELINE_SCHEMA_VERSION = "active-asr-a4-receiver-timeline-v1"
TIMELINE_CONTRACT_SCHEMA_VERSION = "active-asr-a4-receiver-timeline-contract-v1"
TIMELINE_ALGORITHM_IDENTITY = "active-asr-a4-common-receiver-timeline-v1"
TIMELINE_MASK_SCHEMA_VERSION = "active-asr-a4-receiver-time-mask-v1"
TIMELINE_MASK_ALGORITHM_IDENTITY = "active-asr-a4-dry-mask-to-receiver-time-v1"
SOURCE_TIME_CONVENTION_IDENTITY = "active-asr-a4-target-dry-onset-zero-v1"
A2_DIRECT_ONSET_CONVENTION_IDENTITY = (
    "active-asr-a2-direct-window-first-absolute-sample-10-percent-peak-v1"
)
CHANNEL_ORDER = ("L", "R")


class TimelineError(ValueError):
    """Raised when a common receiver timeline cannot be constructed."""


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TimelineError("{} must be a non-empty string".format(path))
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TimelineError("{} must be a positive integer".format(path))
    return int(value)


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TimelineError("{} must be a non-negative integer".format(path))
    return int(value)


def _int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimelineError("{} must be an integer".format(path))
    return int(value)


def _exact_fields(payload: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(payload, Mapping):
        raise TimelineError("{} must be a mapping".format(path))
    unknown = sorted(set(payload) - set(expected))
    missing = sorted(set(expected) - set(payload))
    if unknown or missing:
        raise TimelineError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _source_waveform(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise TimelineError("{} must be a one-dimensional numpy array".format(path))
    if value.dtype != np.dtype(np.float32):
        raise TimelineError("{} must use float32 samples".format(path))
    if value.size <= 0 or not np.isfinite(value).all():
        raise TimelineError("{} must be non-empty and finite".format(path))
    return np.ascontiguousarray(value)


def _rir(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != 2:
        raise TimelineError("{} must have shape (N, 2)".format(path))
    if value.shape[0] <= 0 or not np.issubdtype(value.dtype, np.floating):
        raise TimelineError("{} must be a non-empty floating-point array".format(path))
    if not np.isfinite(value).all():
        raise TimelineError("{} must be finite".format(path))
    return np.ascontiguousarray(value, dtype=np.float32)


def _stable_component_id(namespace: str, array: np.ndarray, label: str) -> str:
    return stable_id(
        namespace,
        {"label": label, "dtype": str(array.dtype), "shape": list(array.shape), "sha256": _array_sha256(array)},
    )


@dataclass(frozen=True)
class TimelineContract:
    """Versioned timeline parameters and the inherited A2 onset convention."""

    sample_rate_hz: int = 16000
    channel_order: Tuple[str, str] = CHANNEL_ORDER
    schema_version: str = TIMELINE_CONTRACT_SCHEMA_VERSION
    algorithm_identity: str = TIMELINE_ALGORITHM_IDENTITY
    source_time_convention_identity: str = SOURCE_TIME_CONVENTION_IDENTITY
    direct_onset_convention_identity: str = A2_DIRECT_ONSET_CONVENTION_IDENTITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_rate_hz", _positive_int(self.sample_rate_hz, "sample_rate_hz"))
        object.__setattr__(self, "channel_order", tuple(self.channel_order))
        if self.sample_rate_hz != 16000:
            raise TimelineError("A4-2B1 native receiver timeline requires 16000 Hz")
        if self.channel_order != CHANNEL_ORDER:
            raise TimelineError("channel_order must be [L, R]")
        for field in (
            "schema_version",
            "algorithm_identity",
            "source_time_convention_identity",
            "direct_onset_convention_identity",
        ):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        if self.schema_version != TIMELINE_CONTRACT_SCHEMA_VERSION:
            raise TimelineError("schema_version is not the A4-2B1 timeline contract")
        if self.algorithm_identity != TIMELINE_ALGORITHM_IDENTITY:
            raise TimelineError("algorithm_identity is not the A4 common timeline")
        if self.source_time_convention_identity != SOURCE_TIME_CONVENTION_IDENTITY:
            raise TimelineError("source_time_convention_identity is not target-onset-zero")
        if self.direct_onset_convention_identity != A2_DIRECT_ONSET_CONVENTION_IDENTITY:
            raise TimelineError("direct_onset_convention_identity is not the frozen A2 convention")

    def to_payload(self) -> dict:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "channel_order": list(self.channel_order),
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "source_time_convention_identity": self.source_time_convention_identity,
            "direct_onset_convention_identity": self.direct_onset_convention_identity,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TimelineContract":
        expected = (
            "sample_rate_hz",
            "channel_order",
            "schema_version",
            "algorithm_identity",
            "source_time_convention_identity",
            "direct_onset_convention_identity",
        )
        _exact_fields(payload, expected, "timeline_contract")
        return cls(**dict(payload))

    @property
    def identity(self) -> str:
        return stable_id("receiver-timeline-contract", self.to_payload())


def a2_direct_onset_samples(rir: np.ndarray, sample_rate_hz: int = 16000) -> Mapping[str, int]:
    """Return A2-equivalent per-channel first absolute-sample onsets.

    This is the frozen A2 direct-window rule: search [0, 50 ms), use the first
    absolute sample at or above 10% of that channel's peak, and reject low
    energy/onset-less impulses.  No channel is shifted to the other channel.
    """

    rate = _positive_int(sample_rate_hz, "sample_rate_hz")
    array = _rir(rir, "rir")
    search_count = int(round(0.050 * rate))
    search = np.abs(array[: min(array.shape[0], search_count), :])
    if search.shape[0] <= 0:
        raise TimelineError("DIRECT_SEARCH_WINDOW_EMPTY")
    peaks = np.max(search, axis=0)
    if float(np.sum(np.square(search))) <= 1.0e-12 or float(np.max(peaks)) < 1.0e-8:
        raise TimelineError("LOW_ENERGY")
    onsets = {}
    for channel_index, channel in enumerate(CHANNEL_ORDER):
        threshold = max(float(peaks[channel_index]) * 0.10, 1.0e-8)
        candidates = np.flatnonzero(search[:, channel_index] >= threshold)
        if candidates.size == 0:
            raise TimelineError("NO_DIRECT_ONSET_{}".format(channel))
        onsets[channel] = int(candidates[0])
    return onsets


@dataclass(frozen=True)
class CommonReceiverTimeline:
    """Two independently propagated components placed on one receiver axis."""

    contract_identity: str
    target_source_segment_identity: str
    noise_source_segment_identity: str
    target_source_start_sample: int
    target_source_end_sample_exclusive: int
    noise_source_start_sample: int
    noise_source_end_sample_exclusive: int
    receiver_start_sample: int
    receiver_end_sample_exclusive: int
    target_receiver_offset_samples: int
    noise_receiver_offset_samples: int
    target_propagation_length_samples: int
    noise_propagation_length_samples: int
    target_rir_identity: str
    noise_rir_identity: str
    target_direct_onset_samples: Mapping[str, int]
    noise_direct_onset_samples: Mapping[str, int]
    target_binaural: np.ndarray
    noise_binaural: np.ndarray
    algorithm_identity: str = TIMELINE_ALGORITHM_IDENTITY
    schema_version: str = TIMELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            validate_stable_id(self.contract_identity, "receiver-timeline-contract", "contract_identity")
            validate_stable_id(self.target_rir_identity, "rir", "target_rir_identity")
            validate_stable_id(self.noise_rir_identity, "rir", "noise_rir_identity")
        except ValueError as exc:
            raise TimelineError(str(exc)) from exc
        for field in (
            "target_source_segment_identity",
            "noise_source_segment_identity",
            "algorithm_identity",
            "schema_version",
        ):
            _string(getattr(self, field), field)
        if self.algorithm_identity != TIMELINE_ALGORITHM_IDENTITY:
            raise TimelineError("algorithm_identity is not the A4 common timeline")
        if self.schema_version != TIMELINE_SCHEMA_VERSION:
            raise TimelineError("schema_version is not the A4 receiver timeline schema")
        for field in (
            "target_source_start_sample",
            "target_source_end_sample_exclusive",
            "noise_source_start_sample",
            "noise_source_end_sample_exclusive",
            "receiver_start_sample",
            "receiver_end_sample_exclusive",
            "target_receiver_offset_samples",
            "noise_receiver_offset_samples",
            "target_propagation_length_samples",
            "noise_propagation_length_samples",
        ):
            if field in (
                "target_source_start_sample",
                "noise_source_start_sample",
                "receiver_start_sample",
            ):
                _int(getattr(self, field), field)
            else:
                _nonnegative_int(getattr(self, field), field)
        if self.target_source_end_sample_exclusive <= self.target_source_start_sample:
            raise TimelineError("target source support must be non-empty")
        if self.noise_source_end_sample_exclusive <= self.noise_source_start_sample:
            raise TimelineError("noise source support must be non-empty")
        if self.receiver_end_sample_exclusive <= self.receiver_start_sample:
            raise TimelineError("receiver support must be non-empty")
        receiver_length = self.receiver_end_sample_exclusive - self.receiver_start_sample
        if self.target_receiver_offset_samples != self.target_source_start_sample - self.receiver_start_sample:
            raise TimelineError("target receiver offset is inconsistent")
        if self.noise_receiver_offset_samples != self.noise_source_start_sample - self.receiver_start_sample:
            raise TimelineError("noise receiver offset is inconsistent")
        if self.target_propagation_length_samples <= 0 or self.noise_propagation_length_samples <= 0:
            raise TimelineError("propagation supports must be non-empty")
        if self.target_receiver_offset_samples + self.target_propagation_length_samples > receiver_length:
            raise TimelineError("target propagation exceeds receiver support")
        if self.noise_receiver_offset_samples + self.noise_propagation_length_samples > receiver_length:
            raise TimelineError("noise propagation exceeds receiver support")
        if not isinstance(self.target_binaural, np.ndarray) or not isinstance(self.noise_binaural, np.ndarray):
            raise TimelineError("timeline components must be numpy arrays")
        for name, array in (("target_binaural", self.target_binaural), ("noise_binaural", self.noise_binaural)):
            if array.shape != (self.receiver_end_sample_exclusive - self.receiver_start_sample, 2):
                raise TimelineError("{} does not match common receiver shape".format(name))
            if array.dtype != np.dtype(np.float32) or not np.isfinite(array).all():
                raise TimelineError("{} must be finite float32".format(name))
            readonly = np.ascontiguousarray(array.copy())
            readonly.setflags(write=False)
            object.__setattr__(self, name, readonly)
        for name, onsets in (
            ("target_direct_onset_samples", self.target_direct_onset_samples),
            ("noise_direct_onset_samples", self.noise_direct_onset_samples),
        ):
            if dict(onsets).keys() != set(CHANNEL_ORDER):
                raise TimelineError("{} must contain L and R".format(name))
            for channel in CHANNEL_ORDER:
                _nonnegative_int(onsets[channel], "{}.{}".format(name, channel))
            object.__setattr__(self, name, MappingProxyType(dict(onsets)))

    def identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "contract_identity": self.contract_identity,
            "target_source_segment_identity": self.target_source_segment_identity,
            "noise_source_segment_identity": self.noise_source_segment_identity,
            "target_source_start_sample": self.target_source_start_sample,
            "target_source_end_sample_exclusive": self.target_source_end_sample_exclusive,
            "noise_source_start_sample": self.noise_source_start_sample,
            "noise_source_end_sample_exclusive": self.noise_source_end_sample_exclusive,
            "receiver_start_sample": self.receiver_start_sample,
            "receiver_end_sample_exclusive": self.receiver_end_sample_exclusive,
            "target_receiver_offset_samples": self.target_receiver_offset_samples,
            "noise_receiver_offset_samples": self.noise_receiver_offset_samples,
            "target_propagation_length_samples": self.target_propagation_length_samples,
            "noise_propagation_length_samples": self.noise_propagation_length_samples,
            "target_rir_identity": self.target_rir_identity,
            "noise_rir_identity": self.noise_rir_identity,
            "target_direct_onset_samples": dict(self.target_direct_onset_samples),
            "noise_direct_onset_samples": dict(self.noise_direct_onset_samples),
            "receiver_shape": list(self.target_binaural.shape),
        }

    @property
    def timeline_id(self) -> str:
        return stable_id("receiver-timeline", self.identity_payload())

    def to_payload(self) -> dict:
        payload = dict(self.identity_payload())
        payload["timeline_id"] = self.timeline_id
        return payload


@dataclass(frozen=True)
class ReceiverTimeMask:
    """A dry target mask expanded on the common receiver-time axis."""

    timeline_identity: str
    dry_mask_identity: str
    target_direct_onset_samples: Mapping[str, int]
    receiver_start_sample: int
    receiver_end_sample_exclusive: int
    mask: Tuple[bool, ...]
    active_sample_count: int
    active_start_sample: int
    active_end_sample_exclusive: int
    algorithm_identity: str = TIMELINE_MASK_ALGORITHM_IDENTITY
    schema_version: str = TIMELINE_MASK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            validate_stable_id(self.timeline_identity, "receiver-timeline", "timeline_identity")
            validate_stable_id(self.dry_mask_identity, "active-mask", "dry_mask_identity")
        except ValueError as exc:
            raise TimelineError(str(exc)) from exc
        if self.algorithm_identity != TIMELINE_MASK_ALGORITHM_IDENTITY:
            raise TimelineError("algorithm_identity is not the A4 receiver-mask mapping")
        if self.schema_version != TIMELINE_MASK_SCHEMA_VERSION:
            raise TimelineError("schema_version is not the A4 receiver-mask schema")
        start = _int(self.receiver_start_sample, "receiver_start_sample")
        end = _int(self.receiver_end_sample_exclusive, "receiver_end_sample_exclusive")
        mask = tuple(self.mask)
        object.__setattr__(self, "mask", mask)
        if end <= start or len(mask) != end - start:
            raise TimelineError("receiver mask shape does not match timeline support")
        if any(not isinstance(value, bool) for value in mask):
            raise TimelineError("receiver mask must contain booleans")
        count = sum(mask)
        if count <= 0 or count != self.active_sample_count:
            raise TimelineError("receiver active sample count is inconsistent")
        active_indices = [index for index, value in enumerate(mask) if value]
        expected_start = start + active_indices[0]
        expected_end = start + active_indices[-1] + 1
        if self.active_start_sample != expected_start or self.active_end_sample_exclusive != expected_end:
            raise TimelineError("receiver mask active bounds are inconsistent")
        if dict(self.target_direct_onset_samples).keys() != set(CHANNEL_ORDER):
            raise TimelineError("target_direct_onset_samples must contain L and R")
        for channel in CHANNEL_ORDER:
            _nonnegative_int(self.target_direct_onset_samples[channel], "target_direct_onset_samples." + channel)
        object.__setattr__(self, "target_direct_onset_samples", MappingProxyType(dict(self.target_direct_onset_samples)))

    def identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "timeline_identity": self.timeline_identity,
            "dry_mask_identity": self.dry_mask_identity,
            "target_direct_onset_samples": dict(self.target_direct_onset_samples),
            "receiver_start_sample": self.receiver_start_sample,
            "receiver_end_sample_exclusive": self.receiver_end_sample_exclusive,
            "mask": list(self.mask),
            "active_sample_count": self.active_sample_count,
            "active_start_sample": self.active_start_sample,
            "active_end_sample_exclusive": self.active_end_sample_exclusive,
        }

    @property
    def mask_id(self) -> str:
        return stable_id("receiver-mask", self.identity_payload())

    def to_payload(self) -> dict:
        payload = dict(self.identity_payload())
        payload["mask_id"] = self.mask_id
        return payload


def _full_convolve(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.convolve(source.astype(np.float32), rir[:, index].astype(np.float32), mode="full") for index in range(2)]
    ).astype(np.float32, copy=False)


def build_common_receiver_timeline(
    target_source: np.ndarray,
    noise_source: np.ndarray,
    target_rir: np.ndarray,
    noise_rir: np.ndarray,
    target_source_segment_identity: str,
    noise_source_segment_identity: str,
    noise_source_start_sample: int,
    timeline_contract: TimelineContract,
) -> CommonReceiverTimeline:
    """Convolve target/noise independently and place both on one sample axis."""

    if not isinstance(timeline_contract, TimelineContract):
        raise TimelineError("timeline_contract must be TimelineContract")
    target = _source_waveform(target_source, "target_source")
    noise = _source_waveform(noise_source, "noise_source")
    target_ir = _rir(target_rir, "target_rir")
    noise_ir = _rir(noise_rir, "noise_rir")
    target_start = 0
    noise_start = _int(noise_source_start_sample, "noise_source_start_sample")
    if noise_start > 0:
        raise TimelineError("noise_source_start_sample must be at or before target onset zero")
    target_end = target_start + int(target.size)
    noise_end = noise_start + int(noise.size)
    target_propagated = _full_convolve(target, target_ir)
    noise_propagated = _full_convolve(noise, noise_ir)
    receiver_start = min(target_start, noise_start)
    receiver_end = max(
        target_start + target_propagated.shape[0],
        noise_start + noise_propagated.shape[0],
    )
    receiver_shape = (receiver_end - receiver_start, 2)
    target_output = np.zeros(receiver_shape, dtype=np.float32)
    noise_output = np.zeros(receiver_shape, dtype=np.float32)
    target_offset = target_start - receiver_start
    noise_offset = noise_start - receiver_start
    target_output[target_offset : target_offset + target_propagated.shape[0], :] = target_propagated
    noise_output[noise_offset : noise_offset + noise_propagated.shape[0], :] = noise_propagated
    target_onsets = a2_direct_onset_samples(target_ir, timeline_contract.sample_rate_hz)
    noise_onsets = a2_direct_onset_samples(noise_ir, timeline_contract.sample_rate_hz)
    return CommonReceiverTimeline(
        contract_identity=timeline_contract.identity,
        target_source_segment_identity=_string(target_source_segment_identity, "target_source_segment_identity"),
        noise_source_segment_identity=_string(noise_source_segment_identity, "noise_source_segment_identity"),
        target_source_start_sample=target_start,
        target_source_end_sample_exclusive=target_end,
        noise_source_start_sample=noise_start,
        noise_source_end_sample_exclusive=noise_end,
        receiver_start_sample=receiver_start,
        receiver_end_sample_exclusive=receiver_end,
        target_receiver_offset_samples=target_offset,
        noise_receiver_offset_samples=noise_offset,
        target_propagation_length_samples=target_propagated.shape[0],
        noise_propagation_length_samples=noise_propagated.shape[0],
        target_rir_identity=_stable_component_id("rir", target_ir, "target"),
        noise_rir_identity=_stable_component_id("rir", noise_ir, "noise"),
        target_direct_onset_samples=target_onsets,
        noise_direct_onset_samples=noise_onsets,
        target_binaural=target_output,
        noise_binaural=noise_output,
    )


def map_dry_mask_to_receiver_time(
    dry_mask: Any,
    timeline: CommonReceiverTimeline,
) -> ReceiverTimeMask:
    """Map one dry target mask using the common target direct-onset sample."""

    if not hasattr(dry_mask, "mask") or not hasattr(dry_mask, "mask_id"):
        raise TimelineError("dry_mask must be an ActiveMaskRecord")
    if int(dry_mask.sample_rate_hz) != 16000 or int(dry_mask.sample_count) != timeline.target_source_end_sample_exclusive:
        raise TimelineError("dry mask does not match target source support")
    if len(tuple(dry_mask.mask)) != int(dry_mask.sample_count):
        raise TimelineError("dry mask payload length does not match sample_count")
    target_delay = min(int(timeline.target_direct_onset_samples[channel]) for channel in CHANNEL_ORDER)
    base_global = timeline.target_source_start_sample + target_delay
    output_mask = [False] * (timeline.receiver_end_sample_exclusive - timeline.receiver_start_sample)
    for dry_index, active in enumerate(tuple(dry_mask.mask)):
        if not isinstance(active, bool):
            raise TimelineError("dry mask must contain booleans")
        if not active:
            continue
        global_sample = base_global + dry_index
        output_index = global_sample - timeline.receiver_start_sample
        if output_index < 0 or output_index >= len(output_mask):
            raise TimelineError("receiver mask active sample is outside common timeline")
        output_mask[output_index] = True
    if not any(output_mask):
        raise TimelineError("receiver mask has zero active samples")
    active_indices = [index for index, active in enumerate(output_mask) if active]
    return ReceiverTimeMask(
        timeline_identity=timeline.timeline_id,
        dry_mask_identity=dry_mask.mask_id,
        target_direct_onset_samples=dict(timeline.target_direct_onset_samples),
        receiver_start_sample=timeline.receiver_start_sample,
        receiver_end_sample_exclusive=timeline.receiver_end_sample_exclusive,
        mask=tuple(output_mask),
        active_sample_count=len(active_indices),
        active_start_sample=timeline.receiver_start_sample + active_indices[0],
        active_end_sample_exclusive=timeline.receiver_start_sample + active_indices[-1] + 1,
    )


__all__ = [
    "A2_DIRECT_ONSET_CONVENTION_IDENTITY",
    "CHANNEL_ORDER",
    "CommonReceiverTimeline",
    "ReceiverTimeMask",
    "SOURCE_TIME_CONVENTION_IDENTITY",
    "TIMELINE_ALGORITHM_IDENTITY",
    "TIMELINE_CONTRACT_SCHEMA_VERSION",
    "TIMELINE_MASK_ALGORITHM_IDENTITY",
    "TIMELINE_MASK_SCHEMA_VERSION",
    "TIMELINE_SCHEMA_VERSION",
    "TimelineContract",
    "TimelineError",
    "a2_direct_onset_samples",
    "build_common_receiver_timeline",
    "map_dry_mask_to_receiver_time",
]
