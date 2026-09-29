"""Pure dry-speech activity masks for later calibration energy estimation.

The result is an auditable boolean sample mask.  This module never normalizes
or modifies the waveform and has no RIR, ASR, WER, or speech-model input.
"""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.identity import stable_id, validate_stable_id


ACTIVE_MASK_CONTRACT_SCHEMA_VERSION = "active-asr-a4-active-mask-contract-v1"
ACTIVE_MASK_SCHEMA_VERSION = "active-asr-a4-active-mask-v1"
ACTIVE_MASK_ALGORITHM_IDENTITY = "active-asr-a4-dry-speech-active-mask-v1"
ACTIVE_MASK_INPUT_IDENTITY = "dry_target_speech_only_v1"
ACTIVE_MASK_FRAME_RMS_IDENTITY = "frame_rms_no_padding_v1"
ACTIVE_MASK_PERCENTILE_IDENTITY = "linear_interpolated_finite_percentile_v1"
ACTIVE_MASK_THRESHOLD_IDENTITY = "reference_plus_db_offset_v1"
ACTIVE_MASK_EXPANSION_IDENTITY = "frame_to_sample_half_open_union_v1"


class ActiveMaskError(ValueError):
    """Raised when dry speech cannot produce a usable deterministic mask."""


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActiveMaskError("{} must be a non-empty string".format(path))
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActiveMaskError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise ActiveMaskError("{} must be finite".format(path))
    return result


def _exact_fields(payload: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(payload, Mapping):
        raise ActiveMaskError("{} must be a mapping".format(path))
    unknown = sorted(set(payload) - set(expected))
    missing = sorted(set(expected) - set(payload))
    if unknown or missing:
        raise ActiveMaskError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _milliseconds_to_samples(value: float, sample_rate_hz: int, path: str) -> int:
    scaled = value * sample_rate_hz / 1000.0
    rounded = int(round(scaled))
    if rounded <= 0 or not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ActiveMaskError("{} does not convert to an exact positive sample count".format(path))
    return rounded


@dataclass(frozen=True)
class ActiveMaskContract:
    """Versioned mask parameters; values can be rebound by later config."""

    frame_duration_ms: float = 25.0
    hop_duration_ms: float = 10.0
    reference_percentile: float = 95.0
    threshold_db: float = -40.0
    schema_version: str = ACTIVE_MASK_CONTRACT_SCHEMA_VERSION
    algorithm_identity: str = ACTIVE_MASK_ALGORITHM_IDENTITY

    def __post_init__(self) -> None:
        for field in ("frame_duration_ms", "hop_duration_ms", "reference_percentile", "threshold_db"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if self.frame_duration_ms <= 0.0 or self.hop_duration_ms <= 0.0:
            raise ActiveMaskError("frame and hop durations must be positive")
        if not 0.0 <= self.reference_percentile <= 100.0:
            raise ActiveMaskError("reference_percentile must be in [0, 100]")
        if self.threshold_db >= 0.0:
            raise ActiveMaskError("threshold_db must be negative")
        object.__setattr__(self, "schema_version", _string(self.schema_version, "schema_version"))
        object.__setattr__(self, "algorithm_identity", _string(self.algorithm_identity, "algorithm_identity"))
        if self.schema_version != ACTIVE_MASK_CONTRACT_SCHEMA_VERSION:
            raise ActiveMaskError("schema_version is not the A4-2A active-mask contract")
        if self.algorithm_identity != ACTIVE_MASK_ALGORITHM_IDENTITY:
            raise ActiveMaskError("algorithm_identity is not the A4-2A active-mask algorithm")

    def to_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "frame_duration_ms": self.frame_duration_ms,
            "hop_duration_ms": self.hop_duration_ms,
            "reference_percentile": self.reference_percentile,
            "threshold_db": self.threshold_db,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ActiveMaskContract":
        expected = (
            "schema_version", "algorithm_identity", "frame_duration_ms", "hop_duration_ms",
            "reference_percentile", "threshold_db",
        )
        _exact_fields(payload, expected, "active_mask_contract")
        return cls(**dict(payload))

    @property
    def identity(self) -> str:
        return stable_id("active-mask-contract", self.to_payload())


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ActiveMaskError("no finite frame RMS values")
    rank = percentile / 100.0 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True)
class ActiveMaskRecord:
    """Serialized mask metrics and the deterministic sample-level mask."""

    sample_rate_hz: int
    sample_count: int
    frame_samples: int
    hop_samples: int
    frame_count: int
    reference_percentile: float
    reference_rms: float
    threshold_db: float
    threshold_rms: float
    active_frame_indices: Tuple[int, ...]
    mask: Tuple[bool, ...]
    contract_identity: str
    algorithm_identity: str = ACTIVE_MASK_ALGORITHM_IDENTITY
    schema_version: str = ACTIVE_MASK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate_hz, bool) or not isinstance(self.sample_rate_hz, int) or self.sample_rate_hz <= 0:
            raise ActiveMaskError("sample_rate_hz must be a positive integer")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ActiveMaskError("sample_count must be a positive integer")
        for field in ("frame_samples", "hop_samples", "frame_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ActiveMaskError("{} must be a positive integer".format(field))
        if self.sample_count < self.frame_samples:
            raise ActiveMaskError("sample_count has no complete no-padding analysis frame")
        expected_frame_count = ((self.sample_count - self.frame_samples) // self.hop_samples) + 1
        if self.frame_count != expected_frame_count:
            raise ActiveMaskError("frame_count does not match no-padding frame definition")
        for field in ("reference_percentile", "reference_rms", "threshold_db", "threshold_rms"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if not 0.0 <= self.reference_percentile <= 100.0:
            raise ActiveMaskError("reference_percentile must be in [0, 100]")
        if self.threshold_db >= 0.0:
            raise ActiveMaskError("threshold_db must be negative")
        expected_threshold = self.reference_rms * (10.0 ** (self.threshold_db / 20.0))
        if not math.isclose(self.threshold_rms, expected_threshold, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ActiveMaskError("threshold_rms does not match reference_rms and threshold_db")
        object.__setattr__(self, "active_frame_indices", tuple(self.active_frame_indices))
        object.__setattr__(self, "mask", tuple(self.mask))
        if len(self.mask) != self.sample_count or any(not isinstance(item, bool) for item in self.mask):
            raise ActiveMaskError("mask must be a boolean sequence matching sample_count")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= self.frame_count
            for index in self.active_frame_indices
        ):
            raise ActiveMaskError("active_frame_indices are invalid")
        if tuple(sorted(set(self.active_frame_indices))) != self.active_frame_indices:
            raise ActiveMaskError("active_frame_indices must be sorted and unique")
        expected_mask = [False] * self.sample_count
        for index in self.active_frame_indices:
            start = index * self.hop_samples
            for sample_index in range(start, min(start + self.frame_samples, self.sample_count)):
                expected_mask[sample_index] = True
        if tuple(expected_mask) != self.mask:
            raise ActiveMaskError("mask is not the exact active-frame half-open union")
        if self.active_sample_count != sum(self.mask):
            raise ActiveMaskError("active_sample_count does not match mask")
        if self.active_sample_count <= 0:
            raise ActiveMaskError("active mask must contain nonzero samples")
        object.__setattr__(self, "contract_identity", _string(self.contract_identity, "contract_identity"))
        try:
            validate_stable_id(self.contract_identity, "active-mask-contract", "contract_identity")
        except ValueError as exc:
            raise ActiveMaskError(str(exc)) from exc
        object.__setattr__(self, "algorithm_identity", _string(self.algorithm_identity, "algorithm_identity"))
        object.__setattr__(self, "schema_version", _string(self.schema_version, "schema_version"))
        if self.algorithm_identity != ACTIVE_MASK_ALGORITHM_IDENTITY:
            raise ActiveMaskError("algorithm_identity is not the A4-2A active-mask algorithm")
        if self.schema_version != ACTIVE_MASK_SCHEMA_VERSION:
            raise ActiveMaskError("schema_version is not the A4-2A active-mask schema")

    @property
    def active_sample_count(self) -> int:
        return sum(self.mask)

    def identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "contract_identity": self.contract_identity,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "frame_samples": self.frame_samples,
            "hop_samples": self.hop_samples,
            "frame_count": self.frame_count,
            "reference_percentile": self.reference_percentile,
            "reference_rms": self.reference_rms,
            "threshold_db": self.threshold_db,
            "threshold_rms": self.threshold_rms,
            "active_frame_indices": list(self.active_frame_indices),
            "mask": list(self.mask),
        }

    @property
    def mask_id(self) -> str:
        return stable_id("active-mask", self.identity_payload())

    def to_payload(self) -> dict:
        payload = dict(self.identity_payload())
        payload["active_sample_count"] = self.active_sample_count
        payload["mask_id"] = self.mask_id
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ActiveMaskRecord":
        expected = tuple(list(ActiveMaskRecord.__dataclass_fields__) + ["active_sample_count", "mask_id"])
        _exact_fields(payload, expected, "active_mask")
        values = dict(payload)
        actual_count = values.pop("active_sample_count")
        actual_id = values.pop("mask_id")
        record = cls(**values)
        if isinstance(actual_count, bool) or not isinstance(actual_count, int) or actual_count != record.active_sample_count:
            raise ActiveMaskError("active_sample_count does not match mask")
        if actual_id != record.mask_id:
            raise ActiveMaskError("mask_id does not match semantic payload")
        return record


def build_active_mask(
    dry_speech: Sequence[float],
    sr: int,
    mask_contract: ActiveMaskContract,
) -> ActiveMaskRecord:
    """Build a deterministic dry-speech mask without changing ``dry_speech``."""

    if isinstance(sr, bool) or not isinstance(sr, int) or sr <= 0:
        raise ActiveMaskError("sr must be a positive integer")
    if not isinstance(mask_contract, ActiveMaskContract):
        raise ActiveMaskError("mask_contract must be ActiveMaskContract")
    try:
        samples = tuple(_finite(value, "dry_speech[{}]".format(index)) for index, value in enumerate(dry_speech))
    except TypeError as exc:
        raise ActiveMaskError("dry_speech must be a finite numeric sequence") from exc
    if not samples:
        raise ActiveMaskError("SILENT_OR_UNUSABLE_SPEECH: dry_speech is empty")
    frame_samples = _milliseconds_to_samples(mask_contract.frame_duration_ms, sr, "frame_duration_ms")
    hop_samples = _milliseconds_to_samples(mask_contract.hop_duration_ms, sr, "hop_duration_ms")
    if len(samples) < frame_samples:
        raise ActiveMaskError("SILENT_OR_UNUSABLE_SPEECH: speech has no complete analysis frame")
    starts = tuple(range(0, len(samples) - frame_samples + 1, hop_samples))
    rms_values = tuple(
        math.sqrt(sum(value * value for value in samples[start:start + frame_samples]) / frame_samples)
        for start in starts
    )
    if not rms_values or any(not math.isfinite(value) for value in rms_values):
        raise ActiveMaskError("SILENT_OR_UNUSABLE_SPEECH: frame RMS is unusable")
    reference = _percentile(rms_values, mask_contract.reference_percentile)
    if reference <= 0.0:
        raise ActiveMaskError("SILENT_OR_UNUSABLE_SPEECH: reference RMS is zero")
    threshold = reference * (10.0 ** (mask_contract.threshold_db / 20.0))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ActiveMaskError("SILENT_OR_UNUSABLE_SPEECH: threshold RMS is unusable")
    active_frames = tuple(index for index, value in enumerate(rms_values) if value >= threshold)
    mask = [False] * len(samples)
    for index in active_frames:
        start = starts[index]
        for sample_index in range(start, min(start + frame_samples, len(samples))):
            mask[sample_index] = True
    if not any(mask):
        raise ActiveMaskError("SILENT_OR_UNUSABLE_SPEECH: no active samples")
    return ActiveMaskRecord(
        sample_rate_hz=sr,
        sample_count=len(samples),
        frame_samples=frame_samples,
        hop_samples=hop_samples,
        frame_count=len(starts),
        reference_percentile=mask_contract.reference_percentile,
        reference_rms=reference,
        threshold_db=mask_contract.threshold_db,
        threshold_rms=threshold,
        active_frame_indices=active_frames,
        mask=tuple(mask),
        contract_identity=mask_contract.identity,
    )


__all__ = [
    "ACTIVE_MASK_ALGORITHM_IDENTITY",
    "ACTIVE_MASK_CONTRACT_SCHEMA_VERSION",
    "ACTIVE_MASK_EXPANSION_IDENTITY",
    "ACTIVE_MASK_FRAME_RMS_IDENTITY",
    "ACTIVE_MASK_INPUT_IDENTITY",
    "ACTIVE_MASK_PERCENTILE_IDENTITY",
    "ACTIVE_MASK_SCHEMA_VERSION",
    "ACTIVE_MASK_THRESHOLD_IDENTITY",
    "ActiveMaskContract",
    "ActiveMaskError",
    "ActiveMaskRecord",
    "build_active_mask",
]
