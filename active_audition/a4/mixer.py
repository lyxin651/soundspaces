"""Pure A4-2B2 dual-source reconstruction and integrity binding.

The mixer consumes independently propagated target and noise components that
already occupy one :class:`CommonReceiverTimeline`.  It does not render RIRs,
select noise parents, normalize waveforms, or perform calibration.  The
selection-only :class:`CalibrationArtifact` is the sole authority for alpha.
"""

import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from active_audition.a4.identity import stable_id, validate_sha256, validate_stable_id
from active_audition.a4.noise_segments import NoiseSegmentError, NoiseSegmentRecord
from active_audition.a4.records import (
    BlockRecord,
    CalibrationArtifact,
    EpisodeRecord,
    validate_calibration_artifact,
)
from active_audition.a4.timeline import CHANNEL_ORDER, CommonReceiverTimeline


MIXTURE_SCHEMA_VERSION = "active-asr-a4-mixture-v1"
MIXTURE_CONTRACT_SCHEMA_VERSION = "active-asr-a4-mixture-contract-v1"
MIXER_ALGORITHM_IDENTITY = "active-asr-a4-dual-source-linear-mixer-v1"
RECONSTRUCTION_IDENTITY = "active-asr-a4-linear-reconstruction-v1"
ARITHMETIC_IDENTITY = "active-asr-a4-float32-single-cast-arithmetic-v1"
GLOBAL_GAIN_SCHEMA_VERSION = "active-asr-a4-global-gain-v1"
GLOBAL_GAIN_ALGORITHM_IDENTITY = "active-asr-a4-block-global-gain-v1"
RESIDUAL_GATE_THRESHOLD = 1.0e-6
SAMPLE_RATE_HZ = 16000
CHANNEL_ORDER_LIST = ("L", "R")

_DIAGNOSTIC_FIELDS = (
    "target_peak_lr",
    "noise_peak_lr",
    "mixture_peak_lr",
    "peak_over_unit",
    "peak_over_unit_max_abs",
    "target_left_power",
    "target_right_power",
    "noise_left_power",
    "noise_right_power",
    "mixture_left_power",
    "mixture_right_power",
    "target_two_ear_power",
    "noise_two_ear_power",
    "mixture_two_ear_power",
    "target_mean_lr_power",
    "noise_mean_lr_power",
    "mixture_mean_lr_power",
    "diagnostic_snr_db_two_ear",
    "diagnostic_snr_db_mean_lr",
)


class MixerError(ValueError):
    """Raised when mixer inputs, bindings, or reconstruction integrity fail."""


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise MixerError("{} must be a non-empty string".format(path))
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MixerError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise MixerError("{} must be finite".format(path))
    return result


def _positive(value: Any, path: str) -> float:
    result = _finite(value, path)
    if result <= 0.0:
        raise MixerError("{} must be positive".format(path))
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MixerError("{} must be a positive integer".format(path))
    return int(value)


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MixerError("{} must be a non-negative integer".format(path))
    return int(value)


def _exact_fields(payload: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(payload, Mapping):
        raise MixerError("{} must be a mapping".format(path))
    unknown = sorted(set(payload) - set(expected))
    missing = sorted(set(expected) - set(payload))
    if unknown or missing:
        raise MixerError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _copy_readonly(array: np.ndarray) -> np.ndarray:
    copied = np.ascontiguousarray(array.copy())
    copied.setflags(write=False)
    return copied


def _component_namespace(role: str) -> str:
    if role == "target":
        return "target-component"
    if role == "noise":
        return "noise-component"
    raise MixerError("component role must be target or noise")


def _validate_stable(value: Any, namespace: str, path: str) -> None:
    try:
        validate_stable_id(value, namespace, path)
    except ValueError as exc:
        raise MixerError(str(exc)) from exc


@dataclass(frozen=True)
class GlobalGainSpec:
    """Block-level gain binding; the numeric value is not an A4 constant."""

    gain: float
    schema_version: str = GLOBAL_GAIN_SCHEMA_VERSION
    algorithm_identity: str = GLOBAL_GAIN_ALGORITHM_IDENTITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain", _positive(self.gain, "global_gain.gain"))
        if self.schema_version != GLOBAL_GAIN_SCHEMA_VERSION:
            raise MixerError("global_gain.schema_version is invalid")
        if self.algorithm_identity != GLOBAL_GAIN_ALGORITHM_IDENTITY:
            raise MixerError("global_gain.algorithm_identity is invalid")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "gain": self.gain,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GlobalGainSpec":
        _exact_fields(payload, ("schema_version", "algorithm_identity", "gain"), "global_gain")
        return cls(**dict(payload))

    @property
    def identity(self) -> str:
        return stable_id("global-gain", self.to_payload())


@dataclass(frozen=True)
class MixtureContract:
    """Versioned arithmetic and output policy for the pure mixer."""

    sample_rate_hz: int = SAMPLE_RATE_HZ
    channel_order: Tuple[str, str] = CHANNEL_ORDER_LIST
    schema_version: str = MIXTURE_CONTRACT_SCHEMA_VERSION
    mixer_algorithm_identity: str = MIXER_ALGORITHM_IDENTITY
    reconstruction_identity: str = RECONSTRUCTION_IDENTITY
    arithmetic_identity: str = ARITHMETIC_IDENTITY
    residual_gate_threshold: float = RESIDUAL_GATE_THRESHOLD
    dtype_policy: str = "native16_float32_finite_v1"
    channel_policy: str = "canonical_binaural_lr_v1"
    normalization_policy: str = "no_limiter_no_source_pose_channel_normalization_v1"

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate_hz, bool) or not isinstance(self.sample_rate_hz, int):
            raise MixerError("mixture_contract.sample_rate_hz must be an integer")
        if self.sample_rate_hz != SAMPLE_RATE_HZ:
            raise MixerError("mixture_contract requires native16")
        if tuple(self.channel_order) != CHANNEL_ORDER_LIST:
            raise MixerError("mixture_contract channel order must be [L, R]")
        if self.schema_version != MIXTURE_CONTRACT_SCHEMA_VERSION:
            raise MixerError("mixture_contract.schema_version is invalid")
        if self.mixer_algorithm_identity != MIXER_ALGORITHM_IDENTITY:
            raise MixerError("mixture_contract mixer algorithm is invalid")
        if self.reconstruction_identity != RECONSTRUCTION_IDENTITY:
            raise MixerError("mixture_contract reconstruction identity is invalid")
        if self.arithmetic_identity != ARITHMETIC_IDENTITY:
            raise MixerError("mixture_contract arithmetic identity is invalid")
        if _finite(self.residual_gate_threshold, "mixture_contract.residual_gate_threshold") != RESIDUAL_GATE_THRESHOLD:
            raise MixerError("mixture_contract residual gate is invalid")
        if self.dtype_policy != "native16_float32_finite_v1":
            raise MixerError("mixture_contract dtype policy is invalid")
        if self.channel_policy != "canonical_binaural_lr_v1":
            raise MixerError("mixture_contract channel policy is invalid")
        if self.normalization_policy != "no_limiter_no_source_pose_channel_normalization_v1":
            raise MixerError("mixture_contract normalization policy is invalid")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "channel_order": list(self.channel_order),
            "schema_version": self.schema_version,
            "mixer_algorithm_identity": self.mixer_algorithm_identity,
            "reconstruction_identity": self.reconstruction_identity,
            "arithmetic_identity": self.arithmetic_identity,
            "residual_gate_threshold": self.residual_gate_threshold,
            "dtype_policy": self.dtype_policy,
            "channel_policy": self.channel_policy,
            "normalization_policy": self.normalization_policy,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MixtureContract":
        _exact_fields(payload, tuple(cls.__dataclass_fields__), "mixture_contract")
        return cls(**dict(payload))

    @property
    def identity(self) -> str:
        return stable_id("mixture-contract", self.to_payload())


@dataclass(frozen=True)
class WaveformComponent:
    """A real float32 timeline component whose identity is hash-bound."""

    role: str
    timeline_identity: str
    payload: np.ndarray
    sample_rate_hz: int = SAMPLE_RATE_HZ
    channel_order: Tuple[str, str] = CHANNEL_ORDER_LIST
    payload_sha256: str = ""
    component_identity: str = ""

    def __post_init__(self) -> None:
        _component_namespace(self.role)
        _validate_stable(self.timeline_identity, "receiver-timeline", "component.timeline_identity")
        if self.sample_rate_hz != SAMPLE_RATE_HZ:
            raise MixerError("component.sample_rate_hz must be native16")
        if tuple(self.channel_order) != CHANNEL_ORDER_LIST:
            raise MixerError("component.channel_order must be [L, R]")
        if not isinstance(self.payload, np.ndarray) or self.payload.ndim != 2 or self.payload.shape[1] != 2:
            raise MixerError("component payload must have shape (N, 2)")
        if self.payload.shape[0] <= 0 or self.payload.dtype != np.dtype(np.float32):
            raise MixerError("component payload must be non-empty float32")
        if not np.isfinite(self.payload).all():
            raise MixerError("component payload must be finite")
        payload = _copy_readonly(self.payload)
        object.__setattr__(self, "payload", payload)
        payload_sha256 = _array_sha256(payload)
        if self.payload_sha256 and self.payload_sha256 != payload_sha256:
            raise MixerError("component payload_sha256 does not match waveform")
        object.__setattr__(self, "payload_sha256", payload_sha256)
        expected = stable_id("{}-component".format(self.role), self.identity_payload())
        if self.component_identity and self.component_identity != expected:
            raise MixerError("component_identity does not match waveform payload")
        object.__setattr__(self, "component_identity", expected)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "timeline_identity": self.timeline_identity,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_order": list(self.channel_order),
            "dtype": "float32",
            "shape": list(self.payload.shape),
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_array(cls, role: str, timeline_identity: str, payload: np.ndarray) -> "WaveformComponent":
        return cls(role=role, timeline_identity=timeline_identity, payload=payload)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "timeline_identity": self.timeline_identity,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_order": list(self.channel_order),
            "dtype": "float32",
            "shape": list(self.payload.shape),
            "payload_sha256": self.payload_sha256,
            "component_identity": self.component_identity,
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _channel_powers(array: np.ndarray) -> Tuple[float, float, float, float]:
    values = array.astype(np.float64, copy=False)
    left = float(np.mean(np.square(values[:, 0]), dtype=np.float64))
    right = float(np.mean(np.square(values[:, 1]), dtype=np.float64))
    two_ear = (left + right) / 2.0
    mean_lr = float(np.mean(np.square((values[:, 0] + values[:, 1]) / 2.0), dtype=np.float64))
    return left, right, two_ear, mean_lr


def _diagnostics(target: np.ndarray, noise: np.ndarray, mixture: np.ndarray, alpha: float) -> Dict[str, Any]:
    target_left, target_right, target_two, target_mean = _channel_powers(target)
    noise_left, noise_right, noise_two, noise_mean = _channel_powers(noise)
    mixture_left, mixture_right, mixture_two, mixture_mean = _channel_powers(mixture)
    target_peak = tuple(float(value) for value in np.max(np.abs(target.astype(np.float64)), axis=0))
    noise_peak = tuple(float(value) for value in np.max(np.abs(noise.astype(np.float64)), axis=0))
    mixture_peak = tuple(float(value) for value in np.max(np.abs(mixture.astype(np.float64)), axis=0))
    max_peak = max(mixture_peak)
    diagnostic_two = None if noise_two <= 0.0 else float(10.0 * math.log10(target_two / (alpha * alpha * noise_two)))
    diagnostic_mean = None if noise_mean <= 0.0 else float(10.0 * math.log10(target_mean / (alpha * alpha * noise_mean)))
    return {
        "target_peak_lr": target_peak,
        "noise_peak_lr": noise_peak,
        "mixture_peak_lr": mixture_peak,
        "peak_over_unit": bool(max_peak > 1.0),
        "peak_over_unit_max_abs": max_peak,
        "target_left_power": target_left,
        "target_right_power": target_right,
        "noise_left_power": noise_left,
        "noise_right_power": noise_right,
        "mixture_left_power": mixture_left,
        "mixture_right_power": mixture_right,
        "target_two_ear_power": target_two,
        "noise_two_ear_power": noise_two,
        "mixture_two_ear_power": mixture_two,
        "target_mean_lr_power": target_mean,
        "noise_mean_lr_power": noise_mean,
        "mixture_mean_lr_power": mixture_mean,
        "diagnostic_snr_db_two_ear": diagnostic_two,
        "diagnostic_snr_db_mean_lr": diagnostic_mean,
    }


def _validate_diagnostics(value: Mapping[str, Any], path: str = "mixture.diagnostics") -> Mapping[str, Any]:
    _exact_fields(value, _DIAGNOSTIC_FIELDS, path)
    for key in _DIAGNOSTIC_FIELDS:
        item = value[key]
        if key.endswith("_peak_lr"):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise MixerError("{}.{} must contain L and R".format(path, key))
            for index, number in enumerate(item):
                _finite(number, "{}.{}[{}]".format(path, key, index))
        elif key == "peak_over_unit":
            if not isinstance(item, bool):
                raise MixerError("{}.{} must be boolean".format(path, key))
        elif key.startswith("diagnostic_snr"):
            if item is not None:
                _finite(item, "{}.{}".format(path, key))
        else:
            _finite(item, "{}.{}".format(path, key))
    return value


def _artifact_identity_payload_from_values(values: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": values["schema_version"],
        "mixer_algorithm_identity": values["mixer_algorithm_identity"],
        "reconstruction_identity": values["reconstruction_identity"],
        "arithmetic_identity": values["arithmetic_identity"],
        "block_id": values["block_id"],
        "episode_id": values["episode_id"],
        "pose_id": values["pose_id"],
        "timeline_id": values["timeline_id"],
        "calibration_artifact_id": values["calibration_artifact_id"],
        "global_gain_identity": values["global_gain_identity"],
        "global_gain": values["global_gain"],
        "alpha": values["alpha"],
        "target_component_identity": values["target_component_identity"],
        "target_payload_sha256": values["target_payload_sha256"],
        "noise_component_identity": values["noise_component_identity"],
        "noise_payload_sha256": values["noise_payload_sha256"],
        "mixture_payload_sha256": values["mixture_payload_sha256"],
        "sample_rate_hz": values["sample_rate_hz"],
        "channel_order": list(values["channel_order"]),
        "shape": list(values["mixture_binaural"].shape),
        "dtype": values["dtype"],
        "target_receiver_offset_samples": values["target_receiver_offset_samples"],
        "noise_receiver_offset_samples": values["noise_receiver_offset_samples"],
        "receiver_start_sample": values["receiver_start_sample"],
        "receiver_end_sample_exclusive": values["receiver_end_sample_exclusive"],
        "diagnostics": _plain(values["diagnostics"]),
        "max_abs_residual": values["max_abs_residual"],
        "relative_residual": values["relative_residual"],
        "residual_gate_threshold": values["residual_gate_threshold"],
        "reconstruction_status": values["reconstruction_status"],
    }


def _artifact_identity_payload(record: "MixtureArtifact") -> Dict[str, Any]:
    return _artifact_identity_payload_from_values({
        "schema_version": record.schema_version,
        "mixer_algorithm_identity": record.mixer_algorithm_identity,
        "reconstruction_identity": record.reconstruction_identity,
        "arithmetic_identity": record.arithmetic_identity,
        "block_id": record.block_id,
        "episode_id": record.episode_id,
        "pose_id": record.pose_id,
        "timeline_id": record.timeline_id,
        "calibration_artifact_id": record.calibration_artifact_id,
        "global_gain_identity": record.global_gain_identity,
        "global_gain": record.global_gain,
        "alpha": record.alpha,
        "target_component_identity": record.target_component_identity,
        "target_payload_sha256": record.target_payload_sha256,
        "noise_component_identity": record.noise_component_identity,
        "noise_payload_sha256": record.noise_payload_sha256,
        "mixture_payload_sha256": record.mixture_payload_sha256,
        "sample_rate_hz": record.sample_rate_hz,
        "channel_order": record.channel_order,
        "dtype": record.dtype,
        "target_receiver_offset_samples": record.target_receiver_offset_samples,
        "noise_receiver_offset_samples": record.noise_receiver_offset_samples,
        "receiver_start_sample": record.receiver_start_sample,
        "receiver_end_sample_exclusive": record.receiver_end_sample_exclusive,
        "mixture_binaural": record.mixture_binaural,
        "diagnostics": record.diagnostics,
        "max_abs_residual": record.max_abs_residual,
        "relative_residual": record.relative_residual,
        "residual_gate_threshold": record.residual_gate_threshold,
        "reconstruction_status": record.reconstruction_status,
    })


@dataclass(frozen=True)
class MixtureArtifact:
    """Immutable output and provenance of one dual-source reconstruction."""

    schema_version: str
    mixture_artifact_id: str
    mixer_algorithm_identity: str
    reconstruction_identity: str
    arithmetic_identity: str
    block_id: str
    episode_id: str
    pose_id: str
    timeline_id: str
    calibration_artifact_id: str
    global_gain_identity: str
    global_gain: float
    alpha: float
    target_component_identity: str
    target_payload_sha256: str
    noise_component_identity: str
    noise_payload_sha256: str
    mixture_payload_sha256: str
    sample_rate_hz: int
    channel_order: Tuple[str, str]
    dtype: str
    target_receiver_offset_samples: int
    noise_receiver_offset_samples: int
    receiver_start_sample: int
    receiver_end_sample_exclusive: int
    mixture_binaural: np.ndarray
    diagnostics: Mapping[str, Any]
    max_abs_residual: float
    relative_residual: float
    residual_gate_threshold: float
    reconstruction_status: str

    def __post_init__(self) -> None:
        if self.schema_version != MIXTURE_SCHEMA_VERSION:
            raise MixerError("mixture.schema_version is invalid")
        for field, namespace in (
            ("mixture_artifact_id", "mixture"),
            ("block_id", "block"),
            ("episode_id", "episode"),
            ("pose_id", "pose"),
            ("timeline_id", "receiver-timeline"),
            ("calibration_artifact_id", "calibration"),
            ("global_gain_identity", "global-gain"),
            ("target_component_identity", "target-component"),
            ("noise_component_identity", "noise-component"),
        ):
            _validate_stable(getattr(self, field), namespace, "mixture." + field)
        if self.mixer_algorithm_identity != MIXER_ALGORITHM_IDENTITY:
            raise MixerError("mixture mixer algorithm is invalid")
        if self.reconstruction_identity != RECONSTRUCTION_IDENTITY:
            raise MixerError("mixture reconstruction identity is invalid")
        if self.arithmetic_identity != ARITHMETIC_IDENTITY:
            raise MixerError("mixture arithmetic identity is invalid")
        global_gain = _positive(self.global_gain, "mixture.global_gain")
        alpha = _positive(self.alpha, "mixture.alpha")
        if self.sample_rate_hz != SAMPLE_RATE_HZ or tuple(self.channel_order) != CHANNEL_ORDER_LIST:
            raise MixerError("mixture sample rate/channel order is invalid")
        if self.dtype != "float32":
            raise MixerError("mixture dtype must be float32")
        for field in (
            "target_receiver_offset_samples", "noise_receiver_offset_samples",
            "receiver_start_sample", "receiver_end_sample_exclusive",
        ):
            if field == "receiver_start_sample":
                if isinstance(getattr(self, field), bool) or not isinstance(getattr(self, field), int):
                    raise MixerError("mixture.receiver_start_sample must be an integer")
            else:
                _nonnegative_int(getattr(self, field), "mixture." + field)
        if self.receiver_end_sample_exclusive <= self.receiver_start_sample:
            raise MixerError("mixture receiver support must be non-empty")
        if not isinstance(self.mixture_binaural, np.ndarray) or self.mixture_binaural.ndim != 2:
            raise MixerError("mixture waveform must have shape (N, 2)")
        if self.mixture_binaural.shape[1] != 2 or self.mixture_binaural.shape[0] <= 0:
            raise MixerError("mixture waveform must have shape (N, 2)")
        if self.mixture_binaural.dtype != np.dtype(np.float32) or not np.isfinite(self.mixture_binaural).all():
            raise MixerError("mixture waveform must be finite float32")
        if self.mixture_binaural.shape[0] != self.receiver_end_sample_exclusive - self.receiver_start_sample:
            raise MixerError("mixture waveform does not match receiver support")
        array = _copy_readonly(self.mixture_binaural)
        object.__setattr__(self, "mixture_binaural", array)
        actual_hash = _array_sha256(array)
        try:
            validate_sha256(self.target_payload_sha256, "mixture.target_payload_sha256")
            validate_sha256(self.noise_payload_sha256, "mixture.noise_payload_sha256")
            validate_sha256(self.mixture_payload_sha256, "mixture.mixture_payload_sha256")
        except ValueError as exc:
            raise MixerError(str(exc)) from exc
        if actual_hash != self.mixture_payload_sha256:
            raise MixerError("mixture_payload_sha256 does not match waveform")
        _validate_diagnostics(self.diagnostics)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        max_residual = _finite(self.max_abs_residual, "mixture.max_abs_residual")
        relative_residual = _finite(self.relative_residual, "mixture.relative_residual")
        threshold = _finite(self.residual_gate_threshold, "mixture.residual_gate_threshold")
        if min(max_residual, relative_residual) < 0.0 or threshold != RESIDUAL_GATE_THRESHOLD:
            raise MixerError("mixture residual metadata is invalid")
        if self.reconstruction_status not in ("PASS", "FAIL"):
            raise MixerError("mixture.reconstruction_status is invalid")
        expected_id = stable_id("mixture", _artifact_identity_payload(self))
        if self.mixture_artifact_id != expected_id:
            raise MixerError("mixture_artifact_id does not match canonical metadata")
        if self.reconstruction_status == "PASS" and self.relative_residual > threshold:
            raise MixerError("PASS mixture exceeds residual threshold")
        object.__setattr__(self, "global_gain", global_gain)
        object.__setattr__(self, "alpha", alpha)

    def identity_payload(self) -> Dict[str, Any]:
        return _artifact_identity_payload(self)

    def to_payload(self) -> Dict[str, Any]:
        payload = dict(self.identity_payload())
        payload["mixture_artifact_id"] = self.mixture_artifact_id
        payload["mixture_samples"] = self.mixture_binaural.tolist()
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MixtureArtifact":
        expected = (
            "schema_version", "mixture_artifact_id", "mixer_algorithm_identity",
            "reconstruction_identity", "arithmetic_identity", "block_id", "episode_id",
            "pose_id", "timeline_id", "calibration_artifact_id", "global_gain_identity",
            "global_gain", "alpha", "target_component_identity", "target_payload_sha256",
            "noise_component_identity", "noise_payload_sha256", "mixture_payload_sha256",
            "sample_rate_hz", "channel_order", "shape", "dtype",
            "target_receiver_offset_samples", "noise_receiver_offset_samples",
            "receiver_start_sample", "receiver_end_sample_exclusive", "diagnostics",
            "max_abs_residual", "relative_residual", "residual_gate_threshold",
            "reconstruction_status", "mixture_samples",
        )
        _exact_fields(payload, expected, "mixture")
        shape = payload["shape"]
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise MixerError("mixture.shape must contain two dimensions")
        samples = np.asarray(payload["mixture_samples"], dtype=np.float32)
        if list(samples.shape) != list(shape):
            raise MixerError("mixture_samples shape does not match shape")
        values = dict(payload)
        values.pop("shape")
        values.pop("mixture_samples")
        values["channel_order"] = tuple(values["channel_order"])
        values["mixture_binaural"] = samples
        return cls(**values)


def _validate_component_against_timeline(
    component: WaveformComponent,
    role: str,
    timeline: CommonReceiverTimeline,
) -> None:
    if component.role != role:
        raise MixerError("{} component role is invalid".format(role))
    if component.timeline_identity != timeline.timeline_id:
        raise MixerError("{} component timeline identity does not match".format(role))
    expected = timeline.target_binaural if role == "target" else timeline.noise_binaural
    if component.payload.shape != expected.shape or not np.array_equal(component.payload, expected):
        raise MixerError("{} component waveform is not the supplied common-timeline component".format(role))


def _validate_bindings(
    block: BlockRecord,
    episode: EpisodeRecord,
    pose_id: str,
    timeline: CommonReceiverTimeline,
    target_component: WaveformComponent,
    noise_component: WaveformComponent,
    calibration_artifact: CalibrationArtifact,
    global_gain: GlobalGainSpec,
    contract: MixtureContract,
) -> None:
    if not isinstance(block, BlockRecord) or not isinstance(episode, EpisodeRecord):
        raise MixerError("block and episode records are required")
    if not isinstance(timeline, CommonReceiverTimeline):
        raise MixerError("timeline must be CommonReceiverTimeline")
    if not isinstance(calibration_artifact, CalibrationArtifact):
        raise MixerError("calibration_artifact must be CalibrationArtifact")
    if not isinstance(global_gain, GlobalGainSpec):
        raise MixerError("global_gain must be GlobalGainSpec")
    if not isinstance(contract, MixtureContract):
        raise MixerError("contract must be MixtureContract")
    _validate_stable(pose_id, "pose", "pose_id")
    if episode.block_id != block.block_id:
        raise MixerError("episode.block_id does not match block.block_id")
    if calibration_artifact.status != "CALIBRATED":
        raise MixerError("only CALIBRATED selection artifact may supply alpha")
    validate_calibration_artifact(calibration_artifact)
    if calibration_artifact.block_id != block.block_id:
        raise MixerError("calibration artifact belongs to a different block")
    if global_gain.identity != block.global_gain_identity:
        raise MixerError("global gain identity does not match block binding")
    _validate_component_against_timeline(target_component, "target", timeline)
    _validate_component_against_timeline(noise_component, "noise", timeline)
    try:
        segment = NoiseSegmentRecord.from_payload(dict(episode.fixed_dry_noise_segment_identity))
    except (NoiseSegmentError, TypeError) as exc:
        raise MixerError("episode noise segment is invalid: {}".format(exc)) from exc
    if timeline.noise_source_segment_identity != segment.segment_id:
        raise MixerError("timeline noise segment does not match episode fixed segment")
    if timeline.target_binaural.shape != timeline.noise_binaural.shape:
        raise MixerError("timeline target/noise shapes do not match")


def _render(target: np.ndarray, noise: np.ndarray, alpha: float, gain: float) -> np.ndarray:
    # A4-2B2 arithmetic identity: float64 intermediate expression, one final
    # float32 cast, no normalization or clipping.
    rendered = gain * (target.astype(np.float64) + alpha * noise.astype(np.float64))
    if not np.isfinite(rendered).all():
        raise MixerError("linear mixer produced non-finite samples")
    result = np.asarray(rendered, dtype=np.float32)
    if not np.isfinite(result).all():
        raise MixerError("linear mixer float32 result is non-finite")
    return result


def _make_artifact(
    block: BlockRecord,
    episode: EpisodeRecord,
    pose_id: str,
    timeline: CommonReceiverTimeline,
    target_component: WaveformComponent,
    noise_component: WaveformComponent,
    calibration_artifact: CalibrationArtifact,
    global_gain: GlobalGainSpec,
    contract: MixtureContract,
) -> MixtureArtifact:
    mixture = _render(target_component.payload, noise_component.payload, calibration_artifact.alpha, global_gain.gain)
    expected = mixture.astype(np.float64)
    actual = mixture.astype(np.float64)
    residual = float(np.max(np.abs(actual - expected)))
    relative = residual / max(1.0, float(np.max(np.abs(expected))))
    diagnostics = _diagnostics(target_component.payload, noise_component.payload, mixture, calibration_artifact.alpha)
    values = {
        "schema_version": MIXTURE_SCHEMA_VERSION,
        "mixer_algorithm_identity": contract.mixer_algorithm_identity,
        "reconstruction_identity": contract.reconstruction_identity,
        "arithmetic_identity": contract.arithmetic_identity,
        "block_id": block.block_id,
        "episode_id": episode.episode_id,
        "pose_id": pose_id,
        "timeline_id": timeline.timeline_id,
        "calibration_artifact_id": calibration_artifact.calibration_artifact_id,
        "global_gain_identity": global_gain.identity,
        "global_gain": global_gain.gain,
        "alpha": calibration_artifact.alpha,
        "target_component_identity": target_component.component_identity,
        "target_payload_sha256": target_component.payload_sha256,
        "noise_component_identity": noise_component.component_identity,
        "noise_payload_sha256": noise_component.payload_sha256,
        "mixture_payload_sha256": _array_sha256(mixture),
        "sample_rate_hz": contract.sample_rate_hz,
        "channel_order": tuple(contract.channel_order),
        "dtype": "float32",
        "target_receiver_offset_samples": timeline.target_receiver_offset_samples,
        "noise_receiver_offset_samples": timeline.noise_receiver_offset_samples,
        "receiver_start_sample": timeline.receiver_start_sample,
        "receiver_end_sample_exclusive": timeline.receiver_end_sample_exclusive,
        "mixture_binaural": mixture,
        "diagnostics": diagnostics,
        "max_abs_residual": residual,
        "relative_residual": relative,
        "residual_gate_threshold": contract.residual_gate_threshold,
        "reconstruction_status": "PASS" if relative <= contract.residual_gate_threshold else "FAIL",
    }
    values["mixture_artifact_id"] = stable_id("mixture", _artifact_identity_payload_from_values(values))
    return MixtureArtifact(**values)


def build_mixture(
    block: BlockRecord,
    episode: EpisodeRecord,
    pose_id: str,
    timeline: CommonReceiverTimeline,
    target_component: WaveformComponent,
    noise_component: WaveformComponent,
    calibration_artifact: CalibrationArtifact,
    global_gain: GlobalGainSpec,
    contract: MixtureContract = MixtureContract(),
) -> MixtureArtifact:
    """Build one pose mixture from the block's fixed alpha and gain.

    There is intentionally no ``alpha`` argument and no RIR argument.  Both
    propagated components must already be independently present on ``timeline``.
    """

    if not isinstance(target_component, WaveformComponent) or not isinstance(noise_component, WaveformComponent):
        raise MixerError("target_component and noise_component must be WaveformComponent values")
    _validate_bindings(
        block, episode, pose_id, timeline, target_component, noise_component,
        calibration_artifact, global_gain, contract,
    )
    artifact = _make_artifact(
        block, episode, pose_id, timeline, target_component, noise_component,
        calibration_artifact, global_gain, contract,
    )
    validate_mixture_reconstruction(
        artifact, block, episode, pose_id, timeline, target_component,
        noise_component, calibration_artifact, global_gain, contract,
    )
    return artifact


def _compare_diagnostics(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    _validate_diagnostics(actual)
    for key in _DIAGNOSTIC_FIELDS:
        left = actual[key]
        right = expected[key]
        if isinstance(right, tuple):
            if tuple(left) != tuple(right):
                raise MixerError("mixture diagnostic {} is inconsistent".format(key))
        elif isinstance(right, bool) or right is None:
            if left != right:
                raise MixerError("mixture diagnostic {} is inconsistent".format(key))
        elif not math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise MixerError("mixture diagnostic {} is inconsistent".format(key))


def validate_mixture_reconstruction(
    artifact: MixtureArtifact,
    block: BlockRecord,
    episode: EpisodeRecord,
    pose_id: str,
    timeline: CommonReceiverTimeline,
    target_component: WaveformComponent,
    noise_component: WaveformComponent,
    calibration_artifact: CalibrationArtifact,
    global_gain: GlobalGainSpec,
    contract: MixtureContract = MixtureContract(),
) -> MixtureArtifact:
    """Recompute the expected waveform and enforce the 1e-6 residual gate."""

    if not isinstance(artifact, MixtureArtifact):
        raise MixerError("artifact must be MixtureArtifact")
    _validate_bindings(
        block, episode, pose_id, timeline, target_component, noise_component,
        calibration_artifact, global_gain, contract,
    )
    if artifact.block_id != block.block_id or artifact.episode_id != episode.episode_id:
        raise MixerError("mixture block/episode binding is inconsistent")
    if artifact.pose_id != pose_id or artifact.timeline_id != timeline.timeline_id:
        raise MixerError("mixture pose/timeline binding is inconsistent")
    if artifact.calibration_artifact_id != calibration_artifact.calibration_artifact_id:
        raise MixerError("mixture calibration binding is inconsistent")
    if artifact.global_gain_identity != global_gain.identity or not math.isclose(artifact.global_gain, global_gain.gain, rel_tol=0.0, abs_tol=0.0):
        raise MixerError("mixture global gain binding is inconsistent")
    if artifact.alpha != calibration_artifact.alpha:
        raise MixerError("mixture alpha is not the calibration artifact alpha")
    if artifact.target_component_identity != target_component.component_identity or artifact.target_payload_sha256 != target_component.payload_sha256:
        raise MixerError("mixture target component binding is inconsistent")
    if artifact.noise_component_identity != noise_component.component_identity or artifact.noise_payload_sha256 != noise_component.payload_sha256:
        raise MixerError("mixture noise component binding is inconsistent")
    if artifact.mixture_binaural.shape != target_component.payload.shape:
        raise MixerError("mixture shape is inconsistent")
    expected = _render(target_component.payload, noise_component.payload, calibration_artifact.alpha, global_gain.gain)
    difference = np.abs(artifact.mixture_binaural.astype(np.float64) - expected.astype(np.float64))
    max_abs = float(np.max(difference))
    relative = max_abs / max(1.0, float(np.max(np.abs(expected.astype(np.float64)))))
    if artifact.mixture_payload_sha256 != _array_sha256(artifact.mixture_binaural):
        raise MixerError("mixture payload hash is inconsistent")
    if not math.isclose(artifact.max_abs_residual, max_abs, rel_tol=0.0, abs_tol=1.0e-12):
        raise MixerError("mixture max_abs_residual is inconsistent")
    if not math.isclose(artifact.relative_residual, relative, rel_tol=0.0, abs_tol=1.0e-12):
        raise MixerError("mixture relative_residual is inconsistent")
    if artifact.residual_gate_threshold != contract.residual_gate_threshold:
        raise MixerError("mixture residual threshold is inconsistent")
    expected_status = "PASS" if relative <= contract.residual_gate_threshold else "FAIL"
    if artifact.reconstruction_status != expected_status:
        raise MixerError("mixture reconstruction status is inconsistent")
    if expected_status != "PASS":
        raise MixerError("mixture reconstruction residual exceeds hard gate")
    _compare_diagnostics(
        artifact.diagnostics,
        _diagnostics(target_component.payload, noise_component.payload, expected, calibration_artifact.alpha),
    )
    return artifact


__all__ = [
    "ARITHMETIC_IDENTITY",
    "CHANNEL_ORDER_LIST",
    "GLOBAL_GAIN_ALGORITHM_IDENTITY",
    "GLOBAL_GAIN_SCHEMA_VERSION",
    "MIXER_ALGORITHM_IDENTITY",
    "MIXTURE_CONTRACT_SCHEMA_VERSION",
    "MIXTURE_SCHEMA_VERSION",
    "MixerError",
    "MixtureArtifact",
    "MixtureContract",
    "GlobalGainSpec",
    "RECONSTRUCTION_IDENTITY",
    "RESIDUAL_GATE_THRESHOLD",
    "WaveformComponent",
    "build_mixture",
    "validate_mixture_reconstruction",
]
