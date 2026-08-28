"""Strict, model-independent ClassDOA V1 schema primitives."""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


DATASET_FAMILY = "soundspaces_binaural_foa_clsdoa_v1"
NAMESPACE = "binaural_foa_clsdoa_v1"
SCHEMA_VERSION = "clsdoa_v1.0"
SAMPLE_RATE_HZ = 24000
CLIP_DURATION_SEC = 5.0
NUM_SAMPLES = 120000
CLASS_COUNT = 12
REPRESENTATIONS = ("binaural", "foa")


class SchemaError(ValueError):
    """Raised when a V1 schema object is invalid."""


@dataclass(frozen=True)
class RenderPolicy:
    """Build/storage policy for optional RIR persistence."""

    save_rir: bool = True
    require_rir: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.save_rir, bool) or not isinstance(self.require_rir, bool):
            raise SchemaError("save_rir and require_rir must be boolean")
        if self.require_rir and not self.save_rir:
            raise SchemaError("require_rir cannot be true when save_rir is false")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RenderPolicy":
        policy = config.get("storage", config)
        save_rir = policy.get("save_rir", True)
        require_rir = policy.get("require_rir", save_rir)
        return cls(save_rir=save_rir, require_rir=require_rir)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SchemaError("{} must be finite".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError("{} must be numeric".format(name)) from exc
    if not math.isfinite(result):
        raise SchemaError("{} must be finite".format(name))
    return result


def _vector(value: Any, name: str) -> Tuple[float, float, float]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise SchemaError("{} must have length 3".format(name)) from exc
    if len(values) != 3:
        raise SchemaError("{} must have length 3".format(name))
    return tuple(_finite_float(item, name) for item in values)


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("{} must be a non-empty string".format(name))
    return value


def validate_class_id(value: Any, name: str = "class_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < CLASS_COUNT:
        raise SchemaError("{} must be an integer in [0, 11]".format(name))
    return int(value)


def _representations(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str) or value is None:
        raise SchemaError("representations must contain binaural and foa")
    values = tuple(str(item) for item in value)
    if set(values) != set(REPRESENTATIONS) or len(values) != len(REPRESENTATIONS):
        raise SchemaError("representations must contain exactly binaural and foa")
    return tuple(item for item in REPRESENTATIONS if item in values)


@dataclass(frozen=True)
class RenderRecord:
    """Actual payload metadata, separate from the immutable EpisodeRecipe."""

    episode_id: str
    representation: str
    audio_path: Optional[str]
    rir_path: Optional[str]
    sample_rate_hz: int
    num_channels: int
    num_samples: int
    dtype: str
    format: str
    render_status: str
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        _required_string(self.episode_id, "episode_id")
        if self.representation not in REPRESENTATIONS:
            raise SchemaError("unsupported representation: {}".format(self.representation))
        if self.render_status not in ("complete", "failed"):
            raise SchemaError("render_status must be complete or failed")
        if self.render_status == "complete":
            if not isinstance(self.audio_path, str) or not self.audio_path:
                raise SchemaError("complete render requires audio_path")
            if int(self.sample_rate_hz) != SAMPLE_RATE_HZ:
                raise SchemaError("V1 sample rate must be 24000 Hz")
            expected_channels = 2 if self.representation == "binaural" else 4
            if int(self.num_channels) != expected_channels:
                raise SchemaError("invalid channel count for {}".format(self.representation))
            if int(self.num_samples) != NUM_SAMPLES:
                raise SchemaError("V1 render must contain 120000 samples")
            if str(self.dtype) != "float32":
                raise SchemaError("V1 render dtype must be float32")
            expected_format = "WAV" if self.representation == "binaural" else "AmbiX ACN/SN3D"
            if self.format != expected_format:
                raise SchemaError("invalid format for {}".format(self.representation))
            if self.failure_reason is not None:
                raise SchemaError("complete render cannot have failure_reason")
        elif not isinstance(self.failure_reason, str) or not self.failure_reason:
            raise SchemaError("failed render requires failure_reason")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "episode_id": self.episode_id,
            "representation": self.representation,
            "audio_path": self.audio_path,
            "rir_path": self.rir_path,
            "sample_rate_hz": int(self.sample_rate_hz),
            "num_channels": int(self.num_channels),
            "num_samples": int(self.num_samples),
            "dtype": self.dtype,
            "format": self.format,
            "render_status": self.render_status,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RenderRecord":
        required = ("episode_id", "representation", "audio_path", "rir_path", "sample_rate_hz", "num_channels", "num_samples", "dtype", "format", "render_status")
        missing = [key for key in required if key not in value]
        if missing:
            raise SchemaError("RenderRecord missing fields: {}".format(", ".join(missing)))
        return cls(**{key: value[key] for key in required}, failure_reason=value.get("failure_reason"))
