"""Result-independent propagated-component SNR diagnostics for O1."""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from active_audition.a4.identity import identity_sha256, stable_id, validate_sha256, validate_stable_id


COMPONENT_SNR_SCHEMA_VERSION = "active-asr-o1-component-snr-v1"
COMPONENT_SNR_ALGORITHM_IDENTITY = "active-asr-o1-target-active-mask-two-ear-component-snr-v1"
COMPONENT_SNR_POWER_IDENTITY = "two_ear_mean_square_same_receiver_mask_v1"


class ComponentSnrError(ValueError):
    pass


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ComponentSnrError("{} must be finite numeric".format(path))
    return float(value)


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ComponentSnrError("{} must be positive integer".format(path))
    return int(value)


def _exact(value: Mapping[str, Any], fields: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise ComponentSnrError("{} must be mapping".format(path))
    unknown = sorted(set(value) - set(fields))
    missing = sorted(set(fields) - set(value))
    if unknown or missing:
        raise ComponentSnrError("{} fields invalid: unknown={}, missing={}".format(path, unknown, missing))


@dataclass(frozen=True)
class O1ComponentSnrRecord:
    block_id: str
    episode_id: str
    role: str
    utterance_id: str
    pose_id: str
    timeline_identity: str
    target_component_identity: str
    noise_component_identity: str
    receiver_mask_identity: str
    calibration_artifact_identity: str
    alpha: float
    active_sample_count: int
    target_power: float
    noise_power: float
    component_snr_db: float
    schema_version: str = COMPONENT_SNR_SCHEMA_VERSION
    algorithm_identity: str = COMPONENT_SNR_ALGORITHM_IDENTITY
    power_identity: str = COMPONENT_SNR_POWER_IDENTITY
    record_id: str = ""
    record_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != COMPONENT_SNR_SCHEMA_VERSION or self.algorithm_identity != COMPONENT_SNR_ALGORITHM_IDENTITY:
            raise ComponentSnrError("unsupported component-SNR schema/algorithm")
        if self.power_identity != COMPONENT_SNR_POWER_IDENTITY:
            raise ComponentSnrError("unsupported component-SNR power convention")
        for field in ("block_id", "episode_id", "role", "utterance_id", "pose_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ComponentSnrError("{} must be non-empty string".format(field))
        for field, namespace in (("timeline_identity", "receiver-timeline"), ("target_component_identity", "target-component"), ("noise_component_identity", "noise-component"), ("receiver_mask_identity", "receiver-mask"), ("calibration_artifact_identity", "calibration")):
            try:
                validate_stable_id(getattr(self, field), namespace, field)
            except ValueError as exc:
                raise ComponentSnrError(str(exc)) from exc
        if self.role not in ("selection", "evaluation"):
            raise ComponentSnrError("role is invalid")
        _finite(self.alpha, "alpha")
        if self.alpha <= 0.0:
            raise ComponentSnrError("alpha must be positive")
        _positive_int(self.active_sample_count, "active_sample_count")
        for field in ("target_power", "noise_power", "component_snr_db"):
            _finite(getattr(self, field), field)
        if self.target_power <= 0.0 or self.noise_power <= 0.0:
            raise ComponentSnrError("target/noise power must be positive")
        expected = 10.0 * math.log10(self.target_power / (self.alpha * self.alpha * self.noise_power))
        if not math.isclose(self.component_snr_db, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ComponentSnrError("component_snr_db does not match powers and alpha")
        expected_id = stable_id("o1-component-snr", self.identity_payload())
        expected_sha = identity_sha256(self.identity_payload())
        if self.record_id and self.record_id != expected_id:
            raise ComponentSnrError("record_id mismatch")
        if self.record_sha256 and self.record_sha256 != expected_sha:
            raise ComponentSnrError("record_sha256 mismatch")
        object.__setattr__(self, "record_id", expected_id)
        object.__setattr__(self, "record_sha256", expected_sha)

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "power_identity": self.power_identity,
            "block_id": self.block_id,
            "episode_id": self.episode_id,
            "role": self.role,
            "utterance_id": self.utterance_id,
            "pose_id": self.pose_id,
            "timeline_identity": self.timeline_identity,
            "target_component_identity": self.target_component_identity,
            "noise_component_identity": self.noise_component_identity,
            "receiver_mask_identity": self.receiver_mask_identity,
            "calibration_artifact_identity": self.calibration_artifact_identity,
            "alpha": self.alpha,
            "active_sample_count": self.active_sample_count,
            "target_power": self.target_power,
            "noise_power": self.noise_power,
            "component_snr_db": self.component_snr_db,
        }

    def to_payload(self) -> Mapping[str, Any]:
        return dict(self.identity_payload(), record_id=self.record_id, record_sha256=self.record_sha256)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1ComponentSnrRecord":
        fields = tuple(cls.__dataclass_fields__)
        _exact(payload, fields, "o1_component_snr")
        return cls(**dict(payload))


def build_component_snr_record(
    *, block_id: str, episode_id: str, role: str, utterance_id: str, pose_id: str,
    timeline_identity: str, target_component_identity: str, noise_component_identity: str,
    receiver_mask_identity: str, calibration_artifact_identity: str, alpha: float,
    target_binaural: np.ndarray, noise_binaural: np.ndarray, receiver_mask: Sequence[bool],
) -> O1ComponentSnrRecord:
    target = np.asarray(target_binaural)
    noise = np.asarray(noise_binaural)
    mask = np.asarray(tuple(receiver_mask), dtype=bool)
    if target.dtype != np.dtype(np.float32) or noise.dtype != np.dtype(np.float32) or target.ndim != 2 or target.shape[1] != 2 or noise.shape != target.shape or mask.shape != (target.shape[0],):
        raise ComponentSnrError("target/noise/mask shape or dtype mismatch")
    if not np.isfinite(target).all() or not np.isfinite(noise).all() or int(mask.sum()) <= 0:
        raise ComponentSnrError("nonfinite components or empty receiver mask")
    target_values = (np.square(target[:, 0].astype(np.float64)) + np.square(target[:, 1].astype(np.float64))) / 2.0
    noise_values = (np.square(noise[:, 0].astype(np.float64)) + np.square(noise[:, 1].astype(np.float64))) / 2.0
    count = int(mask.sum())
    target_power = float(np.sum(target_values[mask], dtype=np.float64) / count)
    noise_power = float(np.sum(noise_values[mask], dtype=np.float64) / count)
    if target_power <= 0.0 or noise_power <= 0.0:
        raise ComponentSnrError("target/noise active power must be positive")
    alpha_value = _finite(alpha, "alpha")
    if alpha_value <= 0.0:
        raise ComponentSnrError("alpha must be positive")
    return O1ComponentSnrRecord(
        block_id=block_id, episode_id=episode_id, role=role, utterance_id=utterance_id, pose_id=pose_id,
        timeline_identity=timeline_identity, target_component_identity=target_component_identity,
        noise_component_identity=noise_component_identity, receiver_mask_identity=receiver_mask_identity,
        calibration_artifact_identity=calibration_artifact_identity, alpha=alpha_value,
        active_sample_count=count, target_power=target_power, noise_power=noise_power,
        component_snr_db=10.0 * math.log10(target_power / (alpha_value * alpha_value * noise_power)),
    )


__all__ = [
    "COMPONENT_SNR_ALGORITHM_IDENTITY", "COMPONENT_SNR_POWER_IDENTITY", "COMPONENT_SNR_SCHEMA_VERSION",
    "ComponentSnrError", "O1ComponentSnrRecord", "build_component_snr_record",
]
