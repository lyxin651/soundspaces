"""Selection-only initial-pose noise-gain calibration for A4-2B1.

Calibration consumes already-propagated, common-timeline target/noise
components.  It never renders RIRs, mixes waveforms, decodes ASR, or accepts
evaluation results.  One immutable artifact is produced for one BlockRecord.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from active_audition.a4.identity import stable_id, validate_stable_id
from active_audition.a4.records import (
    CALIBRATION_ALGORITHM_IDENTITY,
    CALIBRATION_MASK_IDENTITY,
    CALIBRATION_MEASURED_SNR_TOLERANCE_DB,
    CALIBRATION_POWER_IDENTITY,
    CALIBRATION_SCOPE_IDENTITY,
    INITIAL_POSE_SCOPE_IDENTITY,
    BlockRecord,
    CalibrationArtifact,
)
from active_audition.a4.timeline import ReceiverTimeMask


CALIBRATION_CONTRACT_SCHEMA_VERSION = "active-asr-a4-calibration-contract-v1"


class CalibrationError(ValueError):
    """Raised when selection-only calibration inputs are invalid."""


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError("{} must be finite".format(path))
    return result


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationError("{} must be a non-empty string".format(path))
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError("{} must be a positive integer".format(path))
    return int(value)


def _exact_fields(payload: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(payload, Mapping):
        raise CalibrationError("{} must be a mapping".format(path))
    unknown = sorted(set(payload) - set(expected))
    missing = sorted(set(expected) - set(payload))
    if unknown or missing:
        raise CalibrationError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


@dataclass(frozen=True)
class CalibrationContract:
    """Versioned calibration values; nominal SNR is block/config input."""

    nominal_snr_db: float
    sample_rate_hz: int = 16000
    measured_snr_tolerance_db: float = CALIBRATION_MEASURED_SNR_TOLERANCE_DB
    schema_version: str = CALIBRATION_CONTRACT_SCHEMA_VERSION
    algorithm_identity: str = CALIBRATION_ALGORITHM_IDENTITY
    power_identity: str = CALIBRATION_POWER_IDENTITY
    scope_identity: str = CALIBRATION_SCOPE_IDENTITY
    mask_identity: str = CALIBRATION_MASK_IDENTITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "nominal_snr_db", _finite(self.nominal_snr_db, "nominal_snr_db"))
        object.__setattr__(self, "sample_rate_hz", _positive_int(self.sample_rate_hz, "sample_rate_hz"))
        object.__setattr__(
            self,
            "measured_snr_tolerance_db",
            _finite(self.measured_snr_tolerance_db, "measured_snr_tolerance_db"),
        )
        if self.sample_rate_hz != 16000:
            raise CalibrationError("A4-2B1 calibration requires 16000 Hz")
        if self.measured_snr_tolerance_db != CALIBRATION_MEASURED_SNR_TOLERANCE_DB:
            raise CalibrationError("measured_snr_tolerance_db must be exactly 0.1 dB")
        for field in (
            "schema_version",
            "algorithm_identity",
            "power_identity",
            "scope_identity",
            "mask_identity",
        ):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        if self.schema_version != CALIBRATION_CONTRACT_SCHEMA_VERSION:
            raise CalibrationError("schema_version is not the A4 calibration contract")
        if self.algorithm_identity != CALIBRATION_ALGORITHM_IDENTITY:
            raise CalibrationError("algorithm_identity is not selection-only calibration")
        if self.power_identity != CALIBRATION_POWER_IDENTITY:
            raise CalibrationError("power_identity is not two-ear mean-square")
        if self.scope_identity != CALIBRATION_SCOPE_IDENTITY:
            raise CalibrationError("scope_identity is not selection-only initial-pose")
        if self.mask_identity != CALIBRATION_MASK_IDENTITY:
            raise CalibrationError("mask_identity is not shared receiver-mask calibration")

    def to_payload(self) -> dict:
        return {
            "nominal_snr_db": self.nominal_snr_db,
            "sample_rate_hz": self.sample_rate_hz,
            "measured_snr_tolerance_db": self.measured_snr_tolerance_db,
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "power_identity": self.power_identity,
            "scope_identity": self.scope_identity,
            "mask_identity": self.mask_identity,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CalibrationContract":
        expected = (
            "nominal_snr_db",
            "sample_rate_hz",
            "measured_snr_tolerance_db",
            "schema_version",
            "algorithm_identity",
            "power_identity",
            "scope_identity",
            "mask_identity",
        )
        _exact_fields(payload, expected, "calibration_contract")
        return cls(**dict(payload))

    @property
    def identity(self) -> str:
        return stable_id("calibration-contract", self.to_payload())


@dataclass(frozen=True)
class SelectionComponentAtInitial:
    """One selection episode's already-aligned initial-pose components."""

    episode_id: str
    block_id: str
    role: str
    utterance_id: str
    pose_scope_identity: str
    timeline_identity: str
    dry_mask_identity: str
    receiver_mask_identity: str
    target_component_identity: str
    noise_component_identity: str
    target_direct_onset_samples: Mapping[str, int]
    target_binaural: np.ndarray
    noise_binaural: np.ndarray
    receiver_time_mask: ReceiverTimeMask

    def __post_init__(self) -> None:
        for field in (
            "episode_id",
            "block_id",
            "role",
            "utterance_id",
            "pose_scope_identity",
            "timeline_identity",
            "dry_mask_identity",
            "receiver_mask_identity",
            "target_component_identity",
            "noise_component_identity",
        ):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        try:
            validate_stable_id(self.episode_id, "episode", "episode_id")
            validate_stable_id(self.block_id, "block", "block_id")
            validate_stable_id(self.timeline_identity, "receiver-timeline", "timeline_identity")
            validate_stable_id(self.dry_mask_identity, "active-mask", "dry_mask_identity")
            validate_stable_id(self.receiver_mask_identity, "receiver-mask", "receiver_mask_identity")
            validate_stable_id(self.target_component_identity, "target-component", "target_component_identity")
            validate_stable_id(self.noise_component_identity, "noise-component", "noise_component_identity")
        except ValueError as exc:
            raise CalibrationError(str(exc)) from exc
        if self.role != "selection":
            raise CalibrationError("calibration accepts selection episodes only")
        if self.pose_scope_identity != INITIAL_POSE_SCOPE_IDENTITY:
            raise CalibrationError("calibration accepts initial-pose components only")
        if not isinstance(self.target_binaural, np.ndarray) or not isinstance(self.noise_binaural, np.ndarray):
            raise CalibrationError("binaural components must be numpy arrays")
        if self.target_binaural.shape != self.noise_binaural.shape or self.target_binaural.ndim != 2:
            raise CalibrationError("target/noise components must have identical two-dimensional shape")
        if self.target_binaural.shape[1] != 2:
            raise CalibrationError("binaural components must have shape (N, 2) [L, R]")
        if self.target_binaural.dtype != np.dtype(np.float32) or self.noise_binaural.dtype != np.dtype(np.float32):
            raise CalibrationError("binaural components must use float32 samples")
        if not np.isfinite(self.target_binaural).all() or not np.isfinite(self.noise_binaural).all():
            raise CalibrationError("binaural components must be finite")
        if not isinstance(self.receiver_time_mask, ReceiverTimeMask):
            raise CalibrationError("receiver_time_mask must be a ReceiverTimeMask")
        if self.receiver_time_mask.timeline_identity != self.timeline_identity:
            raise CalibrationError("receiver_time_mask timeline does not match component")
        if self.receiver_time_mask.dry_mask_identity != self.dry_mask_identity:
            raise CalibrationError("receiver_time_mask dry mask does not match component")
        if self.receiver_time_mask.mask_id != self.receiver_mask_identity:
            raise CalibrationError("receiver_time_mask identity does not match component")
        if self.receiver_time_mask.target_direct_onset_samples != dict(self.target_direct_onset_samples):
            raise CalibrationError("receiver_time_mask onset provenance does not match component")
        if len(self.receiver_time_mask.mask) != self.target_binaural.shape[0]:
            raise CalibrationError("receiver_time_mask must match component sample count")
        onset = dict(self.target_direct_onset_samples)
        if set(onset) != {"L", "R"}:
            raise CalibrationError("target_direct_onset_samples must contain L and R")
        for channel in ("L", "R"):
            _positive_int(int(onset[channel]) + 1, "target_direct_onset_samples." + channel)
        object.__setattr__(self, "target_direct_onset_samples", MappingProxyType(dict(onset)))
        for field in ("target_binaural", "noise_binaural"):
            copied = np.ascontiguousarray(getattr(self, field).copy())
            copied.setflags(write=False)
            object.__setattr__(self, field, copied)

    @property
    def active_sample_count(self) -> int:
        return self.receiver_time_mask.active_sample_count

    def identity_payload(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "block_id": self.block_id,
            "role": self.role,
            "utterance_id": self.utterance_id,
            "pose_scope_identity": self.pose_scope_identity,
            "timeline_identity": self.timeline_identity,
            "dry_mask_identity": self.dry_mask_identity,
            "receiver_mask_identity": self.receiver_mask_identity,
            "target_component_identity": self.target_component_identity,
            "noise_component_identity": self.noise_component_identity,
            "target_direct_onset_samples": dict(self.target_direct_onset_samples),
            "sample_count": int(self.target_binaural.shape[0]),
            "active_sample_count": self.active_sample_count,
        }


def _two_ear_power(array: np.ndarray, mask: np.ndarray) -> float:
    values = (np.square(array[:, 0].astype(np.float64)) + np.square(array[:, 1].astype(np.float64))) / 2.0
    return float(np.sum(values[mask], dtype=np.float64))


def recompute_measured_snr_db(ps: float, pn: float, alpha: float) -> float:
    ps = _finite(ps, "ps")
    pn = _finite(pn, "pn")
    alpha = _finite(alpha, "alpha")
    if ps <= 0.0 or pn <= 0.0 or alpha <= 0.0:
        raise CalibrationError("Ps, Pn, and alpha must be positive")
    return float(10.0 * math.log10(ps / (alpha * alpha * pn)))


def calibrate_block_noise_gain(
    block: BlockRecord,
    selection_components_at_initial: Sequence[SelectionComponentAtInitial],
    calibration_contract: CalibrationContract,
) -> CalibrationArtifact:
    """Calibrate one block once from exactly its two initial selection episodes."""

    if not isinstance(block, BlockRecord):
        raise CalibrationError("block must be BlockRecord")
    if not isinstance(calibration_contract, CalibrationContract):
        raise CalibrationError("calibration_contract must be CalibrationContract")
    if not math.isclose(
        calibration_contract.nominal_snr_db,
        block.nominal_initial_snr_db,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise CalibrationError("calibration contract nominal SNR does not match block")
    if not isinstance(selection_components_at_initial, (list, tuple)) or len(selection_components_at_initial) != 2:
        raise CalibrationError("calibration requires exactly two selection episodes")
    components = tuple(selection_components_at_initial)
    if any(not isinstance(item, SelectionComponentAtInitial) for item in components):
        raise CalibrationError("calibration inputs must be SelectionComponentAtInitial values")
    if any(item.block_id != block.block_id for item in components):
        raise CalibrationError("all selection components must belong to the block")
    if set(item.utterance_id for item in components) != set(block.selection_utterance_ids):
        raise CalibrationError("selection components must exactly match block selection utterances")
    if len({item.episode_id for item in components}) != 2:
        raise CalibrationError("selection episode IDs must be unique")
    ordered = tuple(sorted(components, key=lambda item: block.selection_utterance_ids.index(item.utterance_id)))
    total_target_energy = 0.0
    total_noise_energy = 0.0
    total_active_samples = 0
    for item in ordered:
        if item.role != "selection":
            raise CalibrationError("evaluation episodes cannot enter calibration")
        if item.active_sample_count <= 0:
            raise CalibrationError("selection episode has zero active samples")
        receiver_mask = np.asarray(item.receiver_time_mask.mask, dtype=bool)
        total_target_energy += _two_ear_power(item.target_binaural, receiver_mask)
        total_noise_energy += _two_ear_power(item.noise_binaural, receiver_mask)
        total_active_samples += item.active_sample_count
    if total_active_samples <= 0:
        raise CalibrationError("calibration has no active samples")
    ps = total_target_energy / float(total_active_samples)
    pn = total_noise_energy / float(total_active_samples)
    if ps <= 0.0:
        raise CalibrationError("Ps must be positive")
    if pn <= 0.0:
        raise CalibrationError("Pn must be positive")
    alpha = math.sqrt(ps / (pn * (10.0 ** (calibration_contract.nominal_snr_db / 10.0))))
    measured = recompute_measured_snr_db(ps, pn, alpha)
    if abs(measured - calibration_contract.nominal_snr_db) > calibration_contract.measured_snr_tolerance_db:
        raise CalibrationError("measured SNR exceeds calibration tolerance")
    input_identities = {
        "target_component_identities": [item.target_component_identity for item in ordered],
        "noise_component_identities": [item.noise_component_identity for item in ordered],
        "timeline_identities": [item.timeline_identity for item in ordered],
        "dry_mask_identities": [item.dry_mask_identity for item in ordered],
        "receiver_mask_identities": [item.receiver_mask_identity for item in ordered],
    }
    provenance = {
        "algorithm_identity": calibration_contract.algorithm_identity,
        "power_identity": calibration_contract.power_identity,
        "scope_identity": calibration_contract.scope_identity,
        "mask_identity": calibration_contract.mask_identity,
        "pose_scope_identity": INITIAL_POSE_SCOPE_IDENTITY,
        "sample_rate_hz": calibration_contract.sample_rate_hz,
        "measurement_tolerance_db": calibration_contract.measured_snr_tolerance_db,
        "timeline_identities": [item.timeline_identity for item in ordered],
        "dry_mask_identities": [item.dry_mask_identity for item in ordered],
        "target_direct_onset_samples": [dict(item.target_direct_onset_samples) for item in ordered],
        "receiver_mask_records": [
            {
                "timeline_identity": item.timeline_identity,
                "dry_mask_identity": item.dry_mask_identity,
                "receiver_mask_identity": item.receiver_mask_identity,
                "receiver_start_sample": item.receiver_time_mask.receiver_start_sample,
                "receiver_end_sample_exclusive": item.receiver_time_mask.receiver_end_sample_exclusive,
                "active_start_sample": item.receiver_time_mask.active_start_sample,
                "active_end_sample_exclusive": item.receiver_time_mask.active_end_sample_exclusive,
                "active_sample_count": item.active_sample_count,
                "sample_count": len(item.receiver_time_mask.mask),
            }
            for item in ordered
        ],
        "total_active_sample_count": total_active_samples,
    }
    artifact_payload = {
        "schema_version": "active-asr-a4-calibration-v1",
        "block_id": block.block_id,
        "calibration_contract_identity": calibration_contract.identity,
        "selection_episode_ids": [item.episode_id for item in ordered],
        "input_component_identities": input_identities,
        "ps": ps,
        "pn": pn,
        "active_sample_count": total_active_samples,
        "alpha": alpha,
        "nominal_snr_db": calibration_contract.nominal_snr_db,
        "measured_snr_db": measured,
        "status": "CALIBRATED",
        "provenance": provenance,
    }
    return CalibrationArtifact(
        calibration_artifact_id=stable_id("calibration", artifact_payload),
        **artifact_payload,
    )


__all__ = [
    "CALIBRATION_ALGORITHM_IDENTITY",
    "CALIBRATION_CONTRACT_SCHEMA_VERSION",
    "CALIBRATION_MASK_IDENTITY",
    "CALIBRATION_MEASURED_SNR_TOLERANCE_DB",
    "CALIBRATION_POWER_IDENTITY",
    "CALIBRATION_SCOPE_IDENTITY",
    "CalibrationContract",
    "CalibrationError",
    "SelectionComponentAtInitial",
    "calibrate_block_noise_gain",
    "recompute_measured_snr_db",
]
