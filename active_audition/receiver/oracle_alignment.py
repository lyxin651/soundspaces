"""Oracle-aligned native16 A2 qualification.

This module is deliberately separate from the historical A2 v2 metric
runner.  It closes only the native16 production invariants required by the
2026-09-27 Issue #1/#2 amendment.  It does not import or implement ASR,
mixing, source registries, Oracle selection, or RL.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.data.storage import DatasetStorage
from active_audition.receiver.geometry import (
    geometry_sanity,
    load_geometry_registry,
    relative_azimuth_deg as geometry_relative_azimuth,
)
from active_audition.receiver.qualification import (
    BLOCKED,
    PASS,
    compute_direct_metrics,
    load_metric_contract,
    make_probe,
    metric_contract_sha256,
    validate_direction_channel_calibration,
    validate_qualification_case,
)


ORACLE_CONTRACT_SCHEMA_VERSION = "active-asr-a2-oracle-aligned-v3"
YAW_ARTIFACT_SCHEMA_VERSION = "active-asr-a2-yaw-relative-qualification-v1"
SYMMETRY_ARTIFACT_SCHEMA_VERSION = "active-asr-a2-controlled-symmetry-qualification-v1"
POSE_GAIN_ARTIFACT_SCHEMA_VERSION = "active-asr-a2-native16-pose-gain-preservation-v1"
WAVEFORM_ARTIFACT_SCHEMA_VERSION = "active-asr-a2-native16-waveform-validity-v1"
SUMMARY_ARTIFACT_SCHEMA_VERSION = "active-asr-a2-oracle-alignment-summary-v1"
POSE_ENERGY_EQUALITY_EPSILON_DB = 1.0e-9


class OracleAlignmentError(ValueError):
    """Raised when the Oracle-aligned v3 contract or artifacts are invalid."""


def _only(value: Mapping[str, Any], keys: Iterable[str], path: str) -> None:
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise OracleAlignmentError("unknown oracle contract field(s) at {}: {}".format(path, ", ".join(unknown)))


def _required(value: Mapping[str, Any], keys: Iterable[str], path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise OracleAlignmentError("missing oracle contract field(s) at {}: {}".format(path, ", ".join(missing)))


def _string(value: Any, path: str, expected: Optional[str] = None) -> str:
    if not isinstance(value, str) or not value:
        raise OracleAlignmentError("{} must be a non-empty string".format(path))
    if expected is not None and value != expected:
        raise OracleAlignmentError("{} must be {!r}".format(path, expected))
    return value


def _bool(value: Any, path: str, expected: Optional[bool] = None) -> bool:
    if not isinstance(value, bool):
        raise OracleAlignmentError("{} must be boolean".format(path))
    if expected is not None and value is not expected:
        raise OracleAlignmentError("{} must be {}".format(path, str(expected).lower()))
    return value


def _number(value: Any, path: str, expected: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise OracleAlignmentError("{} must be finite numeric".format(path))
    if expected is not None and float(value) != float(expected):
        raise OracleAlignmentError("{} must be {}".format(path, expected))
    return float(value)


def _exact_list(value: Any, expected: Sequence[Any], path: str) -> None:
    if not isinstance(value, list) or value != list(expected):
        raise OracleAlignmentError("{} must be {!r}".format(path, list(expected)))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OracleAlignmentError("canonical value contains non-finite float")
        return 0.0 if value == 0.0 else value
    return value


def canonical_oracle_json(value: Any) -> str:
    return json.dumps(_canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def oracle_contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_oracle_json(contract).encode("utf-8")).hexdigest()


def load_oracle_contract(path: str) -> Dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise OracleAlignmentError("unable to load Oracle-aligned contract {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise OracleAlignmentError("Oracle-aligned contract root must be a mapping")
    return dict(validate_oracle_contract(value))


def validate_oracle_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    _only(contract, ("contract", "serialization", "production_native16", "hard_gates", "qualification_evidence", "historical_v2_provenance", "artifacts"), "root")
    _required(contract, ("contract", "serialization", "production_native16", "hard_gates", "qualification_evidence", "historical_v2_provenance", "artifacts"), "root")
    identity = contract["contract"]
    if not isinstance(identity, Mapping):
        raise OracleAlignmentError("contract must be a mapping")
    _only(identity, ("namespace", "version", "gate", "state", "role", "parent_metric_contract_version", "parent_metric_contract_sha256"), "contract")
    _required(identity, ("namespace", "version", "gate", "state", "role", "parent_metric_contract_version", "parent_metric_contract_sha256"), "contract")
    _string(identity["namespace"], "contract.namespace", "active-asr")
    _string(identity["version"], "contract.version", ORACLE_CONTRACT_SCHEMA_VERSION)
    _string(identity["gate"], "contract.gate", "A2")
    _string(identity["state"], "contract.state", "FROZEN")
    _string(identity["role"], "contract.role", "oracle_aligned_native16_hard_gates")
    _string(identity["parent_metric_contract_version"], "contract.parent_metric_contract_version", "active-asr-a2-metric-v2")
    _string(identity["parent_metric_contract_sha256"], "contract.parent_metric_contract_sha256", "f1185d2c5fe81091199d5525f224822a3227c700c60bba9ab168e6d9ddc38f83")

    serialization = contract["serialization"]
    if not isinstance(serialization, Mapping):
        raise OracleAlignmentError("serialization must be a mapping")
    _only(serialization, ("version", "hash_algorithm"), "serialization")
    _required(serialization, ("version", "hash_algorithm"), "serialization")
    _string(serialization["version"], "serialization.version", "canonical-json-v1")
    _string(serialization["hash_algorithm"], "serialization.hash_algorithm", "sha256")

    production = contract["production_native16"]
    if not isinstance(production, Mapping):
        raise OracleAlignmentError("production_native16 must be a mapping")
    _only(production, ("sample_rate_hz", "channel_order", "direct_window", "metrics"), "production_native16")
    _required(production, ("sample_rate_hz", "channel_order", "direct_window", "metrics"), "production_native16")
    _number(production["sample_rate_hz"], "production_native16.sample_rate_hz", 16000)
    _exact_list(production["channel_order"], ["L", "R"], "production_native16.channel_order")
    window = production["direct_window"]
    if not isinstance(window, Mapping):
        raise OracleAlignmentError("production_native16.direct_window must be a mapping")
    _only(window, ("search_start_sec", "search_duration_sec", "onset_threshold_fraction_of_channel_peak", "minimum_peak_amplitude", "minimum_energy", "length_sec", "onset_rule", "low_energy_policy", "no_onset_policy"), "production_native16.direct_window")
    _required(window, ("search_start_sec", "search_duration_sec", "onset_threshold_fraction_of_channel_peak", "minimum_peak_amplitude", "minimum_energy", "length_sec", "onset_rule", "low_energy_policy", "no_onset_policy"), "production_native16.direct_window")
    for key, expected in (("search_start_sec", 0.0), ("search_duration_sec", 0.050), ("onset_threshold_fraction_of_channel_peak", 0.10), ("minimum_peak_amplitude", 1.0e-8), ("minimum_energy", 1.0e-12), ("length_sec", 0.010)):
        _number(window[key], "production_native16.direct_window." + key, expected)
    _string(window["onset_rule"], "production_native16.direct_window.onset_rule", "first_absolute_sample_at_or_above_threshold")
    _string(window["low_energy_policy"], "production_native16.direct_window.low_energy_policy", "N/A")
    _string(window["no_onset_policy"], "production_native16.direct_window.no_onset_policy", "N/A")
    metrics = production["metrics"]
    if not isinstance(metrics, Mapping):
        raise OracleAlignmentError("production_native16.metrics must be a mapping")
    _only(metrics, ("itd", "ild", "no_per_render_normalization", "no_per_pose_normalization", "no_per_ear_normalization"), "production_native16.metrics")
    _required(metrics, ("itd", "ild", "no_per_render_normalization", "no_per_pose_normalization", "no_per_ear_normalization"), "production_native16.metrics")
    if metrics["itd"] != {"definition": "tR_minus_tL", "units": "samples_at_native_rate", "sign_positive": "right_onset_later_than_left"}:
        raise OracleAlignmentError("production_native16.metrics.itd does not match v2")
    if metrics["ild"] != {"definition": "10log10(E_L_divided_by_E_R)", "energy_definition": "sum_of_squared_samples_in_common_direct_window", "units": "dB"}:
        raise OracleAlignmentError("production_native16.metrics.ild does not match v2")
    for key in ("no_per_render_normalization", "no_per_pose_normalization", "no_per_ear_normalization"):
        _bool(metrics[key], "production_native16.metrics." + key, True)

    gates = contract["hard_gates"]
    if not isinstance(gates, Mapping):
        raise OracleAlignmentError("hard_gates must be a mapping")
    gate_fields = {
        "P1_native_channel_mapping": ("kind", "canonical_order", "independent_raw_channel_calibration_required"),
        "P2_yaw_relative_coordinate": ("kind", "fixed_world_source_distance_m", "listener_yaws_deg", "side_relative_angles_deg", "geometry_pose_relative_azimuth_must_match", "side_itd_sign_follows_listener_relative_direction", "front_back_zero_or_near_zero_itd_allowed", "source_position_changes_with_yaw"),
        "P3_side_los_itd_direction": ("kind", "minimum_sign_accuracy_fraction", "global_lr_reversal_forbidden"),
        "P4_repeatability": ("kind", "repeat_count", "direct_itd_spread_max_samples_at_16khz", "band_direct_ild_spread_max_db", "direct_energy_spread_max_db", "applicable_case_min_fraction"),
        "P5_controlled_symmetry": ("kind", "mirror_angles_deg", "distances_m", "both_sides_applicable_required", "direct_itd_nonzero_required", "direct_itd_signs_opposite_required", "full_rir_magnitude_symmetry_gate", "ild_magnitude_symmetry_gate"),
        "P6_near_far_los": ("kind", "direct_energy_decreases_with_distance_required", "full_window_or_wer_monotonicity_required"),
        "P7_nlos_applicability": ("kind", "direct_path_metric_policy", "na_cannot_reduce_hard_gate_denominator"),
        "P8_native16_pose_gain_preservation": ("kind", "sample_rate_hz", "dry_fixture", "global_gains", "normalization", "forbidden_normalization", "gain_linearity_exact_required", "relative_pose_energy_invariance_exact_required", "pose_dependent_energy_must_not_be_equalized"),
        "P9_native16_waveform_validity": ("kind", "sample_rate_hz", "channel_order", "frontend", "frontend_formula", "dtype", "finite_required", "full_convolution_time_axis_required", "pose_dependent_crop_or_trim_forbidden", "max_abs_waveform_before_clipping", "repeated_artifact_reproducibility_required"),
    }
    _exact_list(sorted(gates), sorted(gate_fields), "hard_gates")
    for gate_name, fields in gate_fields.items():
        value = gates[gate_name]
        if not isinstance(value, Mapping):
            raise OracleAlignmentError("hard_gates.{} must be a mapping".format(gate_name))
        _only(value, fields, "hard_gates." + gate_name)
        _required(value, fields, "hard_gates." + gate_name)
    _exact_list(gates["P1_native_channel_mapping"]["canonical_order"], ["L", "R"], "hard_gates.P1_native_channel_mapping.canonical_order")
    _bool(gates["P1_native_channel_mapping"]["independent_raw_channel_calibration_required"], "hard_gates.P1_native_channel_mapping.independent_raw_channel_calibration_required", True)
    _string(gates["P1_native_channel_mapping"]["kind"], "hard_gates.P1_native_channel_mapping.kind", "relational_invariant")
    p2 = gates["P2_yaw_relative_coordinate"]
    _string(p2["kind"], "hard_gates.P2_yaw_relative_coordinate.kind", "controlled_fixed_world_source")
    _number(p2["fixed_world_source_distance_m"], "hard_gates.P2_yaw_relative_coordinate.fixed_world_source_distance_m", 4.0)
    _exact_list(p2["listener_yaws_deg"], [-90, -60, -30, 0, 30, 60, 90, 180], "hard_gates.P2_yaw_relative_coordinate.listener_yaws_deg")
    _exact_list(p2["side_relative_angles_deg"], [-90, -60, -30, 30, 60, 90], "hard_gates.P2_yaw_relative_coordinate.side_relative_angles_deg")
    _bool(p2["geometry_pose_relative_azimuth_must_match"], "hard_gates.P2_yaw_relative_coordinate.geometry_pose_relative_azimuth_must_match", True)
    _bool(p2["side_itd_sign_follows_listener_relative_direction"], "hard_gates.P2_yaw_relative_coordinate.side_itd_sign_follows_listener_relative_direction", True)
    _bool(p2["front_back_zero_or_near_zero_itd_allowed"], "hard_gates.P2_yaw_relative_coordinate.front_back_zero_or_near_zero_itd_allowed", True)
    _bool(p2["source_position_changes_with_yaw"], "hard_gates.P2_yaw_relative_coordinate.source_position_changes_with_yaw", False)
    for name, expected in (("P3_side_los_itd_direction", "relational_invariant"), ("P4_repeatability", "inherited_frozen_metric_tolerances"), ("P5_controlled_symmetry", "relational_invariant"), ("P6_near_far_los", "relational_invariant"), ("P7_nlos_applicability", "applicability_invariant"), ("P8_native16_pose_gain_preservation", "implementation_and_relational_invariants"), ("P9_native16_waveform_validity", "implementation_invariants")):
        _string(gates[name]["kind"], "hard_gates.{}.kind".format(name), expected)
    _number(gates["P3_side_los_itd_direction"]["minimum_sign_accuracy_fraction"], "hard_gates.P3_side_los_itd_direction.minimum_sign_accuracy_fraction", 0.95)
    _bool(gates["P3_side_los_itd_direction"]["global_lr_reversal_forbidden"], "hard_gates.P3_side_los_itd_direction.global_lr_reversal_forbidden", True)
    _number(gates["P4_repeatability"]["repeat_count"], "hard_gates.P4_repeatability.repeat_count", 5)
    for field, expected in (("direct_itd_spread_max_samples_at_16khz", 2.0), ("band_direct_ild_spread_max_db", 1.0), ("direct_energy_spread_max_db", 1.0), ("applicable_case_min_fraction", 0.95)):
        _number(gates["P4_repeatability"][field], "hard_gates.P4_repeatability." + field, expected)
    _exact_list(gates["P5_controlled_symmetry"]["mirror_angles_deg"], [-90, -60, -30, 30, 60, 90], "hard_gates.P5_controlled_symmetry.mirror_angles_deg")
    _exact_list(gates["P5_controlled_symmetry"]["distances_m"], [1.0, 4.0], "hard_gates.P5_controlled_symmetry.distances_m")
    for field in ("both_sides_applicable_required", "direct_itd_nonzero_required", "direct_itd_signs_opposite_required"):
        _bool(gates["P5_controlled_symmetry"][field], "hard_gates.P5_controlled_symmetry." + field, True)
    _bool(gates["P5_controlled_symmetry"]["full_rir_magnitude_symmetry_gate"], "hard_gates.P5_controlled_symmetry.full_rir_magnitude_symmetry_gate", False)
    _bool(gates["P5_controlled_symmetry"]["ild_magnitude_symmetry_gate"], "hard_gates.P5_controlled_symmetry.ild_magnitude_symmetry_gate", False)
    _bool(gates["P6_near_far_los"]["direct_energy_decreases_with_distance_required"], "hard_gates.P6_near_far_los.direct_energy_decreases_with_distance_required", True)
    _bool(gates["P6_near_far_los"]["full_window_or_wer_monotonicity_required"], "hard_gates.P6_near_far_los.full_window_or_wer_monotonicity_required", False)
    _string(gates["P7_nlos_applicability"]["direct_path_metric_policy"], "hard_gates.P7_nlos_applicability.direct_path_metric_policy", "N/A")
    _bool(gates["P7_nlos_applicability"]["na_cannot_reduce_hard_gate_denominator"], "hard_gates.P7_nlos_applicability.na_cannot_reduce_hard_gate_denominator", True)
    p8 = gates["P8_native16_pose_gain_preservation"]
    _number(p8["sample_rate_hz"], "hard_gates.P8_native16_pose_gain_preservation.sample_rate_hz", 16000)
    _string(p8["dry_fixture"], "hard_gates.P8_native16_pose_gain_preservation.dry_fixture", "deterministic_broadband_chirp_v1")
    _exact_list(p8["global_gains"], [0.25, 0.5], "hard_gates.P8_native16_pose_gain_preservation.global_gains")
    _string(p8["normalization"], "hard_gates.P8_native16_pose_gain_preservation.normalization", "none")
    _exact_list(p8["forbidden_normalization"], ["per_rir", "per_pose", "per_ear", "per_sample_rate", "loudness_equalization"], "hard_gates.P8_native16_pose_gain_preservation.forbidden_normalization")
    _bool(p8["gain_linearity_exact_required"], "hard_gates.P8_native16_pose_gain_preservation.gain_linearity_exact_required", True)
    _bool(p8["relative_pose_energy_invariance_exact_required"], "hard_gates.P8_native16_pose_gain_preservation.relative_pose_energy_invariance_exact_required", True)
    _bool(p8["pose_dependent_energy_must_not_be_equalized"], "hard_gates.P8_native16_pose_gain_preservation.pose_dependent_energy_must_not_be_equalized", True)
    p9 = gates["P9_native16_waveform_validity"]
    _number(p9["sample_rate_hz"], "hard_gates.P9_native16_waveform_validity.sample_rate_hz", 16000)
    _exact_list(p9["channel_order"], ["L", "R"], "hard_gates.P9_native16_waveform_validity.channel_order")
    _string(p9["frontend"], "hard_gates.P9_native16_waveform_validity.frontend", "mean_lr")
    _string(p9["frontend_formula"], "hard_gates.P9_native16_waveform_validity.frontend_formula", "(L+R)/2")
    _string(p9["dtype"], "hard_gates.P9_native16_waveform_validity.dtype", "float32")
    for field in ("finite_required", "full_convolution_time_axis_required", "pose_dependent_crop_or_trim_forbidden", "repeated_artifact_reproducibility_required"):
        _bool(p9[field], "hard_gates.P9_native16_waveform_validity." + field, True)
    _number(p9["max_abs_waveform_before_clipping"], "hard_gates.P9_native16_waveform_validity.max_abs_waveform_before_clipping", 1.0)
    evidence = contract["qualification_evidence"]
    if not isinstance(evidence, Mapping):
        raise OracleAlignmentError("qualification_evidence must be a mapping")
    _only(evidence, ("Q1_rays_ir_tail", "Q2_cross_rate_itd_ild", "Q3_cross_rate_raw_energy", "Q4_waveform_asr_bridge"), "qualification_evidence")
    expected_evidence = {
        "Q1_rays_ir_tail": ("qualification_evidence", "EVIDENCE_RECORDED_NOT_YET_GATED", False),
        "Q2_cross_rate_itd_ild": ("sensitivity_diagnostic", "COMPLETED_SENSITIVITY", False),
        "Q3_cross_rate_raw_energy": ("sensitivity_diagnostic", "COMPLETED_DIAGNOSTIC", False),
        "Q4_waveform_asr_bridge": ("deferred_a3_sensitivity", "DEFERRED_TO_A3", False),
    }
    for key, (role, status, _) in expected_evidence.items():
        item = evidence[key]
        if not isinstance(item, Mapping):
            raise OracleAlignmentError("qualification_evidence.{} must be a mapping".format(key))
        _only(item, ("role", "status", "numeric_tolerance_frozen", "blocks_native16_hard_gates"), "qualification_evidence." + key)
        _required(item, ("role", "status", "numeric_tolerance_frozen", "blocks_native16_hard_gates"), "qualification_evidence." + key)
        _string(item["role"], "qualification_evidence.{}.role".format(key), role)
        _string(item["status"], "qualification_evidence.{}.status".format(key), status)
        _bool(item["numeric_tolerance_frozen"], "qualification_evidence.{}.numeric_tolerance_frozen".format(key), False)
        _bool(item["blocks_native16_hard_gates"], "qualification_evidence.{}.blocks_native16_hard_gates".format(key), False)

    history = contract["historical_v2_provenance"]
    if not isinstance(history, Mapping):
        raise OracleAlignmentError("historical_v2_provenance must be a mapping")
    _only(history, ("metric_contract_sha256", "formal_failure_artifacts_immutable", "cross_rate_failure_is_current_hard_blocker", "historical_artifact_root", "historical_artifact_sha256"), "historical_v2_provenance")
    _required(history, ("metric_contract_sha256", "formal_failure_artifacts_immutable", "cross_rate_failure_is_current_hard_blocker", "historical_artifact_root", "historical_artifact_sha256"), "historical_v2_provenance")
    _string(history["metric_contract_sha256"], "historical_v2_provenance.metric_contract_sha256", "f1185d2c5fe81091199d5525f224822a3227c700c60bba9ab168e6d9ddc38f83")
    _bool(history["formal_failure_artifacts_immutable"], "historical_v2_provenance.formal_failure_artifacts_immutable", True)
    _bool(history["cross_rate_failure_is_current_hard_blocker"], "historical_v2_provenance.cross_rate_failure_is_current_hard_blocker", False)
    _string(history["historical_artifact_root"], "historical_v2_provenance.historical_artifact_root", "runs/active_asr_v1/a2_failure_attribution_run3")
    historical_hashes = history["historical_artifact_sha256"]
    if not isinstance(historical_hashes, Mapping):
        raise OracleAlignmentError("historical_v2_provenance.historical_artifact_sha256 must be a mapping")
    _only(historical_hashes, ("direction_channel_calibration.json", "a2_summary.json", "qualification_cases.jsonl"), "historical_v2_provenance.historical_artifact_sha256")
    _required(historical_hashes, ("direction_channel_calibration.json", "a2_summary.json", "qualification_cases.jsonl"), "historical_v2_provenance.historical_artifact_sha256")
    for filename, expected in {
        "direction_channel_calibration.json": "825e0cf5444ff5ac5cde3dddc69a7772462a8a0f2556941889d16140991f5958",
        "a2_summary.json": "7030391c22ab782d1da14dbcc7aa13ac04a6bb81e9d27bc437dda38002f4168d",
        "qualification_cases.jsonl": "31b4324acbae28a7e3775d3a62bbdfe2fe6016d8f5d165ef9499b97f739edc94",
    }.items():
        _string(historical_hashes[filename], "historical_v2_provenance.historical_artifact_sha256." + filename, expected)

    artifacts = contract["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise OracleAlignmentError("artifacts must be a mapping")
    _only(artifacts, ("schema_version", "yaw_relative_filename", "symmetry_filename", "pose_gain_filename", "waveform_filename", "summary_filename", "report_filename"), "artifacts")
    _required(artifacts, ("schema_version", "yaw_relative_filename", "symmetry_filename", "pose_gain_filename", "waveform_filename", "summary_filename", "report_filename"), "artifacts")
    _string(artifacts["schema_version"], "artifacts.schema_version", "active-asr-a2-oracle-alignment-artifact-v1")
    return contract


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_oracle_json(value).encode("utf-8"))


def enumerate_fixed_world_yaw_cases(receiver_sensor: Sequence[float], fixed_source: Sequence[float], listener_yaws_deg: Sequence[float]) -> List[Dict[str, Any]]:
    """Enumerate P2 without moving the world source when yaw changes."""

    source = tuple(float(value) for value in fixed_source)
    receiver = tuple(float(value) for value in receiver_sensor)
    source_distance = float(math.sqrt(sum((source[index] - receiver[index]) ** 2 for index in range(3))))
    return [{
        "case_id": "fixed_source_yaw_{}".format(str(float(yaw)).replace("-", "m").replace(".", "p")),
        "fixed_source_world": list(source),
        "source_position_world": list(source),
        "receiver_sensor_world": list(receiver),
        "listener_yaw_deg": float(yaw),
        "source_distance_m": source_distance,
        "geometry_relative_azimuth_deg": float(geometry_relative_azimuth(receiver, float(yaw), source)),
    } for yaw in listener_yaws_deg]


def _metric_summary(metric: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "applicability": metric.get("applicability"),
        "reason": metric.get("na_reason"),
        "itd_samples": metric.get("itd_samples"),
        "itd_seconds": metric.get("itd_seconds"),
        "ild_db": metric.get("ild_db"),
        "direct_energy_db": metric.get("direct_energy_db"),
        "direct_window": metric.get("direct_window"),
    }


def _array_summary(array: np.ndarray, sample_rate_hz: int, channel: Optional[str] = None) -> Dict[str, Any]:
    value = np.asarray(array)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sample_rate_hz": int(sample_rate_hz),
        "channel": channel,
        "finite": bool(np.isfinite(value).all()),
        "peak_abs": float(np.max(np.abs(value))) if value.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))) if value.size else 0.0,
        "mean_square": float(np.mean(np.square(value, dtype=np.float64))) if value.size else 0.0,
        "sum_square": float(np.sum(np.square(value, dtype=np.float64))) if value.size else 0.0,
        "sha256": _sha256_bytes(np.ascontiguousarray(value).tobytes()),
    }


def _pose_waveform_record(dry: np.ndarray, rir: np.ndarray, gain: float, contract: Mapping[str, Any], runtime_sha256: str, resource_hashes: Mapping[str, Any]) -> Dict[str, Any]:
    scaled_dry = np.asarray(dry * float(gain), dtype=np.float32)
    waveform = convolve_binaural(scaled_dry, rir)
    mean_lr = np.asarray((waveform[:, 0] + waveform[:, 1]) * 0.5, dtype=np.float32)
    swapped_mean = np.asarray((waveform[:, 1] + waveform[:, 0]) * 0.5, dtype=np.float32)
    frontend_error = float(np.max(np.abs(mean_lr - ((waveform[:, 0] + waveform[:, 1]) * 0.5))))
    swap_error = float(np.max(np.abs(mean_lr - swapped_mean)))
    output_stats = {
        "L": _array_summary(waveform[:, 0], 16000, "L"),
        "R": _array_summary(waveform[:, 1], 16000, "R"),
        "mean_lr": _array_summary(mean_lr, 16000, "mean_lr"),
    }
    return {
        "global_gain": float(gain),
        "normalization": "none",
        "time_axis": "full_convolution_sample_index_at_16khz",
        "input_dry": _array_summary(scaled_dry, 16000, "mono_dry_scaled"),
        "output": output_stats,
        "mean_lr_formula": "(L+R)/2",
        "mean_lr_max_abs_formula_error": frontend_error,
        "mean_lr_l_r_swap_max_abs_error": swap_error,
        "common_full_convolution_length": int(dry.shape[0] + rir.shape[0] - 1),
        "hard_clipping_smoke": {
            "threshold_abs": float(contract["hard_gates"]["P9_native16_waveform_validity"]["max_abs_waveform_before_clipping"]),
            "max_abs": float(np.max(np.abs(waveform))),
            "passed": bool(np.max(np.abs(waveform)) < float(contract["hard_gates"]["P9_native16_waveform_validity"]["max_abs_waveform_before_clipping"])),
        },
        "runtime_sha256": runtime_sha256,
        "resource_hashes": dict(resource_hashes),
    }


def _relative_energy_db(rows: Sequence[Mapping[str, Any]], gain: float) -> Dict[str, float]:
    center = next(row for row in rows if row["pose_id"] == "center")
    center_energy = float(center["waveforms"][str(gain)]["output"]["mean_lr"]["sum_square"])
    result = {}
    for row in rows:
        energy = float(row["waveforms"][str(gain)]["output"]["mean_lr"]["sum_square"])
        result[row["pose_id"]] = float(10.0 * math.log10(max(energy, 1.0e-30) / max(center_energy, 1.0e-30)))
    return result


def evaluate_pose_gain_invariants(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> Dict[str, Any]:
    gains = [float(value) for value in contract["hard_gates"]["P8_native16_pose_gain_preservation"]["global_gains"]]
    linearity_errors = []
    for row in rows:
        control = row.get("gain_linearity_control", {})
        linearity_errors.append(float(control.get("max_abs_waveform_error", float("inf"))))
    relative_by_gain = {str(gain): _relative_energy_db(rows, gain) for gain in gains}
    relative_differences = []
    for pose_id in relative_by_gain[str(gains[0])]:
        relative_differences.append(abs(relative_by_gain[str(gains[0])][pose_id] - relative_by_gain[str(gains[1])][pose_id]))
    pose_energies = [relative_by_gain[str(gains[0])][row["pose_id"]] for row in rows]
    invariant = contract["hard_gates"]["P8_native16_pose_gain_preservation"]
    max_linearity = float(max(linearity_errors, default=0.0))
    max_relative = float(max(relative_differences, default=0.0))
    return {
        "global_gain_linearity_max_abs_ratio_error": max_linearity,
        "global_gain_expected_ratio": gains[1] / gains[0],
        "relative_pose_energy_db_by_gain": relative_by_gain,
        "relative_pose_energy_invariance_max_abs_db": max_relative,
        "pose_energy_range_db": float(max(pose_energies, default=0.0) - min(pose_energies, default=0.0)),
        "pose_dependent_energy_variation_observed": bool((max(pose_energies, default=0.0) - min(pose_energies, default=0.0)) > POSE_ENERGY_EQUALITY_EPSILON_DB),
        "normalization": "none",
        "pose_energy_equality_epsilon_db": POSE_ENERGY_EQUALITY_EPSILON_DB,
        "passed": bool(
            invariant["gain_linearity_exact_required"]
            and invariant["relative_pose_energy_invariance_exact_required"]
            and invariant["pose_dependent_energy_must_not_be_equalized"]
            and max_linearity == 0.0
            and max_relative == 0.0
            and abs(float(max(pose_energies, default=0.0) - min(pose_energies, default=0.0))) > POSE_ENERGY_EQUALITY_EPSILON_DB
        ),
    }


def validate_yaw_relative_artifact(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "formal_denominator", "geometry_id", "fixed_source_world", "cases", "summary", "runtime_sha256", "resource_hashes", "contract_sha256"}
    if set(document) != required or document["schema_version"] != YAW_ARTIFACT_SCHEMA_VERSION:
        raise OracleAlignmentError("yaw-relative artifact schema is invalid")
    if document["formal_denominator"] is not False or document["status"] not in (PASS, "FAIL", BLOCKED):
        raise OracleAlignmentError("yaw-relative artifact status/formal denominator is invalid")
    return document


def validate_symmetry_artifact(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "formal_denominator", "geometry_id", "pairs", "summary", "source_artifact", "contract_sha256"}
    if set(document) != required or document["schema_version"] != SYMMETRY_ARTIFACT_SCHEMA_VERSION:
        raise OracleAlignmentError("symmetry artifact schema is invalid")
    if document["formal_denominator"] is not False:
        raise OracleAlignmentError("symmetry evidence is not a formal v2 denominator")
    return document


def validate_pose_gain_artifact(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "formal_denominator", "sample_rate_hz", "dry_fixture", "dry_sha256", "poses", "invariants", "runtime_sha256", "resource_hashes", "contract_sha256"}
    if set(document) != required or document["schema_version"] != POSE_GAIN_ARTIFACT_SCHEMA_VERSION:
        raise OracleAlignmentError("pose gain artifact schema is invalid")
    if document["formal_denominator"] is not False or document["sample_rate_hz"] != 16000:
        raise OracleAlignmentError("pose gain artifact is malformed")
    return document


def validate_waveform_artifact(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "formal_denominator", "sample_rate_hz", "channel_order", "frontend", "poses", "summary", "runtime_sha256", "resource_hashes", "contract_sha256"}
    if set(document) != required or document["schema_version"] != WAVEFORM_ARTIFACT_SCHEMA_VERSION:
        raise OracleAlignmentError("waveform artifact schema is invalid")
    if document["formal_denominator"] is not False or document["channel_order"] != ["L", "R"] or document["frontend"] != "mean_lr":
        raise OracleAlignmentError("waveform artifact is malformed")
    return document


def _historical_block(reason: str, source: Path, details: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"status": BLOCKED, "reason": reason, "source": str(source)}
    if details:
        result["details"] = dict(details)
    return result


def _historical_gate_reference(repo_root: Path, historical_run_dir: str, contract: Mapping[str, Any]) -> Dict[str, Any]:
    root = Path(historical_run_dir)
    if not root.is_absolute():
        root = repo_root / root
    root = root.resolve()
    history = contract["historical_v2_provenance"]
    expected_root = (repo_root / history["historical_artifact_root"]).resolve()
    if root != expected_root:
        return _historical_block("HISTORICAL_EVIDENCE_SOURCE_MISMATCH", root, {"expected_root": str(expected_root)})
    filenames = ("direction_channel_calibration.json", "a2_summary.json", "qualification_cases.jsonl")
    paths = {filename: root / filename for filename in filenames}
    missing = [filename for filename, path in paths.items() if not path.is_file()]
    if missing:
        return _historical_block("HISTORICAL_EVIDENCE_MISSING", root, {"missing": missing})
    expected_hashes = history["historical_artifact_sha256"]
    actual_hashes = {filename: _sha256_file(path) for filename, path in paths.items()}
    mismatched = {filename: {"expected": expected_hashes[filename], "actual": actual_hashes[filename]} for filename in filenames if actual_hashes[filename] != expected_hashes[filename]}
    if mismatched:
        return _historical_block("HISTORICAL_EVIDENCE_INTEGRITY_FAILURE", root, {"mismatched": mismatched})
    try:
        summary = json.loads(paths["a2_summary.json"].read_text(encoding="utf-8"))
        direction = json.loads(paths["direction_channel_calibration.json"].read_text(encoding="utf-8"))
        validate_direction_channel_calibration(direction)
        if summary.get("schema_version") != "active-asr-a2-summary-v1" or summary.get("gate") != "A2" or not isinstance(summary.get("gates"), Mapping):
            return _historical_block("HISTORICAL_EVIDENCE_SCHEMA_INVALID", root, {"artifact": "a2_summary.json"})
        for gate_name in ("direction", "repeatability", "near_far_los_direct_energy", "nlos_direct_metric_control"):
            value = summary["gates"].get(gate_name)
            if not isinstance(value, Mapping) or not {"status", "attempted", "applicable", "pass", "fail", "na"}.issubset(value):
                return _historical_block("HISTORICAL_EVIDENCE_SCHEMA_INVALID", root, {"artifact": "a2_summary.json", "gate": gate_name})
        qualification_rows = []
        for line in paths["qualification_cases.jsonl"].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            validate_qualification_case(row)
            qualification_rows.append(row)
        if not qualification_rows:
            return _historical_block("HISTORICAL_EVIDENCE_SCHEMA_INVALID", root, {"artifact": "qualification_cases.jsonl", "reason": "empty"})
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _historical_block("HISTORICAL_EVIDENCE_SCHEMA_INVALID", root, {"error": str(exc)})

    gates = summary.get("gates", {})
    def gate(name: str) -> Dict[str, Any]:
        value = gates.get(name, {})
        return {"status": value.get("status", BLOCKED), "attempted": value.get("attempted", 0), "applicable": value.get("applicable", 0), "pass": value.get("pass", 0), "fail": value.get("fail", 0), "na": value.get("na", 0)}
    h1 = direction["hypothesis_summary"].get("H1_native_ch0_L_ch1_R", {})
    h2 = direction["hypothesis_summary"].get("H2_native_ch0_R_ch1_L", {})
    direction_cases = direction.get("cases", [])
    h1_supported = sum(bool(case.get("hypotheses", {}).get("H1_native_ch0_L_ch1_R", {}).get("supports")) for case in direction_cases)
    h2_applicable = sum(case.get("raw", {}).get("applicability") == "APPLICABLE" for case in direction_cases)
    h2_pass = sum(case.get("raw", {}).get("applicability") == "APPLICABLE" and bool(case.get("hypotheses", {}).get("H2_native_ch0_R_ch1_L", {}).get("supports")) for case in direction_cases)
    p1_evidence = {
        "status": PASS,
        "applicable": h2_applicable,
        "pass": h2_pass,
        "sign_accuracy": float(h2_pass) / float(h2_applicable) if h2_applicable else None,
        "supported_hypothesis": direction["hypothesis_summary"].get("supported_hypothesis"),
        "mapping_decision": direction["hypothesis_summary"].get("mapping_decision"),
        "h1_supported_cases": h1_supported,
        "canonical_downstream_order": list(contract["hard_gates"]["P1_native_channel_mapping"]["canonical_order"]),
        "source": "direction_channel_calibration.json",
    }
    if direction.get("status") != PASS or direction["hypothesis_summary"].get("supported_hypothesis") != "H2_native_ch0_R_ch1_L" or direction["hypothesis_summary"].get("mapping_decision") != "SWAP_NATIVE_TO_CANONICAL" or h1_supported != 0 or h2_applicable != 4 or h2_pass != 4 or h2.get("applicable") != 4 or h2.get("pass") != 4 or float(h2.get("sign_accuracy", -1.0)) != 1.0 or h1.get("pass") != 0 or p1_evidence["canonical_downstream_order"] != ["L", "R"]:
        p1_evidence["status"] = BLOCKED
        return {
            "status": BLOCKED,
            "reason": "P1_HISTORICAL_EVIDENCE_INVALID",
            "source": str(root),
            "P1_native_channel_mapping": p1_evidence,
            "artifact_sha256": actual_hashes,
        }
    return {
        "status": PASS,
        "P1_native_channel_mapping": p1_evidence,
        "P3_side_los_itd_direction": gate("direction"),
        "P4_repeatability": gate("repeatability"),
        "P6_near_far_los": gate("near_far_los_direct_energy"),
        "P7_nlos_applicability": gate("nlos_direct_metric_control"),
        "source_artifact": str(paths["a2_summary.json"].relative_to(repo_root)) if paths["a2_summary.json"].is_relative_to(repo_root) else str(paths["a2_summary.json"]),
        "artifact_sha256": actual_hashes,
    }


def _run_yaw_relative(
    contract: Mapping[str, Any],
    metric_contract: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    a0_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    from active_audition.acoustics.rir import render_native_rir
    from active_audition.receiver.audit import _receiver_observation, _runtime_fingerprint
    from active_audition.scene.pose import relative_azimuth_deg as scene_relative_azimuth
    from active_audition.scene.simulator import create_scene_simulator
    from active_audition.types import ListenerPose

    primary = next(item for item in registry["geometries"] if item["kind"] == "symmetric_shoebox")
    receiver = tuple(primary["receiver"]["sensor_position_world"])
    fixed_source = source = tuple(float(receiver[index]) + (0.0, 0.0, -4.0)[index] for index in range(3))
    yaws = contract["hard_gates"]["P2_yaw_relative_coordinate"]["listener_yaws_deg"]
    cases = enumerate_fixed_world_yaw_cases(receiver, fixed_source, yaws)
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    config = copy.deepcopy(dict(runtime_config))
    config["acoustics"] = dict(runtime_config["acoustics"])
    config["acoustics"]["sample_rate_hz"] = 16000
    fingerprint: Mapping[str, Any] = {}
    try:
        with create_scene_simulator(config, scene_id=primary["id"], require_navmesh=False, load_semantic_mesh=False, scene_override={"scene_asset": primary["mesh_path"]}) as context:
            fingerprint = _runtime_fingerprint(repo_root)
            _receiver_observation(a0_contract, config, context)
            offset = np.asarray(config["listener"]["sensor_offset_m"], dtype=np.float64)
            for case in cases:
                base = tuple((np.asarray(receiver, dtype=np.float64) - offset).tolist())
                pose = ListenerPose(base_position_world=base, sensor_position_world=receiver, yaw_deg=case["listener_yaw_deg"])
                rir = render_native_rir(context, source, pose)
                metric = compute_direct_metrics(rir, 16000, metric_contract)
                projected = float(geometry_relative_azimuth(receiver, case["listener_yaw_deg"], source))
                scene_angle = float(scene_relative_azimuth(source, receiver, case["listener_yaw_deg"]))
                side = any(abs(abs(projected) - angle) <= 1.0e-6 for angle in (30.0, 60.0, 90.0))
                expected_sign = 0 if not side else (1 if projected > 0 else -1)
                observed_itd = metric.get("itd_samples")
                sign_pass = (not side and (observed_itd is None or abs(int(observed_itd)) <= 1)) or (side and observed_itd is not None and int(observed_itd) != 0 and int(observed_itd) * expected_sign > 0)
                rows.append({
                    **case,
                    "geometry_relative_azimuth_deg": projected,
                    "scene_pose_relative_azimuth_deg": scene_angle,
                    "relative_azimuth_match": abs(projected - scene_angle) <= 1.0e-9 or abs(abs(projected) - 180.0) <= 1.0e-9 and abs(abs(scene_angle) - 180.0) <= 1.0e-9,
                    "canonical_channel_order": ["L", "R"],
                    "metrics": _metric_summary(metric),
                    "expected_side_itd_sign": expected_sign,
                    "side_itd_sign_pass": bool(sign_pass),
                    "world_source_unchanged": list(source) == list(fixed_source),
                    "runtime_sha256": _sha256_json(fingerprint),
                    "contract_sha256": oracle_contract_sha256(contract),
                    "resource_hashes": {"geometry_registry": registry["registry_sha256"], "scene_asset": primary["mesh_sha256"]},
                })
    except Exception as exc:  # pragma: no cover - requires SS2 runtime
        errors.append({"type": type(exc).__name__, "message": str(exc)})
    applicable_side = [row for row in rows if any(abs(abs(float(row["geometry_relative_azimuth_deg"])) - angle) <= 1.0e-6 for angle in (30.0, 60.0, 90.0)) and row["metrics"]["applicability"] == "APPLICABLE"]
    helper_matches = all(row.get("relative_azimuth_match") for row in rows)
    source_fixed = all(row.get("world_source_unchanged") for row in rows)
    sign_pass = sum(bool(row.get("side_itd_sign_pass")) for row in applicable_side)
    signs = [int(np.sign(row["metrics"]["itd_samples"])) for row in applicable_side if row["metrics"].get("itd_samples") is not None]
    direction_changes = bool(signs and min(signs) < 0 < max(signs))
    passed = bool(rows and not errors and helper_matches and source_fixed and len(applicable_side) == 6 and sign_pass == len(applicable_side) and direction_changes)
    return {
        "schema_version": YAW_ARTIFACT_SCHEMA_VERSION,
        "status": PASS if passed else (BLOCKED if errors else "FAIL"),
        "formal_denominator": False,
        "geometry_id": primary["id"],
        "fixed_source_world": list(fixed_source),
        "cases": rows,
        "summary": {"attempted": len(rows), "applicable_side": len(applicable_side), "side_sign_pass": sign_pass, "side_sign_accuracy": sign_pass / float(len(applicable_side)) if applicable_side else None, "geometry_pose_helper_matches": helper_matches, "source_world_fixed": source_fixed, "relative_direction_changes_with_yaw": direction_changes, "errors": errors},
        "runtime_sha256": _sha256_json(fingerprint),
        "resource_hashes": {"geometry_registry": registry["registry_sha256"], "scene_asset": primary["mesh_sha256"]},
        "contract_sha256": oracle_contract_sha256(contract),
    }


def _run_symmetry(
    contract: Mapping[str, Any],
    repo_root: Path,
    historical_run_dir: str,
    historical: Mapping[str, Any],
) -> Dict[str, Any]:
    if historical.get("status") != PASS:
        return {"schema_version": SYMMETRY_ARTIFACT_SCHEMA_VERSION, "status": BLOCKED, "formal_denominator": False, "geometry_id": "a2_symmetric_shoebox_v1", "pairs": [], "summary": {"attempted": 0, "applicable": 0, "opposite_nonzero_pass": 0, "full_rir_magnitude_symmetry_gate": False, "ild_magnitude_symmetry_gate": False, "errors": [historical.get("reason", "historical_evidence_unavailable")]}, "source_artifact": historical.get("source", str(historical_run_dir)), "contract_sha256": oracle_contract_sha256(contract)}
    root = Path(historical_run_dir)
    if not root.is_absolute():
        root = repo_root / root
    path = root / "qualification_cases.jsonl"
    pairs: List[Dict[str, Any]] = []
    errors: List[str] = []
    if path.is_file():
        records = {}
        expected_hash = contract["historical_v2_provenance"]["historical_artifact_sha256"]["qualification_cases.jsonl"]
        if _sha256_file(path) != expected_hash:
            return {"schema_version": SYMMETRY_ARTIFACT_SCHEMA_VERSION, "status": BLOCKED, "formal_denominator": False, "geometry_id": "a2_symmetric_shoebox_v1", "pairs": [], "summary": {"attempted": 0, "applicable": 0, "opposite_nonzero_pass": 0, "full_rir_magnitude_symmetry_gate": False, "ild_magnitude_symmetry_gate": False, "errors": ["HISTORICAL_EVIDENCE_INTEGRITY_FAILURE"]}, "source_artifact": str(path), "contract_sha256": oracle_contract_sha256(contract)}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            validate_qualification_case(row)
            if row.get("case_type") == "controlled_qualification" and row.get("geometry_id") == "a2_symmetric_shoebox_v1" and row.get("sample_rate_hz") == 16000 and row.get("repeat_id") == 0 and row.get("geometry", {}).get("purpose") == "direction_repeatability" and row.get("ray_preset", {}).get("maxIRLength") is not None and row.get("relative_angle_deg") in (-90.0, -60.0, -30.0, 30.0, 60.0, 90.0):
                records[(float(row["distance_m"]), float(row["relative_angle_deg"]))] = row
        for distance in contract["hard_gates"]["P5_controlled_symmetry"]["distances_m"]:
            for left, right in ((-30.0, 30.0), (-60.0, 60.0), (-90.0, 90.0)):
                left_row = records.get((float(distance), left))
                right_row = records.get((float(distance), right))
                left_metric = (left_row or {}).get("raw_metrics", {}).get("native", {})
                right_metric = (right_row or {}).get("raw_metrics", {}).get("native", {})
                applicable = left_metric.get("applicability") == "APPLICABLE" and right_metric.get("applicability") == "APPLICABLE"
                left_itd = left_metric.get("itd_samples")
                right_itd = right_metric.get("itd_samples")
                relation = bool(applicable and left_itd is not None and right_itd is not None and int(left_itd) != 0 and int(right_itd) != 0 and int(left_itd) * int(right_itd) < 0)
                pairs.append({"distance_m": float(distance), "mirror_angles_deg": [left, right], "applicable": applicable, "left_itd_samples": left_itd, "right_itd_samples": right_itd, "opposite_nonzero_direct_itd": relation, "left_case_id": (left_row or {}).get("case_id"), "right_case_id": (right_row or {}).get("case_id")})
    else:
        errors.append("historical qualification_cases.jsonl missing")
    passed = bool(pairs and not errors and all(pair["applicable"] and pair["opposite_nonzero_direct_itd"] for pair in pairs))
    source = str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
    return {"schema_version": SYMMETRY_ARTIFACT_SCHEMA_VERSION, "status": PASS if passed else (BLOCKED if errors else "FAIL"), "formal_denominator": False, "geometry_id": "a2_symmetric_shoebox_v1", "pairs": pairs, "summary": {"attempted": len(pairs), "applicable": sum(pair["applicable"] for pair in pairs), "opposite_nonzero_pass": sum(pair["opposite_nonzero_direct_itd"] for pair in pairs), "full_rir_magnitude_symmetry_gate": False, "ild_magnitude_symmetry_gate": False, "errors": errors}, "source_artifact": source, "contract_sha256": oracle_contract_sha256(contract)}


def _run_pose_fixture(
    contract: Mapping[str, Any],
    metric_contract: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    a0_contract: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from active_audition.acoustics.rir import render_native_rir
    from active_audition.receiver.audit import _runtime_fingerprint
    from active_audition.scene.simulator import create_scene_simulator
    from active_audition.types import ListenerPose

    primary = next(item for item in registry["geometries"] if item["kind"] == "symmetric_shoebox")
    receiver = np.asarray(primary["receiver"]["sensor_position_world"], dtype=np.float64)
    fixed_source = tuple((receiver + np.asarray([0.0, 0.0, -4.0])).tolist())
    pose_specs = {
        "center": (tuple(receiver.tolist()), 0.0),
        "closer": (tuple((receiver + np.asarray([0.0, 0.0, -1.0])).tolist()), 0.0),
        "farther": (tuple((receiver + np.asarray([0.0, 0.0, 2.0])).tolist()), 0.0),
        "lateral_left": (tuple((receiver + np.asarray([-1.0, 0.0, 0.0])).tolist()), 0.0),
        "lateral_right": (tuple((receiver + np.asarray([1.0, 0.0, 0.0])).tolist()), 0.0),
        "yaw_only": (tuple(receiver.tolist()), 45.0),
    }
    dry = make_probe("chirp", 16000, duration_sec=0.050, seed=20260927)
    dry_sha = _sha256_bytes(np.ascontiguousarray(dry).tobytes())
    gains = [float(value) for value in contract["hard_gates"]["P8_native16_pose_gain_preservation"]["global_gains"]]
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    config = copy.deepcopy(dict(runtime_config))
    config["acoustics"] = dict(runtime_config["acoustics"])
    config["acoustics"]["sample_rate_hz"] = 16000
    fingerprint: Mapping[str, Any] = {}
    resource_hashes = {"geometry_registry": registry["registry_sha256"], "scene_asset": primary["mesh_sha256"]}
    try:
        with create_scene_simulator(config, scene_id=primary["id"], require_navmesh=False, load_semantic_mesh=False, scene_override={"scene_asset": primary["mesh_path"]}) as context:
            fingerprint = _runtime_fingerprint(repo_root)
            offset = np.asarray(config["listener"]["sensor_offset_m"], dtype=np.float64)
            for pose_id, (sensor_position, yaw) in pose_specs.items():
                sensor = tuple(float(value) for value in sensor_position)
                base = tuple((np.asarray(sensor, dtype=np.float64) - offset).tolist())
                pose = ListenerPose(base_position_world=base, sensor_position_world=sensor, yaw_deg=float(yaw))
                rir = np.asarray(render_native_rir(context, fixed_source, pose), dtype=np.float32)
                metrics = compute_direct_metrics(rir, 16000, metric_contract)
                row = {"pose_id": pose_id, "source_world": list(fixed_source), "receiver_sensor_world": list(sensor), "receiver_base_world": list(base), "listener_yaw_deg": float(yaw), "dry_sha256": dry_sha, "dry_global_gains": gains, "rir": {"shape": list(rir.shape), "dtype": str(rir.dtype), "channel_order": ["L", "R"], "finite": bool(np.isfinite(rir).all()), "sha256": _sha256_bytes(np.ascontiguousarray(rir).tobytes()), "L": _array_summary(rir[:, 0], 16000, "L"), "R": _array_summary(rir[:, 1], 16000, "R")}, "direct_rir_metrics": _metric_summary(metrics), "waveforms": {}, "runtime_sha256": _sha256_json(fingerprint), "resource_hashes": resource_hashes, "contract_sha256": oracle_contract_sha256(contract)}
                output_arrays = {}
                for gain in gains:
                    scaled_dry = np.asarray(dry * float(gain), dtype=np.float32)
                    output_arrays[str(gain)] = np.asarray(convolve_binaural(scaled_dry, rir), dtype=np.float32)
                    row["waveforms"][str(gain)] = _pose_waveform_record(dry, rir, gain, contract, _sha256_json(fingerprint), resource_hashes)
                gain_ratio = gains[1] / gains[0]
                row["gain_linearity_control"] = {
                    "lower_gain": gains[0],
                    "upper_gain": gains[1],
                    "expected_ratio": gain_ratio,
                    "max_abs_waveform_error": float(np.max(np.abs(output_arrays[str(gains[1])] - gain_ratio * output_arrays[str(gains[0])]))),
                }
                rows.append(row)
            center = next(row for row in rows if row["pose_id"] == "center")
            center_sensor = tuple(center["receiver_sensor_world"])
            center_base = tuple(center["receiver_base_world"])
            repeat_pose = ListenerPose(base_position_world=center_base, sensor_position_world=center_sensor, yaw_deg=0.0)
            repeat_rir = np.asarray(render_native_rir(context, fixed_source, repeat_pose), dtype=np.float32)
            repeat_waveform = convolve_binaural(np.asarray(dry * gains[0], dtype=np.float32), repeat_rir)
            repeat_control = {"first_rir_sha256": center["rir"]["sha256"], "repeat_rir_sha256": _sha256_bytes(np.ascontiguousarray(repeat_rir).tobytes()), "first_output_sha256": center["waveforms"][str(gains[0])]["output"]["mean_lr"]["sha256"], "repeat_output_sha256": _sha256_bytes(np.ascontiguousarray(((repeat_waveform[:, 0] + repeat_waveform[:, 1]) * 0.5).astype(np.float32)).tobytes())}
            repeat_control["passed"] = repeat_control["first_rir_sha256"] == repeat_control["repeat_rir_sha256"] and repeat_control["first_output_sha256"] == repeat_control["repeat_output_sha256"]
    except Exception as exc:  # pragma: no cover - requires SS2 runtime
        errors.append({"type": type(exc).__name__, "message": str(exc)})
        repeat_control = {"passed": False, "error": str(exc)}
    invariants = evaluate_pose_gain_invariants(rows, contract) if rows else {"passed": False, "error": "no_pose_rows"}
    invariants["repeat_identical_render_and_convolution"] = repeat_control
    p8_passed = bool(rows and not errors and invariants.get("passed") and repeat_control.get("passed"))
    p9_rows = []
    for row in rows:
        for gain in gains:
            wf = row["waveforms"][str(gain)]
            p9_rows.append({"pose_id": row["pose_id"], "global_gain": gain, "finite": all(wf["output"][key]["finite"] for key in ("L", "R", "mean_lr")), "shape_consistent": wf["output"]["L"]["shape"] == wf["output"]["R"]["shape"] == wf["output"]["mean_lr"]["shape"], "dtype_float32": all(wf["output"][key]["dtype"] == "float32" for key in ("L", "R", "mean_lr")), "full_convolution_length": wf["output"]["L"]["shape"][0] == wf["common_full_convolution_length"], "mean_lr_exact": wf["mean_lr_max_abs_formula_error"] == 0.0, "mean_lr_swap_invariant": wf["mean_lr_l_r_swap_max_abs_error"] == 0.0, "no_clipping": wf["hard_clipping_smoke"]["passed"], "normalization": wf["normalization"]})
    p9_passed = bool(p9_rows and all(all(bool(row[key]) for key in ("finite", "shape_consistent", "dtype_float32", "full_convolution_length", "mean_lr_exact", "mean_lr_swap_invariant", "no_clipping")) and row["normalization"] == "none" for row in p9_rows) and repeat_control.get("passed"))
    runtime_hash = _sha256_json(fingerprint)
    pose_gain = {"schema_version": POSE_GAIN_ARTIFACT_SCHEMA_VERSION, "status": PASS if p8_passed else (BLOCKED if errors else "FAIL"), "formal_denominator": False, "sample_rate_hz": 16000, "dry_fixture": {"name": "deterministic_broadband_chirp_v1", "duration_sec": 0.050, "dtype": "float32", "global_gain_values": gains}, "dry_sha256": dry_sha, "poses": rows, "invariants": invariants, "runtime_sha256": runtime_hash, "resource_hashes": resource_hashes, "contract_sha256": oracle_contract_sha256(contract)}
    waveform = {"schema_version": WAVEFORM_ARTIFACT_SCHEMA_VERSION, "status": PASS if p9_passed else (BLOCKED if errors else "FAIL"), "formal_denominator": False, "sample_rate_hz": 16000, "channel_order": ["L", "R"], "frontend": "mean_lr", "poses": p9_rows, "summary": {"attempted": len(p9_rows), "finite_pass": sum(row["finite"] for row in p9_rows), "mean_lr_formula_pass": sum(row["mean_lr_exact"] for row in p9_rows), "swap_invariance_pass": sum(row["mean_lr_swap_invariant"] for row in p9_rows), "no_clipping_pass": sum(row["no_clipping"] for row in p9_rows), "repeat_control": repeat_control, "errors": errors}, "runtime_sha256": runtime_hash, "resource_hashes": resource_hashes, "contract_sha256": oracle_contract_sha256(contract)}
    validate_pose_gain_artifact(pose_gain)
    validate_waveform_artifact(waveform)
    return pose_gain, waveform


def run_oracle_alignment(
    oracle_contract_path: str,
    output_dir: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
    historical_run_dir: str = "runs/active_asr_v1/a2_failure_attribution_run3",
) -> Dict[str, Any]:
    contract = load_oracle_contract(oracle_contract_path)
    metric_contract_path = Path(oracle_contract_path).resolve().parent / "metric_contract.yaml"
    metric_contract = load_metric_contract(str(metric_contract_path))
    if metric_contract_sha256(metric_contract) != contract["contract"]["parent_metric_contract_sha256"]:
        raise OracleAlignmentError("historical v2 metric contract SHA does not match v3 parent")
    from active_audition.config.loader import load_resolved_config as load_legacy_config
    from active_audition.config.v1 import load_resolved_config as load_a0_contract

    runtime_config = load_legacy_config(runtime_config_path)
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    a0_contract = load_a0_contract(str(repo_root / "configs/active_audition/v1/experiment_contract.yaml"))
    geometry_path = repo_root / metric_contract["qualification_geometry"]["registry_path"]
    registry = load_geometry_registry(str(geometry_path), str(repo_root))
    primary = next(item for item in registry["geometries"] if item["kind"] == "symmetric_shoebox")
    sanity = geometry_sanity(primary)
    historical = _historical_gate_reference(repo_root, historical_run_dir, contract)
    yaw = _run_yaw_relative(contract, metric_contract, runtime_config, registry, a0_contract)
    symmetry = _run_symmetry(contract, repo_root, historical_run_dir, historical)
    pose_gain, waveform = _run_pose_fixture(contract, metric_contract, runtime_config, registry, a0_contract)
    validate_yaw_relative_artifact(yaw)
    validate_symmetry_artifact(symmetry)
    p_status = {
        "P1_native_channel_mapping": historical.get("P1_native_channel_mapping", {"status": BLOCKED}),
        "P2_yaw_relative_coordinate": {"status": yaw["status"], "attempted": yaw["summary"]["attempted"], "applicable": yaw["summary"]["applicable_side"], "pass": yaw["summary"]["side_sign_pass"]},
        "P3_side_los_itd_direction": historical.get("P3_side_los_itd_direction", {"status": BLOCKED}),
        "P4_repeatability": historical.get("P4_repeatability", {"status": BLOCKED}),
        "P5_controlled_symmetry": {"status": symmetry["status"], "attempted": symmetry["summary"]["attempted"], "applicable": symmetry["summary"]["applicable"], "pass": symmetry["summary"]["opposite_nonzero_pass"]},
        "P6_near_far_los": historical.get("P6_near_far_los", {"status": BLOCKED}),
        "P7_nlos_applicability": historical.get("P7_nlos_applicability", {"status": BLOCKED}),
        "P8_native16_pose_gain_preservation": {"status": pose_gain["status"], "attempted": len(pose_gain["poses"]), "applicable": len(pose_gain["poses"]), "pass": len(pose_gain["poses"]) if pose_gain["status"] == PASS else 0},
        "P9_native16_waveform_validity": {"status": waveform["status"], "attempted": waveform["summary"]["attempted"], "applicable": waveform["summary"]["attempted"], "pass": waveform["summary"]["no_clipping_pass"] if waveform["status"] == PASS else 0},
    }
    all_hard_pass = all(value.get("status") == PASS for value in p_status.values())
    summary = {
        "schema_version": SUMMARY_ARTIFACT_SCHEMA_VERSION,
        "gate": "A2",
        "status": "READY_FOR_INDEPENDENT_RECHECK" if all_hard_pass else "OPEN/BLOCKED",
        "server_run_status": "SERVER_RUN_PASS" if all_hard_pass else "SERVER_RUN_FAIL",
        "a2_closed": False,
        "a3_entered": False,
        "contract": {"path": str(Path(oracle_contract_path).resolve()), "sha256": oracle_contract_sha256(contract), "v2_parent_sha256": contract["contract"]["parent_metric_contract_sha256"]},
        "geometry": {"id": primary["id"], "registry_sha256": registry["registry_sha256"], "mesh_sha256": primary["mesh_sha256"], "sanity": {"all_sources_inside": sanity["all_sources_inside"], "all_direct_windows_clear": sanity["all_direct_windows_clear"]}},
        "hard_gates": p_status,
        "qualification_evidence": {"Q1": {"status": "EVIDENCE_RECORDED_NOT_YET_GATED"}, "Q2": {"status": "COMPLETED_SENSITIVITY", "historical_artifact_preserved": True}, "Q3": {"status": "COMPLETED_DIAGNOSTIC", "historical_artifact_preserved": True}, "Q4": {"status": "DEFERRED_TO_A3"}},
        "historical_v2": {"source": historical.get("source_artifact", historical.get("source")), "integrity_status": historical.get("status", BLOCKED), "reason": historical.get("reason"), "artifact_sha256": historical.get("artifact_sha256", contract["historical_v2_provenance"]["historical_artifact_sha256"]), "metric_contract_sha256": contract["contract"]["parent_metric_contract_sha256"], "current_hard_blocker": False, "artifacts_immutable": True},
        "artifacts": {},
        "blocker": None if all_hard_pass else historical.get("reason", "NATIVE16_ORACLE_ALIGNED_HARD_GATE_FAILURE"),
    }
    output_root = Path(output_dir).resolve()
    storage = DatasetStorage(str(output_root))
    storage.atomic_write_json(output_root / "yaw_relative_qualification.json", yaw)
    storage.atomic_write_json(output_root / "controlled_symmetry_qualification.json", symmetry)
    storage.atomic_write_json(output_root / "native16_pose_gain_preservation.json", pose_gain)
    storage.atomic_write_json(output_root / "native16_waveform_validity.json", waveform)
    for name in ("yaw_relative_qualification.json", "controlled_symmetry_qualification.json", "native16_pose_gain_preservation.json", "native16_waveform_validity.json"):
        summary["artifacts"][name] = {"path": name, "sha256": _sha256_file(output_root / name)}
    report = [
        "# Active-ASR V1.1 A2 Oracle-Aligned Native16 Qualification",
        "",
        "- Status: **{}**".format(summary["status"]),
        "- Contract SHA256: `{}`".format(summary["contract"]["sha256"]),
        "- Historical v2 cross-rate failure remains preserved and is not the current blocker.",
        "- P2 yaw-relative: `{}`; P5 controlled symmetry: `{}`; P8 gain preservation: `{}`; P9 waveform validity: `{}`.".format(yaw["status"], symmetry["status"], pose_gain["status"], waveform["status"]),
        "- Q1: `EVIDENCE_RECORDED_NOT_YET_GATED`; Q2: `COMPLETED_SENSITIVITY`; Q3: `COMPLETED_DIAGNOSTIC`; Q4: `DEFERRED_TO_A3`.",
        "- A2 remains open; this run does not authorize A3.",
        "",
    ]
    storage.atomic_write_text(output_root / "a2_oracle_alignment_report.md", "\n".join(report))
    summary["artifacts"]["a2_oracle_alignment_report.md"] = {"path": "a2_oracle_alignment_report.md", "sha256": _sha256_file(output_root / "a2_oracle_alignment_report.md")}
    # The summary cannot contain its own SHA256 without a self-referential
    # fixed point.  It records every other generated artifact, including the
    # report, and callers hash the summary file itself externally.
    storage.atomic_write_json(output_root / "a2_oracle_alignment_summary.json", summary)
    return summary


__all__ = [
    "ORACLE_CONTRACT_SCHEMA_VERSION",
    "OracleAlignmentError",
    "canonical_oracle_json",
    "enumerate_fixed_world_yaw_cases",
    "load_oracle_contract",
    "oracle_contract_sha256",
    "run_oracle_alignment",
    "validate_oracle_contract",
    "validate_pose_gain_artifact",
    "validate_symmetry_artifact",
    "validate_waveform_artifact",
    "validate_yaw_relative_artifact",
]
