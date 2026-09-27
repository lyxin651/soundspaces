"""A2 metric contract, synthetic fixtures, and bounded qualification runner.

This module is intentionally separate from the A0 contract and from the A1
authority audit.  It can compute the frozen direct-window metrics without
Habitat, and it refuses to turn an ordinary scene into a physics
qualification when no authoritative controlled geometry is registered.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.signal import fftconvolve

from active_audition.acoustics.resampling import resample_array
from active_audition.data.storage import DatasetStorage, json_line
from active_audition.receiver.geometry import (
    GeometryRegistryError,
    MIRROR_PAIRS,
    RELATIVE_AZIMUTHS_DEG,
    enumerate_controlled_cases,
    geometry_sanity,
    load_geometry_registry,
    relative_azimuth_deg as geometry_relative_azimuth,
    source_position_world,
)


METRIC_CONTRACT_SCHEMA_VERSION = "active-asr-a2-metric-v2"
QUALIFICATION_CASE_SCHEMA_VERSION = "active-asr-a2-qualification-case-v2"
A2_RUNTIME_LOCK_SCHEMA_VERSION = "active-asr-a2-runtime-lock-v1"
ENERGY_METRIC_CONVENTION_SCHEMA_VERSION = "active-asr-a2-energy-metric-convention-v1"
RIR_SAMPLE_RATE_CONVENTION_SCHEMA_VERSION = "active-asr-a2-rir-sample-rate-convention-v1"
FRACTIONAL_DELAY_CALIBRATION_SCHEMA_VERSION = "active-asr-a2-fractional-delay-estimator-v1"
SAMPLE_RATE_ATTRIBUTION_SCHEMA_VERSION = "active-asr-a2-sample-rate-attribution-v2"
KNOWN = "KNOWN"
NA = "N/A"
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
EVIDENCE_RECORDED = "EVIDENCE_RECORDED"
FORMAL_GATE_NAMES = ("direction", "repeatability", "sample_rate_ab", "ray_tail_convergence", "mirror_symmetry_direct", "near_far_los_direct_energy")
HARD_GATE_NAMES = ("direction", "repeatability", "sample_rate_ab", "near_far_los_direct_energy")


class QualificationError(ValueError):
    """Raised when a metric contract or qualification artifact is invalid."""


def _path(path: str, key: Any) -> str:
    return "{}.{}".format(path, key) if path else str(key)


def _require_mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise QualificationError("{} must be a mapping".format(path))
    return value


def _only_keys(value: Mapping, keys: Iterable[str], path: str) -> None:
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise QualificationError("unknown metric contract field(s) at {}: {}".format(path, ", ".join(unknown)))


def _require_keys(value: Mapping, keys: Iterable[str], path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise QualificationError("missing metric contract field(s) at {}: {}".format(path, ", ".join(missing)))


def _string(value: Any, path: str, expected: Optional[str] = None) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError("{} must be a non-empty string".format(path))
    if expected is not None and value != expected:
        raise QualificationError("{} must be {!r}".format(path, expected))
    return value


def _bool(value: Any, path: str, expected: Optional[bool] = None) -> bool:
    if not isinstance(value, bool):
        raise QualificationError("{} must be boolean".format(path))
    if expected is not None and value is not expected:
        raise QualificationError("{} must be {}".format(path, str(expected).lower()))
    return value


def _int(value: Any, path: str, expected: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationError("{} must be an integer".format(path))
    if expected is not None and value != expected:
        raise QualificationError("{} must be {}".format(path, expected))
    return value


def _number(value: Any, path: str, expected: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError("{} must be numeric".format(path))
    if not math.isfinite(float(value)):
        raise QualificationError("{} must be finite".format(path))
    if expected is not None and float(value) != float(expected):
        raise QualificationError("{} must be {}".format(path, expected))
    return float(value)


def _list(value: Any, path: str, length: Optional[int] = None) -> List[Any]:
    if not isinstance(value, list):
        raise QualificationError("{} must be a list".format(path))
    if length is not None and len(value) != length:
        raise QualificationError("{} must contain {} values".format(path, length))
    return value


def _exact_list(value: Any, expected: List[Any], path: str) -> None:
    if _list(value, path) != expected:
        raise QualificationError("{} must be {!r}".format(path, expected))


def _validate_band(value: Mapping, path: str) -> None:
    _only_keys(value, ("id", "low_hz", "high_hz"), path)
    _require_keys(value, ("id", "low_hz", "high_hz"), path)
    _string(value["id"], _path(path, "id"))
    low = _number(value["low_hz"], _path(path, "low_hz"))
    high = _number(value["high_hz"], _path(path, "high_hz"))
    if low < 0.0 or high <= low:
        raise QualificationError("{} must have 0 <= low_hz < high_hz".format(path))


def validate_metric_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Strictly validate the versioned A2 metric contract."""

    root = _require_mapping(contract, "root")
    top = (
        "contract",
        "serialization",
        "sample_rate",
        "direct_window",
        "metrics",
        "frequency_bands",
        "probes",
        "qualification_geometry",
        "gates",
        "artifacts",
    )
    _only_keys(root, top, "root")
    _require_keys(root, top, "root")

    metadata = _require_mapping(root["contract"], "contract")
    _only_keys(metadata, ("namespace", "version", "gate", "state"), "contract")
    _require_keys(metadata, ("namespace", "version", "gate", "state"), "contract")
    _string(metadata["namespace"], "contract.namespace", "active-asr")
    _string(metadata["version"], "contract.version", METRIC_CONTRACT_SCHEMA_VERSION)
    _string(metadata["gate"], "contract.gate", "A2")
    _string(metadata["state"], "contract.state", "FROZEN")

    serialization = _require_mapping(root["serialization"], "serialization")
    _only_keys(serialization, ("version", "hash_algorithm"), "serialization")
    _require_keys(serialization, ("version", "hash_algorithm"), "serialization")
    _string(serialization["version"], "serialization.version", "canonical-json-v1")
    _string(serialization["hash_algorithm"], "serialization.hash_algorithm", "sha256")

    sample_rate = _require_mapping(root["sample_rate"], "sample_rate")
    _only_keys(
        sample_rate,
        (
            "primary_native_hz",
            "qualification_reference_native_hz",
            "qualification_target_hz",
            "resampler",
            "anti_aliasing_required",
            "common_band_hz",
            "separate_normalization",
        ),
        "sample_rate",
    )
    _require_keys(
        sample_rate,
        (
            "primary_native_hz",
            "qualification_reference_native_hz",
            "qualification_target_hz",
            "resampler",
            "anti_aliasing_required",
            "common_band_hz",
            "separate_normalization",
        ),
        "sample_rate",
    )
    _int(sample_rate["primary_native_hz"], "sample_rate.primary_native_hz", 16000)
    _int(sample_rate["qualification_reference_native_hz"], "sample_rate.qualification_reference_native_hz", 24000)
    _int(sample_rate["qualification_target_hz"], "sample_rate.qualification_target_hz", 16000)
    _string(sample_rate["resampler"], "sample_rate.resampler", "resample_poly")
    _bool(sample_rate["anti_aliasing_required"], "sample_rate.anti_aliasing_required", True)
    _exact_list(sample_rate["common_band_hz"], [0, 8000], "sample_rate.common_band_hz")
    _bool(sample_rate["separate_normalization"], "sample_rate.separate_normalization", False)

    window = _require_mapping(root["direct_window"], "direct_window")
    window_keys = (
        "search_start_sec",
        "search_duration_sec",
        "onset_threshold_fraction_of_channel_peak",
        "minimum_peak_amplitude",
        "minimum_energy",
        "length_sec",
        "start_rule",
        "end_rule",
        "onset_rule",
        "low_energy_policy",
        "no_onset_policy",
    )
    _only_keys(window, window_keys, "direct_window")
    _require_keys(window, window_keys, "direct_window")
    for key in ("search_start_sec", "search_duration_sec", "onset_threshold_fraction_of_channel_peak", "minimum_peak_amplitude", "minimum_energy", "length_sec"):
        if _number(window[key], _path("direct_window", key)) <= 0.0 and key != "search_start_sec":
            raise QualificationError("direct_window.{} must be positive".format(key))
    _number(window["search_start_sec"], "direct_window.search_start_sec", 0.0)
    _string(window["start_rule"], "direct_window.start_rule", "minimum_channel_onset")
    _string(window["end_rule"], "direct_window.end_rule", "maximum_channel_onset_plus_length")
    _string(window["onset_rule"], "direct_window.onset_rule", "first_absolute_sample_at_or_above_threshold")
    _string(window["low_energy_policy"], "direct_window.low_energy_policy", NA)
    _string(window["no_onset_policy"], "direct_window.no_onset_policy", NA)

    metrics = _require_mapping(root["metrics"], "metrics")
    _only_keys(metrics, ("itd", "ild", "energy", "tail"), "metrics")
    _require_keys(metrics, ("itd", "ild", "energy", "tail"), "metrics")
    itd = _require_mapping(metrics["itd"], "metrics.itd")
    _only_keys(itd, ("definition", "time_reference", "units", "sign_positive", "conversion"), "metrics.itd")
    _require_keys(itd, ("definition", "time_reference", "units", "sign_positive", "conversion"), "metrics.itd")
    _string(itd["definition"], "metrics.itd.definition", "tR_minus_tL")
    _string(itd["time_reference"], "metrics.itd.time_reference", "direct_window_channel_onset")
    _string(itd["units"], "metrics.itd.units", "samples_at_native_rate")
    _string(itd["sign_positive"], "metrics.itd.sign_positive", "right_onset_later_than_left")
    _string(itd["conversion"], "metrics.itd.conversion", "samples_divided_by_sample_rate_hz")
    ild = _require_mapping(metrics["ild"], "metrics.ild")
    _only_keys(ild, ("definition", "energy_definition", "units", "sign_positive", "epsilon_power"), "metrics.ild")
    _require_keys(ild, ("definition", "energy_definition", "units", "sign_positive", "epsilon_power"), "metrics.ild")
    _string(ild["definition"], "metrics.ild.definition", "10log10(E_L_divided_by_E_R)")
    _string(ild["energy_definition"], "metrics.ild.energy_definition", "sum_of_squared_samples_in_common_direct_window")
    _string(ild["units"], "metrics.ild.units", "dB")
    _string(ild["sign_positive"], "metrics.ild.sign_positive", "left_energy_greater_than_right_energy")
    _number(ild["epsilon_power"], "metrics.ild.epsilon_power", 1.0e-12)
    energy = _require_mapping(metrics["energy"], "metrics.energy")
    _only_keys(energy, ("definition", "units", "db_definition", "epsilon_power"), "metrics.energy")
    _require_keys(energy, ("definition", "units", "db_definition", "epsilon_power"), "metrics.energy")
    _string(energy["definition"], "metrics.energy.definition", "sum_of_squared_samples")
    _string(energy["units"], "metrics.energy.units", "sample_squared")
    _string(energy["db_definition"], "metrics.energy.db_definition", "10log10(energy)")
    _number(energy["epsilon_power"], "metrics.energy.epsilon_power", 1.0e-12)
    tail = _require_mapping(metrics["tail"], "metrics.tail")
    _only_keys(tail, ("name", "definition", "interval", "units", "diagnostic_only", "prohibited_aliases"), "metrics.tail")
    _require_keys(tail, ("name", "definition", "interval", "units", "diagnostic_only", "prohibited_aliases"), "metrics.tail")
    _string(tail["name"], "metrics.tail.name", "post_direct_tail_energy_relative_to_direct")
    _string(tail["definition"], "metrics.tail.definition", "10log10(E_post_direct_tail_divided_by_E_direct)")
    _string(tail["interval"], "metrics.tail.interval", "direct_window_end_to_ir_end")
    _string(tail["units"], "metrics.tail.units", "dB")
    _bool(tail["diagnostic_only"], "metrics.tail.diagnostic_only", True)
    _exact_list(tail["prohibited_aliases"], ["DRR", "RT60", "rir_tail_ratio_100ms"], "metrics.tail.prohibited_aliases")

    bands = _require_mapping(root["frequency_bands"], "frequency_bands")
    _only_keys(bands, ("transform", "edge_rule", "bands"), "frequency_bands")
    _require_keys(bands, ("transform", "edge_rule", "bands"), "frequency_bands")
    _string(bands["transform"], "frequency_bands.transform", "rfft_parseval_energy")
    _string(bands["edge_rule"], "frequency_bands.edge_rule", "low_inclusive_high_exclusive")
    band_values = _list(bands["bands"], "frequency_bands.bands")
    if len(band_values) != 4:
        raise QualificationError("frequency_bands.bands must contain four bands")
    for index, band in enumerate(band_values):
        _validate_band(_require_mapping(band, "frequency_bands.bands[{}]".format(index)), "frequency_bands.bands[{}]".format(index))
    if [band["id"] for band in band_values] != ["full_0_8000", "low_0_1000", "mid_1000_4000", "high_4000_8000"]:
        raise QualificationError("frequency_bands.bands ids are not frozen")

    probes = _require_mapping(root["probes"], "probes")
    _only_keys(probes, ("required", "auxiliary", "rir_metric_reference", "fixture_seed"), "probes")
    _require_keys(probes, ("required", "auxiliary", "rir_metric_reference", "fixture_seed"), "probes")
    _exact_list(probes["required"], ["impulse", "broadband_noise", "chirp"], "probes.required")
    _exact_list(probes["auxiliary"], ["voiced_tone"], "probes.auxiliary")
    _string(probes["rir_metric_reference"], "probes.rir_metric_reference", "impulse_response")
    _int(probes["fixture_seed"], "probes.fixture_seed", 20260923)

    geometry = _require_mapping(root["qualification_geometry"], "qualification_geometry")
    _only_keys(
        geometry,
        (
            "registry_path",
            "authoritative_registry_required",
            "selection_rule",
            "relative_angles_deg",
            "line_of_sight_distance_m",
            "scene_conditions",
            "nlos_direct_metric_policy",
            "mirror_yaw_scope",
            "front_back_itd_policy",
            "low_frequency_ild_policy",
        ),
        "qualification_geometry",
    )
    _require_keys(
        geometry,
        (
            "registry_path",
            "authoritative_registry_required",
            "selection_rule",
            "relative_angles_deg",
            "line_of_sight_distance_m",
            "scene_conditions",
            "nlos_direct_metric_policy",
            "mirror_yaw_scope",
            "front_back_itd_policy",
            "low_frequency_ild_policy",
        ),
        "qualification_geometry",
    )
    _string(geometry["registry_path"], "qualification_geometry.registry_path")
    _bool(geometry["authoritative_registry_required"], "qualification_geometry.authoritative_registry_required", True)
    _string(geometry["selection_rule"], "qualification_geometry.selection_rule")
    _exact_list(geometry["relative_angles_deg"], [0, -30, 30, -60, 60, -90, 90, 180], "qualification_geometry.relative_angles_deg")
    _exact_list(geometry["line_of_sight_distance_m"], [1.0, 4.0], "qualification_geometry.line_of_sight_distance_m")
    _exact_list(geometry["scene_conditions"], ["LOS", "NLOS"], "qualification_geometry.scene_conditions")
    _string(geometry["nlos_direct_metric_policy"], "qualification_geometry.nlos_direct_metric_policy", NA)
    _string(geometry["mirror_yaw_scope"], "qualification_geometry.mirror_yaw_scope", "direct_dominant_or_explicitly_symmetric_only")
    _string(geometry["front_back_itd_policy"], "qualification_geometry.front_back_itd_policy", "zero_or_near_zero_is_valid")
    _string(geometry["low_frequency_ild_policy"], "qualification_geometry.low_frequency_ild_policy", "near_zero_is_valid")

    gates = _require_mapping(root["gates"], "gates")
    _only_keys(gates, ("direction", "repeatability", "sample_rate_ab", "convergence"), "gates")
    _require_keys(gates, ("direction", "repeatability", "sample_rate_ab", "convergence"), "gates")
    direction = _require_mapping(gates["direction"], "gates.direction")
    _only_keys(direction, ("side_los_angles_deg", "itd_sign_accuracy_min_fraction", "global_lr_reversal_forbidden"), "gates.direction")
    _require_keys(direction, ("side_los_angles_deg", "itd_sign_accuracy_min_fraction", "global_lr_reversal_forbidden"), "gates.direction")
    _exact_list(direction["side_los_angles_deg"], [-90, -60, -30, 30, 60, 90], "gates.direction.side_los_angles_deg")
    _number(direction["itd_sign_accuracy_min_fraction"], "gates.direction.itd_sign_accuracy_min_fraction", 0.95)
    _bool(direction["global_lr_reversal_forbidden"], "gates.direction.global_lr_reversal_forbidden", True)
    repeat = _require_mapping(gates["repeatability"], "gates.repeatability")
    _only_keys(repeat, ("repeat_count", "direct_itd_spread_max_samples_at_16khz", "band_direct_ild_spread_max_db", "direct_energy_spread_max_db", "applicable_case_min_fraction", "low_energy_is_na"), "gates.repeatability")
    _require_keys(repeat, ("repeat_count", "direct_itd_spread_max_samples_at_16khz", "band_direct_ild_spread_max_db", "direct_energy_spread_max_db", "applicable_case_min_fraction", "low_energy_is_na"), "gates.repeatability")
    _int(repeat["repeat_count"], "gates.repeatability.repeat_count", 5)
    _number(repeat["direct_itd_spread_max_samples_at_16khz"], "gates.repeatability.direct_itd_spread_max_samples_at_16khz", 2.0)
    _number(repeat["band_direct_ild_spread_max_db"], "gates.repeatability.band_direct_ild_spread_max_db", 1.0)
    _number(repeat["direct_energy_spread_max_db"], "gates.repeatability.direct_energy_spread_max_db", 1.0)
    _number(repeat["applicable_case_min_fraction"], "gates.repeatability.applicable_case_min_fraction", 0.95)
    _bool(repeat["low_energy_is_na"], "gates.repeatability.low_energy_is_na", True)
    sample_ab = _require_mapping(gates["sample_rate_ab"], "gates.sample_rate_ab")
    _only_keys(sample_ab, ("native_rate_hz", "reference_native_rate_hz", "resampled_rate_hz", "common_band_hz", "itd_difference_max_samples", "band_ild_difference_max_db", "band_energy_difference_max_db", "applicable_case_min_fraction", "per_render_normalization_forbidden"), "gates.sample_rate_ab")
    _require_keys(sample_ab, ("native_rate_hz", "reference_native_rate_hz", "resampled_rate_hz", "common_band_hz", "itd_difference_max_samples", "band_ild_difference_max_db", "band_energy_difference_max_db", "applicable_case_min_fraction", "per_render_normalization_forbidden"), "gates.sample_rate_ab")
    _int(sample_ab["native_rate_hz"], "gates.sample_rate_ab.native_rate_hz", 16000)
    _int(sample_ab["reference_native_rate_hz"], "gates.sample_rate_ab.reference_native_rate_hz", 24000)
    _int(sample_ab["resampled_rate_hz"], "gates.sample_rate_ab.resampled_rate_hz", 16000)
    _exact_list(sample_ab["common_band_hz"], [0, 8000], "gates.sample_rate_ab.common_band_hz")
    _number(sample_ab["itd_difference_max_samples"], "gates.sample_rate_ab.itd_difference_max_samples", 2.0)
    _number(sample_ab["band_ild_difference_max_db"], "gates.sample_rate_ab.band_ild_difference_max_db", 1.0)
    _number(sample_ab["band_energy_difference_max_db"], "gates.sample_rate_ab.band_energy_difference_max_db", 1.0)
    _number(sample_ab["applicable_case_min_fraction"], "gates.sample_rate_ab.applicable_case_min_fraction", 0.95)
    _bool(sample_ab["per_render_normalization_forbidden"], "gates.sample_rate_ab.per_render_normalization_forbidden", True)
    convergence = _require_mapping(gates["convergence"], "gates.convergence")
    _only_keys(convergence, ("ray_count_multiplier", "ir_length_multiplier", "compare_metrics", "samplewise_identity_required"), "gates.convergence")
    _require_keys(convergence, ("ray_count_multiplier", "ir_length_multiplier", "compare_metrics", "samplewise_identity_required"), "gates.convergence")
    _number(convergence["ray_count_multiplier"], "gates.convergence.ray_count_multiplier", 2.0)
    _number(convergence["ir_length_multiplier"], "gates.convergence.ir_length_multiplier", 2.0)
    _exact_list(convergence["compare_metrics"], ["direct_delay", "itd", "band_ild", "direct_energy", "band_energy", "post_direct_tail_energy"], "gates.convergence.compare_metrics")
    _bool(convergence["samplewise_identity_required"], "gates.convergence.samplewise_identity_required", False)

    artifacts = _require_mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, ("schema_version", "cases_filename", "sample_rate_filename", "summary_filename", "report_filename"), "artifacts")
    _require_keys(artifacts, ("schema_version", "cases_filename", "sample_rate_filename", "summary_filename", "report_filename"), "artifacts")
    _string(artifacts["schema_version"], "artifacts.schema_version", "active-asr-a2-artifact-v1")
    _string(artifacts["cases_filename"], "artifacts.cases_filename", "qualification_cases.jsonl")
    _string(artifacts["sample_rate_filename"], "artifacts.sample_rate_filename", "sample_rate_ab.json")
    _string(artifacts["summary_filename"], "artifacts.summary_filename", "a2_summary.json")
    _string(artifacts["report_filename"], "artifacts.report_filename", "a2_report.md")
    return contract


def _canonical_value(value: Any, path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key], _path(path, key)) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, "{}[{}]".format(path, index)) for index, item in enumerate(value)]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QualificationError("{} must not contain NaN or infinity".format(path))
        return 0.0 if value == 0.0 else value
    raise QualificationError("{} contains unsupported value type {}".format(path, type(value).__name__))


def canonical_json(value: Mapping[str, Any]) -> str:
    validate_metric_contract(value)
    return json.dumps(_canonical_value(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def metric_contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_metric_contract(path: str) -> Dict[str, Any]:
    metric_path = Path(path)
    try:
        with metric_path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise QualificationError("unable to load metric contract {}: {}".format(metric_path, exc)) from exc
    if not isinstance(value, dict):
        raise QualificationError("metric contract root must be a mapping")
    return dict(validate_metric_contract(value))


def _finite_rir(rir: Any) -> np.ndarray:
    array = np.asarray(rir, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise QualificationError("RIR must have shape (N, 2), N > 0")
    if not np.isfinite(array).all():
        raise QualificationError("RIR contains NaN/Inf")
    return array


def _db_power(value: float, epsilon: float) -> Optional[float]:
    if value <= epsilon:
        return None
    return float(10.0 * math.log10(value))


def _band_energy(signal: np.ndarray, sample_rate_hz: int, low_hz: float, high_hz: float) -> float:
    n = int(signal.shape[0])
    spectrum = np.fft.rfft(signal.astype(np.float64), n=n)
    frequencies = np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz))
    selected = (frequencies >= float(low_hz)) & (frequencies < float(high_hz))
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        return 0.0
    values = np.square(np.abs(spectrum[indices]))
    # Parseval energy for a one-sided real FFT.  This is an energy measure,
    # not a per-signal normalization; the A/B comparison therefore preserves
    # absolute gain differences.
    weights = np.ones(indices.size, dtype=np.float64) * 2.0
    if n % 2 == 0:
        weights[frequencies[indices] == 0.0] = 1.0
        weights[frequencies[indices] == float(sample_rate_hz) / 2.0] = 1.0
    else:
        weights[frequencies[indices] == 0.0] = 1.0
    return float(np.sum(values * weights) / float(n))


def _direct_window(rir: np.ndarray, sample_rate_hz: int, contract: Mapping[str, Any]) -> Dict[str, Any]:
    window_contract = contract["direct_window"]
    start = max(0, int(round(float(window_contract["search_start_sec"]) * sample_rate_hz)))
    stop = min(rir.shape[0], start + int(round(float(window_contract["search_duration_sec"]) * sample_rate_hz)))
    if stop <= start:
        return {"applicability": NA, "reason": "DIRECT_SEARCH_WINDOW_EMPTY"}
    search = np.abs(rir[start:stop, :])
    peaks = np.max(search, axis=0)
    energy = float(np.sum(np.square(search)))
    if energy <= float(window_contract["minimum_energy"]) or np.max(peaks) < float(window_contract["minimum_peak_amplitude"]):
        return {"applicability": NA, "reason": "LOW_ENERGY"}
    threshold_fraction = float(window_contract["onset_threshold_fraction_of_channel_peak"])
    onsets: List[int] = []
    for channel in range(2):
        threshold = max(float(peaks[channel]) * threshold_fraction, float(window_contract["minimum_peak_amplitude"]))
        candidates = np.flatnonzero(search[:, channel] >= threshold)
        if candidates.size == 0:
            return {"applicability": NA, "reason": "NO_DIRECT_ONSET_CHANNEL_{}".format(channel)}
        onsets.append(start + int(candidates[0]))
    length = max(1, int(round(float(window_contract["length_sec"]) * sample_rate_hz)))
    direct_start = min(onsets)
    direct_end = min(rir.shape[0], max(onsets) + length)
    if direct_end <= direct_start:
        return {"applicability": NA, "reason": "DIRECT_WINDOW_EMPTY"}
    return {
        "applicability": "APPLICABLE",
        "reason": None,
        "search_start_sample": start,
        "search_end_sample_exclusive": stop,
        "channel_onset_samples": {"L": onsets[0], "R": onsets[1]},
        "start_sample": direct_start,
        "end_sample_exclusive": direct_end,
        "length_samples": direct_end - direct_start,
    }


def compute_direct_metrics(rir: Any, sample_rate_hz: int, contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute only the frozen A2 direct-window metrics.

    The returned ITD is always ``onset_R - onset_L``.  No old whole-RIR lag,
    last-100-ms ratio, DRR, or RT60 name is used here.
    """

    validate_metric_contract(contract)
    if int(sample_rate_hz) <= 0:
        raise QualificationError("sample_rate_hz must be positive")
    array = _finite_rir(rir)
    window = _direct_window(array, int(sample_rate_hz), contract)
    result: Dict[str, Any] = {
        "sample_rate_hz": int(sample_rate_hz),
        "num_samples": int(array.shape[0]),
        "direct_window": window,
        "itd_samples": None,
        "itd_seconds": None,
        "ild_db": None,
        "direct_energy_left": None,
        "direct_energy_right": None,
        "direct_energy_total": None,
        "direct_energy_db": None,
        "bands": {},
        "post_direct_tail_energy_db": None,
    }
    if window["applicability"] != "APPLICABLE":
        result["applicability"] = NA
        result["na_reason"] = window["reason"]
        return result
    start = int(window["start_sample"])
    end = int(window["end_sample_exclusive"])
    direct = array[start:end, :]
    left_energy = float(np.sum(np.square(direct[:, 0])))
    right_energy = float(np.sum(np.square(direct[:, 1])))
    total_energy = left_energy + right_energy
    epsilon = float(contract["metrics"]["energy"]["epsilon_power"])
    onsets = window["channel_onset_samples"]
    result["itd_samples"] = int(onsets["R"] - onsets["L"])
    result["itd_seconds"] = float(result["itd_samples"] / float(sample_rate_hz))
    result["direct_energy_left"] = left_energy
    result["direct_energy_right"] = right_energy
    result["direct_energy_total"] = total_energy
    result["direct_energy_db"] = _db_power(total_energy, epsilon)
    if left_energy > epsilon and right_energy > epsilon:
        result["ild_db"] = float(10.0 * math.log10(left_energy / right_energy))

    for band in contract["frequency_bands"]["bands"]:
        band_id = str(band["id"])
        left_band = _band_energy(direct[:, 0], int(sample_rate_hz), float(band["low_hz"]), float(band["high_hz"]))
        right_band = _band_energy(direct[:, 1], int(sample_rate_hz), float(band["low_hz"]), float(band["high_hz"]))
        band_result: Dict[str, Any] = {
            "low_hz": float(band["low_hz"]),
            "high_hz": float(band["high_hz"]),
            "energy_left": left_band,
            "energy_right": right_band,
            "energy_total": left_band + right_band,
            "energy_db": _db_power(left_band + right_band, epsilon),
            "ild_db": float(10.0 * math.log10(left_band / right_band)) if left_band > epsilon and right_band > epsilon else None,
        }
        result["bands"][band_id] = band_result

    tail = array[end:, :]
    tail_energy = float(np.sum(np.square(tail))) if tail.size else 0.0
    if tail_energy > epsilon and total_energy > epsilon:
        result["post_direct_tail_energy_db"] = float(10.0 * math.log10(tail_energy / total_energy))
    result["applicability"] = "APPLICABLE"
    result["na_reason"] = None
    return result


def make_probe(kind: str, sample_rate_hz: int, duration_sec: float = 0.050, seed: int = 20260923) -> np.ndarray:
    """Return a deterministic probe waveform for the A2 probe contract."""

    kind = str(kind)
    n = max(1, int(round(float(duration_sec) * int(sample_rate_hz))))
    if kind == "impulse":
        result = np.zeros(n, dtype=np.float32)
        result[0] = 1.0
        return result
    rng = np.random.default_rng(int(seed))
    time = np.arange(n, dtype=np.float64) / float(sample_rate_hz)
    if kind == "broadband_noise":
        return np.asarray(rng.standard_normal(n), dtype=np.float32)
    if kind == "chirp":
        phase = 2.0 * math.pi * (200.0 * time + 0.5 * (float(sample_rate_hz) * 0.45 - 200.0) / max(time[-1], 1.0 / sample_rate_hz) * np.square(time))
        return np.asarray(np.sin(phase), dtype=np.float32)
    if kind == "voiced_tone":
        # A deterministic synthetic auxiliary probe.  It is not natural
        # speech, LibriSpeech, an ASR input, or a source registry entry.
        return np.asarray(0.6 * np.sin(2.0 * math.pi * 180.0 * time) + 0.2 * np.sin(2.0 * math.pi * 360.0 * time), dtype=np.float32)
    raise QualificationError("unsupported A2 probe: {}".format(kind))


def _known_impulse_fixture(delay_samples: int, left_gain: float, right_gain: float, sample_rate_hz: int = 16000) -> np.ndarray:
    result = np.zeros((2048, 2), dtype=np.float32)
    left_index = 400
    right_index = left_index + int(delay_samples)
    if right_index < 0 or right_index >= result.shape[0]:
        raise QualificationError("synthetic delay is outside fixture")
    result[left_index, 0] = float(left_gain)
    result[right_index, 1] = float(right_gain)
    # Keep the diagnostic tail outside the frozen direct window so the known
    # gain ratio remains exact in both positive- and negative-delay fixtures.
    result[min(result.shape[0] - 1, max(left_index, right_index) + 200), :] = np.asarray([0.02, 0.01], dtype=np.float32)
    return result


def run_synthetic_metric_fixtures(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Run known delay/gain, N/A, and resampling convention fixtures."""

    expected_ild = 10.0 * math.log10((1.0 ** 2) / (0.5 ** 2))
    positive = compute_direct_metrics(_known_impulse_fixture(6, 1.0, 0.5), 16000, contract)
    negative = compute_direct_metrics(_known_impulse_fixture(-6, 0.5, 1.0), 16000, contract)
    silent = compute_direct_metrics(np.zeros((256, 2), dtype=np.float32), 16000, contract)

    reference = np.zeros((960, 2), dtype=np.float32)
    reference[240, 0] = 0.75
    reference[246, 1] = 0.50
    resampled = resample_array(reference, 24000, 16000)
    peak_indices = np.argmax(np.abs(resampled), axis=0)
    peak_values = np.max(np.abs(resampled), axis=0)
    resampling_fixture = {
        "status": PASS if abs(int(peak_indices[1]) - int(peak_indices[0]) - 4) <= 1 and abs(float(peak_values[0] / peak_values[1]) - 1.5) <= 0.02 else FAIL,
        "source_rate_hz": 24000,
        "target_rate_hz": 16000,
        "expected_peak_delay_samples_at_target": 4,
        "observed_peak_indices": [int(peak_indices[0]), int(peak_indices[1])],
        "observed_peak_gain_ratio_left_divided_by_right": float(peak_values[0] / peak_values[1]),
        "separate_normalization": False,
    }
    checks = {
        "positive_itd": int(positive["itd_samples"] or 0) == 6,
        "positive_ild": positive["ild_db"] is not None and abs(float(positive["ild_db"]) - expected_ild) < 1.0e-6,
        "negative_itd": int(negative["itd_samples"] or 0) == -6,
        "negative_ild": negative["ild_db"] is not None and abs(float(negative["ild_db"]) + expected_ild) < 1.0e-6,
        "low_energy_na": silent["applicability"] == NA and silent["na_reason"] == "LOW_ENERGY",
        "resampling_delay_and_gain": resampling_fixture["status"] == PASS,
    }
    return {
        "status": PASS if all(checks.values()) else FAIL,
        "checks": checks,
        "positive_delay_gain": positive,
        "negative_delay_gain": negative,
        "low_energy": silent,
        "resampling_convention": resampling_fixture,
        "probe_support": {kind: {"status": PASS, "num_samples": int(make_probe(kind, 16000).shape[0])} for kind in ("impulse", "broadband_noise", "chirp", "voiced_tone")},
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(_canonical_value(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _resource_hashes(scene: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    for role, record in sorted(scene.get("resources", {}).items()):
        provenance = record.get("registry_provenance", {})
        result[str(role)] = provenance.get("sha256")
    return result


def _metric_comparison(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "direct_delay_difference_samples": None,
        "itd_difference_samples": None,
        "direct_energy_difference_db": None,
        "post_direct_tail_energy_difference_db": None,
        "bands": {},
    }
    if first.get("applicability") != "APPLICABLE" or second.get("applicability") != "APPLICABLE":
        result["applicability"] = NA
        result["reason"] = "ONE_OR_BOTH_METRICS_NA"
        return result
    first_window = first["direct_window"]
    second_window = second["direct_window"]
    result["direct_delay_difference_samples"] = abs(int(first_window["start_sample"]) - int(second_window["start_sample"]))
    result["itd_difference_samples"] = abs(int(first["itd_samples"]) - int(second["itd_samples"]))
    if first.get("direct_energy_db") is not None and second.get("direct_energy_db") is not None:
        result["direct_energy_difference_db"] = abs(float(first["direct_energy_db"]) - float(second["direct_energy_db"]))
    if first.get("post_direct_tail_energy_db") is not None and second.get("post_direct_tail_energy_db") is not None:
        result["post_direct_tail_energy_difference_db"] = abs(float(first["post_direct_tail_energy_db"]) - float(second["post_direct_tail_energy_db"]))
    for band_id in sorted(set(first.get("bands", {})) & set(second.get("bands", {}))):
        first_band = first["bands"][band_id]
        second_band = second["bands"][band_id]
        result["bands"][band_id] = {
            "ild_difference_db": abs(float(first_band["ild_db"]) - float(second_band["ild_db"])) if first_band.get("ild_db") is not None and second_band.get("ild_db") is not None else None,
            "energy_difference_db": abs(float(first_band["energy_db"]) - float(second_band["energy_db"])) if first_band.get("energy_db") is not None and second_band.get("energy_db") is not None else None,
        }
    result["applicability"] = "APPLICABLE"
    result["reason"] = None
    return result


def _case_row(
    *,
    case_id: str,
    case_type: str,
    geometry_id: Optional[str],
    geometry: Mapping[str, Any],
    source_transform: Mapping[str, Any],
    receiver_transform: Mapping[str, Any],
    line_of_sight: str,
    relative_angle_deg: Optional[float],
    distance_m: Optional[float],
    probe: str,
    probe_domain: str,
    probe_reference: str,
    sample_rate_hz: int,
    ray_preset: Mapping[str, Any],
    ir_length_sec: Optional[float],
    repeat_id: int,
    applicability: str,
    raw_metrics: Optional[Mapping[str, Any]],
    status: str,
    reason: Optional[str],
    runtime_sha256: Optional[str],
    contract_sha256: str,
    resource_hashes: Mapping[str, Optional[str]],
) -> Dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_CASE_SCHEMA_VERSION,
        "case_id": str(case_id),
        "case_type": str(case_type),
        "geometry_id": geometry_id,
        "geometry": dict(geometry),
        "source_transform": dict(source_transform),
        "receiver_transform": dict(receiver_transform),
        "line_of_sight": str(line_of_sight),
        "relative_angle_deg": relative_angle_deg,
        "distance_m": distance_m,
        "probe": str(probe),
        "probe_domain": str(probe_domain),
        "probe_reference": str(probe_reference),
        "sample_rate_hz": int(sample_rate_hz),
        "ray_preset": dict(ray_preset),
        "ir_length_sec": ir_length_sec,
        "repeat_id": int(repeat_id),
        "applicability": str(applicability),
        "raw_metrics": None if raw_metrics is None else dict(raw_metrics),
        "status": str(status),
        "reason": reason,
        "runtime_sha256": runtime_sha256,
        "contract_sha256": contract_sha256,
        "resource_hashes": dict(resource_hashes),
    }


def validate_qualification_case(row: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "case_id",
        "case_type",
        "geometry_id",
        "geometry",
        "source_transform",
        "receiver_transform",
        "line_of_sight",
        "relative_angle_deg",
        "distance_m",
        "probe",
        "probe_domain",
        "probe_reference",
        "sample_rate_hz",
        "ray_preset",
        "ir_length_sec",
        "repeat_id",
        "applicability",
        "raw_metrics",
        "status",
        "reason",
        "runtime_sha256",
        "contract_sha256",
        "resource_hashes",
    }
    if set(row) != required:
        raise QualificationError("qualification case schema keys do not match A2 contract")
    if row["schema_version"] != QUALIFICATION_CASE_SCHEMA_VERSION:
        raise QualificationError("qualification case schema version is invalid")
    if row["applicability"] not in ("APPLICABLE", NA):
        raise QualificationError("qualification case applicability is invalid")
    if row["status"] not in (PASS, FAIL, NA, BLOCKED):
        raise QualificationError("qualification case status is invalid")
    if row["probe_domain"] != "rir" or row["probe_reference"] != "impulse_response" or row["probe"] != "impulse_response":
        raise QualificationError("formal A2 RIR case must identify probe/reference as impulse_response")
    return row


def validate_a2_runtime_lock(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "gate",
        "status",
        "contract_sha256",
        "runtime_fingerprint",
        "receiver_effective",
        "resource_hashes",
        "geometry_registry_reason",
    }
    if set(document) != required:
        raise QualificationError("runtime.lock.json schema keys do not match A2 contract")
    if document["schema_version"] != A2_RUNTIME_LOCK_SCHEMA_VERSION or document["gate"] != "A2":
        raise QualificationError("runtime.lock.json has an invalid A2 schema or gate")
    if document["status"] not in (PASS, BLOCKED):
        raise QualificationError("runtime.lock.json status is invalid")
    if not isinstance(document["runtime_fingerprint"], Mapping):
        raise QualificationError("runtime.lock.json runtime_fingerprint must be a mapping")
    return document


def validate_sample_rate_ab(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "applicability", "reason", "normalization"}
    if not required.issubset(set(document)):
        raise QualificationError("sample_rate_ab.json is missing A2 schema keys")
    if document["schema_version"] != "active-asr-a2-sample-rate-ab-v1":
        raise QualificationError("sample_rate_ab.json schema version is invalid")
    if document["status"] not in (PASS, FAIL, BLOCKED):
        raise QualificationError("sample_rate_ab.json status is invalid")
    if document["applicability"] not in ("APPLICABLE", NA):
        raise QualificationError("sample_rate_ab.json applicability is invalid")
    if document["normalization"] != "none":
        raise QualificationError("sample_rate_ab.json must record no separate normalization")
    return document


def validate_direction_channel_calibration(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "formal_denominator", "cases", "errors", "hypothesis_summary"}
    if set(document) != required and set(document) != required | {"convention_audit"}:
        raise QualificationError("direction channel calibration schema keys do not match A2 contract")
    if document["schema_version"] != "active-asr-a2-direction-channel-calibration-v1":
        raise QualificationError("direction channel calibration schema version is invalid")
    if document["formal_denominator"] is not False:
        raise QualificationError("direction channel calibration must not enter the formal denominator")
    if not isinstance(document["cases"], list) or not isinstance(document["hypothesis_summary"], Mapping):
        raise QualificationError("direction channel calibration cases/summary are invalid")
    return document


def validate_probe_domain_diagnostics(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "domain", "rir_reference", "formal_denominator", "probes"}
    if set(document) != required:
        raise QualificationError("probe domain diagnostic schema keys do not match A2 contract")
    if document["schema_version"] != "active-asr-a2-probe-domain-diagnostic-v1" or document["domain"] != "waveform_diagnostic":
        raise QualificationError("probe domain diagnostic schema/version is invalid")
    if document["formal_denominator"] is not False:
        raise QualificationError("probe domain diagnostics must not enter the formal denominator")
    if not isinstance(document["probes"], list):
        raise QualificationError("probe domain diagnostics probes must be a list")
    return document


def validate_failure_attribution(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "formal_a2_status", "direction", "relative_angle_audit", "sample_rate_ab", "probe_domains", "gate_semantics"}
    if set(document) != required:
        raise QualificationError("failure attribution schema keys do not match A2 contract")
    if document["schema_version"] != "active-asr-a2-failure-attribution-v1":
        raise QualificationError("failure attribution schema version is invalid")
    if not isinstance(document["sample_rate_ab"], Mapping) or not isinstance(document["gate_semantics"], Mapping):
        raise QualificationError("failure attribution sections must be mappings")
    return document


def _load_geometry_registry(contract: Mapping[str, Any], repo_root: Path) -> Tuple[Optional[Path], Optional[str]]:
    relative = Path(str(contract["qualification_geometry"]["registry_path"]))
    path = relative if relative.is_absolute() else repo_root / relative
    try:
        load_geometry_registry(str(path), str(repo_root))
    except GeometryRegistryError as exc:
        return path.resolve(), str(exc)
    return path.resolve(), None


def run_resampler_band_calibration(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Measure the frozen resampler before any real sample-rate A/B case."""

    sample_rate = int(contract["sample_rate"]["qualification_reference_native_hz"])
    target_rate = int(contract["sample_rate"]["qualification_target_hz"])
    duration_sec = 1.0
    n = int(sample_rate * duration_sec)
    time = np.arange(n, dtype=np.float64) / float(sample_rate)
    bands = {
        "low_0_1000": (250.0, 750.0),
        "mid_1000_4000": (1500.0, 3500.0),
        "high_4000_8000": (4500.0, 7500.0),
    }
    rng = np.random.default_rng(int(contract["probes"]["fixture_seed"]))
    noise = rng.standard_normal(n)
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    results: Dict[str, Any] = {}
    for band_id, (low, high) in bands.items():
        tone = np.sin(2.0 * math.pi * (low + high) * 0.5 * time)
        masked = np.where((frequencies >= low) & (frequencies < high), spectrum, 0.0)
        band_noise = np.fft.irfft(masked, n=n)
        def measure(signal: np.ndarray) -> Tuple[float, float]:
            output = resample_array(signal.astype(np.float32), sample_rate, target_rate)
            source_slice = signal[int(0.15 * sample_rate):int(0.85 * sample_rate)]
            output_slice = output[int(0.15 * target_rate):int(0.85 * target_rate)]
            source_rms = float(np.sqrt(np.mean(np.square(source_slice, dtype=np.float64))))
            output_rms = float(np.sqrt(np.mean(np.square(output_slice, dtype=np.float64))))
            amplitude_db = float(20.0 * math.log10(max(output_rms, 1.0e-30) / max(source_rms, 1.0e-30)))
            energy_db = float(10.0 * math.log10(max(output_rms * output_rms, 1.0e-30) / max(source_rms * source_rms, 1.0e-30)))
            return amplitude_db, energy_db
        tone_amplitude_db, tone_energy_db = measure(tone)
        noise_amplitude_db, noise_energy_db = measure(band_noise)
        results[band_id] = {
            "band_hz": [low, high],
            "tone_frequency_hz": (low + high) * 0.5,
            "tone_gain_bias_db": tone_amplitude_db,
            "tone_energy_bias_db": tone_energy_db,
            "band_limited_noise_gain_bias_db": noise_amplitude_db,
            "band_limited_noise_energy_bias_db": noise_energy_db,
            "max_absolute_energy_bias_db": max(abs(tone_energy_db), abs(noise_energy_db)),
        }
    max_bias = max(value["max_absolute_energy_bias_db"] for value in results.values())
    return {
        "schema_version": "active-asr-a2-resampler-calibration-v1",
        "status": PASS if max_bias <= float(contract["gates"]["sample_rate_ab"]["band_energy_difference_max_db"]) else FAIL,
        "source_rate_hz": sample_rate,
        "target_rate_hz": target_rate,
        "algorithm": contract["sample_rate"]["resampler"],
        "separate_normalization": False,
        "fixture": "deterministic_sinusoid_and_fft_band_limited_noise",
        "bands": results,
        "max_absolute_energy_bias_db": max_bias,
        "frozen_gate_reference_db": contract["gates"]["sample_rate_ab"]["band_energy_difference_max_db"],
        "decision": "PROCEED_TO_FORMAL_SAMPLE_RATE_AB" if max_bias <= float(contract["gates"]["sample_rate_ab"]["band_energy_difference_max_db"]) else "STOP_AND_REVIEW_METRIC_CONTRACT",
    }


def energy_metric_convention_audit() -> Dict[str, Any]:
    """Record the two energy conventions used by A2 diagnostics.

    The existing formal metric is deliberately not changed here.  In
    particular, ``sum(x**2)`` is a discrete coefficient energy in sample
    squared units, whereas the earlier resampler calibration measured RMS /
    mean-square gain.  They are related but are not interchangeable when the
    sample count changes with sample rate.
    """

    return {
        "schema_version": ENERGY_METRIC_CONVENTION_SCHEMA_VERSION,
        "formal_rir_metric": {
            "name": "sum_squared_samples",
            "formula": "sum_n(x[n]^2)",
            "mean_square_formula": "sum_n(x[n]^2)/N (not used by formal gate)",
            "db_formula": "10*log10(sum_n(x[n]^2))",
            "units": "sample_squared",
            "band_formula": "one_sided_rfft_parseval_sum_squared_samples",
        },
        "resampler_calibration_metric": {
            "name": "RMS_and_mean_square_gain",
            "rms_formula": "sqrt(mean_n(x[n]^2))",
            "mean_square_formula": "mean_n(x[n]^2)",
            "db_formula": "20*log10(RMS_out/RMS_in) = 10*log10(MS_out/MS_in)",
            "units": "signal_amplitude_ratio_db",
        },
        "optional_sample_rate_weighted_diagnostic": {
            "formula": "sum_n(x[n]^2)/sample_rate_hz",
            "interpretation": "discrete approximation to a continuous-time energy integral",
            "formal_gate": False,
        },
        "same_quantity": False,
        "formal_gate_uses_rms_calibration": False,
        "conclusion": "RMS/mean-square resampler calibration cannot by itself exclude a RIR coefficient sum-square conversion or sample-rate convention effect.",
    }


def validate_energy_metric_convention(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "formal_rir_metric", "resampler_calibration_metric", "optional_sample_rate_weighted_diagnostic", "same_quantity", "formal_gate_uses_rms_calibration", "conclusion"}
    if set(document) != required or document["schema_version"] != ENERGY_METRIC_CONVENTION_SCHEMA_VERSION:
        raise QualificationError("energy metric convention artifact schema is invalid")
    if document["same_quantity"] is not False or document["formal_gate_uses_rms_calibration"] is not False:
        raise QualificationError("energy metric convention artifact must keep formal and RMS metrics distinct")
    return document


def _db_ratio(first: float, second: float, power: bool = True) -> Optional[float]:
    if first <= 1.0e-30 or second <= 1.0e-30:
        return None
    multiplier = 10.0 if power else 20.0
    return float(multiplier * math.log10(first / second))


def _signal_stats(signal: np.ndarray, sample_rate_hz: int, contract: Mapping[str, Any]) -> Dict[str, Any]:
    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise QualificationError("calibration signal is empty or non-finite")
    sum_square = float(np.sum(np.square(array)))
    mean_square = float(np.mean(np.square(array)))
    stats = {
        "num_samples": int(array.size),
        "sample_rate_hz": int(sample_rate_hz),
        "peak_abs": float(np.max(np.abs(array))),
        "rms": float(math.sqrt(mean_square)),
        "mean_square": mean_square,
        "sum_square": sum_square,
        "sum_h2": sum_square,
        "sample_rate_weighted_integral": float(sum_square / float(sample_rate_hz)),
        "bands": {},
    }
    for band in contract["frequency_bands"]["bands"]:
        band_id = str(band["id"])
        power = _band_energy(array, int(sample_rate_hz), float(band["low_hz"]), float(band["high_hz"]))
        stats["bands"][band_id] = {
            "low_hz": float(band["low_hz"]),
            "high_hz": float(band["high_hz"]),
            "sum_square": float(power),
            "mean_square": float(power / float(array.size)),
            "power_db": _db_power(power, 1.0e-30),
        }
    return stats


def _compact_signal_metric(signal: np.ndarray, sample_rate_hz: int, contract: Mapping[str, Any]) -> Dict[str, Any]:
    metric = compute_direct_metrics(signal, int(sample_rate_hz), contract)
    if metric.get("applicability") != "APPLICABLE":
        return {"applicability": NA, "reason": metric.get("na_reason")}
    return {
        "applicability": "APPLICABLE",
        "onset_samples": dict(metric["direct_window"]["channel_onset_samples"]),
        "onset_seconds": {key: float(value / float(sample_rate_hz)) for key, value in metric["direct_window"]["channel_onset_samples"].items()},
        "itd_samples": int(metric["itd_samples"]),
        "itd_seconds": float(metric["itd_seconds"]),
        "ild_db": None if metric.get("ild_db") is None else float(metric["ild_db"]),
    }


def _rate_matched_band_source(sample_rate_hz: int, duration_sec: float = 0.025) -> np.ndarray:
    """Deterministic continuous-time multi-tone source used only for A/B/C."""

    n = int(round(float(sample_rate_hz) * float(duration_sec)))
    time = np.arange(n, dtype=np.float64) / float(sample_rate_hz)
    frequencies = (250.0, 750.0, 1500.0, 3500.0, 4500.0, 7500.0)
    amplitudes = (0.18, 0.16, 0.13, 0.11, 0.09, 0.07)
    phases = (0.1, 0.7, 1.2, 2.0, 2.8, 0.4)
    signal = np.zeros(n, dtype=np.float64)
    for frequency, amplitude, phase in zip(frequencies, amplitudes, phases):
        signal += amplitude * np.sin(2.0 * math.pi * frequency * time + phase)
    return np.asarray(signal, dtype=np.float32)


def _convolve_binaural(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
    return np.column_stack([fftconvolve(source, rir[:, channel], mode="full") for channel in range(2)]).astype(np.float32)


def _compact_rir_stats(rir: np.ndarray, sample_rate_hz: int, contract: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sample_rate_hz": int(sample_rate_hz),
        "channels": {key: _signal_stats(rir[:, index], sample_rate_hz, contract) for index, key in enumerate(("L", "R"))},
    }


def _compare_output_stats(first: np.ndarray, second: np.ndarray, sample_rate_hz: int, contract: Mapping[str, Any]) -> Dict[str, Any]:
    first_stats = [_signal_stats(first[:, index], sample_rate_hz, contract) for index in range(2)]
    second_stats = [_signal_stats(second[:, index], sample_rate_hz, contract) for index in range(2)]
    result: Dict[str, Any] = {"channels": {}, "metrics": {}}
    for index, key in enumerate(("L", "R")):
        result["channels"][key] = {
            "rms_difference_db": _db_ratio(second_stats[index]["rms"], first_stats[index]["rms"], power=False),
            "mean_square_difference_db": _db_ratio(second_stats[index]["mean_square"], first_stats[index]["mean_square"]),
            "sum_square_difference_db": _db_ratio(second_stats[index]["sum_square"], first_stats[index]["sum_square"]),
            "sample_rate_weighted_integral_difference_db": _db_ratio(second_stats[index]["sample_rate_weighted_integral"], first_stats[index]["sample_rate_weighted_integral"]),
        }
    for band in contract["frequency_bands"]["bands"]:
        band_id = str(band["id"])
        first_power = sum(item["bands"][band_id]["sum_square"] for item in first_stats)
        second_power = sum(item["bands"][band_id]["sum_square"] for item in second_stats)
        result["metrics"][band_id] = {"power_difference_db": _db_ratio(second_power, first_power)}
    return result


def _metric_difference(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    if first.get("applicability") != "APPLICABLE" or second.get("applicability") != "APPLICABLE":
        return {"applicability": NA, "reason": "ONE_OR_BOTH_OUTPUT_METRICS_NA"}
    return {
        "applicability": "APPLICABLE",
        "itd_difference_samples": abs(int(first["itd_samples"]) - int(second["itd_samples"])),
        "itd_difference_seconds": abs(float(first["itd_seconds"]) - float(second["itd_seconds"])),
        "ild_difference_db": None if first.get("ild_db") is None or second.get("ild_db") is None else abs(float(first["ild_db"]) - float(second["ild_db"])),
    }


def _formal_energy_offset_attribution(formal_sample_rate_path: Optional[str]) -> Dict[str, Any]:
    """Summarize preserved v2 formal energy differences without rewriting them."""

    result: Dict[str, Any] = {
        "schema_version": "active-asr-a2-formal-energy-offset-attribution-v1",
        "source_artifact": str(formal_sample_rate_path) if formal_sample_rate_path else None,
        "source_available": False,
        "bands": {},
        "by_distance_angle": [],
        "global_offset_assessment": {},
    }
    if not formal_sample_rate_path or not Path(formal_sample_rate_path).is_file():
        result["status"] = "NOT_AVAILABLE"
        return result
    document = json.loads(Path(formal_sample_rate_path).read_text(encoding="utf-8"))
    rows = document.get("comparisons", [])
    values: Dict[str, List[float]] = {}
    for row in rows:
        comparison = row.get("comparison", {})
        for band_id, band in comparison.get("bands", {}).items():
            if band.get("energy_difference_db") is not None:
                values.setdefault(str(band_id), []).append(float(band["energy_difference_db"]))
        result["by_distance_angle"].append({
            "distance_m": row.get("distance_m"),
            "angle_deg": row.get("angle_deg"),
            "band_energy_difference_db": {str(k): v.get("energy_difference_db") for k, v in comparison.get("bands", {}).items()},
            "direct_energy_difference_db": comparison.get("direct_energy_difference_db"),
        })
    for band_id, band_values in sorted(values.items()):
        result["bands"][band_id] = {
            "count": len(band_values),
            "mean_db": float(np.mean(band_values)),
            "std_db": float(np.std(band_values)),
            "min_db": float(np.min(band_values)),
            "max_db": float(np.max(band_values)),
            "spread_db": float(np.max(band_values) - np.min(band_values)),
        }
    all_values = [value for values_for_band in values.values() for value in values_for_band]
    result["source_available"] = True
    result["status"] = "ATTRIBUTION_RECORDED"
    result["global_offset_assessment"] = {
        "all_band_rows_mean_db": float(np.mean(all_values)) if all_values else None,
        "all_band_rows_std_db": float(np.std(all_values)) if all_values else None,
        "global_constant_candidate": bool(all_values and float(np.std(all_values)) <= 0.10),
        "interpretation": "A global constant is only a descriptive test; no RIR gain correction is adopted from formal rows.",
    }
    return result


def _fractional_delay_signal(sample_rate_hz: int, delay_sec: float, gain: float, duration_sec: float = 0.030) -> np.ndarray:
    n = int(round(float(sample_rate_hz) * float(duration_sec)))
    time = np.arange(n, dtype=np.float64) / float(sample_rate_hz)
    sigma = 0.00016
    pulse = np.exp(-0.5 * np.square((time - float(delay_sec)) / sigma))
    return np.asarray(float(gain) * pulse, dtype=np.float32)


def run_fractional_delay_estimator_calibration(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Calibrate the existing 10%-peak onset estimator on rate-matched pulses."""

    rows: List[Dict[str, Any]] = []
    delays_l = (0.004125, 0.0041875, 0.00425, 0.0043125, 0.004375, 0.0044375, 0.0045)
    itd_seconds = (-0.00028125, -0.00015625, 0.0000, 0.00015625, 0.00028125)
    gains = ((1.0, 0.5), (0.5, 1.0), (1.75, 0.35))
    max_channel_shift = 0.0
    max_itd_shift = 0.0
    max_absolute_channel_error = 0.0
    for delay_l in delays_l:
        for itd in itd_seconds:
            for left_gain, right_gain in gains:
                delay_r = float(delay_l + itd)
                if delay_r <= 0.0:
                    continue
                native16 = np.column_stack([
                    _fractional_delay_signal(16000, delay_l, left_gain),
                    _fractional_delay_signal(16000, delay_r, right_gain),
                ])
                native24 = np.column_stack([
                    _fractional_delay_signal(24000, delay_l, left_gain),
                    _fractional_delay_signal(24000, delay_r, right_gain),
                ])
                resampled = resample_array(native24, 24000, 16000)
                metric16 = _compact_signal_metric(native16, 16000, contract)
                metric24_resampled = _compact_signal_metric(resampled, 16000, contract)
                if metric16["applicability"] != "APPLICABLE" or metric24_resampled["applicability"] != "APPLICABLE":
                    continue
                true_onsets = {"L": delay_l * 16000.0, "R": delay_r * 16000.0}
                rate_channel_shifts = {
                    key: float(metric24_resampled["onset_samples"][key] - metric16["onset_samples"][key])
                    for key in ("L", "R")
                }
                absolute_errors = {
                    key: float(metric16["onset_samples"][key] - true_onsets[key])
                    for key in ("L", "R")
                }
                itd_true = itd * 16000.0
                row = {
                    "delay_left_sec": float(delay_l),
                    "delay_right_sec": delay_r,
                    "gain_left": float(left_gain),
                    "gain_right": float(right_gain),
                    "true_onset_samples_at_16khz": true_onsets,
                    "true_itd_samples_at_16khz": float(itd_true),
                    "native16": metric16,
                    "native24_to_16": metric24_resampled,
                    "per_channel_onset_shift_samples_rate_comparison": rate_channel_shifts,
                    "per_channel_absolute_estimator_error_samples_native16": absolute_errors,
                    "itd_shift_samples_rate_comparison": float(metric24_resampled["itd_samples"] - metric16["itd_samples"]),
                    "itd_absolute_error_samples_native16": float(metric16["itd_samples"] - itd_true),
                }
                rows.append(row)
                max_channel_shift = max(max_channel_shift, *(abs(value) for value in rate_channel_shifts.values()))
                max_itd_shift = max(max_itd_shift, abs(float(row["itd_shift_samples_rate_comparison"])))
                max_absolute_channel_error = max(max_absolute_channel_error, *(abs(value) for value in absolute_errors.values()))
    rate_stable = max(max_channel_shift, max_itd_shift) <= 2.0
    return {
        "schema_version": FRACTIONAL_DELAY_CALIBRATION_SCHEMA_VERSION,
        "status": "PASS" if rate_stable else "METRIC_ESTIMATOR_NOT_RATE_STABLE",
        "formal_denominator": False,
        "estimator": {
            "name": "frozen_direct_window_10_percent_peak_onset",
            "threshold_fraction_of_channel_peak": 0.10,
            "tolerance_under_test_samples": 2.0,
        },
        "fixture": {
            "signal": "Gaussian continuous-time pulse sampled at native rate",
            "rate_pair": [16000, 24000],
            "resampler": "resample_poly_kaiser_5_no_gain_correction",
            "independent_of_formal_ss2_cases": True,
        },
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "max_per_channel_onset_shift_samples": float(max_channel_shift),
            "max_itd_shift_samples": float(max_itd_shift),
            "max_absolute_native16_channel_estimator_error_samples": float(max_absolute_channel_error),
            "rate_stability_gate": rate_stable,
            "interpretation": "Rate comparison is independent of formal SS2 rows; absolute threshold-crossing quantization error is reported separately.",
        },
        "v3_recommendation": None if rate_stable else "Consider a fractional-delay/interpolated onset estimator in metric-contract v3; do not alter the frozen v2 tolerance here.",
    }


def validate_fractional_delay_calibration(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "formal_denominator", "estimator", "fixture", "rows", "summary", "v3_recommendation"}
    if set(document) != required or document["schema_version"] != FRACTIONAL_DELAY_CALIBRATION_SCHEMA_VERSION:
        raise QualificationError("fractional-delay calibration schema is invalid")
    if document["formal_denominator"] is not False or not isinstance(document["rows"], list):
        raise QualificationError("fractional-delay calibration must be independent of the formal denominator")
    return document


def validate_rir_sample_rate_convention_calibration(document: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "status", "formal_denominator", "energy_metric_convention", "candidate_conventions", "cases", "summary", "formal_energy_offset_attribution"}
    if set(document) != required or document["schema_version"] != RIR_SAMPLE_RATE_CONVENTION_SCHEMA_VERSION:
        raise QualificationError("RIR sample-rate convention calibration schema is invalid")
    if document["formal_denominator"] is not False or not isinstance(document["cases"], list) or not isinstance(document["candidate_conventions"], Mapping):
        raise QualificationError("RIR sample-rate convention calibration is malformed")
    return document


def run_rir_sample_rate_convention_calibration(
    contract: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    a0_contract: Mapping[str, Any],
    formal_sample_rate_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run independent Path A/B/C convolution evidence on the shoebox."""

    from active_audition.acoustics.rir import render_native_rir
    from active_audition.receiver.audit import _runtime_fingerprint
    from active_audition.scene.simulator import create_scene_simulator
    from active_audition.types import ListenerPose

    repo_root = Path(runtime_config["_repo_root"]).resolve()
    primary = next(item for item in registry["geometries"] if item["kind"] == "symmetric_shoebox")
    selected_angles = (0.0, -60.0, 60.0, -90.0, 90.0)
    selected_distances = (1.0, 4.0)
    candidate_conventions = {
        "no_gain_correction": {"gain": 1.0, "mathematical_basis": "sample values represent the same continuous-time impulse response; sum(h^2)/fs is the comparable energy diagnostic"},
        "fs_source_over_target": {"gain": 1.5, "mathematical_basis": "discrete convolution without an explicit dt factor; h_target = (fs_source/fs_target)*h_resampled preserves a sampled integral convention"},
        "sqrt_fs_source_over_target": {"gain": math.sqrt(1.5), "mathematical_basis": "gain that equalizes unweighted coefficient sum(h^2) after the 24k-to-16k sample-count change"},
    }
    config_base = copy.deepcopy(dict(runtime_config))
    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    fingerprint: Mapping[str, Any] = {}
    for distance in selected_distances:
        for angle in selected_angles:
            source = source_position_world(primary["receiver"]["sensor_position_world"], primary["receiver"]["yaw_deg"], angle, distance)
            receiver_sensor = tuple(primary["receiver"]["sensor_position_world"])
            config16 = copy.deepcopy(config_base)
            config16["acoustics"] = dict(config_base["acoustics"])
            config16["acoustics"]["sample_rate_hz"] = 16000
            config24 = copy.deepcopy(config_base)
            config24["acoustics"] = dict(config_base["acoustics"])
            config24["acoustics"]["sample_rate_hz"] = 24000
            try:
                rendered: Dict[int, np.ndarray] = {}
                effective: Dict[int, Mapping[str, Any]] = {}
                for rate, config in ((16000, config16), (24000, config24)):
                    with create_scene_simulator(config, scene_id=primary["id"], require_navmesh=False, load_semantic_mesh=False, scene_override={"scene_asset": primary["mesh_path"]}) as context:
                        fingerprint = _runtime_fingerprint(repo_root)
                        offset = np.asarray(config["listener"]["sensor_offset_m"], dtype=np.float64)
                        base = tuple((np.asarray(receiver_sensor, dtype=np.float64) - offset).tolist())
                        pose = ListenerPose(base_position_world=base, sensor_position_world=receiver_sensor, yaw_deg=float(primary["receiver"]["yaw_deg"]))
                        rendered[rate] = np.asarray(render_native_rir(context, source, pose), dtype=np.float32)
                        effective[rate] = {"sample_rate_hz": rate}
                h16 = rendered[16000]
                h24 = rendered[24000]
                h24_to_16 = resample_array(h24, 24000, 16000)
                x16 = _rate_matched_band_source(16000)
                x24 = _rate_matched_band_source(24000)
                path_a = _convolve_binaural(x16, h16)
                path_b_native = _convolve_binaural(x24, h24)
                path_b = resample_array(path_b_native, 24000, 16000)
                path_c_by_convention: Dict[str, Any] = {}
                for name, convention in candidate_conventions.items():
                    converted = h24_to_16 * float(convention["gain"])
                    path_c = _convolve_binaural(x16, converted)
                    path_c_by_convention[name] = {
                        "rir_gain_applied": float(convention["gain"]),
                        "rir_stats": _compact_rir_stats(converted, 16000, contract),
                        "output_stats": {"L": _signal_stats(path_c[:, 0], 16000, contract), "R": _signal_stats(path_c[:, 1], 16000, contract)},
                        "output_metric": _compact_signal_metric(path_c, 16000, contract),
                        "vs_path_a": _compare_output_stats(path_a, path_c, 16000, contract),
                        "metric_difference_vs_path_a": _metric_difference(_compact_signal_metric(path_a, 16000, contract), _compact_signal_metric(path_c, 16000, contract)),
                    }
                case = {
                    "geometry_id": primary["id"],
                    "geometry": {"mesh_path": primary["mesh_path"], "mesh_sha256": primary["mesh_sha256"], "registry_sha256": registry["registry_sha256"]},
                    "distance_m": float(distance),
                    "relative_angle_deg": float(angle),
                    "line_of_sight": "LOS",
                    "source_transform": {"position_world": list(source)},
                    "receiver_transform": {"sensor_position_world": list(receiver_sensor), "yaw_deg": float(primary["receiver"]["yaw_deg"])},
                    "source": {"description": "deterministic continuous-time multi-tone band-limited source", "sample_rates_hz": [16000, 24000]},
                    "path_a_x16_convolve_h16": {"rir_stats": _compact_rir_stats(h16, 16000, contract), "output_stats": {"L": _signal_stats(path_a[:, 0], 16000, contract), "R": _signal_stats(path_a[:, 1], 16000, contract)}, "output_metric": _compact_signal_metric(path_a, 16000, contract)},
                    "path_b_x24_convolve_h24_then_resample": {"rir_stats": _compact_rir_stats(h24, 24000, contract), "resampled_rir_stats": _compact_rir_stats(h24_to_16, 16000, contract), "output_stats": {"L": _signal_stats(path_b[:, 0], 16000, contract), "R": _signal_stats(path_b[:, 1], 16000, contract)}, "output_metric": _compact_signal_metric(path_b, 16000, contract), "vs_path_a": _compare_output_stats(path_a, path_b, 16000, contract), "metric_difference_vs_path_a": _metric_difference(_compact_signal_metric(path_a, 16000, contract), _compact_signal_metric(path_b, 16000, contract))},
                    "path_c_x16_convolve_resampled_h24": path_c_by_convention,
                    "native_metric_summary": {"native16": _compact_signal_metric(h16, 16000, contract), "native24": _compact_signal_metric(h24, 24000, contract), "resampled24_to_16": _compact_signal_metric(h24_to_16, 16000, contract)},
                    "effective": effective,
                    "runtime_sha256": _sha256_json(fingerprint),
                    "contract_sha256": metric_contract_sha256(contract),
                    "a0_contract_sha256": _sha256_json(a0_contract),
                    "resource_hashes": {"geometry_registry": registry["registry_sha256"], "scene_asset": primary["mesh_sha256"]},
                }
                cases.append(case)
            except Exception as exc:  # pragma: no cover - requires SS2 runtime
                errors.append({"distance_m": distance, "relative_angle_deg": angle, "type": type(exc).__name__, "message": str(exc)})
    summary: Dict[str, Any] = {"case_count": len(cases), "error_count": len(errors), "errors": errors, "path_b_vs_a": {}, "candidate_c_vs_a": {}}
    for label, source_key in (("path_b", "path_b_x24_convolve_h24_then_resample"),):
        values = []
        for case in cases:
            for channel in ("L", "R"):
                value = case[source_key]["vs_path_a"]["channels"][channel]["mean_square_difference_db"]
                if value is not None:
                    values.append(float(value))
        summary[label] = {"mean_mean_square_difference_db": float(np.mean(values)) if values else None, "std_mean_square_difference_db": float(np.std(values)) if values else None, "max_abs_mean_square_difference_db": float(max((abs(v) for v in values), default=0.0))}
    for candidate in candidate_conventions:
        values = []
        for case in cases:
            comparison = case["path_c_x16_convolve_resampled_h24"][candidate]["vs_path_a"]["channels"]
            values.extend(float(comparison[channel]["mean_square_difference_db"]) for channel in ("L", "R") if comparison[channel]["mean_square_difference_db"] is not None)
        summary["candidate_c_vs_a"][candidate] = {"mean_mean_square_difference_db": float(np.mean(values)) if values else None, "std_mean_square_difference_db": float(np.std(values)) if values else None, "max_abs_mean_square_difference_db": float(max((abs(v) for v in values), default=0.0))}
    summary["deterministic_gain_convention_supported"] = False
    summary["decision"] = "NO_AUTOMATIC_RIR_GAIN_CORRECTION; equivalence evidence is recorded for metric-contract review"
    artifact = {
        "schema_version": RIR_SAMPLE_RATE_CONVENTION_SCHEMA_VERSION,
        "status": "CALIBRATION_COMPLETE" if cases and not errors else ("PARTIAL" if cases else BLOCKED),
        "formal_denominator": False,
        "energy_metric_convention": energy_metric_convention_audit(),
        "candidate_conventions": candidate_conventions,
        "cases": cases,
        "summary": summary,
        "formal_energy_offset_attribution": _formal_energy_offset_attribution(formal_sample_rate_path),
    }
    validate_rir_sample_rate_convention_calibration(artifact)
    return artifact


def run_a2_sample_rate_blocker_calibration(
    metric_contract_path: str,
    output_dir: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
    formal_sample_rate_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run only the non-formal A2 sample-rate attribution calibrations."""

    contract = load_metric_contract(metric_contract_path)
    from active_audition.config.loader import load_resolved_config as load_legacy_config

    runtime_config = load_legacy_config(runtime_config_path)
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    a0_contract = load_a0_contract_for_qualification(runtime_config)
    geometry_path, geometry_reason = _load_geometry_registry(contract, repo_root)
    energy_audit = energy_metric_convention_audit()
    validate_energy_metric_convention(energy_audit)
    fractional = run_fractional_delay_estimator_calibration(contract)
    validate_fractional_delay_calibration(fractional)
    if geometry_reason is not None:
        rir_calibration = {
            "schema_version": RIR_SAMPLE_RATE_CONVENTION_SCHEMA_VERSION,
            "status": BLOCKED,
            "formal_denominator": False,
            "energy_metric_convention": energy_audit,
            "candidate_conventions": {},
            "cases": [],
            "summary": {"case_count": 0, "error_count": 1, "errors": [{"reason": geometry_reason}], "decision": "BLOCKED_BEFORE_RUNTIME"},
            "formal_energy_offset_attribution": _formal_energy_offset_attribution(formal_sample_rate_path),
        }
    else:
        registry = load_geometry_registry(str(geometry_path), str(repo_root))
        rir_calibration = run_rir_sample_rate_convention_calibration(contract, runtime_config, registry, a0_contract, formal_sample_rate_path)
    validate_rir_sample_rate_convention_calibration(rir_calibration)

    formal_offsets = rir_calibration["formal_energy_offset_attribution"]
    summary = {
        "schema_version": SAMPLE_RATE_ATTRIBUTION_SCHEMA_VERSION,
        "gate": "A2",
        "status": "OPEN_BLOCKER_ATTRIBUTION_COMPLETE",
        "formal_a2_status": "OPEN/BLOCKED",
        "metric_contract": {"path": str(Path(metric_contract_path).resolve()), "sha256": metric_contract_sha256(contract), "modified": False},
        "a0_contract_sha256": _sha256_json(a0_contract),
        "geometry": {"path": str(geometry_path), "status": "VALIDATED" if geometry_reason is None else BLOCKED, "reason": geometry_reason},
        "energy_metric_convention_audit": energy_audit,
        "rir_sample_rate_convention_calibration": {"status": rir_calibration["status"], "case_count": len(rir_calibration["cases"]), "formal_denominator": False},
        "fractional_delay_estimator_calibration": fractional["summary"],
        "formal_energy_offset_attribution": formal_offsets,
        "attribution_conclusion": {
            "deterministic_rir_gain_conversion_supported": bool(rir_calibration["summary"].get("deterministic_gain_convention_supported", False)),
            "estimator_rate_stability": fractional["status"],
            "formal_tolerances_changed": False,
            "a2_closed": False,
            "a3_entered": False,
        },
        "blocker": "SAMPLE_RATE_AB_FORMAL_ENERGY_AND_4M_ITD_FAILURE_REQUIRES_INDEPENDENT_CALIBRATION_REVIEW",
    }
    output_root = Path(output_dir).resolve()
    storage = DatasetStorage(str(output_root))
    storage.atomic_write_json(output_root / "energy_metric_convention_audit.json", energy_audit)
    storage.atomic_write_json(output_root / "rir_sample_rate_convention_calibration.json", rir_calibration)
    storage.atomic_write_json(output_root / "fractional_delay_estimator_calibration.json", fractional)
    storage.atomic_write_json(output_root / "formal_energy_offset_attribution.json", formal_offsets)
    summary["artifacts"] = {name: {"path": name, "sha256": _file_sha256(output_root / name)} for name in ("energy_metric_convention_audit.json", "rir_sample_rate_convention_calibration.json", "fractional_delay_estimator_calibration.json", "formal_energy_offset_attribution.json")}
    storage.atomic_write_json(output_root / "a2_sample_rate_calibration_summary.json", summary)
    summary["artifacts"]["a2_sample_rate_calibration_summary.json"] = {"path": "a2_sample_rate_calibration_summary.json", "sha256": _file_sha256(output_root / "a2_sample_rate_calibration_summary.json")}
    report_lines = [
        "# Active-ASR V1.1 A2 Sample-Rate Blocker Attribution",
        "",
        "- Status: **OPEN/BLOCKED; attribution only**",
        "- Metric contract SHA256: `{}` (unchanged)".format(summary["metric_contract"]["sha256"]),
        "- Formal v2 artifacts were read only and were not overwritten.",
        "- Formal RIR energy is `sum(x^2)` / sample-squared; the earlier resampler calibration is RMS/mean-square. They are explicitly different quantities.",
        "- RIR Path A/B/C calibration status: `{}`; cases: `{}`; deterministic gain convention supported: `{}`.".format(rir_calibration["status"], len(rir_calibration["cases"]), rir_calibration["summary"].get("deterministic_gain_convention_supported")),
        "- Fractional-delay estimator status: `{}`; max rate-comparison channel shift: `{}` samples; max ITD shift: `{}` samples.".format(fractional["status"], fractional["summary"]["max_per_channel_onset_shift_samples"], fractional["summary"]["max_itd_shift_samples"]),
        "- Formal energy offset attribution: `{}`; see `formal_energy_offset_attribution.json`.".format(formal_offsets.get("status")),
        "- No tolerance was changed; A2 was not closed and A3 was not entered.",
        "",
    ]
    storage.atomic_write_text(output_root / "a2_sample_rate_calibration_report.md", "\n".join(report_lines))
    summary["artifacts"]["a2_sample_rate_calibration_report.md"] = {"path": "a2_sample_rate_calibration_report.md", "sha256": _file_sha256(output_root / "a2_sample_rate_calibration_report.md")}
    return summary


def _metric_onsets(metric: Optional[Mapping[str, Any]], sample_rate_hz: int) -> Dict[str, Any]:
    if not isinstance(metric, Mapping) or metric.get("applicability") != "APPLICABLE":
        return {"applicability": NA, "reason": (metric or {}).get("na_reason", "METRICS_UNAVAILABLE")}
    onsets = metric["direct_window"]["channel_onset_samples"]
    return {
        "applicability": "APPLICABLE",
        "L": {"samples": int(onsets["L"]), "seconds": float(onsets["L"] / float(sample_rate_hz))},
        "R": {"samples": int(onsets["R"]), "seconds": float(onsets["R"] / float(sample_rate_hz))},
        "itd_samples": int(metric["itd_samples"]),
        "itd_seconds": float(metric["itd_seconds"]),
    }


def _raw_channel_observation(raw_rir: np.ndarray, sample_rate_hz: int, contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute direct-window observations while retaining native channel names."""

    metric = compute_direct_metrics(raw_rir, sample_rate_hz, contract)
    if metric.get("applicability") != "APPLICABLE":
        return {
            "applicability": NA,
            "reason": metric.get("na_reason"),
            "onset": _metric_onsets(metric, sample_rate_hz),
            "direct_energy": {"channel0": None, "channel1": None},
            "ild_like_ch0_div_ch1_db": None,
            "raw_itd_channel1_minus_channel0_samples": None,
        }
    return {
        "applicability": "APPLICABLE",
        "reason": None,
        "onset": {
            "channel0": {"samples": int(metric["direct_window"]["channel_onset_samples"]["L"]), "seconds": float(metric["direct_window"]["channel_onset_samples"]["L"] / float(sample_rate_hz))},
            "channel1": {"samples": int(metric["direct_window"]["channel_onset_samples"]["R"]), "seconds": float(metric["direct_window"]["channel_onset_samples"]["R"] / float(sample_rate_hz))},
        },
        "direct_energy": {"channel0": float(metric["direct_energy_left"]), "channel1": float(metric["direct_energy_right"])},
        "ild_like_ch0_div_ch1_db": metric.get("ild_db"),
        "raw_itd_channel1_minus_channel0_samples": int(metric["itd_samples"]),
    }


def run_probe_domain_diagnostics(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Exercise fixed probes through one deterministic RIR fixture.

    These rows are waveform-domain diagnostics only.  They are deliberately
    separate from formal RIR metric rows and never enter a gate denominator.
    """

    rir = np.zeros((512, 2), dtype=np.float32)
    rir[64, 0] = 1.0
    rir[67, 1] = 0.8
    rir[180, :] = np.asarray([0.08, 0.06], dtype=np.float32)
    rows = []
    for kind in ("broadband_noise", "chirp", "voiced_tone"):
        probe = make_probe(kind, 16000, duration_sec=0.025, seed=int(contract["probes"]["fixture_seed"]))
        waveform = np.column_stack([np.convolve(probe, rir[:, channel]) for channel in range(2)]).astype(np.float32)
        rows.append({
            "probe": kind,
            "domain": "waveform_diagnostic",
            "rir_reference": "deterministic_probe_domain_fixture_v1",
            "formal_denominator": False,
            "sample_rate_hz": 16000,
            "input_num_samples": int(probe.shape[0]),
            "output_num_samples": int(waveform.shape[0]),
            "input_sha256": hashlib.sha256(probe.tobytes()).hexdigest(),
            "output_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
            "output_peak_by_channel": [float(np.max(np.abs(waveform[:, channel]))) for channel in range(2)],
            "output_energy_by_channel": [float(np.sum(np.square(waveform[:, channel], dtype=np.float64))) for channel in range(2)],
        })
    return {
        "schema_version": "active-asr-a2-probe-domain-diagnostic-v1",
        "status": PASS,
        "domain": "waveform_diagnostic",
        "rir_reference": "deterministic_probe_domain_fixture_v1",
        "formal_denominator": False,
        "probes": rows,
    }


def _run_direction_channel_calibration(
    contract: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    a0_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Record raw native channel evidence before canonical [L, R] mapping."""

    from active_audition.acoustics.rir import render_native_rir_raw
    from active_audition.receiver.audit import _receiver_observation, _runtime_fingerprint
    from active_audition.scene.pose import relative_azimuth_deg as scene_relative_azimuth
    from active_audition.scene.simulator import create_scene_simulator
    from active_audition.types import ListenerPose

    primary = next(item for item in registry["geometries"] if item["kind"] == "symmetric_shoebox")
    config = copy.deepcopy(dict(runtime_config))
    config["acoustics"] = dict(runtime_config["acoustics"])
    config["acoustics"]["sample_rate_hz"] = 16000
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    try:
        with create_scene_simulator(
            config,
            scene_id=primary["id"],
            require_navmesh=False,
            load_semantic_mesh=False,
            scene_override={"scene_asset": primary["mesh_path"]},
        ) as context:
            fingerprint = _runtime_fingerprint(repo_root)
            receiver = _receiver_observation(a0_contract, config, context)
            offset = np.asarray(config["listener"]["sensor_offset_m"], dtype=np.float64)
            for yaw in (0.0, 180.0):
                for angle in (-45.0, 45.0):
                    receiver_sensor = tuple(primary["receiver"]["sensor_position_world"])
                    source = source_position_world(receiver_sensor, yaw, angle, 2.0)
                    base = tuple((np.asarray(receiver_sensor, dtype=np.float64) - offset).tolist())
                    pose = ListenerPose(base_position_world=base, sensor_position_world=receiver_sensor, yaw_deg=yaw)
                    raw_rir = render_native_rir_raw(context, source, pose)
                    raw = _raw_channel_observation(raw_rir, 16000, contract)
                    projected_angle = float(geometry_relative_azimuth(receiver_sensor, yaw, source))
                    pose_angle = float(scene_relative_azimuth(source, receiver_sensor, yaw))
                    observed = raw.get("raw_itd_channel1_minus_channel0_samples")
                    h1_expected_sign = 1 if projected_angle > 0 else -1
                    h2_expected_sign = -h1_expected_sign
                    rows.append({
                        "case_id": "direction_channel_calibration_yaw{}_a{}".format(int(yaw), str(int(angle)).replace("-", "m")),
                        "receiver": {"sensor_position_world": list(receiver_sensor), "base_position_world": list(base), "yaw_deg": yaw},
                        "source": {"position_world": list(source), "world_x": float(source[0]), "world_z": float(source[2]), "distance_m": 2.0},
                        "project_relative_angle_deg": projected_angle,
                        "scene_pose_relative_angle_deg": pose_angle,
                        "relative_angle_convention": "positive_left",
                        "raw_native_channel_order": ["channel0", "channel1"],
                        "raw": raw,
                        "hypotheses": {
                            "H1_native_ch0_L_ch1_R": {"expected_raw_itd_sign": h1_expected_sign, "supports": observed is not None and int(observed) * h1_expected_sign > 0},
                            "H2_native_ch0_R_ch1_L": {"expected_raw_itd_sign": h2_expected_sign, "supports": observed is not None and int(observed) * h2_expected_sign > 0},
                        },
                        "runtime_sha256": _sha256_json(fingerprint),
                        "resource_hashes": {"geometry_registry": registry["registry_sha256"], "scene_asset": primary["mesh_sha256"]},
                    })
    except Exception as exc:  # pragma: no cover - requires the real SS2 runtime
        errors.append({"type": type(exc).__name__, "message": str(exc)})

    applicable = [row for row in rows if row["raw"]["applicability"] == "APPLICABLE"]
    h1_pass = sum(bool(row["hypotheses"]["H1_native_ch0_L_ch1_R"]["supports"]) for row in applicable)
    h2_pass = sum(bool(row["hypotheses"]["H2_native_ch0_R_ch1_L"]["supports"]) for row in applicable)
    h1_accuracy = h1_pass / float(len(applicable)) if applicable else None
    h2_accuracy = h2_pass / float(len(applicable)) if applicable else None
    mapping_decision = "SWAP_NATIVE_TO_CANONICAL" if h2_accuracy == 1.0 and (h1_accuracy or 0.0) < h2_accuracy else "NO_MAPPING_DECISION"
    return {
        "schema_version": "active-asr-a2-direction-channel-calibration-v1",
        "status": PASS if rows and not errors and applicable else BLOCKED,
        "formal_denominator": False,
        "cases": rows,
        "errors": errors,
        "hypothesis_summary": {
            "H1_native_ch0_L_ch1_R": {"pass": h1_pass, "applicable": len(applicable), "sign_accuracy": h1_accuracy},
            "H2_native_ch0_R_ch1_L": {"pass": h2_pass, "applicable": len(applicable), "sign_accuracy": h2_accuracy},
            "supported_hypothesis": "H2_native_ch0_R_ch1_L" if mapping_decision == "SWAP_NATIVE_TO_CANONICAL" else None,
            "mapping_decision": mapping_decision,
        },
        "convention_audit": {
            "receiver_geometry_relative_azimuth": "positive_left",
            "scene_pose_relative_azimuth": "positive_left",
            "world_frame": "Y-up, forward=-Z, right=+X at yaw=0",
            "raw_channel_evidence_is_independent_of_canonical_mapping": True,
        },
    }


def _runtime_smoke(
    contract: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    scene_id: str,
    resource_reason: str,
) -> Dict[str, Any]:
    """Run only an explicitly non-qualification office_0 technical smoke."""

    from active_audition.acoustics.rir import render_native_rir
    from active_audition.navigation.pathfinder import PathFinderAdapter
    from active_audition.receiver.audit import _receiver_observation
    from active_audition.receiver.audit import _runtime_fingerprint, _scene_provenance
    from active_audition.scene.episode import fixed_golden_episode
    from active_audition.scene.simulator import create_scene_simulator

    # The import above deliberately keeps habitat_sim behind quaternion via
    # active_audition.scene.simulator.  The alias expression is avoided at
    # runtime below; _resource_hashes is local to this module.
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    base_config = copy.deepcopy(dict(runtime_config))
    base_config["acoustics"] = dict(runtime_config["acoustics"])
    base_config["acoustics"]["sample_rate_hz"] = 16000

    variants = [
        ("native_16k_baseline", base_config, {}),
        ("native_16k_rays_doubled", base_config, None),
        ("native_16k_ir_doubled", base_config, None),
        ("native_24k_reference", dict(base_config, acoustics=dict(base_config["acoustics"], sample_rate_hz=24000)), {}),
    ]
    outputs: Dict[str, Any] = {"status": PASS, "cases": [], "runtime_lock": None, "errors": []}
    baseline_receiver: Optional[Mapping[str, Any]] = None
    baseline_scene: Optional[Mapping[str, Any]] = None
    baseline_fingerprint: Optional[Mapping[str, Any]] = None
    baseline_episode: Any = None
    baseline_rir: Optional[np.ndarray] = None
    variant_rirs: Dict[str, np.ndarray] = {}

    def effective_acoustics(receiver: Mapping[str, Any]) -> Mapping[str, Any]:
        value = receiver["acoustics_config_effective"]
        if isinstance(value, Mapping) and value.get("status") == KNOWN and isinstance(value.get("value"), Mapping):
            return value["value"]
        return value

    for variant_id, config, overrides in variants:
        try:
            if variant_id == "native_16k_rays_doubled":
                if baseline_receiver is None:
                    raise QualificationError("baseline receiver was not established")
                effective = effective_acoustics(baseline_receiver)
                overrides = {field: int(effective[field] * 2) for field in ("directRayCount", "sourceRayCount", "indirectRayCount")}
            elif variant_id == "native_16k_ir_doubled":
                if baseline_receiver is None:
                    raise QualificationError("baseline receiver was not established")
                effective_ir = float(effective_acoustics(baseline_receiver)["maxIRLength"])
                overrides = {"maxIRLength": effective_ir * 2.0}
            with create_scene_simulator(config, scene_id=scene_id, acoustics_overrides=overrides) as context:
                fingerprint = _runtime_fingerprint(repo_root)
                scene = _scene_provenance(config, scene_id, context)
                receiver = _receiver_observation(
                    load_a0_contract_for_qualification(config), config, context
                )
                episode = fixed_golden_episode(config, PathFinderAdapter(context.pathfinder))
                repeat_count = 5 if variant_id == "native_16k_baseline" else 1
                for repeat_id in range(repeat_count):
                    rir = render_native_rir(context, episode.source.position_world, episode.listener_initial)
                    variant_rirs.setdefault(variant_id, rir)
                    metrics = compute_direct_metrics(rir, int(config["acoustics"]["sample_rate_hz"]), contract)
                    case_runtime_sha = _sha256_json(fingerprint)
                    outputs["cases"].append(
                        _case_row(
                            case_id="scene_smoke_{}_r{}".format(variant_id, repeat_id),
                            case_type="scene_technical_smoke",
                            geometry_id=scene_id,
                            geometry={"authority": "UNQUALIFIED_SCENE_SMOKE", "reason": resource_reason},
                            source_transform={"position_world": list(episode.source.position_world)},
                            receiver_transform={
                                "base_position_world": list(episode.listener_initial.base_position_world),
                                "sensor_position_world": list(episode.listener_initial.sensor_position_world),
                                "yaw_deg": float(episode.listener_initial.yaw_deg),
                            },
                            line_of_sight="UNKNOWN",
                            relative_angle_deg=None,
                            distance_m=None,
                            probe="impulse_response",
                            probe_domain="rir",
                            probe_reference="impulse_response",
                            sample_rate_hz=int(config["acoustics"]["sample_rate_hz"]),
                            ray_preset=effective_acoustics(receiver),
                            ir_length_sec=float(effective_acoustics(receiver)["maxIRLength"]),
                            repeat_id=repeat_id,
                            applicability=NA,
                            raw_metrics=metrics,
                            status=NA,
                            reason="UNQUALIFIED_SCENE_SMOKE_NOT_PHYSICS_EVIDENCE",
                            runtime_sha256=case_runtime_sha,
                            contract_sha256=metric_contract_sha256(contract),
                            resource_hashes=_resource_hashes(scene),
                        )
                    )
                    if variant_id == "native_16k_baseline" and repeat_id == 0:
                        baseline_rir = rir
                        baseline_episode = episode
                if variant_id == "native_16k_baseline":
                    baseline_receiver = receiver
                    baseline_scene = scene
                    baseline_fingerprint = fingerprint
        except Exception as exc:  # pragma: no cover - exercised by real runtime
            outputs["status"] = BLOCKED
            outputs["errors"].append({"variant": variant_id, "type": type(exc).__name__, "message": str(exc)})

    if baseline_receiver is None or baseline_scene is None or baseline_fingerprint is None or baseline_rir is None:
        outputs["status"] = BLOCKED
        outputs["runtime_lock"] = {
            "schema_version": A2_RUNTIME_LOCK_SCHEMA_VERSION,
            "gate": "A2",
            "status": BLOCKED,
            "contract_sha256": metric_contract_sha256(contract),
            "runtime_fingerprint": baseline_fingerprint,
            "receiver_effective": None,
            "resource_hashes": {},
            "geometry_registry_reason": resource_reason,
        }
        return outputs

    native_16 = compute_direct_metrics(baseline_rir, 16000, contract)
    native_24 = None
    if "native_24k_reference" in variant_rirs:
        native_24 = compute_direct_metrics(variant_rirs["native_24k_reference"], 24000, contract)
        resampled_24 = resample_array(variant_rirs["native_24k_reference"], 24000, 16000)
        resampled_metrics = compute_direct_metrics(resampled_24, 16000, contract)
    else:
        resampled_24 = None
        resampled_metrics = None
    ray_metrics = compute_direct_metrics(variant_rirs["native_16k_rays_doubled"], 16000, contract) if "native_16k_rays_doubled" in variant_rirs else None
    ir_metrics = compute_direct_metrics(variant_rirs["native_16k_ir_doubled"], 16000, contract) if "native_16k_ir_doubled" in variant_rirs else None
    outputs["sample_rate_ab"] = {
        "schema_version": "active-asr-a2-sample-rate-ab-v1",
        "status": BLOCKED,
        "applicability": NA,
        "reason": resource_reason,
        "native_16khz": native_16,
        "native_24khz": native_24,
        "native_24khz_resampled_to_16khz": resampled_metrics,
        "comparison_native_16_vs_resampled_24": _metric_comparison(native_16, resampled_metrics) if resampled_metrics is not None else {"applicability": NA, "reason": "NATIVE_24K_RENDER_MISSING"},
        "tolerances": {
            "itd_difference_max_samples": contract["gates"]["sample_rate_ab"]["itd_difference_max_samples"],
            "band_ild_difference_max_db": contract["gates"]["sample_rate_ab"]["band_ild_difference_max_db"],
            "band_energy_difference_max_db": contract["gates"]["sample_rate_ab"]["band_energy_difference_max_db"],
        },
        "exceedance_review": {
            "status": "REQUIRES_CAUSE_ANALYSIS_BEFORE_ANY_16KHZ_CONCLUSION",
            "candidate_causes": ["stochastic_rays", "units", "anti_alias_filter", "delay_convention", "gain_convention"],
            "reject_16khz_without_cause_analysis": True,
        },
        "normalization": "none",
    }
    outputs["convergence"] = {
        "status": BLOCKED,
        "applicability": NA,
        "reason": resource_reason,
        "baseline": native_16,
        "rays_doubled": ray_metrics,
        "ir_length_doubled": ir_metrics,
        "baseline_vs_rays_doubled": _metric_comparison(native_16, ray_metrics) if ray_metrics is not None else {"applicability": NA, "reason": "RAY_VARIANT_MISSING"},
        "baseline_vs_ir_length_doubled": _metric_comparison(native_16, ir_metrics) if ir_metrics is not None else {"applicability": NA, "reason": "IR_VARIANT_MISSING"},
        "effective_baseline_ray_preset": dict(effective_acoustics(baseline_receiver)),
        "effective_baseline_ir_length_sec": effective_acoustics(baseline_receiver)["maxIRLength"],
    }
    outputs["runtime_lock"] = {
        "schema_version": A2_RUNTIME_LOCK_SCHEMA_VERSION,
        "gate": "A2",
        "status": BLOCKED,
        "contract_sha256": metric_contract_sha256(contract),
        "runtime_fingerprint": baseline_fingerprint,
        "receiver_effective": dict(effective_acoustics(baseline_receiver)),
        "resource_hashes": _resource_hashes(baseline_scene),
        "geometry_registry_reason": resource_reason,
    }
    return outputs


def _formal_gate_counts(attempted: int, applicable: int, passed: int, failed: int, na: int, status: str, reason: str = "") -> Dict[str, Any]:
    return {"status": status, "attempted": attempted, "applicable": applicable, "pass": passed, "fail": failed, "na": na, "reason": reason}


def _controlled_metric16(record: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    metrics = record.get("metrics16")
    return metrics if isinstance(metrics, Mapping) else None


def _sample_rate_pair_detail(
    key: Tuple[str, float, float],
    pair: Mapping[int, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the auditable 16/24 kHz table row and attribution candidates."""

    native16_record = pair.get(16000)
    native24_record = pair.get(24000)
    native16 = native16_record.get("native_metrics") if native16_record else None
    native24 = native24_record.get("native_metrics") if native24_record else None
    resampled = native24_record.get("resampled_metrics16") if native24_record else None
    comparison = _metric_comparison(native16, resampled) if native16 is not None and resampled is not None else {"applicability": NA, "reason": "RATE_PAIR_MISSING"}

    def criterion_values(field: str) -> List[float]:
        return [float(item[field]) for item in comparison.get("bands", {}).values() if item.get(field) is not None]

    band_ild = criterion_values("ild_difference_db")
    band_energy = criterion_values("energy_difference_db")
    ab_contract = contract["gates"]["sample_rate_ab"]
    itd_pass = comparison.get("itd_difference_samples") is not None and comparison["itd_difference_samples"] <= float(ab_contract["itd_difference_max_samples"])
    ild_pass = bool(band_ild) and max(band_ild) <= float(ab_contract["band_ild_difference_max_db"])
    energy_pass = bool(band_energy) and max(band_energy) <= float(ab_contract["band_energy_difference_max_db"])
    applicable = comparison.get("applicability") == "APPLICABLE"

    native16_onset = _metric_onsets(native16, 16000)
    native24_onset = _metric_onsets(native24, 24000)
    resampled_onset = _metric_onsets(resampled, 16000)
    native_time_delta = {"L": None, "R": None, "itd_seconds_native24_minus_native16": None}
    resample_shift = {"L": None, "R": None, "itd_samples_at_16khz": None}
    if native16_onset.get("applicability") == "APPLICABLE" and native24_onset.get("applicability") == "APPLICABLE":
        native_time_delta = {
            "L": float(native24_onset["L"]["seconds"] - native16_onset["L"]["seconds"]),
            "R": float(native24_onset["R"]["seconds"] - native16_onset["R"]["seconds"]),
            "itd_seconds_native24_minus_native16": float(native24_onset["itd_seconds"] - native16_onset["itd_seconds"]),
        }
    if native24_onset.get("applicability") == "APPLICABLE" and resampled_onset.get("applicability") == "APPLICABLE":
        resample_shift = {
            "L": float(resampled_onset["L"]["samples"] - native24_onset["L"]["seconds"] * 16000.0),
            "R": float(resampled_onset["R"]["samples"] - native24_onset["R"]["seconds"] * 16000.0),
            "itd_samples_at_16khz": float(resampled_onset["itd_samples"] - native24_onset["itd_seconds"] * 16000.0),
        }

    repeat_metrics = []
    for record in records.values():
        spec = record["spec"]
        if (
            spec["purpose"] == "direction_repeatability"
            and spec["line_of_sight"] == "LOS"
            and spec["geometry"]["id"] == key[0]
            and float(spec["distance_m"]) == key[1]
            and float(spec["relative_angle_deg"]) == key[2]
        ):
            metric = _controlled_metric16(record)
            if metric is not None and metric.get("applicability") == "APPLICABLE":
                repeat_metrics.append(metric)
    stochastic_proxy: Dict[str, Any] = {"available": False, "not_isolated_from_all_runtime_variation": True}
    if len(repeat_metrics) >= 2:
        itd_values = [float(item["itd_samples"]) for item in repeat_metrics]
        energy_values = [float(item["direct_energy_db"]) for item in repeat_metrics]
        band_spreads = {}
        for band_id in repeat_metrics[0].get("bands", {}):
            values = [float(item["bands"][band_id]["energy_db"]) for item in repeat_metrics if item["bands"][band_id].get("energy_db") is not None]
            if values:
                band_spreads[band_id] = max(values) - min(values)
        stochastic_proxy = {
            "available": True,
            "repeat_count": len(repeat_metrics),
            "itd_spread_samples": max(itd_values) - min(itd_values),
            "direct_energy_spread_db": max(energy_values) - min(energy_values),
            "band_energy_spread_db": band_spreads,
            "not_isolated_from_all_runtime_variation": True,
        }

    return {
        "geometry": key[0],
        "distance_m": key[1],
        "angle_deg": key[2],
        "native16_onset": native16_onset,
        "native24_onset": native24_onset,
        "resampled24_to_16_onset": resampled_onset,
        "native_time_difference_candidate_A": native_time_delta,
        "resampled_onset_shift_candidate_B": resample_shift,
        "renderer_amplitude_frequency_candidate_C": {
            "direct_energy_difference_db": comparison.get("direct_energy_difference_db"),
            "band_ild_difference_db": {band_id: item.get("ild_difference_db") for band_id, item in comparison.get("bands", {}).items()},
            "band_energy_difference_db": {band_id: item.get("energy_difference_db") for band_id, item in comparison.get("bands", {}).items()},
            "interpretation": "renderer/filter response is not isolated by this pair; no separate normalization was applied",
        },
        "stochastic_ray_difference_candidate_D": stochastic_proxy,
        "comparison": comparison,
        "criterion": {
            "applicable": applicable,
            "itd_pass": bool(applicable and itd_pass),
            "ild_pass": bool(applicable and ild_pass),
            "energy_pass": bool(applicable and energy_pass),
            "joint_pass": bool(applicable and itd_pass and ild_pass and energy_pass),
        },
        "tolerances": {
            "itd_difference_max_samples": ab_contract["itd_difference_max_samples"],
            "band_ild_difference_max_db": ab_contract["band_ild_difference_max_db"],
            "band_energy_difference_max_db": ab_contract["band_energy_difference_max_db"],
        },
    }


def _evaluate_formal_gates(records: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]) -> Dict[str, Any]:
    direction_specs = [record for record in records.values() if record["spec"]["purpose"] == "direction_repeatability" and record["spec"]["line_of_sight"] == "LOS" and record["spec"]["sample_rate_hz"] == 16000 and float(record["spec"]["relative_angle_deg"]) in contract["gates"]["direction"]["side_los_angles_deg"]]
    direction_groups: Dict[Tuple[str, float, float], List[Mapping[str, Any]]] = {}
    for record in direction_specs:
        key = (record["spec"]["geometry"]["id"], float(record["spec"]["distance_m"]), float(record["spec"]["relative_angle_deg"]))
        direction_groups.setdefault(key, []).append(record)
    direction_pass = 0
    direction_na = 0
    direction_fail = 0
    signs = []
    for (_, _, angle), group in sorted(direction_groups.items()):
        metric = _controlled_metric16(group[0])
        if metric is None or metric.get("applicability") != "APPLICABLE":
            direction_na += 1
            continue
        observed = int(metric["itd_samples"])
        expected = 1 if angle > 0.0 else -1
        signs.append(observed * expected)
        if observed * expected > 0:
            direction_pass += 1
        else:
            direction_fail += 1
    attempted = len(direction_groups)
    applicable = direction_pass + direction_fail
    direction_status = PASS if attempted and applicable / float(attempted) >= 0.95 and direction_pass / float(max(1, applicable)) >= float(contract["gates"]["direction"]["itd_sign_accuracy_min_fraction"]) and not (signs and all(value < 0 for value in signs)) else BLOCKED

    repeat_specs = [record for record in records.values() if record["spec"]["purpose"] == "direction_repeatability" and record["spec"]["line_of_sight"] == "LOS" and record["spec"]["sample_rate_hz"] == 16000]
    repeat_groups: Dict[Tuple[str, float, float], List[Mapping[str, Any]]] = {}
    for record in repeat_specs:
        key = (record["spec"]["geometry"]["id"], float(record["spec"]["distance_m"]), float(record["spec"]["relative_angle_deg"]))
        repeat_groups.setdefault(key, []).append(record)
    repeat_pass = repeat_na = repeat_fail = 0
    repeat_details = []
    repeat_contract = contract["gates"]["repeatability"]
    for key, group in sorted(repeat_groups.items()):
        metrics = [_controlled_metric16(record) for record in sorted(group, key=lambda item: item["spec"]["repeat_id"])]
        if len(metrics) != int(repeat_contract["repeat_count"]) or any(metric is None or metric.get("applicability") != "APPLICABLE" for metric in metrics):
            repeat_na += 1
            continue
        itd_values = [float(metric["itd_samples"]) for metric in metrics]
        band_ids = [band_id for band_id in metrics[0]["bands"] if all(item["bands"][band_id].get("ild_db") is not None for item in metrics)]
        ild_spread = max((max(float(item["bands"][band_id]["ild_db"]) for item in metrics) - min(float(item["bands"][band_id]["ild_db"]) for item in metrics) for band_id in band_ids), default=float("inf"))
        energy_values = [float(metric["direct_energy_db"]) for metric in metrics]
        details = {"key": list(key), "itd_spread_samples": max(itd_values) - min(itd_values), "band_ild_spread_db": ild_spread, "direct_energy_spread_db": max(energy_values) - min(energy_values)}
        repeat_details.append(details)
        passed = details["itd_spread_samples"] <= float(repeat_contract["direct_itd_spread_max_samples_at_16khz"]) and details["band_ild_spread_db"] <= float(repeat_contract["band_direct_ild_spread_max_db"]) and details["direct_energy_spread_db"] <= float(repeat_contract["direct_energy_spread_max_db"])
        if passed:
            repeat_pass += 1
        else:
            repeat_fail += 1
    repeat_attempted = len(repeat_groups)
    repeat_applicable = repeat_pass + repeat_fail
    repeat_status = PASS if repeat_attempted and repeat_applicable / float(repeat_attempted) >= 0.95 and repeat_pass == repeat_applicable else BLOCKED

    ab_groups: Dict[Tuple[str, float, float], Dict[int, Mapping[str, Any]]] = {}
    for record in records.values():
        spec = record["spec"]
        if spec["purpose"] == "sample_rate_ab" and spec["line_of_sight"] == "LOS":
            key = (spec["geometry"]["id"], float(spec["distance_m"]), float(spec["relative_angle_deg"]))
            ab_groups.setdefault(key, {})[int(spec["sample_rate_hz"])] = record
    ab_pass = ab_na = ab_fail = 0
    ab_details = []
    ab_contract = contract["gates"]["sample_rate_ab"]
    for key, pair in sorted(ab_groups.items()):
        detail = _sample_rate_pair_detail(key, pair, records, contract)
        if not detail["criterion"]["applicable"]:
            ab_na += 1
            continue
        ab_details.append(detail)
        if detail["criterion"]["joint_pass"]:
            ab_pass += 1
        else:
            ab_fail += 1
    ab_attempted = len(ab_groups)
    ab_applicable = ab_pass + ab_fail
    ab_applicable_fraction = ab_applicable / float(ab_attempted) if ab_attempted else 0.0
    ab_joint_pass_fraction = ab_pass / float(ab_attempted) if ab_attempted else 0.0
    ab_status = PASS if ab_attempted and ab_applicable_fraction >= float(ab_contract["applicable_case_min_fraction"]) and ab_joint_pass_fraction >= float(ab_contract["applicable_case_min_fraction"]) else BLOCKED

    convergence_groups: Dict[Tuple[str, float, float], Dict[Tuple[str, str], Mapping[str, Any]]] = {}
    for record in records.values():
        spec = record["spec"]
        if spec["purpose"] == "ray_tail_convergence" and spec["line_of_sight"] == "LOS":
            key = (spec["geometry"]["id"], float(spec["distance_m"]), float(spec["relative_angle_deg"]))
            convergence_groups.setdefault(key, {})[(spec["ray_variant"], spec["ir_variant"])] = record
    convergence_evidence = convergence_na = 0
    convergence_details = []
    for key, variants in sorted(convergence_groups.items()):
        baseline = variants.get(("baseline", "baseline"))
        rays = variants.get(("rays_x2", "baseline"))
        ir = variants.get(("baseline", "ir_x2"))
        if not baseline or not rays or not ir or any(_controlled_metric16(item) is None or _controlled_metric16(item).get("applicability") != "APPLICABLE" for item in (baseline, rays, ir)):
            convergence_na += 1
            continue
        comparison_rays = _metric_comparison(_controlled_metric16(baseline), _controlled_metric16(rays))
        comparison_ir = _metric_comparison(_controlled_metric16(baseline), _controlled_metric16(ir))
        convergence_details.append({"key": list(key), "baseline_vs_rays_x2": comparison_rays, "baseline_vs_ir_x2": comparison_ir})
        if comparison_rays.get("applicability") == "APPLICABLE" and comparison_ir.get("applicability") == "APPLICABLE":
            convergence_evidence += 1
    convergence_attempted = len(convergence_groups)
    convergence_applicable = convergence_evidence
    convergence_status = EVIDENCE_RECORDED if convergence_attempted and convergence_evidence == convergence_attempted else BLOCKED

    nlos_rows = [record for record in records.values() if record["spec"]["line_of_sight"] == "NLOS"]
    nlos_na = sum(1 for record in nlos_rows if record["row"]["applicability"] == NA and record["row"]["reason"] == "NLOS_DIRECT_PATH_BY_CONTRACT")
    mirror_groups = []
    primary_ids = {record["spec"]["geometry"]["id"] for record in records.values() if record["spec"]["geometry"]["kind"] == "symmetric_shoebox"}
    for geometry_id in sorted(primary_ids):
        for left_angle, right_angle in MIRROR_PAIRS:
            for distance in (1.0, 4.0):
                left = next((record for record in records.values() if record["spec"]["purpose"] == "direction_repeatability" and record["spec"]["geometry"]["id"] == geometry_id and record["spec"]["distance_m"] == distance and record["spec"]["relative_angle_deg"] == left_angle and record["spec"]["repeat_id"] == 0), None)
                right = next((record for record in records.values() if record["spec"]["purpose"] == "direction_repeatability" and record["spec"]["geometry"]["id"] == geometry_id and record["spec"]["distance_m"] == distance and record["spec"]["relative_angle_deg"] == right_angle and record["spec"]["repeat_id"] == 0), None)
                left_metric = _controlled_metric16(left or {})
                right_metric = _controlled_metric16(right or {})
                if left_metric is None or right_metric is None or left_metric.get("applicability") != "APPLICABLE" or right_metric.get("applicability") != "APPLICABLE":
                    mirror_groups.append(False)
                else:
                    mirror_groups.append(int(left_metric["itd_samples"]) * int(right_metric["itd_samples"]) < 0)
    mirror_pass = sum(bool(value) for value in mirror_groups)
    mirror_attempted = len(mirror_groups)
    mirror_gate = _formal_gate_counts(mirror_attempted, mirror_attempted, 0, 0, 0, EVIDENCE_RECORDED if mirror_attempted else BLOCKED, "mirror_symmetry_direct evidence only; current cases do not test listener yaw-relative invariance")
    mirror_gate["evidence_recorded"] = mirror_pass

    near_far_groups = []
    for geometry_id in sorted(primary_ids):
        for angle in RELATIVE_AZIMUTHS_DEG:
            near = next((record for record in records.values() if record["spec"]["purpose"] == "direction_repeatability" and record["spec"]["geometry"]["id"] == geometry_id and record["spec"]["distance_m"] == 1.0 and record["spec"]["relative_angle_deg"] == angle and record["spec"]["repeat_id"] == 0), None)
            far = next((record for record in records.values() if record["spec"]["purpose"] == "direction_repeatability" and record["spec"]["geometry"]["id"] == geometry_id and record["spec"]["distance_m"] == 4.0 and record["spec"]["relative_angle_deg"] == angle and record["spec"]["repeat_id"] == 0), None)
            near_metric = _controlled_metric16(near or {})
            far_metric = _controlled_metric16(far or {})
            if near_metric is None or far_metric is None or near_metric.get("applicability") != "APPLICABLE" or far_metric.get("applicability") != "APPLICABLE":
                near_far_groups.append(None)
            else:
                near_far_groups.append(float(far_metric["direct_energy_db"]) < float(near_metric["direct_energy_db"]))
    nf_pass = sum(value is True for value in near_far_groups)
    nf_fail = sum(value is False for value in near_far_groups)
    nf_na = sum(value is None for value in near_far_groups)
    nf_attempted = len(near_far_groups)
    near_far_gate = _formal_gate_counts(nf_attempted, nf_pass + nf_fail, nf_pass, nf_fail, nf_na, PASS if nf_attempted and nf_na == 0 and nf_fail == 0 else BLOCKED, "LOS direct energy only; no full-window energy or WER monotonicity asserted")
    return {
        "direction": _formal_gate_counts(attempted, applicable, direction_pass, direction_fail, direction_na, direction_status, "side LOS sign expectation from controlled transform geometry"),
        "repeatability": _formal_gate_counts(repeat_attempted, repeat_applicable, repeat_pass, repeat_fail, repeat_na, repeat_status, "five repeats per LOS pose"),
        "sample_rate_ab": _formal_gate_counts(ab_attempted, ab_applicable, ab_pass, ab_fail, ab_na, ab_status, "native 16 kHz versus native 24 kHz resampled to 16 kHz"),
        "ray_tail_convergence": dict(_formal_gate_counts(convergence_attempted, convergence_applicable, 0, 0, convergence_na, convergence_status, "evidence recorded only; no numeric convergence tolerance is frozen"), evidence_recorded=convergence_evidence),
        "nlos_direct_metric_control": _formal_gate_counts(len(nlos_rows), 0, 0, 0, nlos_na, PASS if nlos_rows and nlos_na == len(nlos_rows) else BLOCKED, "NLOS direct-path metrics are N/A by contract"),
        "mirror_symmetry_direct": mirror_gate,
        "near_far_los_direct_energy": near_far_gate,
        "details": {
            "repeatability": repeat_details,
            "sample_rate_ab": ab_details,
            "sample_rate_ab_summary": {
                "itd_pass_count": sum(bool(item["criterion"]["itd_pass"]) for item in ab_details),
                "ild_pass_count": sum(bool(item["criterion"]["ild_pass"]) for item in ab_details),
                "energy_pass_count": sum(bool(item["criterion"]["energy_pass"]) for item in ab_details),
                "joint_pass_count": ab_pass,
                "attempted": ab_attempted,
                "applicable": ab_applicable,
                "applicable_fraction": ab_applicable_fraction,
                "joint_pass_fraction": ab_joint_pass_fraction,
                "standard_interpretation": "at least 95% of attempted cases must be applicable and jointly pass; N/A is reported separately",
            },
            "ray_tail_convergence": convergence_details,
        },
    }


def _run_controlled_qualification(contract: Mapping[str, Any], runtime_config: Mapping[str, Any], registry: Mapping[str, Any], a0_contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Render only the registered deterministic geometry cases."""

    from active_audition.acoustics.rir import render_native_rir
    from active_audition.receiver.audit import _receiver_observation, _runtime_fingerprint
    from active_audition.scene.simulator import create_scene_simulator
    from active_audition.types import ListenerPose

    cases = enumerate_controlled_cases(registry, int(contract["gates"]["repeatability"]["repeat_count"]))
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    config_base = copy.deepcopy(dict(runtime_config))
    records: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []
    baseline_effective: Optional[Mapping[str, Any]] = None
    baseline_fingerprint: Optional[Mapping[str, Any]] = None
    baseline_receiver: Optional[Mapping[str, Any]] = None

    groups: Dict[Tuple[str, int, str, str], List[Mapping[str, Any]]] = {}
    for spec in cases:
        key = (spec["geometry"]["id"], int(spec["sample_rate_hz"]), spec["ray_variant"], spec["ir_variant"])
        groups.setdefault(key, []).append(spec)

    for key, group in sorted(groups.items()):
        geometry = group[0]["geometry"]
        _, rate, ray_variant, ir_variant = key
        config = copy.deepcopy(config_base)
        config["acoustics"] = dict(config_base["acoustics"])
        config["acoustics"]["sample_rate_hz"] = rate
        overrides: Dict[str, Any] = {}
        try:
            if ray_variant != "baseline" and baseline_effective is None:
                raise QualificationError("baseline effective ray preset is unavailable")
            if ray_variant == "rays_x2":
                overrides.update({field: int(baseline_effective[field] * 2) for field in ("directRayCount", "sourceRayCount", "indirectRayCount")})
            if ir_variant == "ir_x2":
                overrides["maxIRLength"] = float(baseline_effective["maxIRLength"]) * 2.0
            scene_override = {"scene_asset": geometry["mesh_path"]}
            with create_scene_simulator(config, scene_id=geometry["id"], acoustics_overrides=overrides, require_navmesh=False, load_semantic_mesh=False, scene_override=scene_override) as context:
                fingerprint = _runtime_fingerprint(repo_root)
                receiver = _receiver_observation(a0_contract, config, context)
                effective_record = receiver["acoustics_config_effective"]
                effective = effective_record.get("value") if isinstance(effective_record, Mapping) and effective_record.get("status") == KNOWN else effective_record
                if not isinstance(effective, Mapping):
                    raise QualificationError("controlled runtime did not expose effective acoustics config")
                if baseline_effective is None and rate == 16000 and ray_variant == "baseline" and ir_variant == "baseline":
                    baseline_effective = dict(effective)
                    baseline_fingerprint = fingerprint
                    baseline_receiver = receiver
                for spec in group:
                    source = spec["transforms"]["source_position_world"]
                    receiver_sensor = spec["transforms"]["receiver_position_world"]
                    offset = np.asarray(config["listener"]["sensor_offset_m"], dtype=np.float64)
                    base = tuple((np.asarray(receiver_sensor, dtype=np.float64) - offset).tolist())
                    pose = ListenerPose(base_position_world=base, sensor_position_world=tuple(receiver_sensor), yaw_deg=float(spec["transforms"]["receiver_yaw_deg"]))
                    rir = render_native_rir(context, source, pose)
                    native_metrics = compute_direct_metrics(rir, rate, contract)
                    metrics16 = native_metrics if rate == 16000 else compute_direct_metrics(resample_array(rir, 24000, 16000), 16000, contract)
                    resource_hashes = {"geometry_registry": registry["registry_sha256"], "scene_asset": geometry["mesh_sha256"]}
                    case_id = "{}_{}_d{}_a{}_r{}_{}_{}".format(geometry["id"], spec["purpose"], str(spec["distance_m"]).replace(".", "p"), str(spec["relative_angle_deg"]).replace("-", "m").replace(".", "p"), spec["repeat_id"], rate, spec["ray_variant"] + "_" + spec["ir_variant"])
                    nlos = spec["line_of_sight"] == "NLOS"
                    applicability = NA if nlos else metrics16["applicability"]
                    reason = "NLOS_DIRECT_PATH_BY_CONTRACT" if nlos else metrics16.get("na_reason")
                    status = NA if applicability == NA else PASS
                    row = _case_row(case_id=case_id, case_type="controlled_qualification", geometry_id=geometry["id"], geometry={"registry_sha256": registry["registry_sha256"], "mesh_path": geometry["mesh_path"], "mesh_sha256": geometry["mesh_sha256"], "purpose": spec["purpose"], "generator_version": registry["generator_version"]}, source_transform={"position_world": list(source)}, receiver_transform={"sensor_position_world": list(receiver_sensor), "base_position_world": list(base), "yaw_deg": float(spec["transforms"]["receiver_yaw_deg"])}, line_of_sight=spec["line_of_sight"], relative_angle_deg=float(spec["relative_angle_deg"]), distance_m=float(spec["distance_m"]), probe="impulse_response", probe_domain="rir", probe_reference="impulse_response", sample_rate_hz=rate, ray_preset=dict(effective), ir_length_sec=float(effective["maxIRLength"]), repeat_id=int(spec["repeat_id"]), applicability=applicability, raw_metrics={"native": native_metrics, "resampled_to_16khz": metrics16 if rate == 24000 else None}, status=status, reason=reason, runtime_sha256=_sha256_json(fingerprint), contract_sha256=metric_contract_sha256(contract), resource_hashes=resource_hashes)
                    records[case_id] = {"spec": spec, "row": row, "native_metrics": native_metrics, "metrics16": metrics16, "resampled_metrics16": metrics16 if rate == 24000 else None}
        except Exception as exc:  # pragma: no cover - requires the real SS2 runtime
            errors.append({"geometry_id": geometry["id"], "sample_rate_hz": rate, "ray_variant": ray_variant, "ir_variant": ir_variant, "type": type(exc).__name__, "message": str(exc)})

    rows = [record["row"] for record in records.values()]
    gates = _evaluate_formal_gates(records, contract) if not errors else {name: _gate_counts(BLOCKED, "CONTROLLED_RUNTIME_RENDER_ERROR") for name in FORMAL_GATE_NAMES}
    geometry_sanity_results = {entry["id"]: geometry_sanity(entry) for entry in registry["geometries"]}
    complete = not errors and len(rows) == len(cases)
    gate_values = [gates[name]["status"] for name in HARD_GATE_NAMES if name in gates]
    overall = PASS if complete and all(value == PASS for value in gate_values) else BLOCKED
    blocker = None if overall == PASS else ("CONTROLLED_RUNTIME_RENDER_ERROR" if errors else "A2_FORMAL_GATE_FAILURE")
    return {
        "status": overall,
        "blocker": blocker,
        "cases": rows,
        "records": records,
        "gates": gates,
        "geometry_sanity": geometry_sanity_results,
        "errors": errors,
        "runtime_lock": {"schema_version": A2_RUNTIME_LOCK_SCHEMA_VERSION, "gate": "A2", "status": overall, "contract_sha256": metric_contract_sha256(contract), "runtime_fingerprint": baseline_fingerprint or {}, "receiver_effective": dict(baseline_effective or {}), "resource_hashes": {entry["id"]: entry["mesh_sha256"] for entry in registry["geometries"]}, "geometry_registry_reason": "VALIDATED"},
        "effective_baseline_ray_preset": dict(baseline_effective or {}),
        "effective_baseline_ir_length_sec": (baseline_effective or {}).get("maxIRLength"),
    }


def load_a0_contract_for_qualification(runtime_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load the A0 contract paired with the legacy runtime config."""

    from active_audition.config.v1 import load_resolved_config

    repo_root = Path(runtime_config["_repo_root"])
    return load_resolved_config(str(repo_root / "configs/active_audition/v1/experiment_contract.yaml"))


def _gate_counts(status: str, reason: str) -> Dict[str, Any]:
    return {
        "status": status,
        "attempted": 0,
        "applicable": 0,
        "pass": 0,
        "fail": 0,
        "na": 0,
        "reason": reason,
    }


def run_a2_qualification(
    metric_contract_path: str,
    output_dir: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
    scene_id: str = "replica.office_0",
) -> Dict[str, Any]:
    """Run the A2 fixtures, calibration, and registered controlled geometry."""

    contract = load_metric_contract(metric_contract_path)
    contract_digest = metric_contract_sha256(contract)
    from active_audition.config.loader import load_resolved_config as load_legacy_config
    from active_audition.config.v1 import contract_sha256 as a0_contract_sha256

    runtime_config = load_legacy_config(runtime_config_path)
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    runtime_contract = load_a0_contract_for_qualification(runtime_config)
    geometry_path, geometry_reason = _load_geometry_registry(contract, repo_root)
    fixtures = run_synthetic_metric_fixtures(contract)
    calibration = run_resampler_band_calibration(contract)
    probe_diagnostics = run_probe_domain_diagnostics(contract)
    smoke: Dict[str, Any] = {"status": "NOT_RUN_FORMAL_CONTROLLED_GEOMETRY", "cases": [], "errors": []}
    formal: Dict[str, Any]
    registry: Optional[Mapping[str, Any]] = None
    direction_channel_calibration: Dict[str, Any] = {
        "schema_version": "active-asr-a2-direction-channel-calibration-v1",
        "status": "NOT_RUN",
        "formal_denominator": False,
        "cases": [],
        "errors": [],
        "hypothesis_summary": {},
    }
    if geometry_reason is not None:
        smoke = _runtime_smoke(contract, runtime_config, scene_id, geometry_reason)
        formal = {"status": BLOCKED, "blocker": geometry_reason, "cases": [], "gates": {name: _gate_counts(BLOCKED, geometry_reason) for name in FORMAL_GATE_NAMES}, "errors": []}
    elif calibration["status"] != PASS:
        formal = {"status": BLOCKED, "blocker": "RESAMPLER_CALIBRATION_EXCEEDS_FROZEN_AB_TOLERANCE", "cases": [], "gates": {name: _gate_counts(BLOCKED, "RESAMPLER_CALIBRATION_EXCEEDS_FROZEN_AB_TOLERANCE") for name in FORMAL_GATE_NAMES}, "errors": []}
        registry = load_geometry_registry(str(geometry_path), str(repo_root))
    else:
        registry = load_geometry_registry(str(geometry_path), str(repo_root))
        direction_channel_calibration = _run_direction_channel_calibration(contract, runtime_config, registry, runtime_contract)
        formal = _run_controlled_qualification(contract, runtime_config, registry, runtime_contract)

    cases: List[Dict[str, Any]] = list(formal.get("cases", []))
    if geometry_reason is not None:
        cases.append(_case_row(case_id="controlled_geometry_required", case_type="controlled_geometry_requirement", geometry_id=None, geometry={"registry_path": str(geometry_path), "authority": "NOT_AVAILABLE"}, source_transform={}, receiver_transform={}, line_of_sight="UNKNOWN", relative_angle_deg=None, distance_m=None, probe="impulse_response", probe_domain="rir", probe_reference="impulse_response", sample_rate_hz=16000, ray_preset={}, ir_length_sec=None, repeat_id=0, applicability=NA, raw_metrics=None, status=NA, reason=geometry_reason, runtime_sha256=_sha256_json(smoke.get("runtime_lock", {})), contract_sha256=contract_digest, resource_hashes=smoke.get("runtime_lock", {}).get("resource_hashes", {})))
    for row in cases:
        validate_qualification_case(row)

    blocker = formal.get("blocker")
    overall_status = formal.get("status", BLOCKED)
    gates = formal.get("gates", {name: _gate_counts(BLOCKED, blocker or "A2_NOT_RUN") for name in FORMAL_GATE_NAMES})
    if registry is not None and "nlos_direct_metric_control" in gates:
        gates = dict(gates)
    runtime_lock = formal.get("runtime_lock")
    if not runtime_lock:
        runtime_lock = smoke.get("runtime_lock", {"schema_version": A2_RUNTIME_LOCK_SCHEMA_VERSION, "gate": "A2", "status": BLOCKED, "contract_sha256": contract_digest, "runtime_fingerprint": {}, "receiver_effective": {}, "resource_hashes": {}, "geometry_registry_reason": geometry_reason or "NOT_RUN"})
    if runtime_lock["status"] == PASS and overall_status != PASS:
        runtime_lock = dict(runtime_lock, status=BLOCKED)
    validate_a2_runtime_lock(runtime_lock)

    comparison_details = gates.get("details", {}).get("sample_rate_ab", []) if isinstance(gates, Mapping) else []
    sample_rate_ab = {
        "schema_version": "active-asr-a2-sample-rate-ab-v1",
        "status": gates.get("sample_rate_ab", {}).get("status", BLOCKED),
        "applicability": "APPLICABLE" if gates.get("sample_rate_ab", {}).get("applicable", 0) else NA,
        "reason": gates.get("sample_rate_ab", {}).get("reason") or blocker,
        "resampler_calibration": calibration,
        "comparisons": comparison_details,
        "criterion_summary": gates.get("details", {}).get("sample_rate_ab_summary", {}),
        "tolerances": {"itd_difference_max_samples": contract["gates"]["sample_rate_ab"]["itd_difference_max_samples"], "band_ild_difference_max_db": contract["gates"]["sample_rate_ab"]["band_ild_difference_max_db"], "band_energy_difference_max_db": contract["gates"]["sample_rate_ab"]["band_energy_difference_max_db"]},
        "normalization": "none",
    }
    validate_sample_rate_ab(sample_rate_ab)
    pattern_by_angle_distance: Dict[str, Any] = {}
    for detail in comparison_details:
        pattern_by_angle_distance["d{}_a{}".format(detail["distance_m"], detail["angle_deg"])] = {
            "distance_m": detail["distance_m"],
            "angle_deg": detail["angle_deg"],
            "itd_pass": detail["criterion"]["itd_pass"],
            "ild_pass": detail["criterion"]["ild_pass"],
            "energy_pass": detail["criterion"]["energy_pass"],
            "joint_pass": detail["criterion"]["joint_pass"],
        }
    failure_attribution = {
        "schema_version": "active-asr-a2-failure-attribution-v1",
        "status": "ATTRIBUTION_RECORDED",
        "formal_a2_status": overall_status,
        "direction": direction_channel_calibration,
        "relative_angle_audit": direction_channel_calibration.get("convention_audit", {}),
        "sample_rate_ab": {
            "diagnostic_rows": comparison_details,
            "criterion_summary": sample_rate_ab["criterion_summary"],
            "pattern_by_angle_distance": pattern_by_angle_distance,
            "candidate_attribution": {
                "A_native16_vs_native24_physical_time": "native onset seconds and native ITD seconds are recorded; this is a timing-base comparison, not a causal physical isolation",
                "B_native24_to_16_resampled_onset_estimator": "resampled onset minus native24 onset projected to 16 kHz is recorded per channel and ITD",
                "C_renderer_amplitude_frequency_response": "direct and per-band ILD/energy differences are recorded without separate normalization; renderer/filter effects are not isolated by this pair",
                "D_stochastic_rays": "five-repeat baseline spread is recorded as a proxy and is explicitly not isolated from other runtime variation",
            },
        },
        "probe_domains": probe_diagnostics,
        "gate_semantics": {
            "sample_rate_applicable_case_min_fraction": contract["gates"]["sample_rate_ab"]["applicable_case_min_fraction"],
            "sample_rate_interpretation": "applicable fraction and joint-pass fraction are both evaluated against the registered 0.95 start gate; no unsupported 100-percent rule is used",
            "ray_tail_convergence": "EVIDENCE_RECORDED / NOT_YET_GATED; no numeric tolerance is frozen",
            "mirror_scope": "mirror_symmetry_direct; no listener-yaw-relative test was run",
        },
    }
    validate_direction_channel_calibration(direction_channel_calibration)
    validate_probe_domain_diagnostics(probe_diagnostics)
    validate_failure_attribution(failure_attribution)
    geometry_status = "VALIDATED" if geometry_reason is None else "BLOCKED"
    geometry_summary: Dict[str, Any] = {"path": str(geometry_path), "status": geometry_status, "reason": geometry_reason}
    if registry is not None:
        geometry_summary.update({"registry_sha256": registry["registry_sha256"], "generator_version": registry["generator_version"], "entries": [{"id": item["id"], "mesh_path": item["mesh_path"], "mesh_sha256": item["mesh_sha256"]} for item in registry["geometries"]], "sanity": formal.get("geometry_sanity", {})})
    summary: Dict[str, Any] = {
        "schema_version": "active-asr-a2-summary-v1",
        "gate": "A2",
        "status": overall_status,
        "contract": {"path": str(Path(metric_contract_path).resolve()), "sha256": contract_digest},
        "a0_contract_sha256": a0_contract_sha256(runtime_contract),
        "qualification_geometry": geometry_summary,
        "synthetic_metric_fixtures": fixtures,
        "resampler_calibration": calibration,
        "probe_domain_diagnostics": probe_diagnostics,
        "direction_channel_calibration": direction_channel_calibration,
        "failure_attribution": {"path": "a2_failure_attribution.json", "status": failure_attribution["status"]},
        "scene_technical_smoke": {"status": smoke.get("status"), "case_count": len(smoke.get("cases", [])), "errors": smoke.get("errors", []), "not_physics_evidence": True},
        "formal_runtime": {"errors": formal.get("errors", []), "case_count": len(formal.get("cases", []))},
        "ir_length_provenance": {"formal_contract_default_sec": {"status": KNOWN, "value": 2.0, "source": "Issue #2 mixed-contract A2 default; A0 schema is unchanged"}, "legacy_config_requested_max_ir_length_sec": {"status": "NOT_REQUESTED", "value": None, "source": "V0 runtime config does not request acousticsConfig.maxIRLength"}, "runtime_effective_max_ir_length_sec": {"status": KNOWN if formal.get("effective_baseline_ir_length_sec") is not None else "UNKNOWN", "value": formal.get("effective_baseline_ir_length_sec"), "source": "A1 public AudioSensor.acousticsConfig readback on controlled geometry"}, "a2_adoption_decision": "QUALIFICATION_EVIDENCE_RECORDED_NO_A0_OR_A1_VALUE_CHANGED"},
        "qualification_case_counts": {"total_case_rows": len(cases), "technical_smoke_na": 0, "controlled_geometry_requirement_na": sum(1 for row in cases if row["case_type"] == "controlled_geometry_requirement"), "formal_applicable": sum(1 for row in cases if row["case_type"] == "controlled_qualification" and row["applicability"] == "APPLICABLE")},
        "gates": gates,
        "n_a_policy": "N/A is reported separately and cannot produce PASS by denominator reduction",
        "a2_boundary": ["no ASR", "no mixing", "no oracle", "no Gymnasium/RL", "no A3 artifact freeze"],
        "blocker": blocker,
    }
    output_root = Path(output_dir).resolve()
    storage = DatasetStorage(str(output_root))
    storage.atomic_write_json(output_root / "runtime.lock.json", runtime_lock)
    storage.atomic_write_text(output_root / "qualification_cases.jsonl", "".join(json_line(row) + "\n" for row in cases))
    storage.atomic_write_json(output_root / "sample_rate_ab.json", sample_rate_ab)
    storage.atomic_write_json(output_root / "direction_channel_calibration.json", direction_channel_calibration)
    storage.atomic_write_json(output_root / "probe_domain_diagnostics.json", probe_diagnostics)
    storage.atomic_write_json(output_root / "a2_failure_attribution.json", failure_attribution)
    summary["artifacts"] = {name: {"sha256": _file_sha256(output_root / name)} for name in ("runtime.lock.json", "qualification_cases.jsonl", "sample_rate_ab.json", "direction_channel_calibration.json", "probe_domain_diagnostics.json", "a2_failure_attribution.json")}
    storage.atomic_write_json(output_root / "a2_summary.json", summary)
    summary_digest = _file_sha256(output_root / "a2_summary.json")
    storage.atomic_write_text(output_root / "a2_report.md", _render_report(summary, summary_digest, smoke, fixtures))
    return {"status": overall_status, "contract_sha256": contract_digest, "output_dir": str(output_root), "artifacts": {name: {"path": str(output_root / name), "sha256": _file_sha256(output_root / name)} for name in ("runtime.lock.json", "qualification_cases.jsonl", "sample_rate_ab.json", "direction_channel_calibration.json", "probe_domain_diagnostics.json", "a2_failure_attribution.json", "a2_summary.json", "a2_report.md")}, "blocker": blocker, "gates": gates}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_report(summary: Mapping[str, Any], summary_sha256: str, smoke: Mapping[str, Any], fixtures: Mapping[str, Any]) -> str:
    geometry = summary.get("qualification_geometry", {})
    calibration = summary.get("resampler_calibration", {})
    lines = [
        "# Active-ASR V1.1 A2 Physics Qualification",
        "",
        "- Status: **{}**".format(summary["status"]),
        "- Metric contract SHA256: `{}`".format(summary["contract"]["sha256"]),
        "- Summary SHA256: `{}`".format(summary_sha256),
        "- Controlled geometry: **{}**{}".format(geometry.get("status", "UNKNOWN"), " — {}".format(summary["blocker"]) if summary.get("blocker") else ""),
        "",
        "## Frozen metric definition",
        "",
        "ITD is `tR - tL` using the first threshold crossing in the frozen direct window. ILD is `10*log10(E_L/E_R)` over the common direct window. The post-direct tail value is a diagnostic named `post_direct_tail_energy_relative_to_direct`; it is not DRR or RT60.",
        "",
        "Direct search: 0–50 ms, first absolute sample at 10% of each channel peak (minimum peak 1e-8), 10 ms window from the earlier onset through the later onset plus the window length. Low energy/no onset is N/A.",
        "",
        "## Synthetic fixtures",
        "",
        "- Fixture status: **{}**".format(fixtures["status"]),
        "- The known-delay/gain impulse, sign reversal, low-energy N/A, and 24 kHz→16 kHz resampling convention were executed before any runtime qualification.",
        "- Formal rows are RIR-domain cases with `probe/reference=impulse_response`; broadband_noise, chirp, and voiced_tone are separate waveform-domain diagnostics only.",
        "",
        "## Geometry and resampler sanity",
        "",
        "- Registry status: `{}`; registry SHA256: `{}`.".format(geometry.get("status"), geometry.get("registry_sha256")),
        "- Minimum analytic first-reflection extra delay: `{:.6f} ms`; direct-window-clear: `{}`.".format(min((item.get("minimum_first_reflection_extra_time_ms", 0.0) for item in geometry.get("sanity", {}).values()), default=0.0), all(item.get("all_direct_windows_clear", False) for item in geometry.get("sanity", {}).values()) if geometry.get("sanity") else False),
        "- Resampler calibration: `{}`; maximum absolute energy bias: `{:.6f} dB`; decision: `{}`.".format(calibration.get("status"), float(calibration.get("max_absolute_energy_bias_db", 0.0)), calibration.get("decision")),
        "- Direction channel calibration: H1 accuracy `{}`; H2 accuracy `{}`; decision `{}`.".format(summary.get("direction_channel_calibration", {}).get("hypothesis_summary", {}).get("H1_native_ch0_L_ch1_R", {}).get("sign_accuracy"), summary.get("direction_channel_calibration", {}).get("hypothesis_summary", {}).get("H2_native_ch0_R_ch1_L", {}).get("sign_accuracy"), summary.get("direction_channel_calibration", {}).get("hypothesis_summary", {}).get("mapping_decision")),
        "- Relative-angle audit: geometry and scene pose use positive-left; raw native channel evidence is stored before canonical mapping.",
        "",
        "## Runtime evidence",
        "",
        "- Runtime smoke status: `{}`; case count: `{}`.".format(smoke.get("status"), len(smoke.get("cases", []))),
        "- office_0 is never included in the formal denominator; this run used the registered controlled geometry only.",
        "- A1 effective ray preset and IR length are recorded in `runtime.lock.json`; no A0/A1 value was silently changed.",
        "- IR provenance remains distinct: Issue #2 formal default 2 s, legacy config did not request maxIRLength, and current A1 runtime readback is recorded separately; this run reports effective controlled-runtime IR length without changing A0/A1.",
        "",
        "## Failure attribution and gate semantics",
        "",
        "- Failure-attribution artifact: `a2_failure_attribution.json`; probe-domain artifact is not part of any formal denominator.",
        "- Sample-rate criterion counts ITD/ILD/energy/joint: `{}/{}/{}/{}`; applicable fraction `{}`; joint-pass fraction `{}`.".format(summary.get("gates", {}).get("details", {}).get("sample_rate_ab_summary", {}).get("itd_pass_count", 0), summary.get("gates", {}).get("details", {}).get("sample_rate_ab_summary", {}).get("ild_pass_count", 0), summary.get("gates", {}).get("details", {}).get("sample_rate_ab_summary", {}).get("energy_pass_count", 0), summary.get("gates", {}).get("details", {}).get("sample_rate_ab_summary", {}).get("joint_pass_count", 0), summary.get("gates", {}).get("details", {}).get("sample_rate_ab_summary", {}).get("applicable_fraction", 0.0), summary.get("gates", {}).get("details", {}).get("sample_rate_ab_summary", {}).get("joint_pass_fraction", 0.0)),
        "- A/B attribution records native-time, resampled-onset, amplitude/frequency, and repeat-ray proxy fields; it does not claim causal isolation.",
        "- Ray/tail is `EVIDENCE_RECORDED / NOT_YET_GATED`; mirror scope is `mirror_symmetry_direct`, not yaw-relative invariance.",
        "",
        "## Gate counts",
        "",
        "| Gate | Attempted | Applicable | PASS | FAIL | N/A | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, gate in summary["gates"].items():
        if not isinstance(gate, Mapping) or "attempted" not in gate:
            continue
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(name, gate["attempted"], gate["applicable"], gate["pass"], gate["fail"], gate["na"], gate["status"]))
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "A2 remains OPEN/BLOCKED. No ITD/ILD physics claim, HRTF/ear-spacing inference, ASR, mixing, Oracle, Gymnasium/RL, or A3 integration was performed.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "A2_RUNTIME_LOCK_SCHEMA_VERSION",
    "BLOCKED",
    "ENERGY_METRIC_CONVENTION_SCHEMA_VERSION",
    "FAIL",
    "FRACTIONAL_DELAY_CALIBRATION_SCHEMA_VERSION",
    "METRIC_CONTRACT_SCHEMA_VERSION",
    "NA",
    "PASS",
    "QUALIFICATION_CASE_SCHEMA_VERSION",
    "QualificationError",
    "RIR_SAMPLE_RATE_CONVENTION_SCHEMA_VERSION",
    "SAMPLE_RATE_ATTRIBUTION_SCHEMA_VERSION",
    "canonical_json",
    "compute_direct_metrics",
    "energy_metric_convention_audit",
    "load_metric_contract",
    "make_probe",
    "metric_contract_sha256",
    "run_a2_qualification",
    "run_a2_sample_rate_blocker_calibration",
    "run_probe_domain_diagnostics",
    "run_fractional_delay_estimator_calibration",
    "run_rir_sample_rate_convention_calibration",
    "run_resampler_band_calibration",
    "run_synthetic_metric_fixtures",
    "validate_a2_runtime_lock",
    "validate_direction_channel_calibration",
    "validate_failure_attribution",
    "validate_energy_metric_convention",
    "validate_fractional_delay_calibration",
    "validate_metric_contract",
    "validate_probe_domain_diagnostics",
    "validate_qualification_case",
    "validate_rir_sample_rate_convention_calibration",
    "validate_sample_rate_ab",
]
