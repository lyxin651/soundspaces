"""Frozen, model-agnostic acoustic QC metrics for Pipeline V0 M3."""

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.io import wavfile


class MetricsError(ValueError):
    """Raised when a canonical WAV/RIR cannot satisfy the M3 metric contract."""


ANALYSIS_SAMPLES = 80000
SILENCE_ABS_THRESHOLD = 1.0e-5
CLIPPING_ABS_THRESHOLD = 1.0
RMS_EPS_AMPLITUDE = 1.0e-12
ILD_EPS_POWER = 1.0e-12
MAX_LAG_SAMPLES = 16
RIR_TAIL_SAMPLES = 1600
RIR_EPS_POWER = 1.0e-12


def _finite_array(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise MetricsError("{} contains NaN/Inf".format(name))
    return array


def waveform_qc_window(waveform: np.ndarray, sample_rate_hz: int = 16000) -> np.ndarray:
    """Return the fixed first-five-second window without mutating input."""

    array = np.asarray(waveform)
    if int(sample_rate_hz) != 16000:
        raise MetricsError("M3 metrics require 16000 Hz")
    if array.ndim != 2 or array.shape[1] != 2:
        raise MetricsError("canonical WAV must have shape (N,2)")
    if array.dtype != np.float32:
        raise MetricsError("canonical WAV must be float32")
    _finite_array(array, "waveform")
    if array.shape[0] < ANALYSIS_SAMPLES:
        raise MetricsError("WAV shorter than fixed 80000-sample analysis window")
    return array[:ANALYSIS_SAMPLES].copy()


def _dbfs(amplitude: float) -> float:
    return float(20.0 * math.log10(max(float(amplitude), RMS_EPS_AMPLITUDE)))


def _lag_correlation(left: np.ndarray, right: np.ndarray, lag: int) -> float:
    if lag > 0:
        left_overlap, right_overlap = left[:-lag], right[lag:]
    elif lag < 0:
        left_overlap, right_overlap = left[-lag:], right[:lag]
    else:
        left_overlap, right_overlap = left, right
    left_centered = left_overlap - np.mean(left_overlap, dtype=np.float64)
    right_centered = right_overlap - np.mean(right_overlap, dtype=np.float64)
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / (denominator + RMS_EPS_AMPLITUDE))


def interaural_correlation_and_lag(
    left: np.ndarray, right: np.ndarray, max_lag_samples: int = MAX_LAG_SAMPLES
):
    correlations = {
        int(lag): _lag_correlation(left, right, int(lag))
        for lag in range(-int(max_lag_samples), int(max_lag_samples) + 1)
    }
    lag = min(correlations, key=lambda item: (-correlations[item], abs(item), item))
    return float(correlations[lag]), int(lag)


def compute_waveform_metrics(
    waveform: np.ndarray,
    sample_rate_hz: int = 16000,
    episode_id: str = "",
    viewpoint_id: str = "",
) -> Mapping[str, float]:
    """Compute fixed-window QC plus explicitly named full-signal diagnostics."""

    array = np.asarray(waveform)
    window = waveform_qc_window(array, sample_rate_hz)
    left, right = window[:, 0].astype(np.float64), window[:, 1].astype(np.float64)
    left_power, right_power = float(np.mean(left * left)), float(np.mean(right * right))
    rms_left, rms_right = math.sqrt(left_power), math.sqrt(right_power)
    rms_mean = math.sqrt(float(np.mean((left * left + right * right) / 2.0)))
    correlation, lag = interaural_correlation_and_lag(left, right)
    peak_left, peak_right = float(np.max(np.abs(left))), float(np.max(np.abs(right)))
    silent = np.maximum(np.abs(left), np.abs(right)) < SILENCE_ABS_THRESHOLD
    over_unit = np.abs(window.reshape(-1)) >= CLIPPING_ABS_THRESHOLD
    return {
        "episode_id": str(episode_id),
        "viewpoint_id": str(viewpoint_id),
        "sample_rate_hz": int(sample_rate_hz),
        "analysis_samples": int(ANALYSIS_SAMPLES),
        "rms_left_dbfs": _dbfs(rms_left),
        "rms_right_dbfs": _dbfs(rms_right),
        "rms_mean_dbfs": _dbfs(rms_mean),
        "ild_db": float(10.0 * math.log10((left_power + ILD_EPS_POWER) / (right_power + ILD_EPS_POWER))),
        "peak_left": peak_left,
        "peak_right": peak_right,
        "peak_max": max(peak_left, peak_right),
        "silence_fraction": float(np.mean(silent)),
        "clipping_fraction": float(np.mean(over_unit)),
        "interaural_correlation": correlation,
        "interaural_lag_samples": lag,
        "interaural_lag_sec": float(lag / float(sample_rate_hz)),
        "full_energy_left": float(np.sum(array[:, 0].astype(np.float64) ** 2)),
        "full_energy_right": float(np.sum(array[:, 1].astype(np.float64) ** 2)),
        "full_energy_total": float(np.sum(array.astype(np.float64) ** 2)),
        "full_num_samples": int(array.shape[0]),
        "wav_num_samples": int(array.shape[0]),
        "wav_duration_sec": float(array.shape[0] / float(sample_rate_hz)),
    }


def compute_rir_metrics(rir: np.ndarray, sample_rate_hz: int = 16000) -> Mapping[str, float]:
    array = np.asarray(rir)
    if int(sample_rate_hz) != 16000:
        raise MetricsError("M3 RIR metrics require 16000 Hz")
    if array.ndim != 2 or array.shape[1] != 2 or array.dtype != np.float32:
        raise MetricsError("canonical RIR must be float32 with shape (N,2)")
    _finite_array(array, "RIR")
    total_energy = float(np.sum(array.astype(np.float64) ** 2))
    tail_n = min(RIR_TAIL_SAMPLES, int(array.shape[0]))
    tail_energy = float(np.sum(array[-tail_n:].astype(np.float64) ** 2)) if tail_n else 0.0
    ratio = tail_energy / (total_energy + RIR_EPS_POWER)
    return {
        "rir_num_samples": int(array.shape[0]),
        "rir_tail_energy_ratio_100ms": float(ratio),
        "rir_tail_energy_db_100ms": float(10.0 * math.log10(ratio + RIR_EPS_POWER)),
    }


def compute_viewpoint_metrics(
    dataset_root: str, viewpoint: Mapping[str, object], sample_rate_hz: int = 16000
) -> Mapping[str, object]:
    root = Path(dataset_root)
    audio_path = root / str(viewpoint["audio_path"])
    rir_path = root / str(viewpoint["rir_path"])
    actual_sample_rate, waveform = wavfile.read(str(audio_path))
    if int(actual_sample_rate) != int(sample_rate_hz):
        raise MetricsError("WAV sample rate mismatch: {}".format(viewpoint.get("viewpoint_id")))
    with np.load(str(rir_path), allow_pickle=False) as payload:
        rir = payload["rir"]
        rir_sample_rate = int(payload["sample_rate_hz"])
    if rir_sample_rate != int(sample_rate_hz):
        raise MetricsError("RIR sample rate mismatch: {}".format(viewpoint.get("viewpoint_id")))
    result = dict(compute_waveform_metrics(waveform, actual_sample_rate, str(viewpoint["episode_id"]), str(viewpoint["viewpoint_id"])))
    result.update(compute_rir_metrics(rir, rir_sample_rate))
    return result
