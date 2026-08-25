"""Frozen M0.1 dry-audio preparation contract."""

from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


class DryAudioError(ValueError):
    """Raised when a dry-audio segment violates the frozen contract."""


GOLDEN_PROBE_SEED = 20260824
GOLDEN_PROBE_SAMPLE_RATE_HZ = 16000
GOLDEN_PROBE_NUM_SAMPLES = 80000
GOLDEN_PROBE_SCALE = 0.05


def generate_golden_probe(
    seed: int = GOLDEN_PROBE_SEED,
    num_samples: int = GOLDEN_PROBE_NUM_SAMPLES,
    fixed_scale: float = GOLDEN_PROBE_SCALE,
) -> np.ndarray:
    """Generate the deterministic V0 synthetic probe without normalization."""

    rng = np.random.Generator(np.random.PCG64(seed))
    probe = rng.standard_normal(num_samples).astype(np.float32)
    probe *= np.float32(fixed_scale)
    return probe


def _as_mono_float32(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        mono = array
    elif array.ndim == 2:
        mono = array.mean(axis=1, dtype=np.float32)
    else:
        raise DryAudioError("dry audio must be one- or two-dimensional")
    if not np.isfinite(mono).all():
        raise DryAudioError("dry audio contains NaN/Inf")
    return mono


def resample_waveform(
    waveform: np.ndarray,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int = GOLDEN_PROBE_SAMPLE_RATE_HZ,
    expected_num_samples: Optional[int] = None,
) -> np.ndarray:
    """Convert numeric audio to canonical mono float32 using resample_poly."""

    if int(source_sample_rate_hz) <= 0 or int(target_sample_rate_hz) <= 0:
        raise DryAudioError("sample rates must be positive")
    mono = _as_mono_float32(waveform)
    source_rate = int(source_sample_rate_hz)
    target_rate = int(target_sample_rate_hz)
    if source_rate == target_rate:
        canonical = mono.astype(np.float32, copy=False)
    else:
        divisor = gcd(source_rate, target_rate)
        canonical = resample_poly(
            mono,
            up=target_rate // divisor,
            down=source_rate // divisor,
            axis=0,
            window=("kaiser", 5.0),
            padtype="constant",
        ).astype(np.float32)
    if expected_num_samples is not None and canonical.shape[0] != int(expected_num_samples):
        raise DryAudioError(
            "canonical sample count mismatch: expected {}, got {}".format(
                int(expected_num_samples), canonical.shape[0]
            )
        )
    if not np.isfinite(canonical).all():
        raise DryAudioError("canonical dry audio contains NaN/Inf")
    return canonical


def prepare_dry_segment(
    waveform: np.ndarray,
    source_sample_rate_hz: int,
    segment_start_sec: float,
    segment_duration_sec: float,
    target_sample_rate_hz: int = GOLDEN_PROBE_SAMPLE_RATE_HZ,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Prepare an exact source-domain segment, then canonicalize and gain it."""

    source_rate = int(source_sample_rate_hz)
    if source_rate <= 0 or segment_start_sec < 0 or segment_duration_sec <= 0:
        raise DryAudioError("invalid source segment or sample rate")
    source_start = int(round(float(segment_start_sec) * source_rate))
    source_count = int(round(float(segment_duration_sec) * source_rate))
    source = np.asarray(waveform)
    if source.ndim not in (1, 2) or source.shape[0] < source_start + source_count:
        raise DryAudioError("source audio is shorter than the requested segment")
    segment = source[source_start : source_start + source_count]
    target_count = int(round(float(segment_duration_sec) * int(target_sample_rate_hz)))
    canonical = resample_waveform(
        segment,
        source_rate,
        target_sample_rate_hz,
        expected_num_samples=target_count,
    )
    if float(gain_db) != 0.0:
        canonical = (canonical * np.float32(10.0 ** (float(gain_db) / 20.0))).astype(
            np.float32
        )
    return canonical


def load_dry_segment(
    path: str,
    segment_start_sec: float,
    segment_duration_sec: float,
    target_sample_rate_hz: int = GOLDEN_PROBE_SAMPLE_RATE_HZ,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Load a WAV and apply the frozen dry-audio preparation contract."""

    source_rate, waveform = wavfile.read(str(Path(path)))
    return prepare_dry_segment(
        waveform,
        source_rate,
        segment_start_sec,
        segment_duration_sec,
        target_sample_rate_hz,
        gain_db,
    )
