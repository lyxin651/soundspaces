"""One explicit sample-rate contract for dry audio and precomputed RIRs."""

from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly


INTERMEDIATE_SAMPLE_RATE_HZ = 16000
OUTPUT_SAMPLE_RATE_HZ = 24000
RESAMPLING_CONTRACT_VERSION = "v0.5-resample-poly-16k-intermediate-24k-v1"


class ResamplingError(ValueError):
    pass


def resample_array(array: Any, source_sample_rate_hz: int, target_sample_rate_hz: int) -> np.ndarray:
    """Resample the first axis, retaining channels and applying no gain correction."""
    source = int(source_sample_rate_hz)
    target = int(target_sample_rate_hz)
    value = np.asarray(array)
    if source <= 0 or target <= 0 or value.ndim not in (1, 2) or value.shape[0] == 0:
        raise ResamplingError("array and sample rates are invalid")
    if not np.isfinite(value).all():
        raise ResamplingError("array contains NaN/Inf")
    value = np.asarray(value, dtype=np.float32)
    if source == target:
        return value.copy()
    divisor = gcd(source, target)
    result = resample_poly(value, target // divisor, source // divisor, axis=0,
                           window=("kaiser", 5.0), padtype="constant")
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all() or result.shape[0] == 0:
        raise ResamplingError("resampling produced invalid output")
    return result


def canonicalize_rir(rir: Any, original_sample_rate_hz: int) -> np.ndarray:
    """Official RIR -> 16 kHz acoustic intermediate -> 24 kHz interface."""
    value = np.asarray(rir, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] == 0:
        raise ResamplingError("RIR must have shape (N, 2), N > 0")
    intermediate = resample_array(value, int(original_sample_rate_hz), INTERMEDIATE_SAMPLE_RATE_HZ)
    return resample_array(intermediate, INTERMEDIATE_SAMPLE_RATE_HZ, OUTPUT_SAMPLE_RATE_HZ)
