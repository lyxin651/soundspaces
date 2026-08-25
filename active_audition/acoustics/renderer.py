"""Pure canonical dry/RIR convolution and float32 WAV output."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve


class RendererError(ValueError):
    """Raised for invalid canonical acoustic arrays."""


def convolve_binaural(dry: Any, rir: Any) -> np.ndarray:
    dry_array = np.asarray(dry, dtype=np.float32)
    rir_array = np.asarray(rir, dtype=np.float32)
    if dry_array.ndim != 1 or dry_array.size == 0:
        raise RendererError("dry audio must be a non-empty mono vector")
    if rir_array.ndim != 2 or rir_array.shape[1] != 2 or rir_array.shape[0] == 0:
        raise RendererError("canonical RIR must have shape (N, 2), N > 0")
    if not np.isfinite(dry_array).all() or not np.isfinite(rir_array).all():
        raise RendererError("dry audio or RIR contains NaN/Inf")
    result = np.column_stack(
        (
            fftconvolve(dry_array, rir_array[:, 0], mode="full"),
            fftconvolve(dry_array, rir_array[:, 1], mode="full"),
        )
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise RendererError("convolved waveform contains NaN/Inf")
    return result


def write_float32_wav(path: str, waveform: Any, sample_rate_hz: int = 16000) -> None:
    array = np.asarray(waveform)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise RendererError("WAV waveform must have shape (N, 2)")
    if array.dtype != np.float32:
        raise RendererError("WAV waveform must be float32")
    if not np.isfinite(array).all():
        raise RendererError("WAV waveform contains NaN/Inf")
    wavfile.write(str(Path(path)), int(sample_rate_hz), array)
