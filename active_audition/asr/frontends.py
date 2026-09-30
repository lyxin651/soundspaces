"""Frozen, reference-independent binaural ASR frontends."""

from typing import Any

import numpy as np


class FrontendError(ValueError):
    """Raised when a canonical binaural waveform is invalid."""


FRONTENDS = ("mean_lr", "fixed_L", "fixed_R")


def _canonical_binaural(waveform: Any) -> np.ndarray:
    value = np.asarray(waveform)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 2:
        raise FrontendError("binaural waveform must have non-empty shape (T,2)")
    if not np.issubdtype(value.dtype, np.floating):
        raise FrontendError("binaural waveform must use floating-point samples")
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FrontendError("binaural waveform contains NaN/Inf")
    return result


def apply_frontend(waveform: Any, frontend: str) -> np.ndarray:
    """Return one deterministic float32 mono view of canonical ``[L,R]``."""

    value = _canonical_binaural(waveform)
    if frontend == "mean_lr":
        return ((value[:, 0] + value[:, 1]) * np.float32(0.5)).astype(np.float32)
    if frontend == "fixed_L":
        return value[:, 0].copy()
    if frontend == "fixed_R":
        return value[:, 1].copy()
    raise FrontendError("unknown frozen frontend: {}".format(frontend))


def mean_lr(waveform: Any) -> np.ndarray:
    return apply_frontend(waveform, "mean_lr")


def fixed_l(waveform: Any) -> np.ndarray:
    return apply_frontend(waveform, "fixed_L")


def fixed_r(waveform: Any) -> np.ndarray:
    return apply_frontend(waveform, "fixed_R")
