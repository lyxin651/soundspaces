"""ClassDOA V1 Step 1D canonical FOA fixture and WAV loader helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import soundfile as sf
from scipy.io import wavfile


SAMPLE_RATE_HZ = 24000
DURATION_SECONDS = 5.0
NUM_SAMPLES = 120000
NUM_CHANNELS = 4
CHANNEL_ORDER = ("W", "Y", "Z", "X")


def make_canonical_foa_fixture(kind: str = "directional") -> np.ndarray:
    """生成 deterministic 24k/5s/4ch float32 canonical FOA，不做归一化。"""
    sample_index = np.arange(NUM_SAMPLES, dtype=np.float32)
    t = sample_index / np.float32(SAMPLE_RATE_HZ)
    envelope = np.linspace(0.65, 1.0, NUM_SAMPLES, dtype=np.float32)

    if kind == "generic":
        channels = [
            0.08 * np.sin(2.0 * np.pi * 440.0 * t),
            0.03 * np.sin(2.0 * np.pi * 550.0 * t + 0.2),
            0.02 * np.sin(2.0 * np.pi * 660.0 * t + 0.4),
            0.04 * np.sin(2.0 * np.pi * 770.0 * t + 0.6),
        ]
    elif kind == "directional":
        base = np.sin(2.0 * np.pi * 500.0 * t) * envelope
        channels = [
            0.10 * base,
            0.06 * np.sin(2.0 * np.pi * 500.0 * t + 0.35) * envelope,
            0.035 * np.sin(2.0 * np.pi * 250.0 * t + 0.70) * envelope,
            -0.075 * np.sin(2.0 * np.pi * 500.0 * t - 0.20) * envelope,
        ]
    else:
        raise ValueError("unknown fixture kind: {}".format(kind))

    fixture = np.stack(channels, axis=1).astype(np.float32, copy=False)
    if fixture.shape != (NUM_SAMPLES, NUM_CHANNELS):
        raise AssertionError("unexpected fixture shape: {}".format(fixture.shape))
    return fixture


def waveform_stats(waveform: np.ndarray, sample_rate_hz: int) -> Dict[str, object]:
    array = np.asarray(waveform)
    if array.ndim != 2:
        raise ValueError("expected waveform with shape (samples, channels)")
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sample_rate_hz": int(sample_rate_hz),
        "num_samples": int(array.shape[0]),
        "channels": int(array.shape[1]),
        "duration_seconds": float(array.shape[0] / float(sample_rate_hz)),
        "per_channel_min": [float(v) for v in np.min(array, axis=0)],
        "per_channel_max": [float(v) for v in np.max(array, axis=0)],
        "per_channel_rms": [float(v) for v in np.sqrt(np.mean(np.square(array), axis=0))],
        "global_rms": float(np.sqrt(np.mean(np.square(array)))),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "finite": bool(finite.all()),
        "nonzero": bool(np.any(array != 0.0)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_float32_wav(path: Path, waveform: np.ndarray) -> Mapping[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(waveform, dtype=np.float32)
    if array.shape != (NUM_SAMPLES, NUM_CHANNELS):
        raise ValueError("expected shape ({}, {})".format(NUM_SAMPLES, NUM_CHANNELS))
    if not np.isfinite(array).all():
        raise ValueError("waveform must be finite")
    wavfile.write(str(path), SAMPLE_RATE_HZ, array)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "stats": waveform_stats(array, SAMPLE_RATE_HZ),
        "subtype": sf.info(str(path)).subtype,
    }


def load_wav_dtype_aware(path: Path, expected_sample_rate_hz: int = SAMPLE_RATE_HZ) -> Tuple[np.ndarray, int, str]:
    """读取 WAV 并按 dtype 安全转换到 float32，不做峰值/RMS 归一化。"""
    sample_rate_hz, raw = wavfile.read(str(path))
    if int(sample_rate_hz) != int(expected_sample_rate_hz):
        raise ValueError("unexpected sample rate: {}".format(sample_rate_hz))
    if raw.ndim != 2 or raw.shape[1] != NUM_CHANNELS:
        raise ValueError("expected 4-channel WAV, got {}".format(raw.shape))
    raw_dtype = str(raw.dtype)
    if np.issubdtype(raw.dtype, np.floating):
        audio = raw.astype(np.float32, copy=False)
    elif raw.dtype == np.int16:
        audio = raw.astype(np.float32) / np.float32(32768.0)
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float32) / np.float32(2147483648.0)
    else:
        raise ValueError("unsupported WAV dtype: {}".format(raw.dtype))
    if audio.shape[0] != NUM_SAMPLES:
        raise ValueError("expected {} samples, got {}".format(NUM_SAMPLES, audio.shape[0]))
    if not np.isfinite(audio).all():
        raise ValueError("loaded waveform must be finite")
    return audio, int(sample_rate_hz), raw_dtype


def simulate_int16_assumption_loader(path: Path, expected_sample_rate_hz: int = SAMPLE_RATE_HZ) -> Tuple[np.ndarray, int, str]:
    """模拟 `wav.read(...); audio / 32768` 这类 stock loader 假设。"""
    sample_rate_hz, raw = wavfile.read(str(path))
    if int(sample_rate_hz) != int(expected_sample_rate_hz):
        raise ValueError("unexpected sample rate: {}".format(sample_rate_hz))
    return raw.astype(np.float32) / np.float32(32768.0), int(sample_rate_hz), str(raw.dtype)


def scaling_comparison(path: Path) -> Dict[str, object]:
    aware, aware_sr, aware_dtype = load_wav_dtype_aware(path)
    stock, stock_sr, stock_dtype = simulate_int16_assumption_loader(path)
    raw_stats = waveform_stats(aware, aware_sr)
    stock_stats = waveform_stats(stock, stock_sr)
    ratio = float(stock_stats["global_rms"] / raw_stats["global_rms"])
    return {
        "raw": raw_stats,
        "stock_loader": {
            "dtype": stock_dtype,
            "stats": stock_stats,
            "attenuation_ratio": ratio,
            "attenuation_db": float(20.0 * np.log10(ratio)),
        },
        "dtype_aware_adapter": {
            "dtype": aware_dtype,
            "stats": raw_stats,
            "amplitude_preserved": bool(np.allclose(aware, make_canonical_foa_fixture("directional"))),
        },
    }
