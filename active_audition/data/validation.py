"""Hard dataset contract checks for the M2 debug dataset."""

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from scipy.io import wavfile

from active_audition.data.manifest import read_jsonl
from active_audition.data.storage import DatasetStorage


class ValidationError(ValueError):
    """Raised when the dataset has hard contract failures."""


def _failure(message: str, failures: list) -> None:
    failures.append(message)


def payload_is_complete(dataset_root: str, row: Mapping[str, Any], save_rir: bool = True) -> bool:
    """Check one manifest payload for resume decisions without changing it."""

    root = Path(dataset_root)
    audio = root / str(row.get("audio_path", ""))
    if not audio.is_file():
        return False
    try:
        sample_rate, waveform = wavfile.read(str(audio))
    except Exception:
        return False
    if sample_rate != int(row.get("sample_rate_hz", -1)) or waveform.ndim != 2:
        return False
    if waveform.dtype != np.float32 or not np.isfinite(waveform).all():
        return False
    if save_rir:
        rir_path = root / str(row.get("rir_path", ""))
        if not rir_path.is_file():
            return False
        try:
            with np.load(str(rir_path), allow_pickle=False) as payload:
                rir = payload["rir"]
                sr = int(payload["sample_rate_hz"])
                count = int(payload["num_samples"])
        except Exception:
            return False
        if rir.ndim != 2 or rir.shape[1] != 2 or rir.shape[0] != count:
            return False
        if sr != int(row.get("sample_rate_hz", -1)) or rir.dtype != np.float32:
            return False
        if not np.isfinite(rir).all():
            return False
    return True


def validate_dataset(dataset_root: str, config: Mapping[str, Any], require_success: bool = False) -> Dict[str, Any]:
    storage = DatasetStorage(dataset_root)
    failures = []
    warnings = []
    if require_success and not storage.success_path.exists():
        _failure("_SUCCESS is missing", failures)
    try:
        episodes = read_jsonl(str(storage.manifest_path("episodes.jsonl")))
        candidates = read_jsonl(str(storage.manifest_path("candidates.jsonl")))
        viewpoints = read_jsonl(str(storage.manifest_path("viewpoints.jsonl")))
    except Exception as exc:
        _failure("manifest read failure: {}".format(exc), failures)
        return {"status": "FAIL", "failures": failures, "warnings": warnings}

    episode_keys = [row.get("episode_id") for row in episodes]
    viewpoint_keys = [(row.get("episode_id"), row.get("viewpoint_id")) for row in viewpoints]
    if len(episode_keys) != len(set(episode_keys)):
        _failure("duplicate episode key", failures)
    if any(None in key for key in viewpoint_keys) or len(viewpoint_keys) != len(set(viewpoint_keys)):
        _failure("duplicate or missing viewpoint key", failures)
    if not episodes:
        _failure("initial episode manifest is empty", failures)

    expected_sr = int(config["acoustics"]["sample_rate_hz"])
    expected_channels = 2
    expected_dtype = np.dtype(config["acoustics"]["canonical_wav_dtype"])
    expected_dry_samples = int(round(float(config["dry_audio"]["segment_duration_sec"]) * expected_sr))
    episode_map = {row.get("episode_id"): row for row in episodes}
    dry_fingerprints = {}
    for row in viewpoints:
        episode_id = row.get("episode_id")
        if episode_id not in episode_map:
            _failure("viewpoint references missing episode: {}".format(episode_id), failures)
            continue
        if row.get("ground_truth", {}).get("scene_id") != config["scene"]["ids"][0]:
            _failure("scene metadata mismatch: {}".format(row.get("viewpoint_id")), failures)
        audio_path = storage.root / str(row.get("audio_path", ""))
        if not audio_path.is_file():
            _failure("missing audio: {}".format(audio_path), failures)
            continue
        try:
            sample_rate, waveform = wavfile.read(str(audio_path))
        except Exception as exc:
            _failure("unreadable audio {}: {}".format(audio_path, exc), failures)
            continue
        if int(sample_rate) != expected_sr:
            _failure("sample rate mismatch: {}".format(audio_path), failures)
        if waveform.ndim != 2 or waveform.shape[1] != expected_channels:
            _failure("channel mismatch: {}".format(audio_path), failures)
        if waveform.dtype != expected_dtype:
            _failure("dtype mismatch: {}".format(audio_path), failures)
        if not np.isfinite(waveform).all():
            _failure("NaN/Inf audio: {}".format(audio_path), failures)
        expected_count = expected_dry_samples
        rir_path = storage.root / str(row.get("rir_path", ""))
        if bool(config["storage"]["save_rir"]):
            if not rir_path.is_file():
                _failure("missing RIR: {}".format(rir_path), failures)
            else:
                try:
                    with np.load(str(rir_path), allow_pickle=False) as payload:
                        rir = payload["rir"]
                        rir_sr = int(payload["sample_rate_hz"])
                        rir_count = int(payload["num_samples"])
                    if rir.ndim != 2 or rir.shape[1] != 2 or rir.dtype != np.float32:
                        _failure("RIR shape/dtype mismatch: {}".format(rir_path), failures)
                    if not np.isfinite(rir).all():
                        _failure("NaN/Inf RIR: {}".format(rir_path), failures)
                    if rir_sr != expected_sr or rir_count != rir.shape[0]:
                        _failure("RIR metadata mismatch: {}".format(rir_path), failures)
                    expected_count += rir.shape[0] - 1
                except Exception as exc:
                    _failure("unreadable RIR {}: {}".format(rir_path, exc), failures)
        if waveform.shape[0] != expected_count:
            _failure("full convolution length mismatch: {}".format(audio_path), failures)
        if waveform.shape[0] != int(row.get("num_samples", -1)):
            _failure("manifest sample count mismatch: {}".format(audio_path), failures)
        duration = waveform.shape[0] / float(expected_sr)
        if abs(duration - float(row.get("duration_sec", -1.0))) > 1e-6:
            _failure("manifest duration mismatch: {}".format(audio_path), failures)
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.0:
            warnings.append("abs(WAV) peak > 1: {}".format(row.get("viewpoint_id")))
        nonzero_rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        if 0.0 < nonzero_rms < 1e-8:
            warnings.append("very low nonzero RMS: {}".format(row.get("viewpoint_id")))
        dry_fingerprints.setdefault(episode_id, set()).add(
            (
                row.get("ground_truth", {}).get("dry_audio_id"),
                row.get("ground_truth", {}).get("gain_db"),
                row.get("ground_truth", {}).get("segment_start_sec"),
            )
        )
        expected_count = int(row.get("num_samples", -1))
        if expected_count <= 0:
            _failure("invalid manifest sample count", failures)

    for episode_id, fingerprints in dry_fingerprints.items():
        if len(fingerprints) != 1:
            _failure("same Episode dry segment/gain mismatch: {}".format(episode_id), failures)
    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "episode_count": len(episodes),
        "candidate_count": len(candidates),
        "viewpoint_count": len(viewpoints),
        "audio_count": sum(1 for row in viewpoints if (storage.root / str(row.get("audio_path", ""))).is_file()),
        "rir_count": sum(1 for row in viewpoints if (storage.root / str(row.get("rir_path", ""))).is_file()),
    }
