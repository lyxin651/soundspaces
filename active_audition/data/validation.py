"""Hard Dataset contract checks for M2.1."""

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from scipy.io import wavfile

from active_audition.data.manifest import read_jsonl
from active_audition.data.storage import DatasetStorage


class ValidationError(ValueError):
    """Raised for invalid Dataset validation inputs."""


def _failure(message: str, failures: list) -> None:
    failures.append(message)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return list(left) == list(right)
    return left == right


def payload_is_complete(dataset_root: str, row: Mapping[str, Any], save_rir: bool = True) -> bool:
    """Check payload bytes and manifest metadata before a resume skip."""

    root = Path(dataset_root)
    audio = root / str(row.get("audio_path", ""))
    if not audio.is_file():
        return False
    try:
        sample_rate, waveform = wavfile.read(str(audio))
    except Exception:
        return False
    if sample_rate != int(row.get("sample_rate_hz", -1)) or waveform.ndim != 2 or waveform.shape[1] != 2:
        return False
    if waveform.dtype != np.float32 or not np.isfinite(waveform).all():
        return False
    if int(row.get("num_channels", -1)) != int(waveform.shape[1]):
        return False
    if str(row.get("dtype")) != str(waveform.dtype):
        return False
    if waveform.shape[0] != int(row.get("num_samples", -1)):
        return False
    if abs(waveform.shape[0] / float(sample_rate) - float(row.get("duration_sec", -1.0))) > 1e-6:
        return False
    if save_rir:
        rir_path = root / str(row.get("rir_path", ""))
        if not row.get("rir_id") or not row.get("rir_path") or not rir_path.is_file():
            return False
        try:
            with np.load(str(rir_path), allow_pickle=False) as payload:
                rir = payload["rir"]
                sr = int(payload["sample_rate_hz"])
                count = int(payload["num_samples"])
        except Exception:
            return False
        if rir.ndim != 2 or rir.shape[1] != 2 or rir.shape[0] != count or sr != int(row.get("sample_rate_hz", -1)):
            return False
        if rir.dtype != np.float32 or not np.isfinite(rir).all():
            return False
        dry_samples = int(round(float(row.get("ground_truth", {}).get("segment_duration_sec", 0.0)) * sr))
        if dry_samples <= 0 or waveform.shape[0] != dry_samples + rir.shape[0] - 1:
            return False
    return True


def _read_manifests(storage: DatasetStorage) -> Mapping[str, list]:
    return {
        "episodes": read_jsonl(str(storage.manifest_path("episodes.jsonl"))),
        "candidates": read_jsonl(str(storage.manifest_path("candidates.jsonl"))),
        "viewpoints": read_jsonl(str(storage.manifest_path("viewpoints.jsonl"))),
    }


def validate_dataset(dataset_root: str, config: Mapping[str, Any], require_success: bool = False) -> Dict[str, Any]:
    storage = DatasetStorage(dataset_root)
    failures, warnings = [], []
    if require_success and not storage.success_path.exists():
        _failure("_SUCCESS is missing", failures)
    try:
        manifests = _read_manifests(storage)
    except Exception as exc:
        _failure("manifest read failure: {}".format(exc), failures)
        return {"status": "FAIL", "failures": failures, "warnings": warnings}
    episodes, candidates, viewpoints = manifests["episodes"], manifests["candidates"], manifests["viewpoints"]
    episode_ids = [row.get("episode_id") for row in episodes]
    candidate_keys = [(row.get("episode_id"), row.get("candidate_id")) for row in candidates]
    viewpoint_keys = [(row.get("episode_id"), row.get("viewpoint_id")) for row in viewpoints]
    if any(value is None for value in episode_ids) or len(episode_ids) != len(set(episode_ids)):
        _failure("duplicate or missing episode key", failures)
    if any(None in key for key in candidate_keys) or len(candidate_keys) != len(set(candidate_keys)):
        _failure("duplicate or missing candidate key", failures)
    if any(None in key for key in viewpoint_keys) or len(viewpoint_keys) != len(set(viewpoint_keys)):
        _failure("duplicate or missing viewpoint key", failures)
    if not episodes:
        _failure("episode manifest is empty", failures)
    episode_map = {row.get("episode_id"): row for row in episodes}
    candidate_map = dict(zip(candidate_keys, candidates))
    viewpoint_map = dict(zip(viewpoint_keys, viewpoints))
    expected_sr = int(config["acoustics"]["sample_rate_hz"])
    expected_dtype = np.dtype(config["acoustics"]["canonical_wav_dtype"])
    expected_channels = 2
    dry_fingerprints = {}

    for candidate in candidates:
        if candidate.get("episode_id") not in episode_map:
            _failure("candidate references missing episode: {}".format(candidate.get("candidate_id")), failures)

    for episode_id, episode in episode_map.items():
        if config["episode"]["mode"] == "sampled":
            for key in ("source_listener_euclidean_m", "source_listener_geodesic_m"):
                value = episode.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    _failure("sampled Episode diagnostic missing or non-finite: {}".format(key), failures)
        rows = [row for row in viewpoints if row.get("episode_id") == episode_id]
        initial = [row for row in rows if row.get("viewpoint_id") == "initial"]
        if len(initial) != 1:
            _failure("Episode {} must have exactly one initial viewpoint".format(episode_id), failures)
        elif initial[0].get("candidate_id") is not None or initial[0].get("action_type") != "initial":
            _failure("initial viewpoint contract mismatch: {}".format(episode_id), failures)
        for candidate in [row for row in candidates if row.get("episode_id") == episode_id]:
            matching = [row for row in rows if row.get("viewpoint_id") == candidate.get("candidate_id")]
            if bool(candidate.get("valid")) and len(matching) != 1:
                _failure("valid candidate missing viewpoint: {}".format(candidate.get("candidate_id")), failures)
            if not bool(candidate.get("valid")) and matching:
                _failure("invalid candidate has viewpoint: {}".format(candidate.get("candidate_id")), failures)

    for row in viewpoints:
        episode_id, viewpoint_id = row.get("episode_id"), row.get("viewpoint_id")
        episode = episode_map.get(episode_id)
        if episode is None:
            _failure("viewpoint references missing episode: {}".format(episode_id), failures)
            continue
        if viewpoint_id != "initial":
            candidate = candidate_map.get((episode_id, row.get("candidate_id")))
            if candidate is None:
                _failure("viewpoint references missing candidate: {}".format(viewpoint_id), failures)
            else:
                if not bool(candidate.get("valid")) or viewpoint_id != candidate.get("candidate_id") or row.get("action_type") != candidate.get("action_type"):
                    _failure("candidate/viewpoint relation mismatch: {}".format(viewpoint_id), failures)
        ground_truth = row.get("ground_truth")
        source = episode.get("source", {})
        expected_gt = {
            "scene_id": episode.get("scene_id"), "source_position_world": source.get("position_world"),
            "dry_audio_id": source.get("audio_id"), "segment_start_sec": source.get("segment_start_sec"),
            "segment_duration_sec": source.get("segment_duration_sec"), "gain_db": source.get("gain_db"),
        }
        if not isinstance(ground_truth, dict) or any(not _same(ground_truth.get(key), value) for key, value in expected_gt.items()):
            _failure("ground_truth/Episode mismatch: {}".format(viewpoint_id), failures)
        dry_fingerprints.setdefault(episode_id, set()).add(tuple(ground_truth.get(key) if isinstance(ground_truth, dict) else None for key in ("dry_audio_id", "segment_start_sec", "segment_duration_sec", "gain_db", "dry_hash")))
        audio_path = storage.root / str(row.get("audio_path", ""))
        if Path(str(row.get("audio_path", ""))).is_absolute() or not audio_path.is_file():
            _failure("missing manifest audio: {}".format(viewpoint_id), failures)
            continue
        try:
            sample_rate, waveform = wavfile.read(str(audio_path))
        except Exception as exc:
            _failure("unreadable audio {}: {}".format(viewpoint_id, exc), failures)
            continue
        if int(sample_rate) != expected_sr or waveform.ndim != 2 or waveform.shape[1] != expected_channels:
            _failure("WAV sample rate/channel mismatch: {}".format(viewpoint_id), failures)
        if waveform.dtype != expected_dtype or str(row.get("dtype")) != str(expected_dtype):
            _failure("WAV dtype mismatch: {}".format(viewpoint_id), failures)
        if not np.isfinite(waveform).all():
            _failure("WAV NaN/Inf: {}".format(viewpoint_id), failures)
        if int(row.get("sample_rate_hz", -1)) != int(sample_rate) or int(row.get("num_channels", -1)) != waveform.shape[1] or int(row.get("num_samples", -1)) != waveform.shape[0] or abs(float(row.get("duration_sec", -1.0)) - waveform.shape[0] / float(sample_rate)) > 1e-6:
            _failure("manifest/WAV metadata mismatch: {}".format(viewpoint_id), failures)
        rir = None
        if bool(config["storage"]["save_rir"]):
            if not row.get("rir_id") or not row.get("rir_path"):
                _failure("missing RIR reference: {}".format(viewpoint_id), failures)
            else:
                rir_path = storage.root / str(row["rir_path"])
                try:
                    with np.load(str(rir_path), allow_pickle=False) as payload:
                        rir = payload["rir"]
                        rir_sr = int(payload["sample_rate_hz"])
                        rir_count = int(payload["num_samples"])
                    if rir.ndim != 2 or rir.shape[1] != 2 or rir.dtype != np.float32 or not np.isfinite(rir).all() or rir_sr != expected_sr or rir_count != rir.shape[0]:
                        _failure("RIR metadata/array mismatch: {}".format(viewpoint_id), failures)
                    dry_samples = int(round(float(source.get("segment_duration_sec", 0.0)) * expected_sr))
                    if rir is not None and waveform.shape[0] != dry_samples + rir.shape[0] - 1:
                        _failure("full convolution length mismatch: {}".format(viewpoint_id), failures)
                except Exception as exc:
                    _failure("unreadable RIR {}: {}".format(viewpoint_id, exc), failures)
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.0:
            warnings.append("abs(WAV) peak > 1: {}".format(viewpoint_id))
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        if 0.0 < rms < 1e-8:
            warnings.append("very low nonzero RMS: {}".format(viewpoint_id))

    for episode_id, fingerprints in dry_fingerprints.items():
        if len(fingerprints) != 1:
            _failure("same Episode dry segment/duration/gain/hash mismatch: {}".format(episode_id), failures)
    status = "PASS" if not failures else "FAIL"
    return {"status": status, "failures": failures, "warnings": warnings, "episode_count": len(episodes), "candidate_count": len(candidates), "viewpoint_count": len(viewpoints), "audio_count": sum(1 for row in viewpoints if (storage.root / str(row.get("audio_path", ""))).is_file()), "rir_count": sum(1 for row in viewpoints if row.get("rir_path") and (storage.root / str(row.get("rir_path"))).is_file())}
