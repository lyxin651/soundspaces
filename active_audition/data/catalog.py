"""Read-only registry resolution and stable episode seed helpers."""

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np
from scipy.io import wavfile


class RegistryError(ValueError):
    """Raised when a registry entry is missing or inconsistent."""


def episode_seed(global_seed: int, episode_id: str) -> int:
    payload = "{}:{}".format(global_seed, episode_id).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_scene_registry(path: str, repo_root: str) -> Dict[str, Dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.is_absolute():
        registry_path = Path(repo_root) / registry_path
    with registry_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    scenes = raw.get("scenes", {})
    if not isinstance(scenes, dict) or not scenes:
        raise RegistryError("scene registry is empty")
    result = {}
    for scene_id, entry in scenes.items():
        required = ("dataset", "scene_asset", "navmesh", "semantic_info", "materials_mode")
        if any(key not in entry for key in required):
            raise RegistryError("scene {} is missing a required field".format(scene_id))
        resolved = dict(entry)
        for key in ("scene_asset", "navmesh", "semantic_info", "stage_config"):
            if key in resolved and resolved[key] is not None:
                resolved[key] = str(_resolve(Path(repo_root), resolved[key]).resolve())
                if not Path(resolved[key]).is_file():
                    raise RegistryError("scene {} resource missing: {}".format(scene_id, resolved[key]))
        if resolved["materials_mode"] != "off":
            raise RegistryError("M0 requires materials_mode=off")
        result[scene_id] = resolved
    return result


def load_dry_audio_registry(path: str, repo_root: str) -> Dict[str, Dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.is_absolute():
        registry_path = Path(repo_root) / registry_path
    with registry_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = (
        "audio_id",
        "event_class",
        "path",
        "duration_sec",
        "sample_rate_hz",
        "channels",
        "sha256",
        "source_dataset",
    )
    result = {}
    for row in rows:
        if any(not row.get(key) for key in required):
            raise RegistryError("dry audio row is missing a required field")
        audio_id = row["audio_id"]
        if audio_id in result:
            raise RegistryError("duplicate dry audio id: {}".format(audio_id))
        resolved_path = _resolve(Path(repo_root), row["path"]).resolve()
        if not resolved_path.is_file():
            raise RegistryError("dry audio missing: {}".format(resolved_path))
        sample_rate, signal = wavfile.read(str(resolved_path))
        duration = signal.shape[0] / float(sample_rate)
        if duration < 5.0 or signal.size == 0:
            raise RegistryError("dry audio is shorter than 5 seconds: {}".format(resolved_path))
        if not np.isfinite(signal).all():
            raise RegistryError("dry audio contains NaN/Inf: {}".format(resolved_path))
        actual_channels = 1 if signal.ndim == 1 else signal.shape[1]
        if int(row["sample_rate_hz"]) != int(sample_rate):
            raise RegistryError("dry audio sample rate mismatch: {}".format(resolved_path))
        if int(row["channels"]) != int(actual_channels):
            raise RegistryError("dry audio channel count mismatch: {}".format(resolved_path))
        if abs(float(row["duration_sec"]) - duration) > 1e-6:
            raise RegistryError("dry audio duration mismatch: {}".format(resolved_path))
        if sha256_file(resolved_path) != row["sha256"]:
            raise RegistryError("dry audio sha256 mismatch: {}".format(resolved_path))
        result[audio_id] = dict(row, path=str(resolved_path))
    if not result:
        raise RegistryError("dry audio registry is empty")
    return result
