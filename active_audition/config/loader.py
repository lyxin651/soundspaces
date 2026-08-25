"""YAML loading with deterministic V0 defaults and validation."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import validate_config


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "v0_replica_debug",
        "schema_version": "v0.1",
        "global_seed": 20260824,
    },
    "scene": {"ids": ["replica.office_0"]},
    "episode": {"mode": "fixed_or_sampled"},
    "listener": {"sensor_offset_m": [0.0, 1.5, 0.0]},
    "source": {"height_m": 1.5, "gain_db": 0.0},
    "dry_audio": {
        "segment_duration_sec": 5.0,
        "mono_required": True,
        "short_clip_policy": "reject",
        "target_sample_rate_hz": 16000,
        "resampling_algorithm": "resample_poly",
        "resampling_window": ["kaiser", 5.0],
        "resampling_padtype": "constant",
        "normalization": "none",
    },
    "candidate": {
        "translation": {
            "distance_m": 1.0,
            "directions": ["forward", "backward", "left", "right"],
        },
        "rotation": {"angle_deg": 45.0, "directions": ["left", "right"]},
    },
    "navigation": {
        "thresholds_enabled": False,
        "max_snap_error_m": None,
        "min_actual_translation_m": None,
        "max_actual_translation_m": None,
        "max_geodesic_detour_ratio": None,
        "duplicate_position_tolerance_m": None,
    },
    "acoustics": {
        "channel_layout": "binaural",
        "sample_rate_hz": 16000,
        "materials_enabled": False,
        "convolution_mode": "full",
        "canonical_rir_dtype": "float32",
        "canonical_wav_dtype": "float32",
        "per_viewpoint_normalization": False,
    },
    "storage": {"save_audio": True, "save_rir": True},
    "movement": {"mode": "reposition_then_listen"},
    "validation": {"enabled": True},
    "qc": {"analysis_window_sec": 5.0},
    "registries": {
        "scenes_path": "registries/scenes.yaml",
        "dry_audio_path": "registries/dry_audio.csv",
    },
    "golden": {
        "scene_id": "replica.office_0",
        "listener_base_position_world": [0.57759, -0.96887, 1.84196],
        "listener_yaw_deg": 0.0,
        "source_anchor_base_position_world": [1.624053, -0.968869, -0.512549],
        "source_audio_id": "golden_probe_v0",
        "segment_start_sec": 0.0,
        "gain_db": 0.0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_resolved_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    resolved = _deep_merge(DEFAULT_CONFIG, raw)
    validate_config(resolved)
    resolved["_config_path"] = str(config_path)
    resolved["_repo_root"] = str(config_path.parents[2])
    return resolved
