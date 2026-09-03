"""Small schema validator for the frozen V0 debug and Pilot configurations."""

import math
from typing import Any, Dict, Iterable


class ConfigError(ValueError):
    """Raised when a V0 config violates its frozen contract."""


def _require(mapping: Dict[str, Any], keys: Iterable[str], path: str) -> None:
    for key in keys:
        if key not in mapping:
            raise ConfigError("missing config field: {}.{}".format(path, key))


def _number(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("{} must be numeric".format(path))


def _finite_positive(value: Any, path: str) -> None:
    _number(value, path)
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ConfigError("{} must be a finite positive number".format(path))


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigError("config root must be a mapping")

    # V0.5 is deliberately an explicit contract; legacy V0 validation remains unchanged.
    if config.get("acoustics", {}).get("backend") == "soundspaces_precomputed":
        return _validate_precomputed_config(config)

    _require(
        config,
        (
            "experiment",
            "scene",
            "episode",
            "listener",
            "source",
            "dry_audio",
            "candidate",
            "navigation",
            "acoustics",
            "storage",
            "movement",
            "validation",
            "qc",
            "registries",
            "golden",
        ),
        "root",
    )

    experiment = config["experiment"]
    _require(experiment, ("name", "schema_version", "global_seed"), "experiment")
    if not isinstance(experiment["global_seed"], int):
        raise ConfigError("experiment.global_seed must be an int")

    scene = config["scene"]
    _require(scene, ("ids",), "scene")
    if scene["ids"] != ["replica.office_0"]:
        raise ConfigError("M0 debug config must contain exactly replica.office_0")

    episode = config["episode"]
    _require(
        episode,
        ("mode", "count", "source_listener_min_distance_m", "source_listener_max_distance_m"),
        "episode",
    )
    if episode["mode"] not in ("fixed", "sampled", "fixed_or_sampled"):
        raise ConfigError("unsupported episode.mode")
    if not isinstance(episode["count"], int) or episode["count"] < 1:
        raise ConfigError("episode.count must be a positive int")

    listener = config["listener"]
    _require(listener, ("sensor_offset_m",), "listener")
    if listener["sensor_offset_m"] != [0.0, 1.5, 0.0]:
        raise ConfigError("listener.sensor_offset_m must be [0.0, 1.5, 0.0]")

    source = config["source"]
    _require(source, ("height_m", "gain_db"), "source")
    if float(source["height_m"]) != 1.5:
        raise ConfigError("source.height_m must be 1.5")
    _number(source["gain_db"], "source.gain_db")

    dry_audio = config["dry_audio"]
    _require(
        dry_audio,
        (
            "segment_duration_sec",
            "mono_required",
            "short_clip_policy",
            "target_sample_rate_hz",
            "resampling_algorithm",
            "resampling_window",
            "resampling_padtype",
            "normalization",
        ),
        "dry_audio",
    )
    if float(dry_audio["segment_duration_sec"]) != 5.0:
        raise ConfigError("dry_audio.segment_duration_sec must be 5.0")
    if dry_audio["mono_required"] is not True:
        raise ConfigError("dry_audio.mono_required must be true")
    if dry_audio["short_clip_policy"] != "reject":
        raise ConfigError("short clips must be rejected")
    if dry_audio["target_sample_rate_hz"] != 16000:
        raise ConfigError("dry_audio.target_sample_rate_hz must be 16000")
    if dry_audio["resampling_algorithm"] != "resample_poly":
        raise ConfigError("dry_audio.resampling_algorithm must be resample_poly")
    if dry_audio["resampling_window"] != ["kaiser", 5.0]:
        raise ConfigError("dry_audio.resampling_window must be ['kaiser', 5.0]")
    if dry_audio["resampling_padtype"] != "constant":
        raise ConfigError("dry_audio.resampling_padtype must be constant")
    if dry_audio["normalization"] != "none":
        raise ConfigError("dry_audio.normalization must be none")

    candidate = config["candidate"]
    _require(candidate, ("translation", "rotation"), "candidate")
    _require(candidate["translation"], ("distance_m", "directions"), "candidate.translation")
    _require(candidate["rotation"], ("angle_deg", "directions"), "candidate.rotation")
    if float(candidate["translation"]["distance_m"]) != 1.0:
        raise ConfigError("translation distance must be 1.0")
    if candidate["translation"]["directions"] != ["forward", "backward", "left", "right"]:
        raise ConfigError("translation directions are not the frozen V0 set")
    if float(candidate["rotation"]["angle_deg"]) != 45.0:
        raise ConfigError("rotation angle must be 45.0")
    if candidate["rotation"]["directions"] != ["left", "right"]:
        raise ConfigError("rotation directions are not the frozen V0 set")

    navigation = config["navigation"]
    _require(
        navigation,
        (
            "thresholds_enabled",
            "max_snap_error_m",
            "min_actual_translation_m",
            "max_actual_translation_m",
            "max_geodesic_detour_ratio",
            "duplicate_position_tolerance_m",
        ),
        "navigation",
    )
    if navigation["thresholds_enabled"] not in (False, True):
        raise ConfigError("navigation.thresholds_enabled must be boolean")
    for key in (
        "max_snap_error_m",
        "min_actual_translation_m",
        "max_actual_translation_m",
        "max_geodesic_detour_ratio",
        "duplicate_position_tolerance_m",
    ):
        if navigation["thresholds_enabled"] is False:
            if navigation[key] is not None:
                raise ConfigError("deferred threshold {} must be null when thresholds are disabled".format(key))
        else:
            _finite_positive(navigation[key], "navigation." + key)
    if navigation["thresholds_enabled"] and float(navigation["min_actual_translation_m"]) > float(navigation["max_actual_translation_m"]):
        raise ConfigError("navigation.min_actual_translation_m must be <= max_actual_translation_m")
    source_min = episode["source_listener_min_distance_m"]
    source_max = episode["source_listener_max_distance_m"]
    if navigation["thresholds_enabled"]:
        _finite_positive(source_min, "episode.source_listener_min_distance_m")
        _finite_positive(source_max, "episode.source_listener_max_distance_m")
        if float(source_min) >= float(source_max):
            raise ConfigError("episode source listener minimum must be less than maximum")
    elif source_min is not None or source_max is not None:
        raise ConfigError("source listener thresholds must be null when navigation thresholds are disabled")

    acoustics = config["acoustics"]
    _require(
        acoustics,
        (
            "channel_layout",
            "sample_rate_hz",
            "materials_enabled",
            "convolution_mode",
            "canonical_rir_dtype",
            "canonical_wav_dtype",
            "per_viewpoint_normalization",
        ),
        "acoustics",
    )
    expected = {
        "channel_layout": "binaural",
        "sample_rate_hz": 16000,
        "materials_enabled": False,
        "convolution_mode": "full",
        "canonical_rir_dtype": "float32",
        "canonical_wav_dtype": "float32",
        "per_viewpoint_normalization": False,
    }
    for key, value in expected.items():
        if acoustics[key] != value:
            raise ConfigError("acoustics.{} must be {!r}".format(key, value))

    _require(config["storage"], ("dataset_id", "save_audio", "save_rir"), "storage")
    dataset_id = str(config["storage"]["dataset_id"])
    if not dataset_id or dataset_id in (".", "..") or "/" in dataset_id or "\\" in dataset_id:
        raise ConfigError("storage.dataset_id must be a safe relative identifier")
    _require(config["movement"], ("mode",), "movement")
    if config["movement"]["mode"] != "reposition_then_listen":
        raise ConfigError("movement.mode must be reposition_then_listen")
    _require(config["validation"], ("enabled",), "validation")
    _require(
        config["qc"],
        (
            "analysis_window_sec",
            "silence_abs_threshold",
            "clipping_abs_threshold",
            "rms_eps_amplitude",
            "ild_eps_power",
            "interaural_max_lag_samples",
            "rir_tail_window_sec",
            "rir_eps_power",
        ),
        "qc",
    )
    if float(config["qc"]["analysis_window_sec"]) != 5.0:
        raise ConfigError("qc.analysis_window_sec must be 5.0")
    if float(config["qc"]["silence_abs_threshold"]) != 1.0e-5:
        raise ConfigError("qc.silence_abs_threshold must be 1e-5")
    if float(config["qc"]["clipping_abs_threshold"]) != 1.0:
        raise ConfigError("qc.clipping_abs_threshold must be 1.0")
    if float(config["qc"]["rms_eps_amplitude"]) != 1.0e-12:
        raise ConfigError("qc.rms_eps_amplitude must be 1e-12")
    if float(config["qc"]["ild_eps_power"]) != 1.0e-12:
        raise ConfigError("qc.ild_eps_power must be 1e-12")
    if int(config["qc"]["interaural_max_lag_samples"]) != 16:
        raise ConfigError("qc.interaural_max_lag_samples must be 16")
    if float(config["qc"]["rir_tail_window_sec"]) != 0.100:
        raise ConfigError("qc.rir_tail_window_sec must be 0.100")
    if float(config["qc"]["rir_eps_power"]) != 1.0e-12:
        raise ConfigError("qc.rir_eps_power must be 1e-12")

    _require(config["registries"], ("scenes_path", "dry_audio_path"), "registries")
    golden = config["golden"]
    _require(
        golden,
        (
            "scene_id",
            "listener_base_position_world",
            "listener_yaw_deg",
            "source_anchor_base_position_world",
            "source_audio_id",
            "segment_start_sec",
            "gain_db",
        ),
        "golden",
    )
    if golden["scene_id"] != "replica.office_0":
        raise ConfigError("golden.scene_id must be replica.office_0")
    if len(golden["listener_base_position_world"]) != 3:
        raise ConfigError("golden listener position must contain 3 values")
    if len(golden["source_anchor_base_position_world"]) != 3:
        raise ConfigError("golden source position must contain 3 values")
    _number(golden["listener_yaw_deg"], "golden.listener_yaw_deg")
    _number(golden["segment_start_sec"], "golden.segment_start_sec")
    _number(golden["gain_db"], "golden.gain_db")
    return config


def _validate_precomputed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    experiment = config["experiment"]
    if str(experiment.get("schema_version")) != "v0.5":
        raise ConfigError("precomputed backend requires experiment.schema_version v0.5")
    audio = config["dry_audio"]
    if int(audio.get("target_sample_rate_hz", 0)) != 24000:
        raise ConfigError("V0.5 dry_audio.target_sample_rate_hz must be 24000")
    acoustics = config["acoustics"]
    for key, expected in (("channel_layout", "binaural"), ("convolution_mode", "full"),
                          ("canonical_rir_dtype", "float32"), ("canonical_wav_dtype", "float32")):
        if acoustics.get(key) != expected: raise ConfigError("acoustics.{} must be {!r}".format(key, expected))
    if acoustics.get("canonical_output_sample_rate_hz") != 24000 or acoustics.get("rir_intermediate_sample_rate_hz") != 16000:
        raise ConfigError("precomputed canonical rates must be 16000 intermediate and 24000 output")
    if acoustics.get("per_viewpoint_normalization", False):
        raise ConfigError("precomputed backend forbids viewpoint normalization")
    if not config.get("acoustics", {}).get("rir_root"):
        raise ConfigError("acoustics.rir_root is required and must not be hard-coded")
    if config.get("navigation", {}).get("backend") != "soundspaces_graph":
        raise ConfigError("navigation.backend must be soundspaces_graph")
    if config.get("navigation", {}).get("candidate_mode") != "graph_neighbors":
        raise ConfigError("navigation.candidate_mode must be graph_neighbors")
    dataset_id = str(config["storage"].get("dataset_id", ""))
    if not dataset_id.startswith("aa_v05_"):
        raise ConfigError("V0.5 dataset_id must use aa_v05_ namespace")
    return config
