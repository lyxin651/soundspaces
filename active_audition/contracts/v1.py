"""Active-ASR V1.1 A0 contract validation and canonical identity.

This module is deliberately data-only.  It does not import Habitat, an ASR
runtime, a dataset loader, or a renderer.  A0 freezes the experiment boundary
and its serialization identity; later gates own runtime and scientific
qualification code.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


A0_SCHEMA_VERSION = "active-asr-v1.1"
A0_GATE = "A0"
CONTRACT_SERIALIZATION_VERSION = "canonical-json-v1"
CONTRACT_HASH_ALGORITHM = "sha256"
ARTIFACT_SCHEMA_VERSION = "active-asr-artifact-v1.1"
MASTER_SEED = 20260923


class ContractError(ValueError):
    """Raised when an Active-ASR A0 contract is invalid."""


def _path(path: str, key: Any) -> str:
    return "{}.{}".format(path, key) if path else str(key)


def _require_mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ContractError("{} must be a mapping".format(path))
    return value


def _require_keys(mapping: Mapping, keys: Iterable[str], path: str) -> None:
    for key in keys:
        if key not in mapping:
            raise ContractError("missing config field: {}".format(_path(path, key)))


def _only_keys(mapping: Mapping, keys: Iterable[str], path: str) -> None:
    expected = set(keys)
    unknown = sorted(set(mapping) - expected)
    if unknown:
        raise ContractError("unknown config field(s) at {}: {}".format(path, ", ".join(map(str, unknown))))


def _expect_string(value: Any, path: str, expected: str = "") -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("{} must be a non-empty string".format(path))
    if expected and value != expected:
        raise ContractError("{} must be {!r}".format(path, expected))
    return value


def _expect_bool(value: Any, path: str, expected: Any = None) -> bool:
    if not isinstance(value, bool):
        raise ContractError("{} must be boolean".format(path))
    if expected is not None and value is not expected:
        raise ContractError("{} must be {}".format(path, str(expected).lower()))
    return value


def _expect_int(value: Any, path: str, expected: int = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("{} must be an integer".format(path))
    if expected is not None and value != expected:
        raise ContractError("{} must be {}".format(path, expected))
    return value


def _expect_number(value: Any, path: str, expected: float = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("{} must be numeric".format(path))
    if not math.isfinite(float(value)):
        raise ContractError("{} must be finite".format(path))
    if expected is not None and float(value) != float(expected):
        raise ContractError("{} must be {}".format(path, expected))
    return float(value)


def _expect_list(value: Any, path: str, length: int = None) -> List[Any]:
    if not isinstance(value, list):
        raise ContractError("{} must be a list".format(path))
    if length is not None and len(value) != length:
        raise ContractError("{} must contain {} values".format(path, length))
    return value


def _expect_exact_list(value: Any, expected: List[Any], path: str) -> None:
    if _expect_list(value, path) != expected:
        raise ContractError("{} must be {!r}".format(path, expected))


def _validate_contract_metadata(contract: Mapping) -> None:
    path = "contract"
    _only_keys(contract, ("namespace", "version", "gate", "state"), path)
    _require_keys(contract, ("namespace", "version", "gate", "state"), path)
    _expect_string(contract["namespace"], _path(path, "namespace"), "active-asr")
    _expect_string(contract["version"], _path(path, "version"), A0_SCHEMA_VERSION)
    _expect_string(contract["gate"], _path(path, "gate"), A0_GATE)
    _expect_string(contract["state"], _path(path, "state"), "FROZEN")


def _validate_audio(audio: Mapping) -> None:
    path = "audio"
    _only_keys(
        audio,
        (
            "render_sample_rate_hz",
            "asr_sample_rate_hz",
            "render_to_asr",
            "waveform_dtype",
            "dry_shape",
            "binaural_shape",
            "channel_layout",
            "channel_order",
            "asr_frontend",
        ),
        path,
    )
    _require_keys(
        audio,
        (
            "render_sample_rate_hz",
            "asr_sample_rate_hz",
            "render_to_asr",
            "waveform_dtype",
            "dry_shape",
            "binaural_shape",
            "channel_layout",
            "channel_order",
            "asr_frontend",
        ),
        path,
    )
    _expect_int(audio["render_sample_rate_hz"], _path(path, "render_sample_rate_hz"), 24000)
    _expect_int(audio["asr_sample_rate_hz"], _path(path, "asr_sample_rate_hz"), 16000)
    _expect_string(audio["waveform_dtype"], _path(path, "waveform_dtype"), "float32")
    _expect_exact_list(audio["dry_shape"], ["T"], _path(path, "dry_shape"))
    _expect_exact_list(audio["binaural_shape"], ["T", 2], _path(path, "binaural_shape"))
    _expect_string(audio["channel_layout"], _path(path, "channel_layout"), "binaural")
    _expect_exact_list(audio["channel_order"], ["L", "R"], _path(path, "channel_order"))
    _expect_string(audio["asr_frontend"], _path(path, "asr_frontend"), "mean_lr")

    render_to_asr = _require_mapping(audio["render_to_asr"], _path(path, "render_to_asr"))
    render_path = _path(path, "render_to_asr")
    _only_keys(render_to_asr, ("source_sample_rate_hz", "target_sample_rate_hz", "algorithm", "anti_aliasing"), render_path)
    _require_keys(render_to_asr, ("source_sample_rate_hz", "target_sample_rate_hz", "algorithm", "anti_aliasing"), render_path)
    _expect_int(render_to_asr["source_sample_rate_hz"], _path(render_path, "source_sample_rate_hz"), 24000)
    _expect_int(render_to_asr["target_sample_rate_hz"], _path(render_path, "target_sample_rate_hz"), 16000)
    _expect_string(render_to_asr["algorithm"], _path(render_path, "algorithm"), "resample_poly")
    _expect_bool(render_to_asr["anti_aliasing"], _path(render_path, "anti_aliasing"), True)


def _validate_actions(actions: Mapping) -> None:
    path = "actions"
    _only_keys(actions, ("order", "definitions"), path)
    _require_keys(actions, ("order", "definitions"), path)
    _expect_exact_list(actions["order"], ["Forward", "TurnLeft", "TurnRight", "Stop"], _path(path, "order"))
    definitions = _require_mapping(actions["definitions"], _path(path, "definitions"))
    definitions_path = _path(path, "definitions")
    _only_keys(definitions, actions["order"], definitions_path)
    _require_keys(definitions, actions["order"], definitions_path)

    expected = {
        "Forward": {"kind": "translation", "distance_m": 0.25, "local_direction": "forward"},
        "TurnLeft": {"kind": "rotation", "angle_deg": 30.0, "yaw_sign": "positive"},
        "TurnRight": {"kind": "rotation", "angle_deg": 30.0, "yaw_sign": "negative"},
        "Stop": {"kind": "stop"},
    }
    for action_name, fields in expected.items():
        action = _require_mapping(definitions[action_name], _path(definitions_path, action_name))
        action_path = _path(definitions_path, action_name)
        _only_keys(action, fields.keys(), action_path)
        _require_keys(action, fields.keys(), action_path)
        for field, value in fields.items():
            if field in ("distance_m", "angle_deg"):
                _expect_number(action[field], _path(action_path, field), value)
            else:
                _expect_string(action[field], _path(action_path, field), value)


def _validate_coordinates(coordinates: Mapping) -> None:
    path = "coordinates"
    _only_keys(
        coordinates,
        (
            "position_order",
            "world_axes",
            "agent_axes",
            "forward_world",
            "right_world",
            "up_world",
            "yaw_positive",
            "yaw_range_deg",
            "listener_sensor_offset_m",
        ),
        path,
    )
    _require_keys(
        coordinates,
        (
            "position_order",
            "world_axes",
            "agent_axes",
            "forward_world",
            "right_world",
            "up_world",
            "yaw_positive",
            "yaw_range_deg",
            "listener_sensor_offset_m",
        ),
        path,
    )
    _expect_exact_list(coordinates["position_order"], ["x", "y", "z"], _path(path, "position_order"))
    _expect_exact_list(coordinates["forward_world"], [0.0, 0.0, -1.0], _path(path, "forward_world"))
    _expect_exact_list(coordinates["right_world"], [1.0, 0.0, 0.0], _path(path, "right_world"))
    _expect_exact_list(coordinates["up_world"], [0.0, 1.0, 0.0], _path(path, "up_world"))
    _expect_string(coordinates["yaw_positive"], _path(path, "yaw_positive"), "left")
    _expect_exact_list(coordinates["yaw_range_deg"], [-180.0, 180.0], _path(path, "yaw_range_deg"))
    _expect_exact_list(coordinates["listener_sensor_offset_m"], [0.0, 1.5, 0.0], _path(path, "listener_sensor_offset_m"))

    for field, expected in (("world_axes", {"x": "right", "y": "up", "z": "backward"}), ("agent_axes", {"x": "right", "y": "up", "z": "backward"})):
        axes = _require_mapping(coordinates[field], _path(path, field))
        axes_path = _path(path, field)
        _only_keys(axes, ("x", "y", "z"), axes_path)
        _require_keys(axes, ("x", "y", "z"), axes_path)
        for axis, meaning in expected.items():
            _expect_string(axes[axis], _path(axes_path, axis), meaning)


def _validate_sources(sources: Mapping) -> None:
    path = "sources"
    _only_keys(sources, ("target", "noise"), path)
    _require_keys(sources, ("target", "noise"), path)
    target = _require_mapping(sources["target"], _path(path, "target"))
    target_path = _path(path, "target")
    _only_keys(target, ("dataset", "split", "modality", "utterance_unit"), target_path)
    _require_keys(target, ("dataset", "split", "modality", "utterance_unit"), target_path)
    _expect_string(target["dataset"], _path(target_path, "dataset"), "LibriSpeech")
    _expect_string(target["split"], _path(target_path, "split"), "test-clean")
    _expect_string(target["modality"], _path(target_path, "modality"), "speech")
    _expect_string(target["utterance_unit"], _path(target_path, "utterance_unit"), "complete")

    noise = _require_mapping(sources["noise"], _path(path, "noise"))
    noise_path = _path(path, "noise")
    _only_keys(noise, ("dataset", "subset", "modality", "localized_source_approximation"), noise_path)
    _require_keys(noise, ("dataset", "subset", "modality", "localized_source_approximation"), noise_path)
    _expect_string(noise["dataset"], _path(noise_path, "dataset"), "MUSAN")
    _expect_string(noise["subset"], _path(noise_path, "subset"), "noise")
    _expect_string(noise["modality"], _path(noise_path, "modality"), "non_speech")
    _expect_bool(noise["localized_source_approximation"], _path(noise_path, "localized_source_approximation"), True)


def _validate_asr(asr: Mapping) -> None:
    path = "asr"
    _only_keys(asr, ("provider", "model_id", "input_sample_rate_hz", "input_channels", "reference_access"), path)
    _require_keys(asr, ("provider", "model_id", "input_sample_rate_hz", "input_channels", "reference_access"), path)
    _expect_string(asr["provider"], _path(path, "provider"), "SpeechBrain")
    _expect_string(asr["model_id"], _path(path, "model_id"), "speechbrain/asr-transformer-transformerlm-librispeech")
    _expect_int(asr["input_sample_rate_hz"], _path(path, "input_sample_rate_hz"), 16000)
    _expect_string(asr["input_channels"], _path(path, "input_channels"), "mono")
    _expect_string(asr["reference_access"], _path(path, "reference_access"), "forbidden")


def _validate_mix(mix: Mapping) -> None:
    path = "mix"
    _only_keys(mix, ("source_count", "independent_propagation", "normalization", "noise_gain"), path)
    _require_keys(mix, ("source_count", "independent_propagation", "normalization", "noise_gain"), path)
    _expect_int(mix["source_count"], _path(path, "source_count"), 2)
    _expect_bool(mix["independent_propagation"], _path(path, "independent_propagation"), True)

    normalization = _require_mapping(mix["normalization"], _path(path, "normalization"))
    normalization_path = _path(path, "normalization")
    _only_keys(normalization, ("mode", "per_pose", "per_channel", "per_source", "global_gain"), normalization_path)
    _require_keys(normalization, ("mode", "per_pose", "per_channel", "per_source", "global_gain"), normalization_path)
    _expect_string(normalization["mode"], _path(normalization_path, "mode"), "none")
    for field in ("per_pose", "per_channel", "per_source"):
        _expect_bool(normalization[field], _path(normalization_path, field), False)
    _expect_string(normalization["global_gain"], _path(normalization_path, "global_gain"), "fixed")

    noise_gain = _require_mapping(mix["noise_gain"], _path(path, "noise_gain"))
    noise_gain_path = _path(path, "noise_gain")
    _only_keys(noise_gain, ("calibration_pose", "calibration_scope", "reuse_scope", "calibration_frontend"), noise_gain_path)
    _require_keys(noise_gain, ("calibration_pose", "calibration_scope", "reuse_scope", "calibration_frontend"), noise_gain_path)
    _expect_string(noise_gain["calibration_pose"], _path(noise_gain_path, "calibration_pose"), "initial")
    _expect_string(noise_gain["calibration_scope"], _path(noise_gain_path, "calibration_scope"), "per_block")
    _expect_string(noise_gain["reuse_scope"], _path(noise_gain_path, "reuse_scope"), "all_poses_and_evaluation_utterances")
    _expect_string(noise_gain["calibration_frontend"], _path(noise_gain_path, "calibration_frontend"), "binaural_mean_power")


def _validate_seed_policy(seed_policy: Mapping) -> None:
    path = "seed_policy"
    _only_keys(seed_policy, ("master_seed", "derivation", "component_order", "retry_policy"), path)
    _require_keys(seed_policy, ("master_seed", "derivation", "component_order", "retry_policy"), path)
    _expect_int(seed_policy["master_seed"], _path(path, "master_seed"), MASTER_SEED)
    _expect_string(seed_policy["derivation"], _path(path, "derivation"), "sha256-domain-separated-v1")
    _expect_exact_list(
        seed_policy["component_order"],
        ["contract_version", "purpose", "geometry_id", "block_id", "episode_id"],
        _path(path, "component_order"),
    )
    _expect_string(seed_policy["retry_policy"], _path(path, "retry_policy"), "same_inputs_same_seed_no_search")


def _validate_artifacts(artifacts: Mapping) -> None:
    path = "artifacts"
    _only_keys(artifacts, ("schema_version", "serialization", "hash_algorithm"), path)
    _require_keys(artifacts, ("schema_version", "serialization", "hash_algorithm"), path)
    _expect_string(artifacts["schema_version"], _path(path, "schema_version"), ARTIFACT_SCHEMA_VERSION)
    _expect_string(artifacts["serialization"], _path(path, "serialization"), CONTRACT_SERIALIZATION_VERSION)
    _expect_string(artifacts["hash_algorithm"], _path(path, "hash_algorithm"), CONTRACT_HASH_ALGORITHM)


def validate_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return an unchanged A0 contract mapping.

    The validator is strict at every level.  Unknown keys are rejected so a
    future gate cannot silently enter an A0 hash or acquire accidental meaning.
    """

    root = _require_mapping(contract, "root")
    _only_keys(root, ("contract", "audio", "actions", "coordinates", "sources", "asr", "mix", "seed_policy", "artifacts"), "root")
    _require_keys(root, ("contract", "audio", "actions", "coordinates", "sources", "asr", "mix", "seed_policy", "artifacts"), "root")
    _validate_contract_metadata(_require_mapping(root["contract"], "contract"))
    _validate_audio(_require_mapping(root["audio"], "audio"))
    _validate_actions(_require_mapping(root["actions"], "actions"))
    _validate_coordinates(_require_mapping(root["coordinates"], "coordinates"))
    _validate_sources(_require_mapping(root["sources"], "sources"))
    _validate_asr(_require_mapping(root["asr"], "asr"))
    _validate_mix(_require_mapping(root["mix"], "mix"))
    _validate_seed_policy(_require_mapping(root["seed_policy"], "seed_policy"))
    _validate_artifacts(_require_mapping(root["artifacts"], "artifacts"))
    return contract


def _canonical_value(value: Any, path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key], _path(path, key)) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, "{}[{}]".format(path, index)) for index, item in enumerate(value)]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("{} must not contain NaN or infinity".format(path))
        return 0.0 if value == 0.0 else value
    raise ContractError("{} contains unsupported value type {}".format(path, type(value).__name__))


def canonical_json(contract: Dict[str, Any]) -> str:
    """Return the validated contract's stable, newline-free JSON form."""

    validate_contract(contract)
    return json.dumps(
        _canonical_value(contract),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def contract_sha256(contract: Dict[str, Any]) -> str:
    """Return the SHA-256 of :func:`canonical_json` encoded as UTF-8."""

    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def load_contract(path: str) -> Dict[str, Any]:
    """Load and validate a YAML A0 contract without adding path metadata."""

    contract_path = Path(path)
    try:
        with contract_path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError("unable to load contract {}: {}".format(contract_path, exc)) from exc
    if not isinstance(value, dict):
        raise ContractError("contract root must be a mapping")
    return validate_contract(value)
