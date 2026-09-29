"""Strict validator for the A4 infrastructure contract.

The contract freezes long-lived interfaces and points back to the already
closed A0--A3 contracts.  A4-1 supplies the sampler and motion algorithm
identities, while A4-2A supplies source-time segment planning and dry-speech
mask semantics.  Exact smoke sources/geometries and later calibration,
timeline, mixer, and cache execution remain versioned in later A4 slices.
"""

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from active_audition.a4.identity import canonical_json, validate_sha256


A4_CONTRACT_VERSION = "active-asr-a4-infrastructure-v1"
A4_GATE = "A4"


class A4ContractError(ValueError):
    """Raised when the A4 contract is malformed or semantically incomplete."""


def _path(path: str, key: Any) -> str:
    return "{}.{}".format(path, key) if path else str(key)


def _mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise A4ContractError("{} must be a mapping".format(path))
    return value


def _only(value: Mapping, keys: Iterable[str], path: str) -> None:
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise A4ContractError("unknown A4 contract field(s) at {}: {}".format(path, ", ".join(unknown)))


def _require(value: Mapping, keys: Iterable[str], path: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise A4ContractError("missing A4 contract field(s) at {}: {}".format(path, ", ".join(missing)))


def _string(value: Any, path: str, expected: Optional[str] = None) -> str:
    if not isinstance(value, str) or not value:
        raise A4ContractError("{} must be a non-empty string".format(path))
    if expected is not None and value != expected:
        raise A4ContractError("{} must be {!r}".format(path, expected))
    return value


def _bool(value: Any, path: str, expected: Optional[bool] = None) -> bool:
    if not isinstance(value, bool):
        raise A4ContractError("{} must be boolean".format(path))
    if expected is not None and value is not expected:
        raise A4ContractError("{} must be {}".format(path, str(expected).lower()))
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise A4ContractError("{} must be a finite number".format(path))
    return float(value)


def _sha(value: Any, path: str, allow_null: bool = False) -> None:
    if allow_null and value is None:
        return
    try:
        validate_sha256(value, path)
    except ValueError as exc:
        raise A4ContractError(str(exc)) from exc


def _contract_metadata(value: Mapping) -> None:
    path = "contract"
    keys = ("namespace", "version", "gate", "state")
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["namespace"], _path(path, "namespace"), "active-asr-a4")
    _string(value["version"], _path(path, "version"), A4_CONTRACT_VERSION)
    _string(value["gate"], _path(path, "gate"), A4_GATE)
    _string(value["state"], _path(path, "state"))
    if value["state"] not in ("DRAFT", "FROZEN"):
        raise A4ContractError("contract.state must be DRAFT or FROZEN")


def _serialization(value: Mapping) -> None:
    path = "serialization"
    keys = ("version", "encoding", "sort_keys", "allow_nan", "separators", "hash_algorithm")
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["version"], _path(path, "version"), "canonical-json-v1")
    _string(value["encoding"], _path(path, "encoding"), "utf-8")
    _bool(value["sort_keys"], _path(path, "sort_keys"), True)
    _bool(value["allow_nan"], _path(path, "allow_nan"), False)
    if value["separators"] != [",", ":"]:
        raise A4ContractError("serialization.separators must be [',', ':']")
    _string(value["hash_algorithm"], _path(path, "hash_algorithm"), "sha256")


def _schema(value: Mapping) -> None:
    path = "typed_records"
    expected = {
        "geometry": "active-asr-a4-geometry-v1",
        "pose": "active-asr-a4-pose-v2",
        "block": "active-asr-a4-block-v1",
        "episode": "active-asr-a4-episode-v1",
        "calibration": "active-asr-a4-calibration-v1",
        "noise_segment": "active-asr-a4-noise-segment-v1",
        "active_mask": "active-asr-a4-active-mask-v1",
    }
    _only(value, expected, path)
    _require(value, expected, path)
    for key, version in expected.items():
        _string(value[key], _path(path, key), version)


def _lifecycle_identity(value: Any, path: str, deferred: str, state: str) -> None:
    _string(value, path)
    if state == "FROZEN" and value.startswith("DEFERRED_TO_A4_"):
        raise A4ContractError("{} remains unresolved: {}".format(path, value))


def _concrete_or_deferred_identity(value: Any, path: str, deferred: str, concrete: str, state: str) -> None:
    _string(value, path)
    if value == deferred:
        if state == "FROZEN":
            raise A4ContractError("{} remains unresolved: {}".format(path, value))
        return
    if value != concrete:
        raise A4ContractError("{} must be {!r} or {!r}".format(path, concrete, deferred))


def _sampler_boundary(value: Mapping, state: str) -> None:
    path = "sampler_boundary"
    keys = ("source_free", "selection_inputs", "forbidden_inputs", "post_sampling_source_clearance", "exact_lattice")
    _only(value, keys, path)
    _require(value, keys, path)
    _bool(value["source_free"], _path(path, "source_free"), True)
    if value["selection_inputs"] != ["navmesh_legality", "pathfinder", "stable_probe_identity"]:
        raise A4ContractError("sampler_boundary.selection_inputs is not the frozen source-free boundary")
    if value["forbidden_inputs"] != ["target_world_pose", "noise_world_pose", "source_direction", "rir", "snr", "asr", "wer", "oracle_landscape"]:
        raise A4ContractError("sampler_boundary.forbidden_inputs is not the frozen source-free boundary")
    _string(value["post_sampling_source_clearance"], _path(path, "post_sampling_source_clearance"), "legality_annotation_only")
    lattice = _mapping(value["exact_lattice"], _path(path, "exact_lattice"))
    lattice_keys = (
        "schema_version", "algorithm_identity", "coordinate_convention_identity",
        "technical_filter_order", "coverage_selection_identity", "tie_break_identity",
        "parameter_schema",
    )
    _only(lattice, lattice_keys, _path(path, "exact_lattice"))
    _require(lattice, lattice_keys, _path(path, "exact_lattice"))
    _string(lattice["schema_version"], _path(path, "exact_lattice.schema_version"), "active-asr-a4-candidate-contract-v1")
    _string(lattice["algorithm_identity"], _path(path, "exact_lattice.algorithm_identity"), "active-asr-a4-local-polar-sampler-v1")
    _string(lattice["coordinate_convention_identity"], _path(path, "exact_lattice.coordinate_convention_identity"), "active-asr-a0-positive-left-forward-minus-z-v1")
    if lattice["technical_filter_order"] != ["requested_point", "snap_point", "finite_check", "navigability", "snap_error", "shortest_path", "geodesic_radius", "duplicate_filter"]:
        raise A4ContractError("sampler_boundary.exact_lattice.technical_filter_order is invalid")
    _string(lattice["coverage_selection_identity"], _path(path, "exact_lattice.coverage_selection_identity"), "active-asr-a4-farthest-coverage-v1")
    _string(lattice["tie_break_identity"], _path(path, "exact_lattice.tie_break_identity"), "active-asr-a4-probe-id-lexical-tie-break-v1")
    schema = _mapping(lattice["parameter_schema"], _path(path, "exact_lattice.parameter_schema"))
    schema_keys = ("radii_m", "azimuth_offsets_deg", "max_geodesic_radius_m", "max_snap_error_m", "duplicate_position_tolerance_m", "max_noninitial_positions", "yaw_offsets_deg", "source_clearance_m")
    _only(schema, schema_keys, _path(path, "exact_lattice.parameter_schema"))
    _require(schema, schema_keys, _path(path, "exact_lattice.parameter_schema"))
    for key in schema_keys:
        _string(schema[key], _path(path, "exact_lattice.parameter_schema." + key))


def _motion(value: Mapping, state: str) -> None:
    path = "motion_cost"
    keys = ("formula_identity", "parameter_schema", "exact_execution", "distance_authority", "turn_plane", "zero_length_segment_policy", "special_cases")
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["formula_identity"], _path(path, "formula_identity"), "active-asr-a4-motion-cost-v1")
    _string(value["exact_execution"], _path(path, "exact_execution"), "active-asr-a4-polyline-turn-execution-v1")
    _string(value["distance_authority"], _path(path, "distance_authority"), "pathfinder_shortest_path_geodesic_distance_m_v1")
    _string(value["turn_plane"], _path(path, "turn_plane"), "horizontal_xz_positive_left_v1")
    _string(value["zero_length_segment_policy"], _path(path, "zero_length_segment_policy"), "remove_near_zero_horizontal_segments_v1")
    schema = _mapping(value["parameter_schema"], _path(path, "parameter_schema"))
    schema_keys = ("translation_speed_mps", "rotation_speed_dps", "settling_sec", "budget_sec")
    _only(schema, schema_keys, _path(path, "parameter_schema"))
    _require(schema, schema_keys, _path(path, "parameter_schema"))
    for key in schema_keys:
        _string(schema[key], _path(path, "parameter_schema." + key))
    special = _mapping(value["special_cases"], _path(path, "special_cases"))
    special_keys = ("stay", "rotate_only", "translation")
    _only(special, special_keys, _path(path, "special_cases"))
    _require(special, special_keys, _path(path, "special_cases"))
    for key in special_keys:
        _string(special[key], _path(path, "special_cases." + key))


def _geometry_legality(value: Mapping) -> None:
    path = "geometry_legality"
    keys = ("phase", "receiver_reference", "source_reference", "distance_metric", "threshold_parameter", "replacement_sampling", "invalid_reasons")
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["phase"], _path(path, "phase"), "post_sampling_only")
    _string(value["receiver_reference"], _path(path, "receiver_reference"), "actual_listener_sensor_xyz")
    _string(value["source_reference"], _path(path, "source_reference"), "acoustic_source_world_xyz")
    _string(value["distance_metric"], _path(path, "distance_metric"), "euclidean_3d")
    _string(value["threshold_parameter"], _path(path, "threshold_parameter"), "candidate_contract.source_clearance_m")
    _bool(value["replacement_sampling"], _path(path, "replacement_sampling"), False)
    if value["invalid_reasons"] != ["TARGET_SOURCE_CLEARANCE", "NOISE_SOURCE_CLEARANCE", "TARGET_AND_NOISE_SOURCE_CLEARANCE"]:
        raise A4ContractError("geometry_legality.invalid_reasons is invalid")


def _noise_segment_planner(value: Mapping) -> None:
    path = "noise_segment_planner"
    keys = (
        "algorithm_identity", "schema_version", "source_time_convention",
        "sample_index_convention", "episode_order_identity",
        "selection_evaluation_nonoverlap", "parent_ranges_nonoverlap",
        "no_looping", "no_provenance_free_concat", "parameter_schema",
    )
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["algorithm_identity"], _path(path, "algorithm_identity"), "active-asr-a4-noise-segment-planner-v1")
    _string(value["schema_version"], _path(path, "schema_version"), "active-asr-a4-noise-segment-plan-v1")
    _string(value["source_time_convention"], _path(path, "source_time_convention"), "active-asr-a4-target-dry-onset-zero-v1")
    _string(value["sample_index_convention"], _path(path, "sample_index_convention"), "active-asr-a4-half-open-sample-range-v1")
    _string(value["episode_order_identity"], _path(path, "episode_order_identity"), "active-asr-a4-stable-episode-order-v1")
    for key in ("selection_evaluation_nonoverlap", "parent_ranges_nonoverlap", "no_looping", "no_provenance_free_concat"):
        _bool(value[key], _path(path, key), True)
    schema = _mapping(value["parameter_schema"], _path(path, "parameter_schema"))
    schema_keys = ("sample_rate_hz", "pre_roll_sec", "post_roll_sec", "guard_interval_sec", "episode_count")
    _only(schema, schema_keys, _path(path, "parameter_schema"))
    _require(schema, schema_keys, _path(path, "parameter_schema"))
    for key in schema_keys:
        _string(schema[key], _path(path, "parameter_schema." + key))


def _active_mask(value: Mapping) -> None:
    path = "active_mask"
    keys = (
        "algorithm_identity", "schema_version", "input_identity", "frame_rms_identity",
        "percentile_identity", "threshold_identity", "expansion_identity", "parameter_schema",
    )
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["algorithm_identity"], _path(path, "algorithm_identity"), "active-asr-a4-dry-speech-active-mask-v1")
    _string(value["schema_version"], _path(path, "schema_version"), "active-asr-a4-active-mask-v1")
    _string(value["input_identity"], _path(path, "input_identity"), "dry_target_speech_only_v1")
    _string(value["frame_rms_identity"], _path(path, "frame_rms_identity"), "frame_rms_no_padding_v1")
    _string(value["percentile_identity"], _path(path, "percentile_identity"), "linear_interpolated_finite_percentile_v1")
    _string(value["threshold_identity"], _path(path, "threshold_identity"), "reference_plus_db_offset_v1")
    _string(value["expansion_identity"], _path(path, "expansion_identity"), "frame_to_sample_half_open_union_v1")
    schema = _mapping(value["parameter_schema"], _path(path, "parameter_schema"))
    schema_keys = ("frame_duration_ms", "hop_duration_ms", "reference_percentile", "threshold_db")
    _only(schema, schema_keys, _path(path, "parameter_schema"))
    _require(schema, schema_keys, _path(path, "parameter_schema"))
    for key in schema_keys:
        _string(schema[key], _path(path, "parameter_schema." + key))


def _acoustic(value: Mapping) -> None:
    path = "production_acoustics"
    keys = ("sample_rate_hz", "channel_layout", "channel_order", "materials", "convolution", "speech_emission_scale", "normalization", "time_convention")
    _only(value, keys, path)
    _require(value, keys, path)
    if value["sample_rate_hz"] != 16000 or value["channel_layout"] != "binaural" or value["channel_order"] != ["L", "R"]:
        raise A4ContractError("production_acoustics native16/canonical binaural fields are invalid")
    _string(value["materials"], _path(path, "materials"), "OFF")
    _string(value["convolution"], _path(path, "convolution"), "full")
    _number(value["speech_emission_scale"], _path(path, "speech_emission_scale"))
    if float(value["speech_emission_scale"]) != 1.0:
        raise A4ContractError("production_acoustics.speech_emission_scale must be 1.0")
    if value["normalization"] != {"per_pose": False, "per_source": False, "per_channel": False}:
        raise A4ContractError("production_acoustics normalization policy is invalid")
    _string(value["time_convention"], _path(path, "time_convention"), "source_time_common_receiver_time")


def _calibration(value: Mapping, state: str) -> None:
    path = "calibration_boundary"
    keys = ("scope", "active_mask", "selection_episode_count", "algorithm", "artifact_schema")
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["scope"], _path(path, "scope"), "selection_only_initial_pose_once")
    _concrete_or_deferred_identity(
        value["active_mask"], _path(path, "active_mask"),
        "DEFERRED_TO_A4_2", "active-asr-a4-dry-speech-active-mask-v1", state
    )
    if value["selection_episode_count"] != 2:
        raise A4ContractError("calibration_boundary.selection_episode_count must be 2")
    _lifecycle_identity(value["algorithm"], _path(path, "algorithm"), "DEFERRED_TO_A4_2", state)
    _string(value["artifact_schema"], _path(path, "artifact_schema"), "active-asr-a4-calibration-v1")


def _mixture(value: Mapping, state: str) -> None:
    path = "mixture_boundary"
    keys = ("target_noise_propagation", "calibration_input", "timeline", "reconstruction", "algorithm")
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["target_noise_propagation"], _path(path, "target_noise_propagation"), "independent_rirs")
    _string(value["calibration_input"], _path(path, "calibration_input"), "calibration_artifact_identity_only")
    _lifecycle_identity(value["timeline"], _path(path, "timeline"), "DEFERRED_TO_A4_2", state)
    _lifecycle_identity(value["reconstruction"], _path(path, "reconstruction"), "DEFERRED_TO_A4_2", state)
    _lifecycle_identity(value["algorithm"], _path(path, "algorithm"), "DEFERRED_TO_A4_2", state)


def _cache(value: Mapping, state: str) -> None:
    path = "cache_resume"
    keys = ("keying", "integrity", "resume", "algorithm")
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["keying"], _path(path, "keying"), "content_addressed_semantic_payload")
    if value["integrity"] != ["key_recompute", "metadata_schema", "payload_exists", "payload_sha256", "shape_dtype", "semantic_identities"]:
        raise A4ContractError("cache_resume.integrity is incomplete")
    _string(value["resume"], _path(path, "resume"), "reject_incomplete_or_corrupt_entries")
    _lifecycle_identity(value["algorithm"], _path(path, "algorithm"), "DEFERRED_TO_A4_3", state)


def _access(value: Mapping) -> None:
    path = "access_boundary"
    keys = ("selection", "evaluation", "result_dependent_selection")
    _only(value, keys, path)
    _require(value, keys, path)
    _string(value["selection"], _path(path, "selection"), "selection_only_for_calibration_and_pose_choice")
    _string(value["evaluation"], _path(path, "evaluation"), "evaluation_never_changes_pose_choice")
    _bool(value["result_dependent_selection"], _path(path, "result_dependent_selection"), False)


def _parents(value: Mapping) -> None:
    path = "frozen_parents"
    keys = ("a0_contract_sha256", "a2_contract_sha256", "a3_contract_sha256")
    _only(value, keys, path)
    _require(value, keys, path)
    for key in keys:
        _sha(value[key], _path(path, key))


def _reject_deferred(value: Any, path: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_deferred(item, _path(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_deferred(item, "{}[{}]".format(path, index))
    elif isinstance(value, str) and value.startswith("DEFERRED_TO_A4_"):
        raise A4ContractError("{} remains unresolved: {}".format(path, value))


def validate_contract(value: Mapping[str, Any], require_frozen: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise A4ContractError("A4 contract must be a mapping")
    keys = (
        "contract", "serialization", "typed_records", "sampler_boundary", "motion_cost",
        "geometry_legality", "noise_segment_planner", "active_mask", "production_acoustics",
        "calibration_boundary", "mixture_boundary", "cache_resume",
        "access_boundary", "frozen_parents",
    )
    _only(value, keys, "root")
    _require(value, keys, "root")
    _contract_metadata(_mapping(value["contract"], "contract"))
    state = value["contract"]["state"]
    if require_frozen and state != "FROZEN":
        raise A4ContractError("A4 contract must be FROZEN")
    _serialization(_mapping(value["serialization"], "serialization"))
    _schema(_mapping(value["typed_records"], "typed_records"))
    _sampler_boundary(_mapping(value["sampler_boundary"], "sampler_boundary"), state)
    _motion(_mapping(value["motion_cost"], "motion_cost"), state)
    _geometry_legality(_mapping(value["geometry_legality"], "geometry_legality"))
    _noise_segment_planner(_mapping(value["noise_segment_planner"], "noise_segment_planner"))
    _active_mask(_mapping(value["active_mask"], "active_mask"))
    _acoustic(_mapping(value["production_acoustics"], "production_acoustics"))
    _calibration(_mapping(value["calibration_boundary"], "calibration_boundary"), state)
    _mixture(_mapping(value["mixture_boundary"], "mixture_boundary"), state)
    _cache(_mapping(value["cache_resume"], "cache_resume"), state)
    _access(_mapping(value["access_boundary"], "access_boundary"))
    _parents(_mapping(value["frozen_parents"], "frozen_parents"))
    if require_frozen:
        _reject_deferred(value)
    return value


def canonical_contract_json(value: Mapping[str, Any]) -> str:
    validate_contract(value)
    return canonical_json(value)


def contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_contract_json(value).encode("utf-8")).hexdigest()


def load_contract(path: str, require_frozen: bool = False) -> Mapping[str, Any]:
    contract_path = Path(path)
    try:
        with contract_path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise A4ContractError("unable to load A4 contract {}: {}".format(contract_path, exc)) from exc
    validate_contract(value, require_frozen=require_frozen)
    return value


__all__ = [
    "A4ContractError",
    "A4_CONTRACT_VERSION",
    "A4_GATE",
    "canonical_contract_json",
    "contract_sha256",
    "load_contract",
    "validate_contract",
]
