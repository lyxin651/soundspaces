"""A3-v2 DISCOVER/FREEZE preparation, with no acoustic or ASR execution.

The public ``freeze_a3_v2`` entry point only performs:

* Replica asset inventory and renderer/navmesh scene-load smoke;
* deterministic geometry-only held-out case selection;
* metadata-only held-out LibriSpeech selection;
* creation and freezing of the versioned A3-v2 contract.

No audio sensor, RIR render, waveform convolution, SpeechBrain call, decoder,
WER, or acoustic score is reachable from this module.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from active_audition.asr.contract import asr_contract_sha256, load_asr_contract
from active_audition.asr.contract_v2 import (
    A3_V1_SHA256,
    A3_V2_VERSION,
    a3_v2_contract_sha256,
    validate_a3_v2_contract,
)
from active_audition.asr.static_mesh_los import (
    STATIC_MESH_EPSILON_M,
    STATIC_MESH_LOS_METHOD,
    StaticPlyMesh,
    synthetic_sanity_results,
)
from active_audition.receiver.geometry import relative_azimuth_deg, source_position_world


INVENTORY_SCHEMA = "active-asr-a3-v2-replica-scene-inventory-v2"
SCENE_MANIFEST_SCHEMA = "active-asr-a3-v2-realistic-domain-scene-manifest-v2"
SPEECH_MANIFEST_SCHEMA = "active-asr-a3-v2-realistic-domain-speech-manifest-v1"
SELECTION_SEED = 20260928
SMOKE_TIMEOUT_SEC = 180
SENSOR_OFFSET = (0.0, 1.5, 0.0)
# Replica's binary mesh_semantic.ply stores coordinates as (x, depth, height),
# while the Habitat world convention is (x, height, -depth).  This is an
# asset-format conversion, not a stage transform.  The stage node is still
# audited and must be identity.
REPLICA_PLY_TO_HABITAT = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)
HABITAT_TO_REPLICA_PLY = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)
STATIC_MESH_SANITY_CASE_COUNT = 3
REPLICA_ROOT = "data/scene_datasets/replica"
INVENTORY_PATH = "registries/active_asr_a3_v2/replica_scene_inventory.json"
SCENE_MANIFEST_PATH = "registries/active_asr_a3_v2/a3_v2_realistic_domain_scene_manifest.json"
SPEECH_MANIFEST_PATH = "registries/active_asr_a3_v2/a3_v2_realistic_domain_speech_manifest.json"
V2_CONTRACT_PATH = "configs/active_audition/v1/asr_contract_v2.yaml"


class A3V2FreezeError(RuntimeError):
    """Raised when DISCOVER/FREEZE cannot produce authoritative inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    path.write_bytes(payload + b"\n")
    return sha256_file(path)


def _strict_keys(value: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise A3V2FreezeError("invalid {} keys: unknown={} missing={}".format(path, unknown, missing))


def _validate_file_identity(value: Any, path: str, allow_missing: bool = False) -> None:
    if not isinstance(value, Mapping):
        raise A3V2FreezeError("{} must be a mapping".format(path))
    _strict_keys(value, ("path", "exists", "sha256"), path)
    if not isinstance(value["path"], str) or not value["path"] or not isinstance(value["exists"], bool):
        raise A3V2FreezeError("{} has invalid path/exists".format(path))
    if value["exists"]:
        if not isinstance(value["sha256"], str) or len(value["sha256"]) != 64:
            raise A3V2FreezeError("{} existing resource must have SHA256".format(path))
    elif not allow_missing and value["sha256"] is not None:
        raise A3V2FreezeError("{} missing resource must have null SHA256".format(path))


def validate_scene_inventory(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _strict_keys(value, ("schema_version", "discovery_root", "selection_policy", "scenes", "asr_run_before_inventory", "wer_run_before_inventory"), "scene_inventory")
    if value["schema_version"] != INVENTORY_SCHEMA or value["discovery_root"] != REPLICA_ROOT:
        raise A3V2FreezeError("scene inventory identity is invalid")
    _strict_keys(value["selection_policy"], ("technical_eligibility_inputs", "forbidden_inputs", "prior_asr_scene_exclusions", "selection_rule"), "scene_inventory.selection_policy")
    if value["selection_policy"]["technical_eligibility_inputs"] != [
        "file_completeness",
        "scene_load",
        "pathfinder",
        "audio_sensor_construction",
        "native16",
        "binaural",
        "materials_off",
    ]:
        raise A3V2FreezeError("scene inventory technical eligibility inputs are invalid")
    if value["selection_policy"]["forbidden_inputs"] != ["RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "Oracle"]:
        raise A3V2FreezeError("scene inventory forbiddens are invalid")
    if value["asr_run_before_inventory"] is not False or value["wer_run_before_inventory"] is not False:
        raise A3V2FreezeError("scene inventory was not frozen before ASR")
    if not isinstance(value["scenes"], list) or not value["scenes"]:
        raise A3V2FreezeError("scene inventory must contain scenes")
    for row in value["scenes"]:
        _strict_keys(row, ("scene_id", "assets", "file_completeness", "runtime_scene_load_smoke", "technical_eligibility", "exclusion_reason"), "scene_inventory.scenes[]")
        if not isinstance(row["scene_id"], str) or not row["scene_id"].startswith("replica."):
            raise A3V2FreezeError("invalid Replica scene_id")
        _strict_keys(row["assets"], ("scene_asset", "navmesh", "semantic_info", "stage_config"), "scene_inventory.assets")
        for key in row["assets"]:
            _validate_file_identity(row["assets"][key], "scene_inventory.assets." + key)
        smoke_keys = (
            "status", "scene_created", "pathfinder_loaded", "pathfinder_loaded_before_explicit_load",
            "explicit_navmesh_load_requested", "process_returncode", "timeout_sec", "stdout_tail",
            "stderr_tail", "audio_sensor_created", "requested_sample_rate_hz",
            "effective_sample_rate_hz", "requested_channel_layout", "effective_channel_layout",
            "requested_channel_count", "effective_channel_count", "materials_requested",
            "materials_effective", "renderer_construction_status", "rir_rendered", "asr_run", "wer_run",
        )
        unknown_smoke = sorted(set(row["runtime_scene_load_smoke"]) - set(smoke_keys))
        if unknown_smoke:
            raise A3V2FreezeError("invalid scene_inventory.runtime_scene_load_smoke keys: unknown={}".format(unknown_smoke))
        smoke = row["runtime_scene_load_smoke"]
        for key in ("scene_created", "pathfinder_loaded", "audio_sensor_created", "materials_requested", "rir_rendered", "asr_run", "wer_run"):
            if key in smoke and not isinstance(smoke[key], bool):
                raise A3V2FreezeError("scene smoke {} must be boolean".format(key))
        if smoke.get("materials_effective") not in (None, True, False):
            raise A3V2FreezeError("scene smoke materials_effective must be boolean or null")
        if smoke.get("rir_rendered") is not False or smoke.get("asr_run") is not False or smoke.get("wer_run") is not False:
            raise A3V2FreezeError("scene smoke contains a forbidden operation")
        if smoke.get("requested_sample_rate_hz") not in (None, 16000) or smoke.get("effective_sample_rate_hz") not in (None, 16000):
            raise A3V2FreezeError("scene smoke sample rate is invalid")
        if smoke.get("requested_channel_layout") not in (None, "binaural") or smoke.get("effective_channel_layout") not in (None, "binaural"):
            raise A3V2FreezeError("scene smoke channel layout is invalid")
        if smoke.get("requested_channel_count") not in (None, 2) or smoke.get("effective_channel_count") not in (None, 2):
            raise A3V2FreezeError("scene smoke channel count is invalid")
        if smoke.get("materials_requested") not in (None, False) or smoke.get("materials_effective") not in (None, False):
            raise A3V2FreezeError("scene smoke materials state is invalid")
        if smoke.get("renderer_construction_status") not in (None, "PASS", "FAIL", "NOT_RUN"):
            raise A3V2FreezeError("scene smoke renderer construction status is invalid")
        if not isinstance(row["technical_eligibility"], bool) or not isinstance(row["exclusion_reason"], str):
            raise A3V2FreezeError("scene inventory eligibility fields are invalid")
        if row["technical_eligibility"] and not (
            row["file_completeness"] == "PASS"
            and smoke.get("status") == "PASS"
            and smoke.get("scene_created") is True
            and smoke.get("pathfinder_loaded") is True
            and smoke.get("audio_sensor_created") is True
            and smoke.get("renderer_construction_status") == "PASS"
            and smoke.get("effective_sample_rate_hz") == 16000
            and smoke.get("effective_channel_layout") == "binaural"
            and smoke.get("effective_channel_count") == 2
            and smoke.get("materials_requested") is False
            and smoke.get("materials_effective") is False
        ):
            raise A3V2FreezeError("technical eligibility is not backed by effective renderer construction evidence")
    return value


def validate_scene_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _strict_keys(value, ("schema_version", "scene_inventory", "selected_scene_ids", "selection_policy", "visibility_sanity", "scenes", "asr_run_before_freeze", "wer_run_before_freeze"), "scene_manifest")
    if value["schema_version"] != SCENE_MANIFEST_SCHEMA:
        raise A3V2FreezeError("scene manifest identity is invalid")
    _strict_keys(value["scene_inventory"], ("path", "sha256", "schema_version"), "scene_manifest.scene_inventory")
    if value["scene_inventory"]["schema_version"] != INVENTORY_SCHEMA:
        raise A3V2FreezeError("scene manifest inventory schema mismatch")
    if len(value["scene_inventory"]["sha256"]) != 64:
        raise A3V2FreezeError("scene manifest inventory hash is invalid")
    _strict_keys(
        value["selection_policy"],
        (
            "selection_rule", "case_categories", "geometry_inputs", "forbidden_inputs", "grid_step_m",
            "listener_yaw_deg", "sensor_offset_m", "visibility_method", "visibility_epsilon_m",
            "coordinate_convention", "stage_transform_policy",
        ),
        "scene_manifest.selection_policy",
    )
    if value["selection_policy"]["case_categories"] != ["near_front_like", "moderate_front_like", "far_front_like", "far_off_axis"]:
        raise A3V2FreezeError("scene manifest categories are not frozen")
    if value["selection_policy"]["forbidden_inputs"] != ["RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "Oracle"]:
        raise A3V2FreezeError("scene manifest forbiddens are invalid")
    if value["asr_run_before_freeze"] is not False or value["wer_run_before_freeze"] is not False:
        raise A3V2FreezeError("scene manifest was not frozen before ASR")
    if value["selected_scene_ids"] != sorted(value["selected_scene_ids"]) or len(value["selected_scene_ids"]) != 2:
        raise A3V2FreezeError("scene selection must contain two lexical scene IDs")
    if len(value["scenes"]) != 2 or [row["scene_id"] for row in value["scenes"]] != value["selected_scene_ids"]:
        raise A3V2FreezeError("scene manifest selected scene rows are inconsistent")
    if value["selection_policy"]["visibility_method"] != STATIC_MESH_LOS_METHOD:
        raise A3V2FreezeError("scene manifest visibility method is not the static mesh verifier")
    if float(value["selection_policy"]["visibility_epsilon_m"]) != STATIC_MESH_EPSILON_M:
        raise A3V2FreezeError("scene manifest visibility epsilon is not frozen")
    if value["selection_policy"]["coordinate_convention"] != "replica_ply_xyz_to_habitat_xyz_x_z_neg_y":
        raise A3V2FreezeError("scene manifest coordinate convention is invalid")
    if value["selection_policy"]["stage_transform_policy"] != "stage_node_identity_verified":
        raise A3V2FreezeError("scene manifest stage transform policy is invalid")
    if not isinstance(value.get("visibility_sanity"), Mapping):
        raise A3V2FreezeError("scene manifest visibility sanity is missing")
    _strict_keys(value["visibility_sanity"], ("synthetic", "mesh_provenance", "floor_checks", "determinism"), "scene_manifest.visibility_sanity")
    if value["visibility_sanity"]["synthetic"].get("passed") is not True or value["visibility_sanity"]["determinism"].get("passed") is not True:
        raise A3V2FreezeError("static mesh synthetic/determinism sanity failed")
    if not value["visibility_sanity"]["floor_checks"] or not all(item.get("passed") is True for item in value["visibility_sanity"]["floor_checks"]):
        raise A3V2FreezeError("static mesh floor sanity failed")
    if len(value["visibility_sanity"]["mesh_provenance"]) != 2:
        raise A3V2FreezeError("static mesh provenance must cover two selected scenes")
    case_keys = (
        "category", "listener_base_position_world", "listener_sensor_position_world", "listener_yaw_deg",
        "source_base_position_world", "source_position_world", "distance_m", "relative_azimuth_deg",
        "los_status", "route_evidence", "visibility_evidence", "coordinate_provenance", "selection_score",
        "fallback_used", "selection_rule", "scene_id", "case_id", "scene_asset_sha256", "navmesh_sha256",
        "semantic_info_sha256", "stage_config_sha256",
    )
    for scene in value["scenes"]:
        _strict_keys(scene, ("scene_id", "cases"), "scene_manifest.scenes[]")
        if len(scene["cases"]) != 4:
            raise A3V2FreezeError("each held-out scene must have four cases")
        for case in scene["cases"]:
            _strict_keys(case, case_keys, "scene_manifest.cases[]")
            if case["scene_id"] != scene["scene_id"] or case["fallback_used"] is not False or case["los_status"] != "VERIFIED_GEOMETRIC_LOS_STATIC_MESH":
                raise A3V2FreezeError("scene case provenance is invalid")
            for key in ("scene_asset_sha256", "navmesh_sha256", "semantic_info_sha256", "stage_config_sha256"):
                if not isinstance(case[key], str) or len(case[key]) != 64:
                    raise A3V2FreezeError("scene case resource hash is invalid")
            if not isinstance(case["route_evidence"], Mapping):
                raise A3V2FreezeError("scene case route evidence is invalid")
            _strict_keys(case["route_evidence"], ("found", "geodesic_distance_m", "euclidean_distance_m", "route_delta_m", "route_point_count"), "scene_manifest.route_evidence")
            visibility = case["visibility_evidence"]
            if not isinstance(visibility, Mapping):
                raise A3V2FreezeError("scene case visibility evidence is invalid")
            _strict_keys(
                visibility,
                (
                    "visibility_method", "mesh_sha256", "origin", "target", "mesh_origin", "mesh_target",
                    "segment_length_m", "intersection_count", "first_intersection_distance_m",
                    "clear_line_of_sight", "epsilon_m", "runtime_provenance",
                ),
                "scene_manifest.visibility_evidence",
            )
            if visibility["visibility_method"] != STATIC_MESH_LOS_METHOD or visibility["mesh_sha256"] != case["scene_asset_sha256"]:
                raise A3V2FreezeError("scene case mesh visibility binding is invalid")
            if visibility["clear_line_of_sight"] is not True or visibility["intersection_count"] != 0:
                raise A3V2FreezeError("scene case is not verified geometric LOS")
            if float(visibility["epsilon_m"]) != STATIC_MESH_EPSILON_M:
                raise A3V2FreezeError("scene case visibility epsilon is invalid")
            _strict_keys(
                visibility["runtime_provenance"],
                ("geometry_library", "geometry_library_version", "backend", "backend_version", "python", "coordinate_convention", "stage_transform"),
                "scene_manifest.visibility_evidence.runtime_provenance",
            )
            if visibility["runtime_provenance"]["backend"] != STATIC_MESH_LOS_METHOD or visibility["runtime_provenance"]["stage_transform"] != "identity":
                raise A3V2FreezeError("scene case visibility runtime provenance is invalid")
            coordinate = case["coordinate_provenance"]
            if not isinstance(coordinate, Mapping):
                raise A3V2FreezeError("scene case coordinate provenance is invalid")
            _strict_keys(
                coordinate,
                (
                    "mesh_to_habitat_matrix", "habitat_to_mesh_matrix", "stage_transform", "runtime_scene_asset",
                    "runtime_scene_asset_sha256", "transform_source",
                ),
                "scene_manifest.coordinate_provenance",
            )
            if coordinate["runtime_scene_asset_sha256"] != case["scene_asset_sha256"] or coordinate["stage_transform"] != "identity":
                raise A3V2FreezeError("scene case coordinate binding is invalid")
    return value


def validate_speech_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _strict_keys(value, ("schema_version", "speech_registry", "selection_policy", "speaker_audit", "records", "asr_run_before_freeze", "wer_run_before_freeze"), "speech_manifest")
    if value["schema_version"] != SPEECH_MANIFEST_SCHEMA:
        raise A3V2FreezeError("speech manifest identity is invalid")
    _strict_keys(value["speech_registry"], ("path", "sha256", "schema_version"), "speech_manifest.speech_registry")
    if len(value["speech_registry"]["sha256"]) != 64 or value["speech_registry"]["schema_version"] != "active-asr-a3-speech-registry-v1":
        raise A3V2FreezeError("speech registry binding is invalid")
    policy_keys = ("selection_time", "method", "seed", "constraints", "required_counts", "forbidden_inputs", "historical_utterance_exclusion", "historical_speaker_exclusion_count")
    _strict_keys(value["selection_policy"], policy_keys, "speech_manifest.selection_policy")
    constraints = value["selection_policy"]["constraints"]
    _strict_keys(constraints, ("duration_sec", "minimum_reference_words", "complete_utterance"), "speech_manifest.constraints")
    if constraints["duration_sec"] != [4.0, 15.0] or constraints["minimum_reference_words"] != 10 or constraints["complete_utterance"] is not True:
        raise A3V2FreezeError("speech constraints are invalid")
    if value["selection_policy"]["forbidden_inputs"] != ["audio_listening", "ASR", "WER", "decoder_score", "RIR", "energy", "Oracle"]:
        raise A3V2FreezeError("speech selection forbiddens are invalid")
    for history in value["selection_policy"]["historical_utterance_exclusion"]:
        _strict_keys(history, ("path", "sha256", "utterance_count", "utterance_ids"), "speech_manifest.history[]")
        if len(history["sha256"]) != 64 or not isinstance(history["utterance_ids"], list):
            raise A3V2FreezeError("speech historical exclusion is invalid")
    _strict_keys(value["speaker_audit"], ("selected_speaker_ids", "historical_excluded_speaker_ids", "selected_historical_speaker_overlap", "speaker_disjoint_from_history"), "speech_manifest.speaker_audit")
    if value["speaker_audit"]["selected_historical_speaker_overlap"] or value["speaker_audit"]["speaker_disjoint_from_history"] is not True:
        raise A3V2FreezeError("speech speaker-disjoint audit failed")
    if value["asr_run_before_freeze"] is not False or value["wer_run_before_freeze"] is not False:
        raise A3V2FreezeError("speech manifest was not frozen before ASR")
    record_keys = ("utterance_id", "split", "speaker_id", "chapter_id", "relative_source_path", "source_file_sha256", "decoded_waveform_sha256", "samples", "duration_sec", "normalized_transcript", "reference_word_count", "complete_utterance")
    records = value["records"]
    if len(records) != 24:
        raise A3V2FreezeError("speech manifest must contain 24 records")
    if sum(row["split"] == "dev-clean" for row in records) != 12 or sum(row["split"] == "dev-other" for row in records) != 12:
        raise A3V2FreezeError("speech manifest must contain 12 dev-clean and 12 dev-other records")
    if len({row["utterance_id"] for row in records}) != 24:
        raise A3V2FreezeError("speech manifest contains duplicate utterances")
    for row in records:
        _strict_keys(row, record_keys, "speech_manifest.records[]")
        if not row["complete_utterance"] or not (4.0 <= float(row["duration_sec"]) <= 15.0) or int(row["reference_word_count"]) < 10:
            raise A3V2FreezeError("speech record violates held-out eligibility")
        for key in ("source_file_sha256", "decoded_waveform_sha256"):
            if len(row[key]) != 64:
                raise A3V2FreezeError("speech record hash is invalid")
    return value


def _relative(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _file_identity(path: Path, repo: Path) -> Dict[str, Any]:
    exists = path.is_file()
    return {
        "path": _relative(path, repo),
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
    }


def _scene_paths(repo: Path, scene_dir: Path) -> Dict[str, Path]:
    habitat = scene_dir / "habitat"
    return {
        "scene_asset": habitat / "mesh_semantic.ply",
        "navmesh": habitat / "mesh_semantic.navmesh",
        "semantic_info": habitat / "info_semantic.json",
        "stage_config": habitat / "replica_stage.stage_config.json",
    }


def _smoke_child(scene_asset: str, navmesh: str) -> None:
    """Run isolated scene/navmesh/audio-sensor construction only."""

    # This import order is required by the SS2 Habitat-Sim installation.
    import quaternion  # noqa: F401
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(Path(scene_asset).resolve())
    backend.enable_physics = False
    backend.load_semantic_mesh = True
    agent = habitat_sim.agent.AgentConfiguration()
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    try:
        pathfinder_loaded_before = bool(simulator.pathfinder.is_loaded)
        explicit_navmesh_load = False
        if not pathfinder_loaded_before:
            explicit_navmesh_load = True
            if not simulator.pathfinder.load_nav_mesh(str(Path(navmesh).resolve())):
                raise A3V2FreezeError("PathFinder.load_nav_mesh returned false")
        audio_spec = habitat_sim.AudioSensorSpec()
        audio_spec.uuid = "audio_sensor"
        audio_spec.enableMaterials = False
        audio_spec.position = list(SENSOR_OFFSET)
        audio_spec.acousticsConfig.sampleRate = 16000
        audio_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        audio_spec.channelLayout.channelCount = 2
        simulator.add_sensor(audio_spec)
        sensor = simulator.get_agent(0)._sensors["audio_sensor"]
        effective_spec = sensor.specification()
        effective_layout = effective_spec.channelLayout.type
        layout_is_binaural = effective_layout == habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        effective_sample_rate = float(effective_spec.acousticsConfig.sampleRate)
        effective_sample_rate = int(effective_sample_rate) if effective_sample_rate.is_integer() else effective_sample_rate
        effective_channel_count = int(effective_spec.channelLayout.channelCount)
        effective_materials = bool(effective_spec.enableMaterials)
        effective_state_verified = (
            effective_sample_rate == 16000
            and layout_is_binaural
            and effective_channel_count == 2
            and effective_materials is False
            and list(effective_spec.position) == list(SENSOR_OFFSET)
        )
        result = {
            "status": "PASS",
            "scene_created": True,
            "pathfinder_loaded": bool(simulator.pathfinder.is_loaded),
            "pathfinder_loaded_before_explicit_load": pathfinder_loaded_before,
            "explicit_navmesh_load_requested": explicit_navmesh_load,
            "audio_sensor_created": True,
            "requested_sample_rate_hz": 16000,
            "effective_sample_rate_hz": effective_sample_rate,
            "requested_channel_layout": "binaural",
            "effective_channel_layout": "binaural" if layout_is_binaural else str(effective_layout),
            "requested_channel_count": 2,
            "effective_channel_count": effective_channel_count,
            "materials_requested": False,
            "materials_effective": effective_materials,
            "renderer_construction_status": "PASS" if effective_state_verified else "FAIL",
            "rir_rendered": False,
            "asr_run": False,
            "wer_run": False,
        }
        print("A3V2_SMOKE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        simulator.close()


def _run_scene_smoke(repo: Path, paths: Mapping[str, Path]) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "active_audition.asr.a3_v2_freeze",
        "_scene-smoke-child",
        "--scene-asset",
        str(paths["scene_asset"].resolve()),
        "--navmesh",
        str(paths["navmesh"].resolve()),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        result = subprocess.run(
            command,
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SMOKE_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "FAIL",
            "scene_created": False,
            "pathfinder_loaded": False,
            "process_returncode": None,
            "timeout_sec": SMOKE_TIMEOUT_SEC,
            "stderr_tail": str(error)[:2000],
            "audio_sensor_created": False,
            "requested_sample_rate_hz": 16000,
            "effective_sample_rate_hz": None,
            "requested_channel_layout": "binaural",
            "effective_channel_layout": None,
            "requested_channel_count": 2,
            "effective_channel_count": None,
            "materials_requested": False,
            "materials_effective": None,
            "renderer_construction_status": "FAIL",
            "rir_rendered": False,
            "asr_run": False,
            "wer_run": False,
        }
    marker = None
    for line in result.stdout.splitlines():
        if line.startswith("A3V2_SMOKE_RESULT "):
            marker = line[len("A3V2_SMOKE_RESULT "):]
    parsed: Dict[str, Any]
    if marker:
        try:
            parsed = dict(json.loads(marker))
        except json.JSONDecodeError:
            parsed = {"status": "FAIL", "scene_created": False, "pathfinder_loaded": False}
    else:
        parsed = {"status": "FAIL", "scene_created": False, "pathfinder_loaded": False}
    parsed.setdefault("audio_sensor_created", False)
    parsed.setdefault("requested_sample_rate_hz", 16000)
    parsed.setdefault("effective_sample_rate_hz", None)
    parsed.setdefault("requested_channel_layout", "binaural")
    parsed.setdefault("effective_channel_layout", None)
    parsed.setdefault("requested_channel_count", 2)
    parsed.setdefault("effective_channel_count", None)
    parsed.setdefault("materials_requested", False)
    parsed.setdefault("materials_effective", None)
    parsed.setdefault("renderer_construction_status", "FAIL")
    parsed.setdefault("rir_rendered", False)
    parsed.setdefault("asr_run", False)
    parsed.setdefault("wer_run", False)
    # Habitat-Sim diagnostics contain wall-clock timestamps.  They are useful
    # while debugging a failed child, but must not enter the frozen inventory
    # hash.  Keep only stable process/effective-state fields in the artifact.
    parsed.update({"process_returncode": result.returncode})
    if result.returncode != 0:
        parsed["status"] = "FAIL"
        parsed["scene_created"] = False
    return parsed


def scan_replica_inventory(repo_root: str, run_smoke: bool = True) -> Dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / REPLICA_ROOT
    if not root.is_dir():
        raise A3V2FreezeError("Replica root is missing: {}".format(root))
    rows: List[Dict[str, Any]] = []
    for scene_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name):
        scene_id = "replica." + scene_dir.name
        paths = _scene_paths(repo, scene_dir)
        file_identities = {key: _file_identity(path, repo) for key, path in paths.items()}
        completeness = all(bool(item["exists"]) for item in file_identities.values())
        prior_exclusion = scene_id == "replica.office_0"
        if prior_exclusion:
            smoke = {
                "status": "NOT_RUN_EXCLUDED_PRIOR_ASR_DOMAIN",
                "scene_created": False,
                "pathfinder_loaded": False,
                "process_returncode": None,
                "audio_sensor_created": False,
                "requested_sample_rate_hz": 16000,
                "effective_sample_rate_hz": None,
                "requested_channel_layout": "binaural",
                "effective_channel_layout": None,
                "requested_channel_count": 2,
                "effective_channel_count": None,
                "materials_requested": False,
                "materials_effective": None,
                "renderer_construction_status": "NOT_RUN",
                "rir_rendered": False,
                "asr_run": False,
                "wer_run": False,
            }
        elif completeness and run_smoke:
            smoke = _run_scene_smoke(repo, paths)
        elif completeness:
            smoke = {
                "status": "NOT_RUN_DISCOVER_ONLY",
                "scene_created": False,
                "pathfinder_loaded": False,
                "process_returncode": None,
                "audio_sensor_created": False,
                "requested_sample_rate_hz": 16000,
                "effective_sample_rate_hz": None,
                "requested_channel_layout": "binaural",
                "effective_channel_layout": None,
                "requested_channel_count": 2,
                "effective_channel_count": None,
                "materials_requested": False,
                "materials_effective": None,
                "renderer_construction_status": "NOT_RUN",
                "rir_rendered": False,
                "asr_run": False,
                "wer_run": False,
            }
        else:
            smoke = {
                "status": "NOT_RUN_INCOMPLETE_ASSET_SET",
                "scene_created": False,
                "pathfinder_loaded": False,
                "process_returncode": None,
                "audio_sensor_created": False,
                "requested_sample_rate_hz": 16000,
                "effective_sample_rate_hz": None,
                "requested_channel_layout": "binaural",
                "effective_channel_layout": None,
                "requested_channel_count": 2,
                "effective_channel_count": None,
                "materials_requested": False,
                "materials_effective": None,
                "renderer_construction_status": "NOT_RUN",
                "rir_rendered": False,
                "asr_run": False,
                "wer_run": False,
            }
        renderer_effective = (
            smoke.get("scene_created") is True
            and smoke.get("pathfinder_loaded") is True
            and smoke.get("audio_sensor_created") is True
            and smoke.get("renderer_construction_status") == "PASS"
            and smoke.get("effective_sample_rate_hz") == 16000
            and smoke.get("effective_channel_layout") == "binaural"
            and smoke.get("effective_channel_count") == 2
            and smoke.get("materials_effective") is False
        )
        eligible = bool(completeness and smoke["status"] == "PASS" and renderer_effective and not prior_exclusion)
        exclusion_reason = ""
        if prior_exclusion:
            exclusion_reason = "PRIOR_ASR_DOMAIN_SCENE_OFFICE_0"
        elif not completeness:
            exclusion_reason = "REQUIRED_REPLICA_ASSET_MISSING"
        elif smoke["status"] != "PASS":
            exclusion_reason = "RUNTIME_SCENE_LOAD_SMOKE_FAILED"
        elif not renderer_effective:
            exclusion_reason = "NATIVE16_BINAURAL_MATERIALS_OFF_AUDIO_SENSOR_EFFECTIVE_STATE_UNVERIFIED"
        rows.append({
            "scene_id": scene_id,
            "assets": file_identities,
            "file_completeness": "PASS" if completeness else "FAIL",
            "runtime_scene_load_smoke": smoke,
            "technical_eligibility": eligible,
            "exclusion_reason": exclusion_reason,
        })
    if not rows:
        raise A3V2FreezeError("Replica inventory is empty")
    return {
        "schema_version": INVENTORY_SCHEMA,
        "discovery_root": REPLICA_ROOT,
        "selection_policy": {
            "technical_eligibility_inputs": [
                "file_completeness", "scene_load", "pathfinder", "audio_sensor_construction",
                "native16", "binaural", "materials_off",
            ],
            "forbidden_inputs": ["RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "Oracle"],
            "prior_asr_scene_exclusions": ["replica.office_0"],
            "selection_rule": "lexical_scene_id_after_technical_eligibility",
        },
        "scenes": rows,
        "asr_run_before_inventory": False,
        "wer_run_before_inventory": False,
    }


def _load_pathfinder(scene_asset: Path, navmesh: Path):
    # Keep the required import ordering local to the SS2-only operation.
    import quaternion  # noqa: F401
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_asset.resolve())
    backend.enable_physics = False
    backend.load_semantic_mesh = False
    agent = habitat_sim.agent.AgentConfiguration()
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    if not simulator.pathfinder.is_loaded and not simulator.pathfinder.load_nav_mesh(str(navmesh.resolve())):
        simulator.close()
        raise A3V2FreezeError("selected scene navmesh did not load: {}".format(navmesh))
    return simulator


def _grid_points(pathfinder: Any, step: float = 0.5) -> List[Tuple[float, float, float]]:
    minimum, maximum = pathfinder.get_bounds()
    minimum = [float(v) for v in minimum]
    maximum = [float(v) for v in maximum]
    y = minimum[1]
    points: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    count_x = int(math.ceil((maximum[0] - minimum[0]) / step)) + 1
    count_z = int(math.ceil((maximum[2] - minimum[2]) / step)) + 1
    for ix in range(count_x):
        x = minimum[0] + ix * step
        for iz in range(count_z):
            z = minimum[2] + iz * step
            snapped = pathfinder.snap_point([x, y, z])
            if snapped is None or not pathfinder.is_navigable(snapped):
                continue
            key = tuple(int(round(float(value) * 1000.0)) for value in snapped)
            points[key] = tuple(float(value) for value in snapped)
    return [points[key] for key in sorted(points)]


def _route(pathfinder: Any, start: Sequence[float], end: Sequence[float]) -> Optional[Dict[str, Any]]:
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = __import__("numpy").asarray(start, dtype="float32")
    path.requested_end = __import__("numpy").asarray(end, dtype="float32")
    found = bool(pathfinder.find_path(path))
    distance = float(path.geodesic_distance)
    euclidean = math.sqrt(sum((float(end[i]) - float(start[i])) ** 2 for i in range(3)))
    if not found or not math.isfinite(distance) or not math.isfinite(euclidean):
        return None
    return {
        "found": True,
        "geodesic_distance_m": distance,
        "euclidean_distance_m": euclidean,
        "route_delta_m": distance - euclidean,
        "route_point_count": len(path.points),
    }


def _coordinate_provenance(repo: Path, paths: Mapping[str, Path], scene_asset_sha256: str) -> Dict[str, Any]:
    """Bind the Replica asset frame and prove the stage transform is identity."""

    stage = json.loads(paths["stage_config"].read_text(encoding="utf-8"))
    if not isinstance(stage, Mapping):
        raise A3V2FreezeError("stage config is not a mapping: {}".format(paths["stage_config"]))
    # The Replica stage config contains asset references but no transform
    # fields.  An unexpected transform-bearing field is a hard stop rather
    # than an implicit identity assumption.
    transform_fields = {"transform", "translation", "rotation", "scale", "position", "orientation"}
    if transform_fields.intersection(stage):
        raise A3V2FreezeError("stage config has an unhandled transform: {}".format(paths["stage_config"]))
    runtime_asset = _relative(paths["scene_asset"], repo)
    return {
        "mesh_to_habitat_matrix": [list(row) for row in REPLICA_PLY_TO_HABITAT],
        "habitat_to_mesh_matrix": [list(row) for row in HABITAT_TO_REPLICA_PLY],
        "stage_transform": "identity",
        "runtime_scene_asset": runtime_asset,
        "runtime_scene_asset_sha256": scene_asset_sha256,
        "transform_source": "Replica mesh_semantic.ply asset frame (x,depth,height)->Habitat (x,height,-depth); stage config has no transform fields",
    }


def _world_to_mesh(point: Sequence[float]) -> List[float]:
    import numpy as np

    matrix = np.asarray(HABITAT_TO_REPLICA_PLY, dtype=np.float64)
    result = matrix.dot(np.asarray(point, dtype=np.float64))
    return [float(value) for value in result]


def _mesh_world_aabb(mesh: StaticPlyMesh) -> Tuple[List[float], List[float]]:
    import itertools
    import numpy as np

    matrix = np.asarray(REPLICA_PLY_TO_HABITAT, dtype=np.float64)
    corners = np.asarray(
        list(itertools.product(*zip(mesh.aabb_min.tolist(), mesh.aabb_max.tolist()))), dtype=np.float64
    )
    world = corners.dot(matrix.T)
    return [float(value) for value in np.min(world, axis=0)], [float(value) for value in np.max(world, axis=0)]


def _assert_world_points_consistent(mesh: StaticPlyMesh, points: Iterable[Sequence[float]], scene_id: str) -> None:
    minimum, maximum = _mesh_world_aabb(mesh)
    for point in points:
        values = [float(value) for value in point]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise A3V2FreezeError("non-finite frozen coordinate in {}".format(scene_id))
        if any(value < minimum[index] - 1.0e-4 or value > maximum[index] + 1.0e-4 for index, value in enumerate(values)):
            raise A3V2FreezeError(
                "frozen coordinate is outside transformed mesh AABB in {}: point={} aabb=({}, {})".format(
                    scene_id, values, minimum, maximum
                )
            )


def _static_visibility(mesh: StaticPlyMesh, origin_world: Sequence[float], target_world: Sequence[float]) -> Dict[str, Any]:
    evidence = mesh.intersect_segment(_world_to_mesh(origin_world), _world_to_mesh(target_world), STATIC_MESH_EPSILON_M)
    provenance = mesh.provenance()
    evidence["origin"] = [float(value) for value in origin_world]
    evidence["target"] = [float(value) for value in target_world]
    evidence["mesh_origin"] = _world_to_mesh(origin_world)
    evidence["mesh_target"] = _world_to_mesh(target_world)
    evidence["runtime_provenance"] = {
        "geometry_library": provenance["geometry_library"],
        "geometry_library_version": provenance["geometry_library_version"],
        "backend": STATIC_MESH_LOS_METHOD,
        "backend_version": "1",
        "python": provenance["python"],
        "coordinate_convention": "replica_ply_xyz_to_habitat_xyz_x_z_neg_y",
    }
    return evidence


def _floor_sanity(repo: Path, scene_id: str, paths: Mapping[str, Path], scene_asset_sha256: str, simulator: Any, mesh: StaticPlyMesh) -> List[Dict[str, Any]]:
    points = _grid_points(simulator.pathfinder)
    if len(points) < STATIC_MESH_SANITY_CASE_COUNT:
        raise A3V2FreezeError("not enough deterministic navigable points for floor sanity: {}".format(scene_id))
    checks = []
    for index, base in enumerate(points[:STATIC_MESH_SANITY_CASE_COUNT]):
        origin = [float(base[i] + SENSOR_OFFSET[i]) for i in range(3)]
        target = [origin[0], origin[1] - 3.0, origin[2]]
        evidence = _static_visibility(mesh, origin, target)
        checks.append({
            "scene_id": scene_id,
            "check_id": "{}_floor_{}".format(scene_id.replace(".", "_"), index + 1),
            "listener_base_position_world": [float(value) for value in base],
            "origin": origin,
            "target": target,
            "mesh_sha256": scene_asset_sha256,
            "passed": evidence["intersection_count"] >= 1 and evidence["first_intersection_distance_m"] is not None,
            "evidence": evidence,
        })
    return checks


def _category_spec(category: str) -> Tuple[float, float, float, float, float]:
    if category == "near_front_like":
        return 1.5, 1.0, 2.0, 0.0, 15.0
    if category == "moderate_front_like":
        return 3.0, 2.25, 3.75, 0.0, 15.0
    if category == "far_front_like":
        return 5.0, 4.0, 6.5, 0.0, 15.0
    if category == "far_off_axis":
        return 5.0, 4.0, 6.5, 60.0, 25.0
    raise A3V2FreezeError("unknown geometry category: {}".format(category))


def _find_case(pathfinder: Any, mesh: StaticPlyMesh, category: str) -> Dict[str, Any]:
    target_distance, minimum_distance, maximum_distance, target_angle, angle_tolerance = _category_spec(category)
    points = _grid_points(pathfinder)
    yaw = 0.0
    candidates: List[Dict[str, Any]] = []
    for listener_base in points:
        requested_source = source_position_world(listener_base, yaw, target_angle, target_distance)
        source_base = pathfinder.snap_point(requested_source)
        if source_base is None or not pathfinder.is_navigable(source_base):
            continue
        route = _route(pathfinder, listener_base, source_base)
        if route is None or route["route_delta_m"] > 0.30:
            continue
        distance = route["euclidean_distance_m"]
        angle = relative_azimuth_deg(listener_base, yaw, source_base)
        angle_error = abs(angle - target_angle)
        if angle_error > 180.0:
            angle_error = 360.0 - angle_error
        if not (minimum_distance <= distance <= maximum_distance and angle_error <= angle_tolerance):
            continue
        candidates.append({
            "listener_base": list(listener_base),
            "source_base": list(source_base),
            "distance": distance,
            "angle": angle,
            "route": route,
            "selection_score": abs(distance - target_distance) + angle_error / 90.0 + route["route_delta_m"],
        })
    if not candidates:
        raise A3V2FreezeError("no geometry-only {} case satisfies the frozen deterministic rule".format(category))
    ordered_candidates = sorted(candidates, key=lambda item: tuple(item["listener_base"]) + tuple(item["source_base"]))
    chosen = None
    visibility = None
    for candidate in ordered_candidates:
        listener_sensor = [candidate["listener_base"][i] + SENSOR_OFFSET[i] for i in range(3)]
        source_sensor = [candidate["source_base"][i] + SENSOR_OFFSET[i] for i in range(3)]
        candidate_visibility = _static_visibility(mesh, listener_sensor, source_sensor)
        if candidate_visibility["clear_line_of_sight"] is True:
            chosen = candidate
            visibility = candidate_visibility
            break
    if chosen is None or visibility is None:
        raise A3V2FreezeError(
            "no static-mesh VERIFIED_GEOMETRIC_LOS {} case satisfies the frozen deterministic rule".format(category)
        )
    listener_base = chosen["listener_base"]
    source_base = chosen["source_base"]
    listener_sensor = [listener_base[i] + SENSOR_OFFSET[i] for i in range(3)]
    source_sensor = [source_base[i] + SENSOR_OFFSET[i] for i in range(3)]
    return {
        "category": category,
        "listener_base_position_world": listener_base,
        "listener_sensor_position_world": listener_sensor,
        "listener_yaw_deg": 0.0,
        "source_base_position_world": source_base,
        "source_position_world": source_sensor,
        "distance_m": chosen["distance"],
        "relative_azimuth_deg": chosen["angle"],
        "los_status": "VERIFIED_GEOMETRIC_LOS_STATIC_MESH",
        "route_evidence": chosen["route"],
        "visibility_evidence": visibility,
        "selection_score": chosen["selection_score"],
        "fallback_used": False,
        "selection_rule": "lexical_first_legal_grid_candidate_within_predeclared_distance_angle_bands",
    }


def build_scene_manifest(repo_root: str, inventory: Mapping[str, Any], inventory_sha256: str) -> Dict[str, Any]:
    repo = Path(repo_root).resolve()
    eligible = sorted(
        [row["scene_id"] for row in inventory["scenes"] if row["technical_eligibility"]],
    )
    if len(eligible) < 2:
        raise A3V2FreezeError("fewer than two technical-eligible held-out Replica scenes: {}".format(eligible))
    selected = eligible[:2]
    by_id = {row["scene_id"]: row for row in inventory["scenes"]}
    scenes: List[Dict[str, Any]] = []
    categories = ("near_front_like", "moderate_front_like", "far_front_like", "far_off_axis")
    synthetic_details = synthetic_sanity_results()
    synthetic_passed = all(item["passed"] for key, item in synthetic_details.items() if isinstance(item, Mapping) and "passed" in item)
    if not synthetic_passed:
        raise A3V2FreezeError("STATIC_MESH_COORDINATE_OR_INTERSECTION_INVALID: synthetic sanity failed")
    floor_checks: List[Dict[str, Any]] = []
    mesh_provenance: List[Dict[str, Any]] = []
    determinism_checks: List[Dict[str, Any]] = []
    for scene_id in selected:
        row = by_id[scene_id]
        paths = _scene_paths(repo, repo / REPLICA_ROOT / scene_id.split(".", 1)[1])
        simulator = _load_pathfinder(paths["scene_asset"], paths["navmesh"])
        mesh = StaticPlyMesh(paths["scene_asset"], expected_sha256=row["assets"]["scene_asset"]["sha256"])
        try:
            provenance = _coordinate_provenance(repo, paths, row["assets"]["scene_asset"]["sha256"])
            mesh_info = mesh.provenance()
            mesh_info["scene_id"] = scene_id
            mesh_info["stage_transform"] = provenance["stage_transform"]
            mesh_info["runtime_scene_asset"] = provenance["runtime_scene_asset"]
            mesh_provenance.append(mesh_info)
            floor_checks.extend(_floor_sanity(repo, scene_id, paths, row["assets"]["scene_asset"]["sha256"], simulator, mesh))
            points = _grid_points(simulator.pathfinder)
            if not points:
                raise A3V2FreezeError("no deterministic navigable points for static mesh determinism: {}".format(scene_id))
            deterministic_origin = [float(points[0][i] + SENSOR_OFFSET[i]) for i in range(3)]
            deterministic_target = [deterministic_origin[0], deterministic_origin[1] - 3.0, deterministic_origin[2]]
            first = _static_visibility(mesh, deterministic_origin, deterministic_target)
            second = _static_visibility(mesh, deterministic_origin, deterministic_target)
            first_distance = first["first_intersection_distance_m"]
            second_distance = second["first_intersection_distance_m"]
            deterministic_passed = (
                first["intersection_count"] == second["intersection_count"]
                and ((first_distance is None and second_distance is None) or abs(first_distance - second_distance) <= 1.0e-9)
            )
            determinism_checks.append({
                "scene_id": scene_id,
                "origin": deterministic_origin,
                "target": deterministic_target,
                "first_intersection_count": first["intersection_count"],
                "second_intersection_count": second["intersection_count"],
                "first_intersection_distance_m": first_distance,
                "second_intersection_distance_m": second_distance,
                "tolerance_m": 1.0e-9,
                "passed": deterministic_passed,
            })
            cases = []
            for index, category in enumerate(categories, 1):
                case = _find_case(simulator.pathfinder, mesh, category)
                case["scene_id"] = scene_id
                case["case_id"] = "{}__R{}".format(scene_id.replace(".", "_"), index)
                case["scene_asset_sha256"] = row["assets"]["scene_asset"]["sha256"]
                case["navmesh_sha256"] = row["assets"]["navmesh"]["sha256"]
                case["semantic_info_sha256"] = row["assets"]["semantic_info"]["sha256"]
                case["stage_config_sha256"] = row["assets"]["stage_config"]["sha256"]
                case["coordinate_provenance"] = provenance
                case["visibility_evidence"]["runtime_provenance"]["stage_transform"] = provenance["stage_transform"]
                _assert_world_points_consistent(
                    mesh,
                    (case["listener_sensor_position_world"], case["source_position_world"]),
                    scene_id,
                )
                cases.append(case)
        finally:
            simulator.close()
        scenes.append({"scene_id": scene_id, "cases": cases})
    return {
        "schema_version": SCENE_MANIFEST_SCHEMA,
        "scene_inventory": {"path": INVENTORY_PATH, "sha256": inventory_sha256, "schema_version": INVENTORY_SCHEMA},
        "selected_scene_ids": selected,
        "selection_policy": {
            "selection_rule": "lexical_scene_id_after_technical_eligibility",
            "case_categories": list(categories),
            "geometry_inputs": ["navmesh_legality", "euclidean_distance", "relative_azimuth", "route_legality", "static_mesh_segment"],
            "forbidden_inputs": ["RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "Oracle"],
            "grid_step_m": 0.5,
            "listener_yaw_deg": 0.0,
            "sensor_offset_m": list(SENSOR_OFFSET),
            "visibility_method": STATIC_MESH_LOS_METHOD,
            "visibility_epsilon_m": STATIC_MESH_EPSILON_M,
            "coordinate_convention": "replica_ply_xyz_to_habitat_xyz_x_z_neg_y",
            "stage_transform_policy": "stage_node_identity_verified",
        },
        "visibility_sanity": {
            "synthetic": {"passed": synthetic_passed, "details": synthetic_details},
            "mesh_provenance": mesh_provenance,
            "floor_checks": floor_checks,
            "determinism": {"passed": all(item["passed"] for item in determinism_checks), "checks": determinism_checks},
        },
        "scenes": scenes,
        "asr_run_before_freeze": False,
        "wer_run_before_freeze": False,
    }


def _extract_utterance_ids(value: Any) -> List[str]:
    result: List[str] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("utterance_id"), str):
            result.append(value["utterance_id"])
        for child in value.values():
            result.extend(_extract_utterance_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_extract_utterance_ids(child))
    return result


def _qualification_speech_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "utterance_id": row["utterance_id"],
        "split": row["split"],
        "speaker_id": row["speaker_id"],
        "chapter_id": row["chapter_id"],
        "relative_source_path": row["relative_source_path"],
        "source_file_sha256": row["source_file_sha256"],
        "decoded_waveform_sha256": row["decoded_waveform_sha256"],
        "samples": row["samples"],
        "duration_sec": row["duration_sec"],
        "normalized_transcript": row["normalized_transcript"],
        "reference_word_count": row["reference_word_count"],
        "complete_utterance": True,
    }


def build_speech_manifest(repo_root: str, registry_path: str = "registries/active_asr_a3/librispeech.jsonl") -> Dict[str, Any]:
    repo = Path(repo_root).resolve()
    registry = repo / registry_path
    if not registry.is_file():
        raise A3V2FreezeError("speech registry is missing: {}".format(registry))
    rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["utterance_id"]: row for row in rows}
    history_paths = [
        "registries/active_asr_a3/clean_qualification_manifest.json",
        "registries/active_asr_a3/ss2_domain_qualification_manifest.json",
        "runs/active_asr_v1/a3_g6_failure_attribution_v1/g6_24utterance_replication.json",
        "runs/active_asr_v1/a3_g6_failure_attribution_v1/real_scene_domain_attribution_v1/replica_office0_materials_off_asr.json",
    ]
    excluded_ids = set()
    history = []
    for relative in history_paths:
        path = repo / relative
        if not path.is_file():
            raise A3V2FreezeError("required historical utterance evidence is missing: {}".format(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = sorted(set(_extract_utterance_ids(payload)))
        if not found:
            raise A3V2FreezeError("historical evidence contains no utterance IDs: {}".format(path))
        excluded_ids.update(found)
        history.append({"path": relative, "sha256": sha256_file(path), "utterance_count": len(found), "utterance_ids": found})
    missing_registry_ids = sorted(excluded_ids - set(by_id))
    if missing_registry_ids:
        raise A3V2FreezeError("historical utterance IDs missing from immutable registry: {}".format(missing_registry_ids))
    excluded_speakers = {by_id[uid]["speaker_id"] for uid in excluded_ids}
    selection_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("dev-clean", "dev-other"):
        candidates = []
        for row in rows:
            if row["split"] != split or not row.get("eligible") or not row.get("complete_utterance"):
                continue
            if row["utterance_id"] in excluded_ids or row["speaker_id"] in excluded_speakers:
                continue
            if not (4.0 <= float(row["duration_sec"]) <= 15.0 and int(row["reference_word_count"]) >= 10):
                continue
            key = sha256_bytes(("a3-v2-speech-selection:{}:{}:{}".format(SELECTION_SEED, split, row["utterance_id"])).encode("utf-8"))
            candidates.append((key, row))
        candidates.sort(key=lambda item: (item[0], item[1]["utterance_id"]))
        by_speaker: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = defaultdict(list)
        for key, row in candidates:
            by_speaker[row["speaker_id"]].append((key, row))
        speaker_order = sorted(by_speaker, key=lambda speaker: (by_speaker[speaker][0][0], speaker))
        chosen: List[Dict[str, Any]] = []
        round_index = 0
        while len(chosen) < 12:
            progressed = False
            for speaker in speaker_order:
                items = by_speaker[speaker]
                if round_index < len(items):
                    chosen.append(dict(items[round_index][1]))
                    progressed = True
                    if len(chosen) == 12:
                        break
            if not progressed:
                break
            round_index += 1
        if len(chosen) != 12:
            raise A3V2FreezeError("{} has only {} held-out speaker-disjoint candidates; need 12".format(split, len(chosen)))
        selection_rows[split] = chosen
    selected = selection_rows["dev-clean"] + selection_rows["dev-other"]
    selected_ids = [row["utterance_id"] for row in selected]
    selected_speakers = [row["speaker_id"] for row in selected]
    if len(set(selected_ids)) != 24 or set(selected_ids) & excluded_ids or set(selected_speakers) & excluded_speakers:
        raise A3V2FreezeError("held-out speech selection violates historical utterance/speaker exclusion")
    return {
        "schema_version": SPEECH_MANIFEST_SCHEMA,
        "speech_registry": {"path": registry_path, "sha256": sha256_file(registry), "schema_version": "active-asr-a3-speech-registry-v1"},
        "selection_policy": {
            "selection_time": "before_any_a3_v2_rir_or_wer",
            "method": "metadata_only_seeded_speaker_round_robin",
            "seed": SELECTION_SEED,
            "constraints": {"duration_sec": [4.0, 15.0], "minimum_reference_words": 10, "complete_utterance": True},
            "required_counts": {"dev-clean": 12, "dev-other": 12},
            "forbidden_inputs": ["audio_listening", "ASR", "WER", "decoder_score", "RIR", "energy", "Oracle"],
            "historical_utterance_exclusion": history,
            "historical_speaker_exclusion_count": len(excluded_speakers),
        },
        "speaker_audit": {
            "selected_speaker_ids": sorted(set(selected_speakers)),
            "historical_excluded_speaker_ids": sorted(excluded_speakers),
            "selected_historical_speaker_overlap": sorted(set(selected_speakers) & excluded_speakers),
            "speaker_disjoint_from_history": not (set(selected_speakers) & excluded_speakers),
        },
        "records": [_qualification_speech_record(row) for row in selected],
        "asr_run_before_freeze": False,
        "wer_run_before_freeze": False,
    }


def _artifact(path: str, repo: Path) -> Dict[str, str]:
    target = repo / path
    if not target.is_file():
        raise A3V2FreezeError("required historical artifact is missing: {}".format(target))
    return {"path": path, "sha256": sha256_file(target)}


def build_v2_contract(repo_root: str, inventory_sha: str, inventory_records: int, scene_manifest_sha: str, speech_manifest_sha: str) -> Dict[str, Any]:
    repo = Path(repo_root).resolve()
    v1_path = repo / "configs/active_audition/v1/asr_contract.yaml"
    v1 = load_asr_contract(str(v1_path), require_frozen=True)
    if asr_contract_sha256(v1) != A3_V1_SHA256:
        raise A3V2FreezeError("A3-v1 contract hash does not match the requested parent")
    model_lock = _artifact(v1["artifacts"]["model_lock"], repo)
    historical_summary = _artifact("runs/active_asr_v1/a3_qualification_07a7769/a3_qualification_summary.json", repo)
    historical_report = _artifact("runs/active_asr_v1/a3_qualification_07a7769/a3_qualification_report.md", repo)
    contract = {
        "contract": {"namespace": "active-asr", "version": A3_V2_VERSION, "gate": "A3", "state": "FROZEN"},
        "serialization": {"version": "canonical-json-v1", "hash_algorithm": "sha256"},
        "parents": {
            "a0_contract_sha256": "d731393cda3ddb29f0bdf58249f104da59f29d012b976eeb2de1f160e1df8107",
            "a2_v2_sha256": "f1185d2c5fe81091199d5525f224822a3227c700c60bba9ab168e6d9ddc38f83",
            "a2_v3_sha256": "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c",
            "a3_v1_contract_sha256": A3_V1_SHA256,
        },
        "production_acoustics": {
            "renderer": "SoundSpaces2_HabitatSim0.2.2_RLRAudioPropagation",
            "sample_rate_hz": 16000,
            "channel_layout": "binaural",
            "channel_order": ["L", "R"],
            "convolution": "full",
            "materials": "OFF",
            "post_convolution_normalization": "none",
            "per_rir_normalization": "none",
            "per_utterance_normalization": "none",
            "primary_frontend": "mean_lr",
            "sensitivity_frontends": ["fixed_L", "fixed_R"],
        },
        "materials_policy": {
            "production": "OFF",
            "materials_on": "DEFERRED_UNSUPPORTED_REALISM_EXTENSION",
            "audit_status": "MATERIALS_ON_PROVENANCE_RECOVERED_BUT_RUNTIME_UNSTABLE",
            "audit_evidence": {
                "provenance": _artifact("runs/active_asr_v1/historical_materials_recovery_v1/historical_materials_provenance.json", repo),
                "capability": _artifact("runs/active_asr_v1/historical_materials_recovery_v1/historical_materials_capability.json", repo),
                "report": _artifact("runs/active_asr_v1/historical_materials_recovery_v1/historical_materials_recovery_report.md", repo),
            },
        },
        "historical_v1": {
            "g6_status": "FAIL",
            "g6_wer_fraction": 0.548689,
            "g6_threshold_fraction": 0.50,
            "g6_manifest": {
                "path": v1["qualification"]["manifests"]["ss2_domain"]["path"],
                "sha256": v1["qualification"]["manifests"]["ss2_domain"]["sha256"],
                "schema_version": v1["qualification"]["manifests"]["ss2_domain"]["schema_version"],
                "records": v1["qualification"]["manifests"]["ss2_domain"]["records"],
            },
            "attribution_evidence": {
                "failure_attribution": _artifact("runs/active_asr_v1/a3_g6_failure_attribution_v1/g6_failure_attribution_summary.json", repo),
                "real_scene_attribution": _artifact("runs/active_asr_v1/a3_g6_failure_attribution_v1/real_scene_domain_attribution_v1/real_scene_domain_attribution_summary.json", repo),
            },
        },
        "office0": {
            "attribution_only": True,
            "forbidden_for_v2_acceptance": True,
            "manifest": _artifact("registries/active_asr_a3/replica_office0_domain_diagnostic_manifest_v1.json", repo),
            "summary": _artifact("runs/active_asr_v1/a3_g6_failure_attribution_v1/real_scene_domain_attribution_v1/real_scene_domain_attribution_summary.json", repo),
        },
        "realistic_domain": {
            "scene_inventory": {"path": INVENTORY_PATH, "sha256": inventory_sha, "schema_version": INVENTORY_SCHEMA, "records": inventory_records},
            "scene_manifest": {"path": SCENE_MANIFEST_PATH, "sha256": scene_manifest_sha, "schema_version": SCENE_MANIFEST_SCHEMA, "records": 8},
            "speech_manifest": {"path": SPEECH_MANIFEST_PATH, "sha256": speech_manifest_sha, "schema_version": SPEECH_MANIFEST_SCHEMA, "records": 24},
            "selected_scene_count": 2,
            "cases_per_scene": 4,
            "expected_records": 192,
            "selection_inputs_forbidden": ["RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "Oracle"],
            "technical_scene_eligibility": [
                "file_completeness", "scene_load", "pathfinder", "audio_sensor_construction",
                "native16", "binaural", "materials_off",
            ],
            "visibility_policy": {
                "required_status": "VERIFIED_GEOMETRIC_LOS_STATIC_MESH",
                "method": STATIC_MESH_LOS_METHOD,
                "navmesh_route_role": "navigation_legality_only",
                "coordinate_convention": "replica_ply_xyz_to_habitat_xyz_x_z_neg_y",
                "stage_transform_policy": "stage_node_identity_verified",
            },
        },
        "model": deepcopy(v1["model"]),
        "environment": deepcopy(v1["environment"]),
        "decoder": deepcopy(v1["decoder"]),
        "input": deepcopy(v1["input"]),
        "frontends": deepcopy(v1["frontends"]),
        "text": deepcopy(v1["text"]),
        "sources": deepcopy(v1["sources"]),
        "qualification": {
            "minimum_nonempty_fraction": 0.80,
            "wer_max_fraction": 0.50,
            "records_expected": 192,
            "new_wer_allowed_after_freeze": True,
        },
        "artifacts": {
            "model_lock": model_lock,
            "model_lock_sha256": model_lock["sha256"],
            "historical_v1_summary": historical_summary,
            "historical_v1_report": historical_report,
        },
    }
    validate_a3_v2_contract(contract, require_frozen=True, repo_root=str(repo))
    return contract


def freeze_a3_v2(repo_root: str, run_smoke: bool = True) -> Dict[str, Any]:
    repo = Path(repo_root).resolve()
    inventory = scan_replica_inventory(str(repo), run_smoke=run_smoke)
    validate_scene_inventory(inventory)
    inventory_path = repo / INVENTORY_PATH
    inventory_sha = write_canonical_json(inventory_path, inventory)
    scene_manifest = build_scene_manifest(str(repo), inventory, inventory_sha)
    validate_scene_manifest(scene_manifest)
    scene_manifest_path = repo / SCENE_MANIFEST_PATH
    scene_manifest_sha = write_canonical_json(scene_manifest_path, scene_manifest)
    speech_manifest = build_speech_manifest(str(repo))
    validate_speech_manifest(speech_manifest)
    speech_manifest_path = repo / SPEECH_MANIFEST_PATH
    speech_manifest_sha = write_canonical_json(speech_manifest_path, speech_manifest)
    contract = build_v2_contract(str(repo), inventory_sha, len(inventory["scenes"]), scene_manifest_sha, speech_manifest_sha)
    contract_path = repo / V2_CONTRACT_PATH
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8")
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    validate_a3_v2_contract(loaded, require_frozen=True, repo_root=str(repo))
    contract_sha = a3_v2_contract_sha256(loaded)
    return {
        "status": "A3_V2_FREEZE_READY_NO_WER",
        "contract": {"path": V2_CONTRACT_PATH, "sha256": contract_sha, "state": "FROZEN"},
        "inventory": {"path": INVENTORY_PATH, "sha256": inventory_sha, "records": len(inventory["scenes"])},
        "scene_manifest": {"path": SCENE_MANIFEST_PATH, "sha256": scene_manifest_sha, "records": 8, "selected_scene_ids": scene_manifest["selected_scene_ids"]},
        "speech_manifest": {"path": SPEECH_MANIFEST_PATH, "sha256": speech_manifest_sha, "records": 24, "utterance_ids": [r["utterance_id"] for r in speech_manifest["records"]]},
        "speaker_audit": speech_manifest["speaker_audit"],
        "asr_run": False,
        "wer_run": False,
    }


def validate_a3_v2_file(path: str, require_frozen: bool = False, repo_root: str = ".") -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_a3_v2_contract(value, require_frozen=require_frozen, repo_root=repo_root)
    return {"status": "PASS", "contract_sha256": a3_v2_contract_sha256(value), "state": value["contract"]["state"]}


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("_scene-smoke-child")
    child.add_argument("--scene-asset", required=True)
    child.add_argument("--navmesh", required=True)
    args = parser.parse_args()
    if args.command == "_scene-smoke-child":
        _smoke_child(args.scene_asset, args.navmesh)


if __name__ == "__main__":
    _main()
