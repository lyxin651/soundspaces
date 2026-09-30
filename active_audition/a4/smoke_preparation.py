"""Deterministic metadata-only preparation helpers for A4-4P.

The caller supplies a real or fake PathFinder.  This module performs no
Habitat import and has no path to RIR, waveform, ASR, or result-dependent
selection.  The ss-only launcher is kept outside this pure module.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.identity import identity_sha256, stable_id
from active_audition.a4.mixer import GlobalGainSpec
from active_audition.a4.noise_segments import (
    EpisodeNoiseRequest,
    NoiseParentMetadata,
    NoiseSegmentPlannerParameters,
    noise_parent_from_a3_provenance,
    plan_noise_segments,
)
from active_audition.a4.pose_sampler import (
    CandidateContract,
    SamplerOutput,
    annotate_geometry_legality,
    make_sampler_context,
    sample_source_free,
)
from active_audition.a4.records import BlockRecord, EpisodeRecord, GeometryRecord
from active_audition.a4.smoke_manifest import EngineeringSmokeManifest, ENGINEERING_SMOKE_MANIFEST_V1_SCHEMA_VERSION


def engineering_candidate_contract() -> CandidateContract:
    """Return the versioned A4 engineering fixture parameterization."""

    return CandidateContract(
        schema_version="active-asr-a4-candidate-contract-v1",
        sampler_algorithm_identity="active-asr-a4-local-polar-sampler-v1",
        coordinate_convention_identity="active-asr-a0-positive-left-forward-minus-z-v1",
        coverage_selection_identity="active-asr-a4-farthest-coverage-v1",
        tie_break_identity="active-asr-a4-probe-id-lexical-tie-break-v1",
        radii_m=(0.5, 1.0, 1.5, 2.0),
        azimuth_offsets_deg=tuple(index * 22.5 for index in range(16)),
        max_geodesic_radius_m=2.0,
        max_snap_error_m=0.05,
        duplicate_position_tolerance_m=0.05,
        max_noninitial_positions=5,
        yaw_offsets_deg=(0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        source_clearance_m=0.5,
        translation_speed_mps=0.25,
        rotation_speed_dps=90.0,
        settling_sec=0.5,
        budget_sec=10.0,
    )


def _speech_identity(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "utterance_id": row["utterance_id"],
        "speaker_id": row["speaker_id"],
        "relative_source_path": row["relative_source_path"],
        "source_file_sha256": row["source_file_sha256"],
        "decoded_waveform_sha256": row["decoded_waveform_sha256"],
        "samples": row["samples"],
        "duration_sec": row["duration_sec"],
        "reference_word_count": row["reference_word_count"],
        "complete_utterance": row["complete_utterance"],
        "engineering_only": True,
    }


def _episode_payload(block: BlockRecord, row: Mapping[str, Any], segment: Any) -> Mapping[str, Any]:
    utterance = _speech_identity(row)
    reference = {
        "reference_sha256": identity_sha256({"normalized_transcript": row["normalized_transcript"]}),
        "normalization_identity": "active-asr-a3-reference-normalization-v1",
    }
    payload = {
        "schema_version": "active-asr-a4-episode-v1",
        "block_id": block.block_id,
        "role": segment.role,
        "utterance_identity": utterance,
        "reference_identity": reference,
        "fixed_dry_noise_segment_identity": segment.to_payload(),
        "target_source_duration_sec": segment.target_duration_sec,
        "noise_source_time_start_sec": segment.source_time_start_sec,
        "noise_source_time_end_sec": segment.source_time_end_sec,
        "noise_segment_duration_sec": segment.source_time_end_sec - segment.source_time_start_sec,
    }
    return dict(payload, episode_id=stable_id("episode", payload))


def build_smoke_block_payload(
    case: Mapping[str, Any],
    target_source_case: Mapping[str, Any],
    noise_source_case: Mapping[str, Any],
    speech_rows: Sequence[Mapping[str, Any]],
    noise_row: Mapping[str, Any],
    source_free_output: SamplerOutput,
    candidate_contract: CandidateContract,
) -> Mapping[str, Any]:
    """Bind one geometry-only sampler output to four source metadata records."""

    if len(speech_rows) != 4 or len({row["speaker_id"] for row in speech_rows}) != 1:
        raise ValueError("one block requires four utterances from one speaker")
    scene_resource_identities = {
        "scene_asset_sha256": case["scene_asset_sha256"],
        "navmesh_sha256": case["navmesh_sha256"],
        "stage_config_sha256": case["stage_config_sha256"],
    }
    target_pose = {"position_xyz": list(target_source_case["source_position_world"]), "yaw_deg": 0.0}
    noise_pose = {"position_xyz": list(noise_source_case["source_position_world"]), "yaw_deg": 0.0}
    geometry_payload = {
        "schema_version": "active-asr-a4-geometry-v1",
        "scene_id": case["scene_id"],
        "scene_resource_identities": scene_resource_identities,
        "initial_listener_requested_base_xyz": list(case["listener_base_position_world"]),
        "actual_listener_base_xyz": list(case["listener_base_position_world"]),
        "sensor_xyz": list(case["listener_sensor_position_world"]),
        "sensor_transform_identity": "active-asr-a0-listener-sensor-offset-v1",
        "initial_yaw_deg": case["listener_yaw_deg"],
        "target_world_pose": target_pose,
        "noise_world_pose": noise_pose,
        "production_acoustic_policy_identity": "active-asr-a4-native16-binaural-materials-off-v1",
        "candidate_contract_identity": candidate_contract.candidate_contract_identity,
    }
    geometry = GeometryRecord(geometry_id=stable_id("geometry", geometry_payload), **geometry_payload)
    parent = noise_parent_from_a3_provenance(
        parent_recording_id=noise_row["parent_recording_id"],
        original_payload_sha256=noise_row["source_file_sha256"],
        decoded_payload_sha256=noise_row["decoded_waveform_sha256"],
        resampled_payload_sha256=noise_row["decoded_waveform_sha256"],
        sample_rate_hz=noise_row["sample_rate_hz"],
        sample_count=noise_row["samples"],
    )
    planner = NoiseSegmentPlannerParameters(16000, 2.0, 2.0, 0.25)
    requests = tuple(
        EpisodeNoiseRequest(role, index, row["utterance_id"], int(row["samples"]))
        for (role, index), row in zip(
            (("selection", 0), ("selection", 1), ("evaluation", 0), ("evaluation", 1)), speech_rows
        )
    )
    plan = plan_noise_segments(parent, requests, planner)
    gain = GlobalGainSpec(1.0)
    block_core = {
        "geometry_id": geometry.geometry_id,
        "speaker_id": speech_rows[0]["speaker_id"],
        "noise_parent_id": parent.noise_parent_id,
        "nominal_initial_snr_db": 0.0,
        "selection_utterance_ids": [speech_rows[0]["utterance_id"], speech_rows[1]["utterance_id"]],
        "evaluation_utterance_ids": [speech_rows[2]["utterance_id"], speech_rows[3]["utterance_id"]],
        "noise_segment_plan_identity": plan.plan_id,
        "global_gain_identity": gain.identity,
    }
    block = BlockRecord(
        schema_version="active-asr-a4-block-v1",
        block_id=stable_id("block", block_core),
        **block_core,
    )
    episodes = tuple(
        EpisodeRecord.from_payload(_episode_payload(block, row, segment))
        for row, segment in zip(speech_rows, plan.segments)
    )
    poses = annotate_geometry_legality(source_free_output, make_sampler_context(
        case["scene_id"], scene_resource_identities,
        case["listener_base_position_world"], case["listener_base_position_world"],
        "active-asr-a0-listener-sensor-offset-v1", case["listener_sensor_position_world"],
        case["listener_yaw_deg"], candidate_contract,
    ), candidate_contract, geometry)
    return {
        "block_record": block.to_payload(),
        "geometry_record": geometry.to_payload(),
        "candidate_contract": candidate_contract.to_payload(),
        "sampler_context": make_sampler_context(
            case["scene_id"], scene_resource_identities,
            case["listener_base_position_world"], case["listener_base_position_world"],
            "active-asr-a0-listener-sensor-offset-v1", case["listener_sensor_position_world"],
            case["listener_yaw_deg"], candidate_contract,
        ).to_payload(),
        "sampler_output": source_free_output.to_payload(),
        "poses": [pose.to_payload() for pose in poses],
        "noise_parent": dict(noise_row, engineering_only=True, a4_noise_parent_id=parent.noise_parent_id),
        "noise_plan": plan.to_payload(),
        "episodes": [episode.to_payload() for episode in episodes],
        "speech_sources": [_speech_identity(row) for row in speech_rows],
        "noise_audit": {
            "engineering_only": True,
            "source_registry_parent_recording_id": noise_row["parent_recording_id"],
            "source_file_sha256": noise_row["source_file_sha256"],
            "decoded_waveform_sha256": noise_row["decoded_waveform_sha256"],
            "sample_rate_hz": noise_row["sample_rate_hz"],
            "sample_count": noise_row["samples"],
            "technical_audit": noise_row["technical_audit"],
            "speech_leakage_audit": noise_row["speech_leakage_audit"],
            "strong_reverberation_audit": noise_row["strong_reverberation_audit"],
        },
        "global_gain": gain.to_payload(),
        "selection_evaluation_policy": {
            "selection": "calibration_and_later_pose_choice_eligible",
            "evaluation": "never_changes_pose_choice",
        },
    }


def build_engineering_smoke_manifest(
    infrastructure_contract_sha256: str,
    blocks: Sequence[Mapping[str, Any]],
    schema_version: str = ENGINEERING_SMOKE_MANIFEST_V1_SCHEMA_VERSION,
) -> EngineeringSmokeManifest:
    return EngineeringSmokeManifest(
        infrastructure_contract_sha256=infrastructure_contract_sha256,
        selection_policy={
            "algorithm": "active-asr-a4-engineering-geometry-metadata-only-selection-v1",
            "tie_break": "scene_id_then_case_id_lexical_after_technical_eligibility",
            "inputs": ["resource_availability", "navmesh_legality", "source_clearance", "relative_direction_pattern", "los_status"],
            "forbidden_inputs": list(("RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "Oracle", "movement_benefit")),
            "close_pattern_block": "target/noise source directions selected from adjacent lexical cases",
            "separated_pattern_block": "target/noise source directions selected from non-adjacent cases",
        },
        blocks=tuple(blocks),
        schema_version=schema_version,
        o2_exclusion={
            "excluded": True,
            "classification": "engineering_only_familiar_exploratory",
            "o2_scene_ids": "not_in_manifest",
            "reason": "pre-smoke infrastructure preparation; no held-out claim",
        },
        o1_reuse_classification_policy="reuse_requires_explicit_familiar_exploratory_label_and_never_held_out_claim",
    )


__all__ = ["build_engineering_smoke_manifest", "build_smoke_block_payload", "engineering_candidate_contract"]
