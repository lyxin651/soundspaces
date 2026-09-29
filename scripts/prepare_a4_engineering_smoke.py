#!/usr/bin/env python
"""Freeze A4-4P metadata from frozen registries and an ss sampler output.

This command is intentionally a metadata hand-off tool.  It verifies source
file identities and reconstructs pure A4 records, but never loads audio into a
waveform pipeline, renders RIR, builds a mixture, or invokes ASR.
"""

import argparse
import hashlib
import json
from pathlib import Path

from active_audition.a4.budget import MotionCost
from active_audition.a4.cache import AsrCacheKey, MixtureCacheKey, RirCacheKey, NOISE_SOURCE_TIME_IDENTITY
from active_audition.a4.cache_resume import CacheExpectedManifest
from active_audition.a4.identity import canonical_json, identity_sha256, stable_id
from active_audition.a4.pose_sampler import (
    CandidateContract, ProbeAttemptRecord, SampledPositionPlan, SampledYawPlan, SamplerContext,
    SamplerOutput,
)
from active_audition.a4.records import GeometryRecord, PoseRecord
from active_audition.a4.smoke_manifest import EngineeringSmokeManifest
from active_audition.a4.smoke_preparation import build_engineering_smoke_manifest, build_smoke_block_payload, engineering_candidate_contract
from active_audition.data.speech_registry import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCENE_MANIFEST = ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_scene_manifest.json"
SPEECH_MANIFEST = ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_speech_manifest.json"
SPEECH_REGISTRY = ROOT / "registries/active_asr_a3/librispeech.jsonl"
NOISE_REGISTRY = ROOT / "registries/active_asr_a3/musan_noise.jsonl"
SOURCES_ROOT = ROOT / "data/active_asr_a3/corpora"
CONTRACT_SHA_PATH = ROOT / "configs/active_audition/v1/a4_infrastructure_contract.sha256"
MANIFEST_PATH = ROOT / "configs/active_audition/v1/a4_engineering_smoke_manifest.json"
CACHE_MANIFEST_PATH = ROOT / "configs/active_audition/v1/a4_engineering_smoke_cache_expected_manifest.json"


def _sampler_output(payload):
    attempts = tuple(ProbeAttemptRecord(**dict(item)) for item in payload["probe_attempts"])
    positions = tuple(SampledPositionPlan(**dict(item)) for item in payload["selected_positions"])
    yaws = tuple(
        SampledYawPlan(
            motion_cost=MotionCost(**dict(item["motion_cost"])),
            **{key: value for key, value in item.items() if key != "motion_cost"},
        )
        for item in payload["yaw_plans"]
    )
    return SamplerOutput(
        schema_version=payload["schema_version"],
        sampler_context_id=payload["sampler_context_id"],
        candidate_contract_identity=payload["candidate_contract_identity"],
        probe_attempts=attempts,
        selected_positions=positions,
        yaw_plans=yaws,
        opportunity_constrained=payload["opportunity_constrained"],
        opportunity_reason=payload["opportunity_reason"],
    )


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _speech_rows():
    rows = _load_jsonl(SPEECH_REGISTRY)
    # The registry rows are authoritative for source hashes; the v2 manifest's
    # selected speaker audit is authoritative for the historical exclusion set.
    v2 = json.loads(SPEECH_MANIFEST.read_text(encoding="utf-8"))
    historical = set(v2["speaker_audit"]["historical_excluded_speaker_ids"])
    grouped = {}
    for row in rows:
        if row["speaker_id"] in historical or not row["eligible"] or not row["complete_utterance"]:
            continue
        if not (4.0 <= float(row["duration_sec"]) <= 15.0 and row["reference_word_count"] >= 10):
            continue
        grouped.setdefault(row["speaker_id"], []).append(row)
    selected = []
    for speaker in sorted(grouped):
        group = sorted(grouped[speaker], key=lambda row: row["utterance_id"])
        if len(group) >= 4:
            selected.append(group[:4])
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError("unable to select two non-historical four-utterance speakers")
    for group in selected:
        for row in group:
            path = SOURCES_ROOT / "LibriSpeech" / row["relative_source_path"]
            if not path.is_file() or sha256_file(path) != row["source_file_sha256"]:
                raise RuntimeError("speech source identity mismatch: {}".format(path))
    return selected


def _noise_rows():
    by_id = {row["parent_recording_id"]: row for row in _load_jsonl(NOISE_REGISTRY)}
    selected = []
    for parent_id in ("noise/free-sound/noise-free-sound-0048", "noise/free-sound/noise-free-sound-0270"):
        row = by_id[parent_id]
        path = SOURCES_ROOT / "musan" / row["relative_source_path"]
        if row["excluded"] or row["sample_rate_hz"] != 16000 or not path.is_file() or sha256_file(path) != row["source_file_sha256"]:
            raise RuntimeError("noise source is not technically eligible: {}".format(parent_id))
        selected.append(row)
    return selected


def _case_index():
    manifest = json.loads(SCENE_MANIFEST.read_text(encoding="utf-8"))
    return {scene["scene_id"]: {case["case_id"]: case for case in scene["cases"]} for scene in manifest["scenes"]}


def _planned_cache_manifest(smoke, contract_sha):
    """Create a deterministic no-payload cache plan for later A4 stages.

    Calibration/timeline identities are plan identities here: no rendered or
    calibrated payload exists at A4-4P.  A4-5 replaces these planned entries
    with the producer's verified artifact identities before writing payloads.
    """
    rir = []
    mixtures = []
    asr = []
    model = {"repo_id": "speechbrain/asr-transformer-transformerlm-librispeech", "revision": "A3-frozen"}
    lm = {"name": "A3-frozen-language-model", "sha256": "0" * 64}
    tok = {"name": "A3-frozen-tokenizer", "sha256": "1" * 64}
    dec = {"beam": "A3-frozen"}
    runtime = {"precision": "float32", "runtime": "A3-frozen"}
    for block in smoke.blocks:
        geometry = GeometryRecord.from_payload(block["geometry_record"])
        poses = [PoseRecord.from_payload(item) for item in block["poses"]]
        for pose in poses:
            receiver = {"position_xyz": list(pose.sensor_xyz), "yaw_deg": pose.yaw_deg}
            source_keys = {}
            for role, source in (("target", geometry.target_world_pose), ("noise", geometry.noise_world_pose)):
                key = RirCacheKey(
                    scene_resource_identities=geometry.scene_resource_identities,
                    acoustic_contract_identity="c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c",
                    renderer_algorithm_identity="SoundSpaces2_HabitatSim0.2.2_RLRAudioPropagation-v1",
                    materials_policy="OFF", source_world_transform=source,
                    receiver_sensor_transform=receiver, receiver_yaw_deg=pose.yaw_deg,
                    replicate_identity="a4-engineering-smoke-replicate-0",
                )
                source_keys[role] = key
                rir.append(key)
            plan = block["noise_plan"]
            for episode in block["episodes"]:
                segment = next(item for item in plan["segments"] if item["segment_id"] == episode["fixed_dry_noise_segment_identity"]["segment_id"])
                segment_sha = identity_sha256({"planned_noise_segment_payload": segment["segment_id"], "parent": plan["decoded_resampled_payload_sha256"]})
                calibration = stable_id("calibration", {"planned_block_id": block["block_record"]["block_id"], "algorithm": "a4-4p-no-payload-plan-v1"})
                timeline = stable_id("receiver-timeline", {"planned_pose_id": pose.pose_id, "episode_id": episode["episode_id"], "algorithm": "a4-4p-no-payload-plan-v1"})
                mixer = stable_id("mixture-contract", {"infrastructure_contract_sha256": contract_sha, "algorithm": "active-asr-a4-dual-source-linear-mixer-v1"})
                mixture = MixtureCacheKey(
                    target_rir_cache_key=source_keys["target"].cache_key,
                    noise_rir_cache_key=source_keys["noise"].cache_key,
                    target_dry_waveform_sha256=episode["utterance_identity"]["decoded_waveform_sha256"],
                    noise_segment_payload_sha256=segment_sha,
                    noise_segment_identity=segment["segment_id"],
                    noise_source_time_identity=NOISE_SOURCE_TIME_IDENTITY,
                    calibration_artifact_identity=calibration,
                    global_gain_identity=block["block_record"]["global_gain_identity"],
                    timeline_identity=timeline,
                    mixer_contract_identity=mixer,
                )
                mixtures.append(mixture)
                for frontend in ("mean_lr", "fixed_L", "fixed_R"):
                    asr.append(AsrCacheKey(
                        mono_payload_sha256=identity_sha256({"planned_mixture": mixture.cache_key, "frontend": frontend}),
                        frontend=frontend, model_identity=model, language_model_identity=lm,
                        tokenizer_identity=tok, decoder_identity=dec,
                        precision_runtime_identity=runtime,
                        asr_contract_identity="70864c814a55db5d184ef8a6835b65cb564c4c90fc86112f8705b7ecf6df1ffe",
                    ))
    manifest = CacheExpectedManifest(
        infrastructure_contract_sha256=contract_sha,
        expected_rir_keys=tuple(rir), expected_mixture_keys=tuple(mixtures), expected_asr_keys=tuple(asr),
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler-output", type=Path, required=True)
    parser.add_argument("--infrastructure-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--cache-output", type=Path, default=CACHE_MANIFEST_PATH)
    args = parser.parse_args()
    contract_sha = args.infrastructure_contract_sha256
    if len(contract_sha) != 64 or any(char not in "0123456789abcdef" for char in contract_sha):
        raise SystemExit("infrastructure contract SHA must be lowercase SHA256")
    sampler_rows = {row["scene_id"]: row for row in json.loads(args.sampler_output.read_text(encoding="utf-8"))}
    cases = _case_index()
    speakers = _speech_rows()
    noises = _noise_rows()
    candidate = engineering_candidate_contract()
    selections = (
        ("replica.apartment_0", "replica_apartment_0__R1", "replica_apartment_0__R1", "replica_apartment_0__R2"),
        ("replica.apartment_1", "replica_apartment_1__R1", "replica_apartment_1__R1", "replica_apartment_1__R4"),
    )
    blocks = []
    for index, (scene_id, listener_case, target_case, noise_case) in enumerate(selections):
        case = cases[scene_id][listener_case]
        sampler = sampler_rows[scene_id]
        context = SamplerContext(**dict(sampler["sampler_context"]))
        if context.candidate_contract_identity != candidate.candidate_contract_identity:
            raise RuntimeError("sampler candidate contract identity mismatch")
        output = _sampler_output(sampler["sampler_output"])
        blocks.append(build_smoke_block_payload(
            case=case, target_source_case=cases[scene_id][target_case], noise_source_case=cases[scene_id][noise_case],
            speech_rows=speakers[index], noise_row=noises[index], source_free_output=output, candidate_contract=candidate,
        ))
    smoke = build_engineering_smoke_manifest(contract_sha, blocks)
    cache_manifest = _planned_cache_manifest(smoke, contract_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(smoke.to_payload()) + "\n", encoding="utf-8")
    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_output.write_text(canonical_json(cache_manifest.to_payload()) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest_id": smoke.manifest_id,
        "manifest_sha256": smoke.manifest_sha256,
        "block_ids": [block["block_record"]["block_id"] for block in blocks],
        "geometry_ids": [block["geometry_record"]["geometry_id"] for block in blocks],
        "speakers": [block["block_record"]["speaker_id"] for block in blocks],
        "utterances": [block["block_record"]["selection_utterance_ids"] + block["block_record"]["evaluation_utterance_ids"] for block in blocks],
        "noise_parents": [block["block_record"]["noise_parent_id"] for block in blocks],
        "pose_counts": [len(block["poses"]) for block in blocks],
        "raw_probe_counts": [len(block["sampler_output"]["probe_attempts"]) for block in blocks],
        "cache_manifest_id": cache_manifest.manifest_id,
        "cache_counts": dict(cache_manifest.expected_counts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
