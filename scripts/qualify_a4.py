#!/usr/bin/env python3
"""Run the read-only A4-5 engineering qualification.

This command consumes the frozen smoke manifest and the strict content
addressed cache.  It never writes cache entries, never ranks poses, and never
uses ASR results to make a scientific or engineering choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from active_audition.a4.budget import compute_motion_cost  # noqa: E402
from active_audition.a4.cache import A2_ACOUSTIC_CONTRACT_SHA256, CacheStore, HIT_VALID, MixtureCacheKey, NOISE_SOURCE_TIME_IDENTITY  # noqa: E402
from active_audition.a4.cache_resume import (  # noqa: E402
    CacheExpectedManifest,
    read_completion_marker,
    reconcile_cache_manifest,
)
from active_audition.a4.calibration import CalibrationContract  # noqa: E402
from active_audition.a4.contract import contract_sha256, load_contract  # noqa: E402
from active_audition.a4.identity import canonical_json, canonical_json_bytes, identity_sha256  # noqa: E402
from active_audition.a4.mixer import (  # noqa: E402
    GlobalGainSpec,
    MixtureContract,
    WaveformComponent,
    build_mixture,
)
from active_audition.a4.noise_segments import NoiseSegmentPlan  # noqa: E402
from active_audition.a4.qualification import (  # noqa: E402
    A4QualificationArtifact,
    PENDING_RESOURCE_PROFILE,
    RESOURCE_PROFILE_ALGORITHM_IDENTITY,
    RESOURCE_PROFILE_SCHEMA_VERSION,
)
from active_audition.a4.records import (  # noqa: E402
    BlockRecord,
    CalibrationArtifact,
    EpisodeRecord,
    GeometryRecord,
    PoseRecord,
)
from active_audition.a4.smoke_manifest import EngineeringSmokeManifest  # noqa: E402
from active_audition.a4.timeline import (  # noqa: E402
    CONVOLUTION_IMPLEMENTATION_IDENTITY,
    TIMELINE_ALGORITHM_IDENTITY,
)
from active_audition.evaluation.asr_metrics import aggregate_error_counts, error_counts  # noqa: E402
from scripts.run_a4_production import (  # noqa: E402
    CACHE_ROOT,
    MANIFEST_PATH,
    _all_rir_keys,
    _array_sha,
    _build_timeline,
    _load_manifest,
    _load_verified_source,
    _rir_key,
    _run_root,
)


EXPECTED_CONTRACT_SHA = "1460899468de01050946107ba59617cd385eed9d8e2adecbf8d7e8cde288ac0b"
EXPECTED_A3_V2_SHA = "70864c814a55db5d184ef8a6835b65cb564c4c90fc86112f8705b7ecf6df1ffe"
EXPECTED_A0_SHA = "d731393cda3ddb29f0bdf58249f104da59f29d012b976eeb2de1f160e1df8107"
EXPECTED_A2_SHA = "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c"
EXPECTED_PARENTS = (
    "noise/free-sound/noise-free-sound-0002",
    "noise/free-sound/noise-free-sound-0020",
)
FRONTENDS = ("mean_lr", "fixed_L", "fixed_R")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return _sha_file(path)


def _write_diagnostic_json(path: Path, value: Any) -> str:
    """Write non-semantic evidence, which may intentionally contain WER/hypotheses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _sha_file(path)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def _load_cache_manifests(manifest: EngineeringSmokeManifest, run_root: Path):
    final = CacheExpectedManifest.from_payload(_json(run_root / "final_cache_expected_manifest.json"))
    rir = CacheExpectedManifest.from_payload(_json(run_root / "rir_expected_manifest.json"))
    mixture = CacheExpectedManifest.from_payload(_json(run_root / "mixture_expected_manifest.json"))
    if final.infrastructure_contract_sha256 != manifest.infrastructure_contract_sha256:
        raise RuntimeError("final cache manifest is bound to the wrong Infrastructure Contract")
    if rir.expected_counts != {"rir": 192, "mixture": 0, "asr": 0}:
        raise RuntimeError("RIR expected manifest count mismatch")
    if mixture.expected_counts != {"rir": 0, "mixture": 384, "asr": 0}:
        raise RuntimeError("mixture expected manifest count mismatch")
    if final.expected_counts != {"rir": 192, "mixture": 384, "asr": 1152}:
        raise RuntimeError("final expected manifest count mismatch")
    if tuple(key.cache_key for key in rir.expected_rir_keys) != tuple(key.cache_key for key in final.expected_rir_keys):
        raise RuntimeError("final RIR key set differs from RIR manifest")
    if tuple(key.cache_key for key in mixture.expected_mixture_keys) != tuple(key.cache_key for key in final.expected_mixture_keys):
        raise RuntimeError("final mixture key set differs from mixture manifest")
    return rir, mixture, final


def _strict_reconcile(store: CacheStore, manifest: CacheExpectedManifest, require_marker: bool = False):
    record = reconcile_cache_manifest(manifest, store)
    if not record.complete or any(record.rebuild_counts.values()):
        raise RuntimeError("strict reconciliation failed: {}".format(record.to_payload()))
    marker = read_completion_marker(store, manifest)
    if not require_marker:
        return record, marker
    if marker is None:
        raise RuntimeError("completion marker is absent")
    if (
        marker.expected_manifest_id != manifest.manifest_id
        or marker.infrastructure_contract_sha256 != manifest.infrastructure_contract_sha256
        or marker.reconciliation_record_id != record.record_id
        or marker.reconciliation_record_sha256 != record.record_sha256
    ):
        raise RuntimeError("completion marker does not match fresh reconciliation")
    return record, marker


def _registry_map() -> dict[str, dict[str, Any]]:
    rows = {}
    with (REPO / "registries/active_asr_a3/librispeech.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["utterance_id"]] = row
    return rows


def _validate_structure(manifest: EngineeringSmokeManifest) -> dict[str, Any]:
    if not manifest.engineering_only or not manifest.o2_exclusion.get("excluded"):
        raise RuntimeError("engineering-only/O2 exclusion boundary is invalid")
    if len(manifest.blocks) != 2:
        raise RuntimeError("manifest does not contain exactly two blocks")
    if tuple(manifest.expected_frontends) != FRONTENDS:
        raise RuntimeError("frontend set is not frozen")
    episode_ids = []
    pose_ids = []
    sampler_evidence = []
    budget_evidence = []
    source_evidence = []
    registry = _registry_map()
    for block_index, block in enumerate(manifest.blocks, start=1):
        geometry = GeometryRecord.from_payload(block["geometry_record"])
        block_record = BlockRecord.from_payload(block["block_record"])
        episodes = tuple(EpisodeRecord.from_payload(item) for item in block["episodes"])
        poses = tuple(PoseRecord.from_payload(item) for item in block["poses"])
        if len(episodes) != 4 or len(poses) != 48:
            raise RuntimeError("block {} structure is not 4 episodes x 48 poses".format(block_index))
        if tuple((item.role, item.fixed_dry_noise_segment_identity["role_index"]) for item in episodes) != (
            ("selection", 0), ("selection", 1), ("evaluation", 0), ("evaluation", 1)
        ):
            raise RuntimeError("block {} episode role/order mismatch".format(block_index))
        if geometry.geometry_id != block_record.geometry_id:
            raise RuntimeError("block {} geometry binding mismatch".format(block_index))
        episode_ids.extend(item.episode_id for item in episodes)
        pose_ids.extend(item.pose_id for item in poses)
        sampler = block["sampler_output"]
        if len(sampler["probe_attempts"]) != 64 or len(sampler["selected_positions"]) != 6 or len(sampler["yaw_plans"]) != 48:
            raise RuntimeError("block {} sampler count mismatch".format(block_index))
        if sampler["opportunity_constrained"] is not False:
            raise RuntimeError("block {} is opportunity constrained".format(block_index))
        sampler_evidence.append({
            "block_index": block_index,
            "raw_probe_count": len(sampler["probe_attempts"]),
            "selected_position_count": len(sampler["selected_positions"]),
            "yaw_expanded_count": len(sampler["yaw_plans"]),
            "opportunity_constrained": False,
            "source_free_mutation_regression": "covered_by_frozen_A4_sampler_tests",
        })
        parameters = block["candidate_contract"]
        motion_parameters = __import__("active_audition.a4.pose_sampler", fromlist=["CandidateContract"]).CandidateContract.from_payload(parameters).motion_parameters
        feasible = 0
        for pose in poses:
            cost = compute_motion_cost(
                geometry.actual_listener_base_xyz,
                geometry.initial_yaw_deg,
                pose.path_polyline,
                pose.yaw_deg,
                pose.geodesic_path_length_m,
                motion_parameters,
            )
            for field, record_field in (
                ("geodesic_path_length_m", "geodesic_path_length_m"),
                ("polyline_length_m", "path_polyline_length_m"),
                ("initial_to_path_turn_deg", "initial_to_path_turn_deg"),
                ("internal_path_turn_deg", "internal_path_turn_deg"),
                ("final_turn_deg", "final_turn_deg"),
                ("settling_sec", "settling_sec"),
                ("translation_sec", "translation_sec"),
                ("rotation_sec", "rotation_sec"),
                ("total_cost_sec", "total_cost_sec"),
                ("budget_sec", "budget_sec"),
            ):
                if not math.isclose(getattr(cost, field), getattr(pose, record_field), rel_tol=1e-9, abs_tol=1e-9):
                    raise RuntimeError("pose {} motion field {} is not reproducible".format(pose.pose_id, field))
            if cost.budget_feasible != pose.budget_feasible:
                raise RuntimeError("pose {} budget flag mismatch".format(pose.pose_id))
            feasible += int(pose.budget_feasible)
        if poses[0].total_cost_sec != 0.0:
            raise RuntimeError("block {} initial Stay cost is not exactly zero".format(block_index))
        budget_evidence.append({"block_index": block_index, "pose_count": 48, "budget_feasible_count": feasible, "budget_infeasible_count": 48 - feasible})
        parent = block["noise_parent"]
        parent_id = parent["parent_recording_id"]
        if parent_id not in EXPECTED_PARENTS:
            raise RuntimeError("unexpected frozen noise parent {}".format(parent_id))
        audit = block["noise_audit_record"]
        if any(audit[field] != "PASS" for field in ("speech_leakage", "strong_reverberation", "indoor_localized_source_compatibility")):
            raise RuntimeError("block {} noise audit is not fully PASS".format(block_index))
        _load_verified_source(REPO, parent, "noise")
        plan = NoiseSegmentPlan.from_payload(block["noise_plan"])
        if len(plan.segments) != 4 or plan.noise_parent_id != block_record.noise_parent_id:
            raise RuntimeError("block {} noise plan binding mismatch".format(block_index))
        ranges = [(segment.start_sample, segment.end_sample) for segment in plan.segments]
        if any(end <= start for start, end in ranges) or any(ranges[index][1] > ranges[index + 1][0] for index in range(3)):
            raise RuntimeError("block {} noise segments overlap".format(block_index))
        source_evidence.append({
            "block_index": block_index,
            "parent_recording_id": parent_id,
            "source_file_sha256": parent["source_file_sha256"],
            "decoded_waveform_sha256": parent["decoded_waveform_sha256"],
            "segment_count": 4,
            "nonoverlap": True,
            "loop_or_concat": False,
            "same_episode_samples_across_poses": True,
        })
        for speech in block["speech_sources"]:
            registry_row = registry.get(speech["utterance_id"])
            if registry_row is None or registry_row["source_file_sha256"] != speech["source_file_sha256"] or registry_row["decoded_waveform_sha256"] != speech["decoded_waveform_sha256"]:
                raise RuntimeError("speech registry provenance mismatch for {}".format(speech["utterance_id"]))
        if block_record.selection_utterance_ids != tuple(item.utterance_identity["utterance_id"] for item in episodes[:2]):
            raise RuntimeError("selection utterance binding mismatch")
        if block_record.evaluation_utterance_ids != tuple(item.utterance_identity["utterance_id"] for item in episodes[2:]):
            raise RuntimeError("evaluation utterance binding mismatch")
    return {
        "blocks": 2,
        "episodes": len(episode_ids),
        "poses": len(pose_ids),
        "sampler": sampler_evidence,
        "budget": budget_evidence,
        "sources": source_evidence,
        "unique_episode_ids": len(set(episode_ids)),
        "unique_pose_ids": len(set(pose_ids)),
        "frozen_parent_contracts": {"a0": EXPECTED_A0_SHA, "a2": EXPECTED_A2_SHA, "a3": EXPECTED_A3_V2_SHA},
    }


def _validate_rirs(store: CacheStore, manifest: EngineeringSmokeManifest, final: CacheExpectedManifest):
    by_key = {key.cache_key: key for key in final.expected_rir_keys}
    timing = []
    source_roles = defaultdict(int)
    for key in final.expected_rir_keys:
        result = store.read_rir(key)
        if result.status != HIT_VALID:
            raise RuntimeError("RIR cache read failed: {} {}".format(result.status, result.reason))
        if key.materials_policy != "OFF" or key.sample_rate_hz != 16000 or tuple(key.channel_order) != ("L", "R") or key.acoustic_contract_identity != A2_ACOUSTIC_CONTRACT_SHA256:
            raise RuntimeError("RIR key runtime policy mismatch")
        provenance = dict(result.metadata.provenance)
        if "render_seconds" not in provenance or float(provenance["render_seconds"]) < 0.0:
            raise RuntimeError("RIR metadata lacks render_seconds provenance")
        timing.append(float(provenance["render_seconds"]))
        source_roles[provenance.get("source_role", "unknown")] += 1
    if len(timing) != 192 or source_roles != {"target": 96, "noise": 96}:
        raise RuntimeError("RIR count/source-role qualification failed")
    values = sorted(timing)
    profile = {
        "count": len(values),
        "total_render_seconds": float(sum(values)),
        "mean_render_seconds": float(np.mean(values)),
        "median_render_seconds": float(median(values)),
        "p95_render_seconds": float(np.percentile(values, 95)),
        "min_render_seconds": float(values[0]),
        "max_render_seconds": float(values[-1]),
        "source_role_counts": dict(source_roles),
    }
    return profile


def _validate_calibrations(repo: Path, manifest: EngineeringSmokeManifest, run_root: Path, mixture: CacheExpectedManifest, store: CacheStore):
    artifacts = []
    for index, block in enumerate(manifest.blocks, start=1):
        artifact = CalibrationArtifact.from_payload(_json(run_root / "calibration_block_{}.json".format(index)))
        episodes = tuple(EpisodeRecord.from_payload(item) for item in block["episodes"])
        if artifact.block_id != block["block_record"]["block_id"] or artifact.status != "CALIBRATED":
            raise RuntimeError("calibration block binding/status mismatch")
        if tuple(artifact.selection_episode_ids) != tuple(item.episode_id for item in episodes[:2]):
            raise RuntimeError("calibration includes non-selection or wrong episodes")
        if abs(artifact.measured_snr_db - artifact.nominal_snr_db) > 0.1:
            raise RuntimeError("calibration measured SNR exceeds 0.1 dB")
        artifacts.append(artifact)
    ids_by_block = defaultdict(set)
    for key in mixture.expected_mixture_keys:
        result = store.read_mixture(key)
        if result.status != HIT_VALID:
            raise RuntimeError("mixture prerequisite failed while checking calibration")
        provenance = result.metadata.provenance
        ids_by_block[int(provenance["block_index"])].add(provenance["calibration_artifact_id"])
    if any(len(ids_by_block[index]) != 1 or ids_by_block[index] != {artifacts[index - 1].calibration_artifact_id} for index in (1, 2)):
        raise RuntimeError("mixtures do not reuse exactly one calibration artifact per block")
    return artifacts


def _reconstruct_mixtures(repo: Path, manifest: EngineeringSmokeManifest, run_root: Path, mixture_manifest: CacheExpectedManifest, store: CacheStore, artifacts):
    from active_audition.a4.active_mask import ActiveMaskContract, build_active_mask
    from active_audition.a4.pose_sampler import CandidateContract

    expected = {key.cache_key for key in mixture_manifest.expected_mixture_keys}
    seen = set()
    checked = 0
    max_residual = 0.0
    for block_index, block in enumerate(manifest.blocks, start=1):
        block_record = BlockRecord.from_payload(block["block_record"])
        episodes = tuple(EpisodeRecord.from_payload(item) for item in block["episodes"])
        speech = {row["utterance_id"]: _load_verified_source(repo, row, "speech") for row in block["speech_sources"]}
        noise_parent = _load_verified_source(repo, block["noise_parent"], "noise")
        masks = {uid: build_active_mask(wave, 16000, ActiveMaskContract()) for uid, wave in speech.items()}
        gain = GlobalGainSpec.from_payload(block["global_gain"])
        mixture_contract = MixtureContract()
        calibration = artifacts[block_index - 1]
        for episode_payload, episode in zip(block["episodes"], episodes):
            uid = episode.utterance_identity["utterance_id"]
            segment = episode.fixed_dry_noise_segment_identity
            dry_noise = noise_parent[int(segment["start_sample"]):int(segment["end_sample"])]
            for pose in block["poses"]:
                timeline = _build_timeline(block, episode_payload, pose, speech[uid], dry_noise, store)
                target_component = WaveformComponent.from_array("target", timeline.timeline_id, timeline.target_binaural)
                noise_component = WaveformComponent.from_array("noise", timeline.timeline_id, timeline.noise_binaural)
                key = MixtureCacheKey(
                    target_rir_cache_key=_rir_key(block, pose, "target").cache_key,
                    noise_rir_cache_key=_rir_key(block, pose, "noise").cache_key,
                    target_component_identity=target_component.component_identity,
                    noise_component_identity=noise_component.component_identity,
                    target_dry_waveform_sha256=_array_sha(speech[uid]),
                    noise_segment_payload_sha256=_array_sha(dry_noise),
                    noise_segment_identity=segment["segment_id"],
                    noise_source_time_identity=NOISE_SOURCE_TIME_IDENTITY,
                    calibration_artifact_identity=calibration.calibration_artifact_id,
                    global_gain_identity=gain.identity,
                    timeline_identity=timeline.timeline_id,
                    mixer_contract_identity=mixture_contract.identity,
                )
                seen.add(key.cache_key)
                result = store.read_mixture(key)
                if result.status != HIT_VALID:
                    raise RuntimeError("mixture key is not a valid cached entry: {}".format(key.cache_key))
                artifact = build_mixture(block_record, episode, pose["pose_id"], timeline, target_component, noise_component, calibration, gain, mixture_contract)
                actual = np.asarray(result.payload, dtype=np.float32)
                expected_waveform = np.asarray(artifact.mixture_binaural, dtype=np.float32)
                residual = float(np.max(np.abs(actual.astype(np.float64) - expected_waveform.astype(np.float64))))
                relative = residual / max(1.0, float(np.max(np.abs(actual.astype(np.float64)))))
                if relative > 1.0e-6 or _array_sha(actual) != _array_sha(expected_waveform):
                    raise RuntimeError("mixture reconstruction mismatch for {}".format(key.cache_key))
                max_residual = max(max_residual, relative)
                checked += 1
    if seen != expected or checked != 384:
        raise RuntimeError("mixture reconstruction count/key mismatch")
    return {"count": checked, "reconstruction_pass": checked, "max_relative_residual": max_residual}


def _write_asr_diagnostics(repo: Path, run_root: Path, manifest: EngineeringSmokeManifest, final: CacheExpectedManifest, store: CacheStore):
    registry = _registry_map()
    episodes = {}
    for block in manifest.blocks:
        for item in block["episodes"]:
            episode = EpisodeRecord.from_payload(item)
            episodes[episode.episode_id] = episode
    rows = []
    by_frontend = defaultdict(list)
    by_block = defaultdict(list)
    by_role = defaultdict(list)
    mixture_keys = {key.cache_key: key for key in final.expected_mixture_keys}
    for key in final.expected_asr_keys:
        result = store.read_asr(key)
        if result.status != HIT_VALID:
            raise RuntimeError("ASR cache read failed: {} {}".format(result.status, result.reason))
        provenance = dict(result.metadata.provenance)
        mixture_key = provenance.get("mixture_cache_key")
        mixture_result = store.read_mixture(mixture_keys[mixture_key])
        if mixture_result.status != HIT_VALID:
            raise RuntimeError("ASR references an invalid mixture")
        mix_prov = dict(mixture_result.metadata.provenance)
        episode = episodes[mix_prov["episode_id"]]
        utterance_id = episode.utterance_identity["utterance_id"]
        reference = registry[utterance_id]["normalized_transcript"]
        metrics = error_counts(reference, result.payload["hypothesis"])
        row = {
            "block_id": episode.block_id,
            "episode_id": episode.episode_id,
            "role": episode.role,
            "utterance_id": utterance_id,
            "pose_id": mix_prov["pose_id"],
            "frontend": result.payload["frontend"],
            "hypothesis": result.payload["hypothesis"],
            "reference": reference,
            "S": metrics["S"], "D": metrics["D"], "I": metrics["I"], "N": metrics["N"], "WER": metrics["WER"],
        }
        rows.append(row)
        by_frontend[row["frontend"]].append(metrics)
        by_block[row["block_id"]].append(metrics)
        by_role[row["role"]].append(metrics)
    if len(rows) != 1152 or set(by_frontend) != set(FRONTENDS):
        raise RuntimeError("ASR diagnostic count/frontend mismatch")
    rows.sort(key=lambda row: (row["block_id"], row["episode_id"], row["pose_id"], row["frontend"]))
    jsonl = run_root / "a4_asr_diagnostics.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "schema_version": "active-asr-a4-asr-engineering-diagnostics-v1",
        "record_count": len(rows),
        "normalization_identity": "active-asr-a3-reference-normalization-v1",
        "by_frontend": {key: aggregate_error_counts(value) for key, value in sorted(by_frontend.items())},
        "by_block": {key: aggregate_error_counts(value) for key, value in sorted(by_block.items())},
        "by_role": {key: aggregate_error_counts(value) for key, value in sorted(by_role.items())},
        "selection_policy": "diagnostic_only_no_pose_selection_or_ranking",
    }
    summary_path = run_root / "a4_asr_diagnostics_summary.json"
    summary_sha = _write_diagnostic_json(summary_path, summary)
    return {"record_count": len(rows), "jsonl": str(jsonl.relative_to(REPO)), "summary": str(summary_path.relative_to(REPO)), "summary_sha256": summary_sha, "by_frontend": summary["by_frontend"]}


REPLAY_PROFILE_FIELDS = (
    "schema_version", "profile_status", "semantic_cache_authority", "expected", "decoded",
    "batch_size", "frontends", "total_audio_seconds", "wall_seconds", "rtf",
    "hypothesis_mismatch_count_against_cached_diagnostic", "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes", "device", "device_name", "cuda_visible_devices", "torch",
    "torch_cuda", "python", "a3_v2_contract_sha256", "final_manifest_id", "final_manifest_sha256",
    "cache_write_count",
)


def _validate_replay_profile(replay: Mapping[str, Any], final_manifest: CacheExpectedManifest, asr_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(replay, Mapping) or set(replay) != set(REPLAY_PROFILE_FIELDS):
        raise RuntimeError("ASR resource replay schema is missing or has unknown fields")
    if replay["schema_version"] != "active-asr-a4-asr-resource-profile-replay-v1":
        raise RuntimeError("ASR resource replay schema version is invalid")
    if replay["profile_status"] != "DIAGNOSTIC_RESOURCE_REPLAY" or replay["semantic_cache_authority"] != "NOT_SEMANTIC_CACHE_AUTHORITY":
        raise RuntimeError("ASR resource replay authority markers are invalid")
    for field, expected in (("expected", 1152), ("decoded", 1152), ("batch_size", 4)):
        if isinstance(replay[field], bool) or not isinstance(replay[field], int) or replay[field] != expected:
            raise RuntimeError("ASR resource replay count/batch contract is invalid")
    if isinstance(replay["cache_write_count"], bool) or not isinstance(replay["cache_write_count"], int) or replay["cache_write_count"] != 0:
        raise RuntimeError("ASR resource replay wrote semantic cache entries")
    if isinstance(replay["hypothesis_mismatch_count_against_cached_diagnostic"], bool) or not isinstance(replay["hypothesis_mismatch_count_against_cached_diagnostic"], int) or replay["hypothesis_mismatch_count_against_cached_diagnostic"] != 0:
        raise RuntimeError("ASR resource replay hypothesis mismatch is non-zero")
    if tuple(replay["frontends"]) != FRONTENDS:
        raise RuntimeError("ASR resource replay frontend set is invalid")
    if replay["final_manifest_id"] != final_manifest.manifest_id or replay["final_manifest_sha256"] != final_manifest.manifest_sha256:
        raise RuntimeError("ASR resource replay final manifest binding is invalid")
    if replay["a3_v2_contract_sha256"] != EXPECTED_A3_V2_SHA:
        raise RuntimeError("ASR resource replay A3-v2 contract binding is invalid")
    if replay["device"] != "cuda:0" or "RTX 4090" not in str(replay["device_name"]):
        raise RuntimeError("ASR resource replay device is not the frozen RTX 4090 cuda:0")
    if isinstance(replay["peak_memory_allocated_bytes"], bool) or not isinstance(replay["peak_memory_allocated_bytes"], int) or replay["peak_memory_allocated_bytes"] <= 0:
        raise RuntimeError("ASR resource replay allocated peak memory is invalid")
    if isinstance(replay["peak_memory_reserved_bytes"], bool) or not isinstance(replay["peak_memory_reserved_bytes"], int) or replay["peak_memory_reserved_bytes"] < replay["peak_memory_allocated_bytes"]:
        raise RuntimeError("ASR resource replay reserved peak memory is invalid")
    if not math.isclose(float(replay["total_audio_seconds"]), float(asr_summary["total_audio_seconds"]), rel_tol=1.0e-9, abs_tol=1.0e-6):
        raise RuntimeError("ASR resource replay audio duration differs from production decode")
    for field in ("wall_seconds", "rtf", "total_audio_seconds"):
        if not isinstance(replay[field], (int, float)) or not math.isfinite(float(replay[field])) or float(replay[field]) < 0.0:
            raise RuntimeError("ASR resource replay {} is invalid".format(field))
    return replay


def _resource_profile_semantic_payload(rir_profile: Mapping[str, Any], asr_summary: Mapping[str, Any], replay: Mapping[str, Any] | None, layer_bytes: Mapping[str, int]) -> dict[str, Any]:
    production_decode = {
        key: asr_summary[key]
        for key in ("expected", "new_decodes", "total_audio_seconds", "decode_wall_seconds", "rtf")
        if key in asr_summary
    }
    resource_replay = dict(replay) if replay is not None else {"status": "PENDING_GPU_REPLAY"}
    return {
        "schema_version": RESOURCE_PROFILE_SCHEMA_VERSION,
        "algorithm_identity": RESOURCE_PROFILE_ALGORITHM_IDENTITY,
        "rir": {
            "count": rir_profile["count"],
            "total_render_seconds": rir_profile["total_render_seconds"],
            "mean_render_seconds": rir_profile["mean_render_seconds"],
            "median_render_seconds": rir_profile["median_render_seconds"],
            "p95_render_seconds": rir_profile["p95_render_seconds"],
            "min_render_seconds": rir_profile["min_render_seconds"],
            "max_render_seconds": rir_profile["max_render_seconds"],
        },
        "production_asr": production_decode,
        "gpu_replay": resource_replay,
        "effective_cache_footprint": {
            "effective_rir_cache_bytes": layer_bytes["rir"],
            "effective_mixture_v3_cache_bytes": layer_bytes["mixture"],
            "effective_asr_cache_bytes": layer_bytes["asr"],
            "effective_cache_total_bytes": sum(layer_bytes.values()),
        },
    }


def _resource_profile_semantic_sha256(payload: Mapping[str, Any]) -> str:
    return identity_sha256(payload)


def _resource_profile(repo: Path, run_root: Path, rir_profile: Mapping[str, Any], asr_summary: Mapping[str, Any], final: CacheExpectedManifest, store: CacheStore):
    layer_bytes = {}
    effective_keys = {
        "rir": {key.cache_key for key in final.expected_rir_keys},
        "mixture": {key.cache_key for key in final.expected_mixture_keys},
        "asr": {key.cache_key for key in final.expected_asr_keys},
    }
    for layer in ("rir", "mixture", "asr"):
        total = 0
        for key in effective_keys[layer]:
            key_obj = next(item for item in final.entries() if item[0] == layer and item[1].cache_key == key)[1]
            directory = store.entry_dir(key_obj)
            total += sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        layer_bytes[layer] = total
    legacy_mixture = 0
    mixture_root = store.root / "mixture"
    if mixture_root.is_dir():
        for directory in mixture_root.iterdir():
            if directory.is_dir() and directory.name not in effective_keys["mixture"]:
                legacy_mixture += sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    run_bytes = sum(path.stat().st_size for path in run_root.rglob("*") if path.is_file())
    replay_path = run_root / "asr_resource_profile_replay.json"
    replay = None
    if replay_path.is_file():
        replay = _validate_replay_profile(_json(replay_path), final, asr_summary)
    production_decode = {key: asr_summary[key] for key in ("expected", "cache_hits", "new_decodes", "total_audio_seconds", "decode_wall_seconds", "rtf") if key in asr_summary}
    resource_replay = dict(replay) if replay is not None else {
        "status": "PENDING_GPU_REPLAY",
        "path": str(replay_path.relative_to(repo)),
    }
    semantic_payload = _resource_profile_semantic_payload(rir_profile, asr_summary, replay, layer_bytes)
    semantic_sha = _resource_profile_semantic_sha256(semantic_payload)
    profile_path = run_root / "a4_resource_profile.json"
    profile = {
        "schema_version": RESOURCE_PROFILE_SCHEMA_VERSION,
        "semantic_payload": semantic_payload,
        "resource_profile_semantic_sha256": semantic_sha,
        "reporting_provenance": {
            "algorithm_identity": RESOURCE_PROFILE_ALGORITHM_IDENTITY,
            "profile_path": str(profile_path.relative_to(repo)),
            "run_evidence_bytes_snapshot": run_bytes,
            "legacy_superseded_mixture_cache_bytes": legacy_mixture,
        },
        "disk_bytes": {
            "effective_rir_cache": layer_bytes["rir"],
            "effective_mixture_cache_v3": layer_bytes["mixture"],
            "effective_asr_cache": layer_bytes["asr"],
            "run_evidence": run_bytes,
            "effective_total": sum(layer_bytes.values()) + run_bytes,
            "legacy_superseded_mixture_cache": legacy_mixture,
        },
        "asr_peak_memory": "RECORDED" if replay is not None else "PENDING_GPU_REPLAY",
    }
    sha = _write_json(profile_path, profile)
    profile["path"] = str(profile_path.relative_to(repo))
    profile["sha256"] = sha
    profile["resource_replay_valid"] = replay is not None
    profile["semantic_payload"] = semantic_payload
    profile["resource_profile_semantic_sha256"] = semantic_sha
    profile["run_evidence_bytes_snapshot"] = run_bytes
    return profile


def qualify(repo: Path = REPO) -> A4QualificationArtifact:
    manifest_payload = _json(repo / MANIFEST_PATH)
    manifest = EngineeringSmokeManifest.from_payload(manifest_payload)
    contract = load_contract(str(repo / "configs/active_audition/v1/a4_infrastructure_contract.yaml"), require_frozen=True)
    actual_contract_sha = contract_sha256(contract)
    if actual_contract_sha != EXPECTED_CONTRACT_SHA or actual_contract_sha != manifest.infrastructure_contract_sha256:
        raise RuntimeError("frozen Infrastructure Contract SHA mismatch")
    frozen_parents = contract.get("frozen_parents", {})
    if frozen_parents != {"a0_contract_sha256": EXPECTED_A0_SHA, "a2_contract_sha256": EXPECTED_A2_SHA, "a3_contract_sha256": EXPECTED_A3_V2_SHA}:
        raise RuntimeError("A0/A2/A3 frozen parent contract identities changed")
    run_root = repo / _run_root(manifest)
    store = CacheStore(str(repo / CACHE_ROOT))
    rir_manifest, mixture_manifest, final_manifest = _load_cache_manifests(manifest, run_root)
    rir_record, _ = _strict_reconcile(store, rir_manifest)
    mixture_record, _ = _strict_reconcile(store, mixture_manifest)
    final_record, marker = _strict_reconcile(store, final_manifest, require_marker=True)
    structure = _validate_structure(manifest)
    rir_profile = _validate_rirs(store, manifest, final_manifest)
    calibrations = _validate_calibrations(repo, manifest, run_root, mixture_manifest, store)
    mixture_profile = _reconstruct_mixtures(repo, manifest, run_root, mixture_manifest, store, calibrations)
    asr_summary = _json(run_root / "asr_stage_summary.json")
    if asr_summary.get("expected") != 1152 or asr_summary.get("new_decodes") != 1152 or asr_summary.get("valid") != 1152 or asr_summary.get("completion_marker_status") != "VALID":
        raise RuntimeError("ASR stage summary does not match the completed manual GPU run")
    diagnostics = _write_asr_diagnostics(repo, run_root, manifest, final_manifest, store)
    resource = _resource_profile(repo, run_root, rir_profile, asr_summary, final_manifest, store)
    profile_ready = bool(resource["resource_replay_valid"])
    calibration_evidence = {
        "blocks": [
            {"block_index": index, "calibration_artifact_id": artifact.calibration_artifact_id, "ps": artifact.ps, "pn": artifact.pn, "alpha": artifact.alpha, "nominal_snr_db": artifact.nominal_snr_db, "measured_snr_db": artifact.measured_snr_db, "error_db": abs(artifact.measured_snr_db - artifact.nominal_snr_db)}
            for index, artifact in enumerate(calibrations, start=1)
        ]
    }
    gates = {
        "G1": {"status": "PASS", "evidence": {"blocks": 2, "episodes": 8, "poses": 96, "engineering_only": True, "o2_excluded": True}},
        "G2": {"status": "PASS", "evidence": structure["sampler"]},
        "G3": {"status": "PASS", "evidence": structure["budget"]},
        "G4": {"status": "PASS", "evidence": structure["sources"]},
        "G5": {"status": "PASS", "evidence": {"rir_valid": 192, "rir_profile": rir_profile}},
        "G6": {"status": "PASS", "evidence": {"calibration": calibration_evidence, "timeline_algorithm_identity": TIMELINE_ALGORITHM_IDENTITY, "convolution_implementation_identity": CONVOLUTION_IMPLEMENTATION_IDENTITY}},
        "G7": {"status": "PASS", "evidence": mixture_profile},
        "G8": {"status": "PASS", "evidence": {"expected_entries": 1728, "reuse": {"rir": 192, "mixture": 384, "asr": 1152}, "completion_marker": marker.marker_id}},
        "G9": {"status": "PASS" if profile_ready else PENDING_RESOURCE_PROFILE, "evidence": {"blocks": 2, "episodes": 8, "mixtures": 384, "asr_records": 1152, "frontends": list(FRONTENDS), "resource_profile": "PASS" if profile_ready else PENDING_RESOURCE_PROFILE, "no_result_dependent_selection": True}},
    }
    artifact = A4QualificationArtifact(
        infrastructure_contract_sha256=manifest.infrastructure_contract_sha256,
        smoke_manifest_sha256=manifest.manifest_sha256,
        code_head=_git_head(),
        gate_records=gates,
        expected_counts={"rir": 192, "mixture": 384, "asr": 1152, "blocks": 2, "episodes": 8, "poses": 96},
        valid_counts={"rir": 192, "mixture": 384, "asr": 1152, "blocks": 2, "episodes": 8, "poses": 96},
        calibration_evidence=calibration_evidence,
        cache_reconciliation_identity={"manifest_id": final_manifest.manifest_id, "record_id": final_record.record_id, "record_sha256": final_record.record_sha256, "expected_counts": dict(final_record.expected_counts)},
        completion_marker_identity={"marker_id": marker.marker_id, "reconciliation_record_id": marker.reconciliation_record_id, "reconciliation_record_sha256": marker.reconciliation_record_sha256},
        resource_profile_identity={
            "schema_version": RESOURCE_PROFILE_SCHEMA_VERSION,
            "semantic_sha256": resource["resource_profile_semantic_sha256"],
        },
        resource_profile_semantic_payload=resource["semantic_payload"],
        resource_profile_references={
            "path": resource["path"],
            "sha256": resource["sha256"],
            "run_evidence_bytes_snapshot": resource["run_evidence_bytes_snapshot"],
        },
        asr_diagnostic_references={"jsonl": diagnostics["jsonl"], "summary": diagnostics["summary"], "summary_sha256": diagnostics["summary_sha256"], "record_count": diagnostics["record_count"]},
        qualification_state="SERVER_RUN_COMPLETE_PENDING_REVIEW" if profile_ready else "RESOURCE_PROFILE_PENDING_GPU_REPLAY",
    )
    qualification_path = run_root / "a4_qualification.json"
    _write_json(qualification_path, artifact.to_payload())
    print(json.dumps({"qualification": str(qualification_path), "qualification_id": artifact.artifact_id, "qualification_sha256": artifact.artifact_sha256, "run_root": str(run_root), "final_manifest": final_manifest.manifest_id}, sort_keys=True))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    args = parser.parse_args()
    qualify(Path(args.repo).resolve())


if __name__ == "__main__":
    main()
