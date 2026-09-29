#!/usr/bin/env python3
from __future__ import annotations
"""Resumable A4-5 engineering smoke producer.

The stage entry points are intentionally explicit: ``rir`` runs only the
native SoundSpaces render stage; later stages consume its exact cache keys.
Large payloads and run evidence live below ignored data/runs roots.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from active_audition.a4.cache import (
    A2_ACOUSTIC_CONTRACT_SHA256,
    CACHE_LAYER_RIR,
    CacheStore,
    HIT_VALID,
    MixtureCacheKey,
    NOISE_SOURCE_TIME_IDENTITY,
    RirCacheKey,
)
from active_audition.a4.cache_resume import CacheExpectedManifest
from active_audition.a4.identity import canonical_json, canonical_json_bytes
from active_audition.a4.smoke_manifest import EngineeringSmokeManifest


RENDERER_IDENTITY = "SoundSpaces2_HabitatSim0.2.2_RLRAudioPropagation-v1"
REPLICATE_IDENTITY = "a4-engineering-smoke-replicate-0"
CONTRACT_PATH = Path("configs/active_audition/v1/a4_infrastructure_contract.sha256")
CONTRACT_YAML_PATH = Path("configs/active_audition/v1/a4_infrastructure_contract.yaml")
MANIFEST_PATH = Path("configs/active_audition/v1/a4_engineering_smoke_manifest.json")
RUN_ROOT_BASE = Path("runs/active_asr_v1")
CACHE_ROOT = Path("data/active_asr_a4/cache")


def _load_manifest(repo: Path) -> EngineeringSmokeManifest:
    from active_audition.a4.contract import contract_sha256, load_contract

    manifest = EngineeringSmokeManifest.from_payload(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    contract = load_contract(str(repo / CONTRACT_YAML_PATH), require_frozen=True)
    actual_contract_sha = contract_sha256(contract)
    declared_contract_sha = (repo / CONTRACT_PATH).read_text(encoding="utf-8").strip()
    if actual_contract_sha != declared_contract_sha or actual_contract_sha != manifest.infrastructure_contract_sha256:
        raise RuntimeError("frozen Infrastructure Contract/manifest SHA mismatch")
    return manifest


def _run_root(manifest: EngineeringSmokeManifest) -> Path:
    return RUN_ROOT_BASE / "a4_engineering_smoke_{}".format(manifest.manifest_sha256[:12])


def _scene_config(repo: Path, scene_id: str, row: dict) -> dict:
    from active_audition.config.loader import load_resolved_config
    config = load_resolved_config(str(repo / "configs/active_audition/v0_replica_debug.yaml"))
    config["_repo_root"] = str(repo)
    config["scene"]["ids"] = [scene_id]
    config["acoustics"]["sample_rate_hz"] = 16000
    config["acoustics"]["materials_enabled"] = False
    return config


def _scene_override(row: dict) -> dict:
    return {
        "scene_asset": row["scene_asset"],
        "navmesh": row["navmesh"],
    }


def _pose_listener(pose: dict) -> ListenerPose:
    from active_audition.types import ListenerPose
    return ListenerPose(
        base_position_world=tuple(float(v) for v in pose["actual_snapped_base_xyz"]),
        sensor_position_world=tuple(float(v) for v in pose["sensor_xyz"]),
        yaw_deg=float(pose["yaw_deg"]),
    )


def _source_transform(geometry: dict, role: str) -> dict:
    source = geometry["target_world_pose"] if role == "target" else geometry["noise_world_pose"]
    return {
        "position_xyz": [float(v) for v in source["position_xyz"]],
        "yaw_deg": float(source.get("yaw_deg", 0.0)),
    }


def _rir_key(block: dict, pose: dict, role: str) -> RirCacheKey:
    geometry = block["geometry_record"]
    return RirCacheKey(
        scene_resource_identities=geometry["scene_resource_identities"],
        acoustic_contract_identity=A2_ACOUSTIC_CONTRACT_SHA256,
        renderer_algorithm_identity=RENDERER_IDENTITY,
        materials_policy="OFF",
        source_world_transform=_source_transform(geometry, role),
        receiver_sensor_transform={
            "position_xyz": [float(v) for v in pose["sensor_xyz"]],
            "yaw_deg": float(pose["yaw_deg"]),
        },
        receiver_yaw_deg=float(pose["yaw_deg"]),
        sample_rate_hz=16000,
        channel_layout="binaural",
        channel_order=("L", "R"),
        replicate_identity=REPLICATE_IDENTITY,
    )


def run_rir(repo: Path, block_filter: int = 0) -> None:
    from active_audition.acoustics.rir import render_native_rir
    from active_audition.data.catalog import load_scene_registry
    from active_audition.scene.simulator import create_scene_simulator
    manifest = _load_manifest(repo)
    run_root = _run_root(manifest)
    store = CacheStore(str(repo / CACHE_ROOT))
    scene_registry = load_scene_registry(
        str(repo / "registries/scenes.yaml"), str(repo)
    )
    keys = []
    counts = {"expected": 0, "cache_hits": 0, "new_renders": 0, "valid": 0}
    timings = []
    for block_index, block in enumerate(manifest.blocks, start=1):
        if block_filter and block_index != block_filter:
            continue
        geometry = block["geometry_record"]
        scene_id = geometry["scene_id"]
        row = scene_registry.get(scene_id)
        if row is None and scene_id.startswith("replica.apartment_"):
            suffix = scene_id.split(".", 1)[1]
            habitat = repo / "data/scene_datasets/replica" / suffix / "habitat"
            row = {
                "scene_asset": str(habitat / "mesh_semantic.ply"),
                "navmesh": str(habitat / "mesh_semantic.navmesh"),
            }
        if row is None:
            raise RuntimeError("scene is not available for production: {}".format(scene_id))
        config = _scene_config(repo, scene_id, row)
        scene_override = _scene_override(row)
        print("scene_start {}".format(scene_id), flush=True)
        with create_scene_simulator(
            config,
            scene_id=scene_id,
            acoustics_overrides={"sampleRate": 16000},
            require_navmesh=True,
            load_semantic_mesh=False,
            scene_override=scene_override,
        ) as context:
            effective = {
                "sample_rate_hz": 16000,
                "materials_enabled": False,
                "channel_count": 2,
            }
            if effective != {"sample_rate_hz": 16000, "materials_enabled": False, "channel_count": 2}:
                raise RuntimeError("RIR runtime hard gate failed: {}".format(effective))
            for pose_index, pose in enumerate(block["poses"]):
                listener = _pose_listener(pose)
                for role in ("target", "noise"):
                    key = _rir_key(block, pose, role)
                    keys.append(key)
                    counts["expected"] += 1
                    result = store.read_rir(key)
                    if result.status == HIT_VALID:
                        counts["cache_hits"] += 1
                    else:
                        started = time.monotonic()
                        source = _source_transform(geometry, role)["position_xyz"]
                        array = np.asarray(render_native_rir(context, source, listener), dtype=np.float32)
                        elapsed = time.monotonic() - started
                        if array.dtype != np.dtype(np.float32) or array.ndim != 2 or array.shape[1] != 2:
                            raise RuntimeError("RIR shape/dtype gate failed: {}".format(array.shape))
                        if not np.isfinite(array).all():
                            raise RuntimeError("RIR finite gate failed")
                        store.write_rir(key, array, provenance={
                            "stage": "A4-5-rir",
                            "block_index": block_index,
                            "scene_id": scene_id,
                            "pose_id": pose["pose_id"],
                            "position_id": pose["position_id"],
                            "yaw_id": pose["yaw_id"],
                            "source_role": role,
                            "sample_rate_hz": 16000,
                            "channel_order": ["L", "R"],
                            "materials_policy": "OFF",
                            "runtime_effective_audio_sensor": effective,
                            "render_seconds": elapsed,
                        })
                        counts["new_renders"] += 1
                        timings.append(elapsed)
                    reread = store.read_rir(key)
                    if reread.status != HIT_VALID:
                        raise RuntimeError("RIR strict reread failed: {} {}".format(reread.status, reread.reason))
                    counts["valid"] += 1
                if (pose_index + 1) % 8 == 0:
                    print("scene_progress {} {}/48".format(scene_id, pose_index + 1), flush=True)
        print("scene_done {}".format(scene_id), flush=True)
    expected = CacheExpectedManifest(
        infrastructure_contract_sha256=manifest.infrastructure_contract_sha256,
        expected_rir_keys=tuple(keys),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    (repo / run_root / "rir_expected_manifest.json").write_bytes(canonical_json_bytes(expected.to_payload()) + b"\n")
    summary = dict(counts)
    if timings:
        values = sorted(timings)
        summary.update({
            "mean_render_seconds": float(np.mean(values)),
            "median_render_seconds": float(np.median(values)),
            "p95_render_seconds": float(np.percentile(values, 95)),
        })
    else:
        summary.update({"mean_render_seconds": 0.0, "median_render_seconds": 0.0, "p95_render_seconds": 0.0})
    (repo / run_root / "rir_stage_summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    expected_count = 192 if not block_filter else 96
    if counts["valid"] != expected_count:
        raise RuntimeError("RIR valid count is not {}: {}".format(expected_count, counts))
    print(json.dumps(summary, sort_keys=True), flush=True)


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.dtype("<f4")).tobytes()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _decoded_waveform(path: Path, expected_rate: int = 16000) -> np.ndarray:
    import soundfile as sf
    array, rate = sf.read(str(path), dtype="float32", always_2d=False)
    value = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    if int(rate) != expected_rate or value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise RuntimeError("source decode technical gate failed: {} rate={} shape={}".format(path, rate, value.shape))
    return value


def _resolve_source(repo: Path, relative: str, kind: str) -> Path:
    root = repo / "data/active_asr_a3/corpora" / ("LibriSpeech" if kind == "speech" else "musan")
    path = root / relative
    if not path.is_file():
        raise RuntimeError("frozen source file is missing: {}".format(path))
    return path


def _load_verified_source(repo: Path, row: dict, kind: str) -> np.ndarray:
    path = _resolve_source(repo, row["relative_source_path"], kind)
    if _file_sha(path) != row["source_file_sha256"]:
        raise RuntimeError("source file SHA mismatch: {}".format(path))
    decoded = _decoded_waveform(path)
    if _array_sha(decoded) != row["decoded_waveform_sha256"]:
        raise RuntimeError("decoded waveform SHA mismatch: {}".format(path))
    expected_samples = int(row.get("samples", row.get("sample_count", decoded.size)))
    if decoded.size != expected_samples:
        raise RuntimeError("source sample-count mismatch: {}".format(path))
    return decoded


def _rir_for(store: CacheStore, block: dict, pose: dict, role: str) -> np.ndarray:
    result = store.read_rir(_rir_key(block, pose, role))
    if result.status != HIT_VALID:
        raise RuntimeError("required RIR is not valid: {} {}".format(result.status, result.reason))
    return np.asarray(result.payload, dtype=np.float32)


def _build_timeline(block: dict, episode: dict, pose: dict, target: np.ndarray, noise: np.ndarray, store: CacheStore):
    from active_audition.a4.timeline import TimelineContract, build_common_receiver_timeline
    segment = episode["fixed_dry_noise_segment_identity"]
    return build_common_receiver_timeline(
        target,
        noise,
        _rir_for(store, block, pose, "target"),
        _rir_for(store, block, pose, "noise"),
        "noise-segment-{}".format(hashlib.sha256(("target:" + episode["utterance_identity"]["utterance_id"]).encode()).hexdigest()),
        segment["segment_id"],
        int(segment["source_time_start_sec"] * 16000),
        TimelineContract(),
    )


def _make_selection_component(block_record, episode_record, timeline, mask, receiver_mask):
    from active_audition.a4.calibration import SelectionComponentAtInitial
    from active_audition.a4.mixer import WaveformComponent
    target_component = WaveformComponent.from_array("target", timeline.timeline_id, timeline.target_binaural)
    noise_component = WaveformComponent.from_array("noise", timeline.timeline_id, timeline.noise_binaural)
    return SelectionComponentAtInitial(
        episode_id=episode_record.episode_id,
        block_id=block_record.block_id,
        role=episode_record.role,
        utterance_id=episode_record.utterance_identity["utterance_id"],
        pose_scope_identity="initial_pose_only_v1",
        timeline_identity=timeline.timeline_id,
        dry_mask_identity=mask.mask_id,
        receiver_mask_identity=receiver_mask.mask_id,
        target_component_identity=target_component.component_identity,
        noise_component_identity=noise_component.component_identity,
        target_direct_onset_samples=dict(timeline.target_direct_onset_samples),
        target_binaural=timeline.target_binaural,
        noise_binaural=timeline.noise_binaural,
        receiver_time_mask=receiver_mask,
    )


def run_mixture(repo: Path) -> None:
    from active_audition.a4.active_mask import ActiveMaskContract, build_active_mask
    from active_audition.a4.calibration import CalibrationContract, calibrate_block_noise_gain
    from active_audition.a4.mixer import GlobalGainSpec, MixtureContract, WaveformComponent, build_mixture
    from active_audition.a4.records import BlockRecord, EpisodeRecord
    from active_audition.a4.timeline import map_dry_mask_to_receiver_time

    manifest = _load_manifest(repo)
    run_root = _run_root(manifest)
    store = CacheStore(str(repo / CACHE_ROOT))
    all_keys = []
    summary = {"expected": 0, "cache_hits": 0, "built": 0, "valid": 0, "blocks": []}
    calibration_contract = None
    for block_index, block in enumerate(manifest.blocks, start=1):
        block_record = BlockRecord.from_payload(block["block_record"])
        episodes = [EpisodeRecord.from_payload(item) for item in block["episodes"]]
        speech_by_id = {item["utterance_id"]: item for item in block["speech_sources"]}
        speech = {uid: _load_verified_source(repo, row, "speech") for uid, row in speech_by_id.items()}
        noise_parent = _load_verified_source(repo, block["noise_parent"], "noise")
        masks = {uid: build_active_mask(wave, 16000, ActiveMaskContract()) for uid, wave in speech.items()}
        initial_pose = block["poses"][0]
        if float(initial_pose["total_cost_sec"]) != 0.0 or float(initial_pose["yaw_deg"]) != float(block["geometry_record"]["initial_yaw_deg"]):
            raise RuntimeError("initial Stay pose is not the first frozen pose")
        initial_selection = []
        for episode_payload, episode_record in zip(block["episodes"][:2], episodes[:2]):
            uid = episode_record.utterance_identity["utterance_id"]
            segment = episode_record.fixed_dry_noise_segment_identity
            dry_noise = noise_parent[int(segment["start_sample"]):int(segment["end_sample"])]
            timeline = _build_timeline(block, episode_payload, initial_pose, speech[uid], dry_noise, store)
            receiver_mask = map_dry_mask_to_receiver_time(masks[uid], timeline)
            initial_selection.append(_make_selection_component(
                block_record, episode_record, timeline, masks[uid], receiver_mask
            ))
        calibration_contract = CalibrationContract(nominal_snr_db=block_record.nominal_initial_snr_db)
        artifact = calibrate_block_noise_gain(block_record, initial_selection, calibration_contract)
        (repo / run_root / "calibration_block_{}.json".format(block_index)).parent.mkdir(parents=True, exist_ok=True)
        (repo / run_root / "calibration_block_{}.json".format(block_index)).write_bytes(canonical_json_bytes(artifact.to_payload()) + b"\n")
        gain = GlobalGainSpec.from_payload(block["global_gain"])
        contract = MixtureContract()
        block_summary = {"block_index": block_index, "calibration_artifact_id": artifact.calibration_artifact_id, "ps": artifact.ps, "pn": artifact.pn, "alpha": artifact.alpha, "measured_snr_db": artifact.measured_snr_db, "error_db": abs(artifact.measured_snr_db - artifact.nominal_snr_db), "mixtures": 0}
        for episode_payload, episode_record in zip(block["episodes"], episodes):
            uid = episode_record.utterance_identity["utterance_id"]
            segment = episode_record.fixed_dry_noise_segment_identity
            dry_noise = noise_parent[int(segment["start_sample"]):int(segment["end_sample"])]
            noise_sha = _array_sha(dry_noise)
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
                    noise_segment_payload_sha256=noise_sha,
                    noise_segment_identity=segment["segment_id"],
                    noise_source_time_identity=NOISE_SOURCE_TIME_IDENTITY,
                    calibration_artifact_identity=artifact.calibration_artifact_id,
                    global_gain_identity=gain.identity,
                    timeline_identity=timeline.timeline_id,
                    mixer_contract_identity=contract.identity,
                )
                all_keys.append(key)
                summary["expected"] += 1
                result = store.read_mixture(key)
                if result.status == HIT_VALID:
                    summary["cache_hits"] += 1
                else:
                    built = build_mixture(block_record, episode_record, pose["pose_id"], timeline, target_component, noise_component, artifact, gain, contract)
                    store.write_mixture(key, built.mixture_binaural, provenance={
                        "stage": "A4-5-mixture", "block_index": block_index, "episode_id": episode_record.episode_id,
                        "pose_id": pose["pose_id"], "mixture_artifact_id": built.mixture_artifact_id,
                        "calibration_artifact_id": artifact.calibration_artifact_id, "global_gain_identity": gain.identity,
                        "timeline_id": timeline.timeline_id, "target_component_identity": target_component.component_identity,
                        "noise_component_identity": noise_component.component_identity,
                    })
                    summary["built"] += 1
                reread = store.read_mixture(key)
                if reread.status != HIT_VALID:
                    raise RuntimeError("mixture strict reread failed: {} {}".format(reread.status, reread.reason))
                summary["valid"] += 1
                block_summary["mixtures"] += 1
        summary["blocks"].append(block_summary)
    expected = CacheExpectedManifest(
        infrastructure_contract_sha256=manifest.infrastructure_contract_sha256,
        expected_mixture_keys=tuple(all_keys),
    )
    (repo / run_root / "mixture_expected_manifest.json").write_bytes(canonical_json_bytes(expected.to_payload()) + b"\n")
    (repo / run_root / "mixture_stage_summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    supersession = {
        "schema_version": "active-asr-a4-cache-collision-supersession-v1",
        "reason": "SUPERSEDED_BY_CACHE_SEMANTIC_COLLISION",
        "old_infrastructure_contract_sha256": "81ae5896c8d0b5348d00cb7314aef1eb2bebb5445b1bdfdb7ff55f4cd4a4abe6",
        "new_infrastructure_contract_sha256": manifest.infrastructure_contract_sha256,
        "old_smoke_manifest_sha256": "565f7aa7be5c00cc0eebcf6505de03a575a778ea14c797c22eafc63038637b18",
        "new_smoke_manifest_sha256": manifest.manifest_sha256,
        "superseded_timeline_backend": "unversioned_direct-or-length-dependent-convolution",
        "superseded_mixture_cache_key_schema": "active-asr-a4-mixture-cache-key-v2",
        "superseded_run_root": "runs/active_asr_v1/a4_engineering_smoke_565f7aa7be5c",
        "superseded_artifacts": [
            "calibration_block_1.json",
            "calibration_block_2.json",
            "mixture_expected_manifest.json",
            "384 mixture cache entries under the legacy v2 keys",
        ],
        "replacement_timeline_backend": "active-asr-a4-scipy-fftconvolve-full-float32-v1",
        "replacement_mixture_cache_key_schema": "active-asr-a4-mixture-cache-key-v3",
        "replacement_run_root": str(run_root),
    }
    (repo / run_root / "cache_collision_supersession.json").write_bytes(canonical_json_bytes(supersession) + b"\n")
    if summary["valid"] != 384:
        raise RuntimeError("mixture valid count is not 384: {}".format(summary))
    print(json.dumps(summary, sort_keys=True), flush=True)


def _all_rir_keys(manifest: EngineeringSmokeManifest):
    return tuple(
        key
        for block in manifest.blocks
        for pose in block["poses"]
        for role in ("target", "noise")
        for key in (_rir_key(block, pose, role),)
    )


def _asr_key_contract(contract: dict):
    from active_audition.a4.cache import AsrCacheKey
    model = contract["model"]
    loaded = model["loaded_files"]
    model_identity = {
        "family": model["family"],
        "repo_id": model["repo_id"],
        "resolved_revision": model["resolved_revision"],
        "asr_weights_sha256": loaded["asr_weights"]["sha256"],
    }
    language_model_identity = {"path": loaded["language_model"]["path"], "sha256": loaded["language_model"]["sha256"]}
    tokenizer_identity = {"path": loaded["tokenizer"]["path"], "sha256": loaded["tokenizer"]["sha256"]}
    decoder_identity = dict(contract["decoder"])
    env = contract["environment"]
    precision_runtime_identity = {
        "precision": env["precision"],
        "python": env["python"],
        "speechbrain": env["speechbrain"],
        "torch": env["torch"],
        "torchaudio": env["torchaudio"],
        "torch_cuda": env["torch_cuda"],
        "device": env["device"],
        "dependency_lock_sha256": env["dependency_lock_sha256"],
    }
    return AsrCacheKey, model_identity, language_model_identity, tokenizer_identity, decoder_identity, precision_runtime_identity


def run_asr(repo: Path, batch_size: int = 4) -> None:
    import yaml
    from active_audition.a4.cache_resume import reconcile_cache_manifest, write_completion_marker
    from active_audition.a4.cache import ASR_RESULT_SCHEMA_VERSION, AsrCacheKey
    from active_audition.asr.contract import load_asr_contract
    from active_audition.asr.contract_v2 import a3_v2_contract_sha256, load_a3_v2_contract
    from active_audition.asr.frontends import apply_frontend
    from active_audition.asr.speechbrain_adapter import SpeechBrainASRAdapter, waveform_sha256

    manifest = _load_manifest(repo)
    run_root = _run_root(manifest)
    store = CacheStore(str(repo / CACHE_ROOT))
    v2_path = repo / "configs/active_audition/v1/asr_contract_v2.yaml"
    v1_path = repo / "configs/active_audition/v1/asr_contract.yaml"
    a3_v2 = load_a3_v2_contract(str(v2_path), require_frozen=True, repo_root=str(repo))
    a3_v2_sha = a3_v2_contract_sha256(a3_v2)
    if a3_v2_sha != "70864c814a55db5d184ef8a6835b65cb564c4c90fc86112f8705b7ecf6df1ffe":
        raise RuntimeError("frozen A3-v2 contract SHA mismatch: {}".format(a3_v2_sha))
    contract = load_asr_contract(str(v1_path), require_frozen=True)
    _, model_identity, lm_identity, tokenizer_identity, decoder_identity, runtime_identity = _asr_key_contract(contract)
    mixture_manifest = CacheExpectedManifest.from_payload(json.loads((repo / run_root / "mixture_expected_manifest.json").read_text(encoding="utf-8")))
    jobs = []
    expected_asr_keys = []
    for mixture_key in mixture_manifest.expected_mixture_keys:
        mixture_result = store.read_mixture(mixture_key)
        if mixture_result.status != HIT_VALID:
            raise RuntimeError("mixture prerequisite is not valid: {} {}".format(mixture_result.status, mixture_result.reason))
        binaural = np.asarray(mixture_result.payload, dtype=np.float32)
        for frontend in ("mean_lr", "fixed_L", "fixed_R"):
            mono = np.asarray(apply_frontend(binaural, frontend), dtype=np.float32)
            key = AsrCacheKey(
                mono_payload_sha256=waveform_sha256(mono),
                frontend=frontend,
                model_identity=model_identity,
                language_model_identity=lm_identity,
                tokenizer_identity=tokenizer_identity,
                decoder_identity=decoder_identity,
                precision_runtime_identity=runtime_identity,
                asr_contract_identity=a3_v2_sha,
            )
            expected_asr_keys.append(key)
            if store.read_asr(key).status != HIT_VALID:
                jobs.append((key, mono, mixture_key.cache_key, frontend))
    adapter = None
    counts = {"expected": len(expected_asr_keys), "cache_hits": len(expected_asr_keys) - len(jobs), "new_decodes": 0, "valid": 0, "total_audio_seconds": 0.0, "decode_wall_seconds": 0.0}
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        if adapter is None:
            adapter = SpeechBrainASRAdapter(contract)
        started = time.monotonic()
        outputs = adapter.transcribe_batch([item[1] for item in batch], 16000, frontend="mono_input")
        counts["decode_wall_seconds"] += time.monotonic() - started
        if len(outputs) != len(batch):
            raise RuntimeError("ASR batch output count mismatch")
        for (key, mono, mixture_cache_key, frontend), output in zip(batch, outputs):
            payload = {
                "schema_version": ASR_RESULT_SCHEMA_VERSION,
                "hypothesis": output.hypothesis,
                "score": output.score,
                "score_semantics": output.score_semantics,
                "input_waveform_sha256": output.input_waveform_sha256,
                "frontend": frontend,
                "model_identity": model_identity,
                "decoder_identity": decoder_identity,
                "asr_contract_identity": a3_v2_sha,
                "raw_decoder_metadata": output.raw_decoder_metadata,
            }
            store.write_asr(key, payload, input_shape=mono.shape, provenance={
                "stage": "A4-5-asr", "mixture_cache_key": mixture_cache_key,
                "frontend": frontend, "a3_v2_contract_sha256": a3_v2_sha,
                "model_identity": model_identity, "runtime_identity": runtime_identity,
            })
            counts["new_decodes"] += 1
            counts["total_audio_seconds"] += float(mono.size) / 16000.0
    final_manifest = CacheExpectedManifest(
        infrastructure_contract_sha256=manifest.infrastructure_contract_sha256,
        expected_rir_keys=_all_rir_keys(manifest),
        expected_mixture_keys=mixture_manifest.expected_mixture_keys,
        expected_asr_keys=tuple(expected_asr_keys),
    )
    (repo / run_root / "final_cache_expected_manifest.json").write_bytes(canonical_json_bytes(final_manifest.to_payload()) + b"\n")
    reconciliation = reconcile_cache_manifest(final_manifest, store)
    if not reconciliation.complete:
        raise RuntimeError("final cache reconciliation incomplete: {}".format(reconciliation.to_payload()))
    marker = write_completion_marker(store, final_manifest)
    second = reconcile_cache_manifest(final_manifest, store)
    if not second.complete:
        raise RuntimeError("completion marker reread reconciliation incomplete")
    counts["valid"] = sum(1 for layer, _ in final_manifest.entries() if layer == "asr" and store.read_asr(_).status == HIT_VALID)
    counts["rtf"] = counts["decode_wall_seconds"] / counts["total_audio_seconds"] if counts["total_audio_seconds"] else 0.0
    counts["final_manifest_id"] = final_manifest.manifest_id
    counts["final_manifest_sha256"] = final_manifest.manifest_sha256
    counts["completion_marker_id"] = marker.marker_id
    counts["completion_marker_status"] = "VALID"
    (repo / run_root / "asr_stage_summary.json").write_bytes(canonical_json_bytes(counts) + b"\n")
    print(json.dumps(counts, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("rir", "mixture", "asr"))
    parser.add_argument("--block", type=int, choices=(1, 2), default=0)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.stage == "rir":
        run_rir(repo, args.block)
    elif args.stage == "mixture":
        run_mixture(repo)
    elif args.stage == "asr":
        run_asr(repo)


if __name__ == "__main__":
    main()
