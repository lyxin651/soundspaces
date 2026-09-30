#!/usr/bin/env python3
"""Resumable production stages for the first O1 exploratory landscape.

The O1 manifest is independent of A4's frozen Engineering Smoke Manifest.
This runner reuses the already-validated cache/timeline/mixer/ASR primitives,
but writes only to the O1 run namespace and never changes A4 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from active_audition.a4.cache import CacheStore, HIT_VALID, MixtureCacheKey, NOISE_SOURCE_TIME_IDENTITY, RirCacheKey
from active_audition.a4.cache_resume import CacheExpectedManifest, reconcile_cache_manifest, write_completion_marker
from active_audition.a4.identity import canonical_json, canonical_json_bytes
from active_audition.a4.mixer import GlobalGainSpec, MixtureContract, WaveformComponent, build_mixture
from active_audition.a4.records import BlockRecord, CalibrationArtifact, EpisodeRecord
from active_audition.a4.smoke_manifest import EXPECTED_FRONTENDS
from active_audition.a4.timeline import map_dry_mask_to_receiver_time
from active_audition.evaluation.asr_metrics import error_counts
from active_audition.o1.manifest import O1ExploratoryManifest, O1ManifestError, O1_MANIFEST_SCHEMA_VERSION_V2
from active_audition.o1.component_snr import build_component_snr_record

from scripts.run_a4_production import (
    A2_ACOUSTIC_CONTRACT_SHA256,
    CACHE_ROOT,
    RENDERER_IDENTITY,
    REPLICATE_IDENTITY,
    _array_sha,
    _asr_key_contract,
    _build_timeline,
    _file_sha,
    _load_verified_source,
    _make_selection_component,
    _pose_listener,
    _rir_for,
    _rir_key,
    _scene_config,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_BASE = ROOT / "runs/active_asr_v1"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_rows(path: Path):
    """Load a frozen JSONL registry without introducing a new registry layer."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _manifest(path: Path) -> O1ExploratoryManifest:
    return O1ExploratoryManifest.from_payload(_json(path))


def _require_scientific_manifest(manifest: O1ExploratoryManifest, stage: str) -> None:
    """Fail closed before any noise-dependent or result-producing O1 stage."""

    if stage == "rir":
        return
    if manifest.schema_version != O1_MANIFEST_SCHEMA_VERSION_V2:
        raise RuntimeError("O1_SCIENTIFIC_MANIFEST_REQUIRED: {} requires manifest v2".format(stage))
    try:
        manifest.require_scientific_noise_audit()
    except O1ManifestError as exc:
        raise RuntimeError("O1_SCIENTIFIC_MANIFEST_REQUIRED: {}".format(exc)) from exc


def _run_root(manifest: O1ExploratoryManifest, supplied: Path | None) -> Path:
    if supplied is not None:
        return supplied
    return RUN_BASE / "o1_replica_apartment_2_{}".format(manifest.manifest_sha256[:12])


@lru_cache(maxsize=1)
def _frozen_asr_key_contract() -> tuple[str, object, object, object, object, object]:
    """Return the exact A3-v2/key identities shared by decode and diagnostics."""

    from active_audition.asr.contract import load_asr_contract
    from active_audition.asr.contract_v2 import a3_v2_contract_sha256, load_a3_v2_contract

    a3_v2 = load_a3_v2_contract(
        str(ROOT / "configs/active_audition/v1/asr_contract_v2.yaml"),
        require_frozen=True,
        repo_root=str(ROOT),
    )
    a3_sha = a3_v2_contract_sha256(a3_v2)
    if a3_sha != "70864c814a55db5d184ef8a6835b65cb564c4c90fc86112f8705b7ecf6df1ffe":
        raise RuntimeError("frozen A3-v2 contract SHA mismatch")
    contract = load_asr_contract(str(ROOT / "configs/active_audition/v1/asr_contract.yaml"), require_frozen=True)
    _, model_identity, lm_identity, tokenizer_identity, decoder_identity, runtime_identity = _asr_key_contract(contract)
    return a3_sha, model_identity, lm_identity, tokenizer_identity, decoder_identity, runtime_identity


def _asr_cache_key_for_mono(mono: np.ndarray, frontend: str):
    from active_audition.a4.cache import AsrCacheKey
    from active_audition.asr.frontends import FRONTENDS as FROZEN_FRONTENDS
    from active_audition.asr.speechbrain_adapter import waveform_sha256

    if frontend not in FROZEN_FRONTENDS:
        raise RuntimeError("unknown frozen frontend: {}".format(frontend))
    a3_sha, model_identity, lm_identity, tokenizer_identity, decoder_identity, runtime_identity = _frozen_asr_key_contract()
    return AsrCacheKey(
        mono_payload_sha256=waveform_sha256(mono),
        frontend=frontend,
        model_identity=model_identity,
        language_model_identity=lm_identity,
        tokenizer_identity=tokenizer_identity,
        decoder_identity=decoder_identity,
        precision_runtime_identity=runtime_identity,
        asr_contract_identity=a3_sha,
    )


def _scene_override() -> dict:
    habitat = ROOT / "data/scene_datasets/replica/apartment_2/habitat"
    return {"scene_asset": str(habitat / "mesh_semantic.ply"), "navmesh": str(habitat / "mesh_semantic.navmesh")}


def _scene_registry_row() -> dict:
    return {
        "scene_asset": str(ROOT / "data/scene_datasets/replica/apartment_2/habitat/mesh_semantic.ply"),
        "navmesh": str(ROOT / "data/scene_datasets/replica/apartment_2/habitat/mesh_semantic.navmesh"),
    }


def run_rir(manifest_path: Path, run_root: Path) -> None:
    from active_audition.acoustics.rir import render_native_rir
    from active_audition.navigation.pathfinder import PathFinderAdapter  # noqa: F401
    from active_audition.scene.simulator import create_scene_simulator

    manifest = _manifest(manifest_path)
    store = CacheStore(str(ROOT / CACHE_ROOT))
    config = _scene_config(ROOT, "replica.apartment_2", _scene_registry_row())
    keys = []
    counts = {"expected": 0, "cache_hits": 0, "new_renders": 0, "valid": 0}
    timings = []
    with create_scene_simulator(
        config,
        scene_id="replica.apartment_2",
        acoustics_overrides={"sampleRate": 16000},
        require_navmesh=True,
        load_semantic_mesh=False,
        scene_override=_scene_override(),
    ) as context:
        effective = {"sample_rate_hz": 16000, "materials_enabled": False, "channel_count": 2}
        for block_index, block in enumerate(manifest.blocks, 1):
            geometry = block["geometry_record"]
            for pose_index, pose in enumerate(block["poses"], 1):
                listener = _pose_listener(pose)
                for role in ("target", "noise"):
                    key = _rir_key(block, pose, role)
                    keys.append(key)
                    counts["expected"] += 1
                    cached = store.read_rir(key)
                    if cached.status == HIT_VALID:
                        counts["cache_hits"] += 1
                    else:
                        started = time.monotonic()
                        source = geometry["target_world_pose" if role == "target" else "noise_world_pose"]["position_xyz"]
                        array = np.asarray(render_native_rir(context, source, listener), dtype=np.float32)
                        elapsed = time.monotonic() - started
                        if array.ndim != 2 or array.shape[1] != 2 or array.dtype != np.dtype(np.float32) or not np.isfinite(array).all():
                            raise RuntimeError("O1 RIR hard gate failed at block {} pose {} {}".format(block_index, pose_index, role))
                        store.write_rir(key, array, provenance={
                            "stage": "O1-1-rir",
                            "o1_manifest_sha256": manifest.manifest_sha256,
                            "block_id": block["block_record"]["block_id"],
                            "pose_id": pose["pose_id"],
                            "source_role": role,
                            "sample_rate_hz": 16000,
                            "channel_order": ["L", "R"],
                            "materials_policy": "OFF",
                            "runtime_effective_audio_sensor": effective,
                            "render_seconds": elapsed,
                        })
                        timings.append(elapsed)
                        counts["new_renders"] += 1
                    if store.read_rir(key).status != HIT_VALID:
                        raise RuntimeError("O1 RIR strict reread failed")
                    counts["valid"] += 1
                if pose_index % 8 == 0:
                    print("block {} progress {}/48".format(block_index, pose_index), flush=True)
    expected = CacheExpectedManifest(infrastructure_contract_sha256=manifest.infrastructure_contract_sha256, expected_rir_keys=tuple(keys))
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "rir_expected_manifest.json").write_bytes(canonical_json_bytes(expected.to_payload()) + b"\n")
    values = sorted(timings)
    summary = dict(counts)
    summary.update({
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.manifest_sha256,
        "mean_render_seconds": float(np.mean(values)) if values else 0.0,
        "median_render_seconds": float(np.median(values)) if values else 0.0,
        "p95_render_seconds": float(np.percentile(values, 95)) if values else 0.0,
        "min_render_seconds": float(min(values)) if values else 0.0,
        "max_render_seconds": float(max(values)) if values else 0.0,
    })
    (run_root / "rir_stage_summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    if counts["valid"] != 384:
        raise RuntimeError("O1 RIR valid count is not 384: {}".format(counts))
    print(json.dumps(summary, sort_keys=True), flush=True)


def run_mixture(manifest_path: Path, run_root: Path) -> None:
    from active_audition.a4.active_mask import ActiveMaskContract, build_active_mask
    from active_audition.a4.calibration import CalibrationContract, calibrate_block_noise_gain

    manifest = _manifest(manifest_path)
    _require_scientific_manifest(manifest, "mixture")
    store = CacheStore(str(ROOT / CACHE_ROOT))
    all_keys = []
    index = []
    summary = {"expected": 0, "cache_hits": 0, "built": 0, "valid": 0, "blocks": []}
    for block_index, block in enumerate(manifest.blocks, 1):
        block_record = BlockRecord.from_payload(block["block_record"])
        episodes = [EpisodeRecord.from_payload(item) for item in block["episodes"]]
        speech_by_id = {item["utterance_id"]: item for item in block["speech_sources"]}
        speech = {uid: _load_verified_source(ROOT, row, "speech") for uid, row in speech_by_id.items()}
        noise_parent = _load_verified_source(ROOT, block["noise_parent"], "noise")
        masks = {uid: build_active_mask(wave, 16000, ActiveMaskContract()) for uid, wave in speech.items()}
        initial_pose = block["poses"][0]
        if float(initial_pose["total_cost_sec"]) != 0.0 or float(initial_pose["yaw_deg"]) != float(block["geometry_record"]["initial_yaw_deg"]):
            raise RuntimeError("O1 initial Stay pose is not the first frozen pose")
        selection_components = []
        for episode_payload, episode_record in zip(block["episodes"][:2], episodes[:2]):
            uid = episode_record.utterance_identity["utterance_id"]
            segment = episode_record.fixed_dry_noise_segment_identity
            dry_noise = noise_parent[int(segment["start_sample"]):int(segment["end_sample"])]
            timeline = _build_timeline(block, episode_payload, initial_pose, speech[uid], dry_noise, store)
            receiver_mask = map_dry_mask_to_receiver_time(masks[uid], timeline)
            selection_components.append(_make_selection_component(block_record, episode_record, timeline, masks[uid], receiver_mask))
        fresh_artifact = calibrate_block_noise_gain(block_record, selection_components, CalibrationContract(nominal_snr_db=block_record.nominal_initial_snr_db))
        run_root.mkdir(parents=True, exist_ok=True)
        calibration_path = run_root / "calibration_block_{}.json".format(block_index)
        artifact = fresh_artifact
        if calibration_path.is_file():
            existing = CalibrationArtifact.from_payload(_json(calibration_path))
            reusable = (
                existing.block_id == fresh_artifact.block_id
                and existing.selection_episode_ids == fresh_artifact.selection_episode_ids
                and existing.calibration_contract_identity == fresh_artifact.calibration_contract_identity
                and existing.input_component_identities == fresh_artifact.input_component_identities
                and existing.active_sample_count == fresh_artifact.active_sample_count
                and math.isclose(existing.ps, fresh_artifact.ps, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(existing.pn, fresh_artifact.pn, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(existing.alpha, fresh_artifact.alpha, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(existing.measured_snr_db, fresh_artifact.measured_snr_db, rel_tol=0.0, abs_tol=1.0e-12)
            )
            if reusable:
                artifact = existing
        (run_root / "calibration_block_{}.json".format(block_index)).write_bytes(canonical_json_bytes(artifact.to_payload()) + b"\n")
        gain = GlobalGainSpec.from_payload(block["global_gain"])
        contract = MixtureContract()
        block_summary = {"block_index": block_index, "block_id": block_record.block_id, "calibration_artifact_id": artifact.calibration_artifact_id, "ps": artifact.ps, "pn": artifact.pn, "alpha": artifact.alpha, "measured_snr_db": artifact.measured_snr_db, "error_db": abs(artifact.measured_snr_db - artifact.nominal_snr_db), "mixtures": 0}
        for episode_payload, episode_record in zip(block["episodes"], episodes):
            uid = episode_record.utterance_identity["utterance_id"]
            segment = episode_record.fixed_dry_noise_segment_identity
            dry_noise = noise_parent[int(segment["start_sample"]):int(segment["end_sample"])]
            noise_sha = _array_sha(dry_noise)
            speech_registry = next(row for row in _registry_rows(ROOT / "registries/active_asr_a3/librispeech.jsonl") if row["utterance_id"] == uid)
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
                cached = store.read_mixture(key)
                if cached.status == HIT_VALID:
                    summary["cache_hits"] += 1
                else:
                    built = build_mixture(block_record, episode_record, pose["pose_id"], timeline, target_component, noise_component, artifact, gain, contract)
                    store.write_mixture(key, built.mixture_binaural, provenance={
                        "stage": "O1-1-mixture", "o1_manifest_sha256": manifest.manifest_sha256, "block_id": block_record.block_id,
                        "episode_id": episode_record.episode_id, "pose_id": pose["pose_id"], "mixture_artifact_id": built.mixture_artifact_id,
                        "calibration_artifact_id": artifact.calibration_artifact_id, "global_gain_identity": gain.identity,
                        "timeline_id": timeline.timeline_id, "target_component_identity": target_component.component_identity,
                        "noise_component_identity": noise_component.component_identity,
                    })
                    summary["built"] += 1
                if store.read_mixture(key).status != HIT_VALID:
                    raise RuntimeError("O1 mixture strict reread failed")
                summary["valid"] += 1
                block_summary["mixtures"] += 1
                index.append({
                    "mixture_cache_key": key.cache_key,
                    "mixture_key_payload": key.to_payload(),
                    "block_id": block_record.block_id,
                    "episode_id": episode_record.episode_id,
                    "role": episode_record.role,
                    "utterance_id": uid,
                    "pose_id": pose["pose_id"],
                    "reference": speech_registry["normalized_transcript"],
                    "motion_cost_sec": pose["total_cost_sec"],
                    "geometry_legality": pose["geometry_legality"],
                })
        summary["blocks"].append(block_summary)
    expected = CacheExpectedManifest(infrastructure_contract_sha256=manifest.infrastructure_contract_sha256, expected_rir_keys=tuple(
        key for block in manifest.blocks for pose in block["poses"] for role in ("target", "noise") for key in (_rir_key(block, pose, role),)
    ), expected_mixture_keys=tuple(all_keys))
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "mixture_expected_manifest.json").write_bytes(canonical_json_bytes(expected.to_payload()) + b"\n")
    (run_root / "o1_mixture_index.json").write_bytes(canonical_json_bytes({"schema_version": "active-asr-o1-mixture-index-v1", "entries": sorted(index, key=lambda row: row["mixture_cache_key"])}) + b"\n")
    summary.update({"manifest_id": manifest.manifest_id, "manifest_sha256": manifest.manifest_sha256})
    (run_root / "mixture_stage_summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    if summary["valid"] != 768:
        raise RuntimeError("O1 mixture valid count is not 768: {}".format(summary))
    print(json.dumps(summary, sort_keys=True), flush=True)


def run_asr(manifest_path: Path, run_root: Path, batch_size: int = 4) -> None:
    import platform
    import torch
    import yaml
    from active_audition.a4.cache import ASR_RESULT_SCHEMA_VERSION
    from active_audition.a4.contract import load_contract
    from active_audition.a4.cache_resume import reconcile_cache_manifest, write_completion_marker
    from active_audition.asr.contract import load_asr_contract
    from active_audition.asr.frontends import apply_frontend
    from active_audition.asr.speechbrain_adapter import SpeechBrainASRAdapter

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("FROZEN_GPU_REQUIRED: O1 ASR requires cuda:0; CPU fallback is forbidden")
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    manifest = _manifest(manifest_path)
    _require_scientific_manifest(manifest, "asr")
    store = CacheStore(str(ROOT / CACHE_ROOT))
    a3_sha, model_identity, lm_identity, tokenizer_identity, decoder_identity, runtime_identity = _frozen_asr_key_contract()
    contract = load_asr_contract(str(ROOT / "configs/active_audition/v1/asr_contract.yaml"), require_frozen=True)
    mixture_manifest = CacheExpectedManifest.from_payload(_json(run_root / "mixture_expected_manifest.json"))
    jobs = []
    expected = []
    for mixture_key in mixture_manifest.expected_mixture_keys:
        result = store.read_mixture(mixture_key)
        if result.status != HIT_VALID:
            raise RuntimeError("O1 ASR mixture prerequisite invalid")
        binaural = np.asarray(result.payload, dtype=np.float32)
        for frontend in EXPECTED_FRONTENDS:
            mono = np.asarray(apply_frontend(binaural, frontend), dtype=np.float32)
            key = _asr_cache_key_for_mono(mono, frontend)
            expected.append(key)
            if store.read_asr(key).status != HIT_VALID:
                jobs.append((key, mono, mixture_key.cache_key, frontend))
    adapter = None
    counts = {"expected": len(expected), "cache_hits": len(expected) - len(jobs), "new_decodes": 0, "valid": 0, "total_audio_seconds": 0.0, "decode_wall_seconds": 0.0, "batch_size": batch_size}
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        if adapter is None:
            adapter = SpeechBrainASRAdapter(contract)
        started = time.monotonic()
        outputs = adapter.transcribe_batch([item[1] for item in batch], 16000, frontend="mono_input")
        counts["decode_wall_seconds"] += time.monotonic() - started
        if len(outputs) != len(batch):
            raise RuntimeError("O1 ASR batch output count mismatch")
        for (key, mono, mixture_cache_key, frontend), output in zip(batch, outputs):
            payload = {"schema_version": ASR_RESULT_SCHEMA_VERSION, "hypothesis": output.hypothesis, "score": output.score, "score_semantics": output.score_semantics, "input_waveform_sha256": output.input_waveform_sha256, "frontend": frontend, "model_identity": model_identity, "decoder_identity": decoder_identity, "asr_contract_identity": a3_sha, "raw_decoder_metadata": output.raw_decoder_metadata}
            store.write_asr(key, payload, input_shape=mono.shape, provenance={"stage": "O1-1-asr", "o1_manifest_sha256": manifest.manifest_sha256, "mixture_cache_key": mixture_cache_key, "frontend": frontend, "a3_v2_contract_sha256": a3_sha})
            counts["new_decodes"] += 1
            counts["total_audio_seconds"] += float(mono.size) / 16000.0
    final = CacheExpectedManifest(infrastructure_contract_sha256=manifest.infrastructure_contract_sha256, expected_rir_keys=mixture_manifest.expected_rir_keys, expected_mixture_keys=mixture_manifest.expected_mixture_keys, expected_asr_keys=tuple(expected))
    (run_root / "final_cache_expected_manifest.json").write_bytes(canonical_json_bytes(final.to_payload()) + b"\n")
    reconciliation = reconcile_cache_manifest(final, store)
    if not reconciliation.complete:
        raise RuntimeError("O1 final reconciliation incomplete")
    marker = write_completion_marker(store, final)
    if not reconcile_cache_manifest(final, store).complete:
        raise RuntimeError("O1 completion reread failed")
    counts["valid"] = len(expected)
    counts["rtf"] = counts["decode_wall_seconds"] / counts["total_audio_seconds"] if counts["total_audio_seconds"] else 0.0
    counts.update({"manifest_id": manifest.manifest_id, "manifest_sha256": manifest.manifest_sha256, "final_manifest_id": final.manifest_id, "completion_marker_id": marker.marker_id, "completion_marker_status": "VALID", "a3_v2_contract_sha256": a3_sha, "device": "cuda:0", "device_name": torch.cuda.get_device_name(0), "python": platform.python_version()})
    (run_root / "asr_stage_summary.json").write_bytes(canonical_json_bytes(counts) + b"\n")
    print(json.dumps(counts, sort_keys=True), flush=True)


def run_component_snr(manifest_path: Path, run_root: Path) -> None:
    """Reconstruct only propagated components for the privileged SNR diagnostic."""
    from active_audition.a4.active_mask import ActiveMaskContract, build_active_mask

    manifest = _manifest(manifest_path)
    _require_scientific_manifest(manifest, "component-snr")
    store = CacheStore(str(ROOT / CACHE_ROOT))
    records = []
    for block_index, block in enumerate(manifest.blocks, 1):
        block_record = BlockRecord.from_payload(block["block_record"])
        episodes = [EpisodeRecord.from_payload(item) for item in block["episodes"]]
        speech_by_id = {item["utterance_id"]: item for item in block["speech_sources"]}
        speech = {uid: _load_verified_source(ROOT, row, "speech") for uid, row in speech_by_id.items()}
        masks = {uid: build_active_mask(wave, 16000, ActiveMaskContract()) for uid, wave in speech.items()}
        calibration = CalibrationArtifact.from_payload(_json(run_root / "calibration_block_{}.json".format(block_index)))
        for episode_payload, episode_record in zip(block["episodes"], episodes):
            uid = episode_record.utterance_identity["utterance_id"]
            segment = episode_record.fixed_dry_noise_segment_identity
            noise_parent = _load_verified_source(ROOT, block["noise_parent"], "noise")
            dry_noise = noise_parent[int(segment["start_sample"]):int(segment["end_sample"])]
            for pose in block["poses"]:
                timeline = _build_timeline(block, episode_payload, pose, speech[uid], dry_noise, store)
                target_component = WaveformComponent.from_array("target", timeline.timeline_id, timeline.target_binaural)
                noise_component = WaveformComponent.from_array("noise", timeline.timeline_id, timeline.noise_binaural)
                receiver_mask = map_dry_mask_to_receiver_time(masks[uid], timeline)
                records.append(build_component_snr_record(
                    block_id=block_record.block_id,
                    episode_id=episode_record.episode_id,
                    role=episode_record.role,
                    utterance_id=uid,
                    pose_id=pose["pose_id"],
                    timeline_identity=timeline.timeline_id,
                    target_component_identity=target_component.component_identity,
                    noise_component_identity=noise_component.component_identity,
                    receiver_mask_identity=receiver_mask.mask_id,
                    calibration_artifact_identity=calibration.calibration_artifact_id,
                    alpha=calibration.alpha,
                    target_binaural=timeline.target_binaural,
                    noise_binaural=timeline.noise_binaural,
                    receiver_mask=receiver_mask.mask,
                ))
    if len(records) != 768:
        raise RuntimeError("O1 component-SNR cardinality is not 768: {}".format(len(records)))
    run_root.mkdir(parents=True, exist_ok=True)
    payload = b"\n".join(canonical_json_bytes(record.to_payload()) for record in sorted(records, key=lambda item: item.record_id)) + b"\n"
    (run_root / "o1_component_snr.jsonl").write_bytes(payload)
    summary = {
        "schema_version": "active-asr-o1-component-snr-summary-v1",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.manifest_sha256,
        "records": len(records),
        "algorithm_identity": "active-asr-o1-target-active-mask-two-ear-component-snr-v1",
        "result_dependent_pose_selection": False,
    }
    (run_root / "o1_component_snr_summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


def build_diagnostics(manifest_path: Path, run_root: Path) -> None:
    from active_audition.asr.frontends import apply_frontend
    manifest = _manifest(manifest_path)
    _require_scientific_manifest(manifest, "diagnostics")
    store = CacheStore(str(ROOT / CACHE_ROOT))
    index = _json(run_root / "o1_mixture_index.json")["entries"]
    rows = []
    for item in index:
        mixture_key = MixtureCacheKey.from_payload(item["mixture_key_payload"])
        mixture = store.read_mixture(mixture_key)
        if mixture.status != HIT_VALID:
            raise RuntimeError("O1 diagnostics mixture invalid")
        binaural = np.asarray(mixture.payload, dtype=np.float32)
        for frontend in EXPECTED_FRONTENDS:
            mono = np.asarray(apply_frontend(binaural, frontend), dtype=np.float32)
            asr_key = _asr_cache_key_for_mono(mono, frontend)
            result = store.read_asr(asr_key)
            if result.status != HIT_VALID:
                raise RuntimeError("O1 diagnostics exact ASR key is not HIT_VALID for {} {}".format(item["mixture_cache_key"], frontend))
            metrics = error_counts(item["reference"], result.payload["hypothesis"])
            rows.append({"block_id": item["block_id"], "episode_id": item["episode_id"], "role": item["role"], "utterance_id": item["utterance_id"], "pose_id": item["pose_id"], "frontend": frontend, "reference": item["reference"], "hypothesis": result.payload["hypothesis"], "S": metrics["S"], "D": metrics["D"], "I": metrics["I"], "N": metrics["N"], "WER": metrics["WER"], "motion_cost_sec": item["motion_cost_sec"], "geometry_legality": item["geometry_legality"]})
    rows.sort(key=lambda row: (row["block_id"], row["episode_id"], row["frontend"], row["pose_id"]))
    (run_root / "o1_asr_diagnostics.jsonl").write_bytes(b"\n".join(canonical_json_bytes(row) for row in rows) + b"\n")
    summary = {"schema_version": "active-asr-o1-asr-diagnostics-summary-v1", "manifest_id": manifest.manifest_id, "manifest_sha256": manifest.manifest_sha256, "records": len(rows), "frontends": list(EXPECTED_FRONTENDS), "result_dependent_pose_selection": False}
    (run_root / "o1_asr_diagnostics_summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    print(json.dumps(summary, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("rir", "mixture", "component-snr", "asr", "diagnostics"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    manifest = _manifest(args.manifest)
    run_root = _run_root(manifest, args.run_root)
    if args.stage == "rir":
        run_rir(args.manifest, run_root)
    elif args.stage == "mixture":
        run_mixture(args.manifest, run_root)
    elif args.stage == "asr":
        run_asr(args.manifest, run_root, args.batch_size)
    elif args.stage == "component-snr":
        run_component_snr(args.manifest, run_root)
    else:
        build_diagnostics(args.manifest, run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
