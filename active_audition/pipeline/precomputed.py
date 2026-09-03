"""Small real-asset V0.5 PLAN/RENDER path; deliberately independent of Habitat/RLRA."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import yaml
from scipy.io import wavfile

from active_audition.acoustics.precomputed_rir import PrecomputedRIRBackend
from active_audition.acoustics.resampling import canonicalize_rir
from active_audition.acoustics.renderer import convolve_binaural, write_float32_wav
from active_audition.data.audio import load_dry_segment
from active_audition.data.manifest import read_plan_manifests, read_viewpoint_manifest, write_plan_manifests, write_viewpoint_manifest
from active_audition.data.storage import DatasetStorage, StorageError
from active_audition.navigation.graph import generate_graph_candidates


def _backend(config: Mapping[str, Any], scene: str) -> PrecomputedRIRBackend:
    root = Path(config["_repo_root"]) / str(config["acoustics"]["rir_root"])
    metadata = Path(config["_repo_root"]) / str(config["acoustics"]["metadata_root"])
    return PrecomputedRIRBackend(str(root), metadata_root=str(metadata))


def _scene_pair(backend: PrecomputedRIRBackend, scene: str) -> Tuple[str, str, int]:
    graph_nodes = set(backend.list_nodes(scene))
    for key in sorted(backend._rows):
        if key[0] == scene and key[1] in graph_nodes and key[2] in graph_nodes and key[3] == 0:
            return key[1], key[2], key[3]
    raise StorageError("no graph/RIR closed pair for {}".format(scene))


def _point(backend: PrecomputedRIRBackend, scene: str, node: str):
    return list(backend._nodes[scene][str(node)])


def plan_precomputed(config_path: str) -> Dict[str, Any]:
    from active_audition.config.loader import load_resolved_config
    config = load_resolved_config(config_path)
    storage = DatasetStorage(Path(config["_repo_root"]) / "datasets/active_audition_v05" / config["storage"]["dataset_id"])
    storage.ensure_writable()
    episodes, candidates = [], []
    for scene in sorted(config["scene"]["ids"]):
        backend = _backend(config, scene)
        if scene not in backend.list_scenes(): raise StorageError("scene not available: {}".format(scene))
        receiver, source, heading = _scene_pair(backend, scene)
        for ordinal in range(int(config["episode"]["count"])):
            episode_id = "{}__ep_{:06d}".format(scene.replace(".", "_"), ordinal + 1)
            episode = {"schema_version": "v0.5", "episode_id": episode_id, "scene_id": scene, "scene_family": scene.split(".", 1)[0], "episode_seed": int(config["experiment"]["global_seed"] + ordinal),
                       "source_node_id": source, "source_xyz": _point(backend, scene, source), "initial_receiver_node_id": receiver, "initial_receiver_xyz": _point(backend, scene, receiver), "initial_heading_index": heading, "initial_heading_deg_dataset": 0.0,
                       "source": {"position_world": _point(backend, scene, source), "audio_id": "golden_probe_v0", "segment_start_sec": 0.0, "segment_duration_sec": 5.0, "gain_db": 0.0},
                       "listener_initial": {"base_position_world": _point(backend, scene, receiver), "sensor_position_world": _point(backend, scene, receiver), "yaw_deg": 0.0}}
            episodes.append(episode)
            rows = generate_graph_candidates(episode, backend, max_neighbors=3)
            candidates.extend(row for row in rows if row["action_type"] != "stay")
    write_plan_manifests(str(storage.root), episodes, candidates)
    return {"dataset_root": str(storage.root), "episode_count": len(episodes), "candidate_count": len(candidates), "valid_candidates": sum(bool(x["valid"]) for x in candidates), "invalid_candidates": sum(not bool(x["valid"]) for x in candidates)}


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def render_precomputed(config_path: str, resume: bool = False) -> Dict[str, Any]:
    from active_audition.config.loader import load_resolved_config
    config = load_resolved_config(config_path)
    storage = DatasetStorage(Path(config["_repo_root"]) / "datasets/active_audition_v05" / config["storage"]["dataset_id"])
    storage.ensure_writable()
    manifests = read_plan_manifests(str(storage.root)); existing = {(x["episode_id"], x["viewpoint_id"]): x for x in (read_viewpoint_manifest(str(storage.root)) if storage.manifest_path("viewpoints.jsonl").exists() else [])}
    registry = {"golden_probe_v0": {"path": str(Path(config["_repo_root"]) / "res/active_audition/golden_probe_v0.wav")}}
    output = list(existing.values()); skipped = 0; rendered = 0
    backends = {scene: _backend(config, scene) for scene in sorted({e["scene_id"] for e in manifests["episodes"]})}
    for episode in manifests["episodes"]:
        backend = backends[episode["scene_id"]]; source = episode["source"]
        dry = load_dry_segment(registry["golden_probe_v0"]["path"], 0.0, 5.0, 24000, 0.0)
        viewpoints = [("initial", None, "initial", episode["initial_receiver_node_id"], episode["initial_heading_index"])]
        for c in manifests["candidates"]:
            if c["episode_id"] == episode["episode_id"] and c["valid"]:
                viewpoints.append((c["candidate_id"], c["candidate_id"], c["action_type"], c["to_receiver_node_id"], c["to_heading"]))
        for viewpoint_id, candidate_id, action, receiver, heading in viewpoints:
            key = (episode["episode_id"], viewpoint_id)
            if resume and key in existing:
                skipped += 1; continue
            asset, rir = backend.read_canonical_rir(episode["scene_id"], receiver, episode["source_node_id"], heading)
            waveform = convolve_binaural(dry, rir)
            audio = storage.viewpoint_audio_path(episode["scene_id"], episode["episode_id"], viewpoint_id)
            rir_path = storage.rir_cache_path(episode["episode_id"] + "__" + viewpoint_id)
            audio.parent.mkdir(parents=True, exist_ok=True); write_float32_wav(str(audio), waveform, 24000)
            storage.atomic_write_npz(rir_path, {"rir": rir, "sample_rate_hz": np.int64(24000), "num_samples": np.int64(rir.shape[0])})
            row = {"episode_id": episode["episode_id"], "viewpoint_id": viewpoint_id, "candidate_id": candidate_id, "action_type": action, "base_position_world": _point(backend, episode["scene_id"], receiver), "sensor_position_world": _point(backend, episode["scene_id"], receiver), "yaw_deg": 0.0, "audio_path": str(audio.relative_to(storage.root)), "rir_id": episode["episode_id"] + "__" + viewpoint_id, "rir_path": str(rir_path.relative_to(storage.root)), "sample_rate_hz": 24000, "num_channels": 2, "num_samples": int(waveform.shape[0]), "duration_sec": waveform.shape[0] / 24000.0, "dtype": "float32", "receiver_node_id": receiver, "source_node_id": episode["source_node_id"], "heading_index": int(heading), "heading_deg_dataset": backend.list_headings(episode["scene_id"])[int(heading)]["heading_deg_dataset"], **backend.asset_manifest_fields(asset), "ground_truth": {"scene_id": episode["scene_id"], "source_position_world": source["position_world"], "dry_audio_id": source["audio_id"], "segment_start_sec": 0.0, "segment_duration_sec": 5.0, "gain_db": 0.0, "dry_hash": hashlib.sha256(dry.tobytes()).hexdigest()}}
            output.append(row); rendered += 1
    write_viewpoint_manifest(str(storage.root), output)
    return {"dataset_root": str(storage.root), "viewpoints": len(output), "rendered": rendered, "skipped": skipped}


def finalize_precomputed(config_path: str, validation: Mapping[str, Any]) -> Dict[str, Any]:
    from active_audition.config.loader import load_resolved_config
    config = load_resolved_config(config_path)
    if validation.get("status") != "PASS":
        raise StorageError("cannot finalize a failed dataset")
    root = Path(config["_repo_root"])
    storage = DatasetStorage(root / "datasets/active_audition_v05" / config["storage"]["dataset_id"])
    storage.ensure_writable()
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    storage.atomic_write_text(storage.path("config_resolved.yaml"), yaml.safe_dump(public, sort_keys=True))
    rows = read_viewpoint_manifest(str(storage.root))
    used_assets = [{"rir_asset_relpath": row.get("rir_asset_relpath"), "rir_asset_sha256": row.get("rir_asset_sha256"), "rir_original_sample_rate_hz": row.get("rir_original_sample_rate_hz")} for row in rows]
    used_assets.sort(key=lambda row: (str(row["rir_asset_relpath"]), str(row["rir_asset_sha256"])))
    aggregate = hashlib.sha256(json.dumps(used_assets, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    lock = {"precomputed_rir_root": str(root / config["acoustics"]["rir_root"]), "metadata_root": str(root / config["acoustics"]["metadata_root"]), "selected_scene_ids": sorted(config["scene"]["ids"]), "official_asset_archives": {"replica_office_0.tar.gz": _sha(root / "source_assets/soundspaces_official/binaural_rirs/replica_office_0.tar.gz"), "mp3d_17DRP5sb8fy.tar.gz": _sha(root / "source_assets/soundspaces_official/binaural_rirs/mp3d_17DRP5sb8fy.tar.gz")}, "used_assets_aggregate_sha256": aggregate, "used_assets": used_assets, "resampling_contract_version": "v0.5-resample-poly-16k-intermediate-24k-v1", "canonical_output_sample_rate_hz": 24000, "rir_intermediate_sample_rate_hz": 16000, "ontology_v2_scene_registry_sha256": "06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508", "metadata_archive_sha256": _sha(root / "source_assets/soundspaces_official/metadata.tar.xz")}
    storage.atomic_write_json(storage.path("resources.lock.json"), lock)
    storage.atomic_write_json(storage.path("identity.json"), {"dataset_id": config["storage"]["dataset_id"], "schema_version": "v0.5", "backend": "soundspaces_precomputed", "config_sha256": _sha(storage.path("config_resolved.yaml")), "used_assets_aggregate_sha256": aggregate})
    storage.atomic_write_text(storage.success_path, "Pipeline V0.5 dataset finalized\n")
    return {"dataset_root": str(storage.root), "dataset_id": config["storage"]["dataset_id"], "status": "PASS", "success": str(storage.success_path), "viewpoints": len(rows), "used_assets": len(used_assets)}
