"""M2.1 V0 orchestration for dataset contract, provenance and recovery."""

import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import yaml

from active_audition.acoustics.renderer import convolve_binaural, write_float32_wav
from active_audition.acoustics.rir import render_native_rir
from active_audition.config.loader import load_resolved_config
from active_audition.data.audio import load_dry_segment
from active_audition.data.catalog import load_dry_audio_registry, load_scene_registry, sha256_file
from active_audition.data.manifest import read_plan_manifests, read_viewpoint_manifest, write_viewpoint_manifest
from active_audition.data.storage import DatasetStorage, StorageError
from active_audition.data.storage import json_line
from active_audition.data.validation import payload_is_complete, validate_dataset
from active_audition.scene.simulator import create_scene_simulator


ACTION_ORDER = {"initial": 0, "translation": 1, "rotation": 2}
SUCCESS_TEXT = "Pipeline V0 M2.1 finalized\n"


def _storage_for_config(config: Mapping[str, Any]) -> DatasetStorage:
    return DatasetStorage.from_config(config["_repo_root"], config)


def _row_pose(row: Mapping[str, Any]) -> Any:
    from active_audition.types import ListenerPose

    base = row["base_position_world"] if "base_position_world" in row else row["snapped_base_position_world"]
    return ListenerPose(tuple(float(x) for x in base), tuple(float(x) for x in row["sensor_position_world"]), float(row["yaw_deg"]))


def _dry_hash(dry: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(dry, dtype=np.float32).tobytes()).hexdigest()


def _write_wav_atomic(storage: DatasetStorage, path: Path, waveform: np.ndarray, sample_rate_hz: int) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".wav-", suffix=".wav", dir=str(storage.root))
    os.close(fd)
    try:
        write_float32_wav(temp_name, waveform, sample_rate_hz)
        storage.atomic_write_bytes(path, Path(temp_name).read_bytes())
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def _viewpoint_row(episode: Mapping[str, Any], viewpoint_id: str, candidate_id: Any, action_type: str, pose: Any, audio_path: Path, rir_id: str, rir_path: Path, sample_rate_hz: int, waveform: np.ndarray, dry_hash: str) -> Dict[str, Any]:
    source = episode["source"]
    dataset_root = Path(episode["_dataset_root"])
    return {
        "episode_id": episode["episode_id"], "viewpoint_id": viewpoint_id, "candidate_id": candidate_id, "action_type": action_type,
        "base_position_world": list(pose.base_position_world), "sensor_position_world": list(pose.sensor_position_world), "yaw_deg": float(pose.yaw_deg),
        "audio_path": str(audio_path.relative_to(dataset_root)), "rir_id": rir_id, "rir_path": str(rir_path.relative_to(dataset_root)),
        "sample_rate_hz": int(sample_rate_hz), "num_channels": 2, "num_samples": int(waveform.shape[0]),
        "duration_sec": float(waveform.shape[0] / float(sample_rate_hz)), "dtype": str(waveform.dtype),
        "ground_truth": {
            "scene_id": episode["scene_id"], "source_position_world": source["position_world"], "dry_audio_id": source["audio_id"],
            "segment_start_sec": source["segment_start_sec"], "segment_duration_sec": source["segment_duration_sec"], "gain_db": source["gain_db"], "dry_hash": dry_hash,
        },
    }


def _ordered_candidate_rows(rows: Iterable[Mapping[str, Any]]) -> list:
    return sorted(rows, key=lambda row: (ACTION_ORDER.get(str(row.get("action_type")), 99), str(row.get("translation_direction") or ""), str(row.get("candidate_id"))))


def _existing_viewpoints(storage: DatasetStorage) -> Dict[tuple, Mapping[str, Any]]:
    path = storage.manifest_path("viewpoints.jsonl")
    if not path.exists():
        return {}
    rows = read_viewpoint_manifest(str(storage.root))
    keys = [(row.get("episode_id"), row.get("viewpoint_id")) for row in rows]
    if any(None in key for key in keys) or len(keys) != len(set(keys)):
        raise StorageError("viewpoints manifest has duplicate or missing keys")
    return dict(zip(keys, rows))


def _orphan_payloads(storage: DatasetStorage, rows: Iterable[Mapping[str, Any]]) -> list:
    rows = list(rows)
    audio_refs = {str(row.get("audio_path")) for row in rows}
    rir_refs = {str(row.get("rir_path")) for row in rows}
    orphans = []
    for path in sorted(storage.root.glob("episodes/*/*/audio/*.wav")):
        if str(path.relative_to(storage.root)) not in audio_refs:
            orphans.append(str(path.relative_to(storage.root)))
    for path in sorted(storage.root.glob("cache/rir/*.npz")):
        if str(path.relative_to(storage.root)) not in rir_refs:
            orphans.append(str(path.relative_to(storage.root)))
    return orphans


def _append_generation_event(storage: DatasetStorage, event: Mapping[str, Any]) -> None:
    """Keep render/resume provenance append-only beside legacy logs."""

    path = storage.path("logs", "generation_events.jsonl")
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    storage.atomic_write_text(path, previous + json_line(dict(event)) + "\n")


def render_dataset(config_path: str, resume: bool = False, overwrite: bool = False) -> Dict[str, Any]:
    if resume and overwrite:
        raise StorageError("--resume and --overwrite are mutually exclusive")
    config = load_resolved_config(config_path)
    repo_root = Path(config["_repo_root"])
    storage = _storage_for_config(config)
    storage.ensure_writable()
    manifests = read_plan_manifests(str(storage.root))
    episodes, candidates = manifests["episodes"], manifests["candidates"]
    existing = _existing_viewpoints(storage)
    orphans = _orphan_payloads(storage, existing.values())
    if orphans:
        raise StorageError("orphan payload rejected: {}".format(orphans))
    if existing and not resume and not overwrite:
        raise StorageError("incomplete dataset already contains rendered viewpoints; use --resume or --overwrite")
    registry = load_dry_audio_registry(config["registries"]["dry_audio_path"], str(repo_root))
    rendered, skipped, recovery, details = [], [], [], []
    generation_stats, render_failures = [], []
    with create_scene_simulator(config) as context:
        for episode_input in sorted(episodes, key=lambda row: str(row["episode_id"])):
            episode_started = time.perf_counter()
            episode = dict(episode_input)
            episode["_dataset_root"] = str(storage.root)
            source = episode["source"]
            dry = load_dry_segment(registry[source["audio_id"]]["path"], float(source["segment_start_sec"]), float(source["segment_duration_sec"]), int(config["acoustics"]["sample_rate_hz"]), float(source["gain_db"]))
            dry_hash = _dry_hash(dry)
            valid_candidates = [row for row in candidates if row.get("episode_id") == episode["episode_id"] and bool(row.get("valid"))]
            viewpoints = [("initial", None, "initial", _row_pose(episode["listener_initial"]))]
            viewpoints.extend((str(candidate["candidate_id"]), str(candidate["candidate_id"]), str(candidate["action_type"]), _row_pose(candidate)) for candidate in _ordered_candidate_rows(valid_candidates))
            episode_render_failures = []
            episode_rendered = 0
            episode_skipped = 0
            for viewpoint_id, candidate_id, action_type, pose in viewpoints:
                key = (episode["episode_id"], viewpoint_id)
                old = existing.get(key)
                if resume and not overwrite and old is not None and payload_is_complete(str(storage.root), old, bool(config["storage"]["save_rir"])):
                    rendered.append(old); skipped.append(viewpoint_id); episode_skipped += 1; continue
                if resume and old is not None:
                    recovery.append(viewpoint_id)
                try:
                    rir = render_native_rir(context, source["position_world"], pose)
                    rir_id = "{}__{}".format(episode["episode_id"], viewpoint_id)
                    rir_path = storage.rir_cache_path(rir_id)
                    if bool(config["storage"]["save_rir"]):
                        storage.atomic_write_npz(rir_path, {"rir": rir, "sample_rate_hz": np.asarray(config["acoustics"]["sample_rate_hz"], dtype=np.int64), "num_samples": np.asarray(rir.shape[0], dtype=np.int64)})
                    waveform = convolve_binaural(dry, rir)
                    audio_path = storage.viewpoint_audio_path(episode["scene_id"], episode["episode_id"], viewpoint_id)
                    if bool(config["storage"]["save_audio"]):
                        _write_wav_atomic(storage, audio_path, waveform, int(config["acoustics"]["sample_rate_hz"]))
                    rendered.append(_viewpoint_row(episode, viewpoint_id, candidate_id, action_type, pose, audio_path, rir_id, rir_path, int(config["acoustics"]["sample_rate_hz"]), waveform, dry_hash))
                    details.append({"episode_id": episode["episode_id"], "viewpoint_id": viewpoint_id, "rir_samples": int(rir.shape[0]), "wav_samples": int(waveform.shape[0]), "wav_peak": float(np.max(np.abs(waveform))), "dtype": str(waveform.dtype)})
                    episode_rendered += 1
                except Exception as exc:
                    failure = {"episode_id": episode["episode_id"], "viewpoint_id": viewpoint_id, "exception_type": type(exc).__name__, "message": str(exc)}
                    episode_render_failures.append(failure)
                    render_failures.append(failure)
            generation_stats.append({
                "episode_id": episode["episode_id"],
                "rendered_viewpoints": int(episode_rendered),
                "skipped_viewpoints": episode_skipped,
                "valid_candidates": len(valid_candidates),
                "render_failures": len(episode_render_failures),
                "failure_details": episode_render_failures,
                "episode_render_runtime_sec": float(time.perf_counter() - episode_started),
            })
    manifest_path = write_viewpoint_manifest(str(storage.root), rendered)
    storage.atomic_write_text(storage.path("logs", "generation.log"), "resume={} overwrite={} rendered={} skipped={} recovery={}\n".format(resume, overwrite, len(rendered) - len(skipped), len(skipped), recovery))
    stats_text = "\n".join(json_line(row) for row in sorted(generation_stats, key=lambda row: row["episode_id"]))
    if stats_text:
        stats_text += "\n"
    storage.atomic_write_text(storage.path("logs", "generation_stats.jsonl"), stats_text)
    mode = "resume" if resume else "overwrite" if overwrite else "render"
    _append_generation_event(storage, {
        "mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episodes),
        "viewpoint_count": len(rendered),
        "rendered": len(rendered) - len(skipped),
        "skipped": len(skipped),
        "recovery_count": len(recovery),
        "render_failures": len(render_failures),
        "generation_stats": generation_stats,
    })
    return {"dataset_root": str(storage.root), "dataset_id": config["storage"]["dataset_id"], "episode_count": len(episodes), "viewpoints": len(rendered), "rendered": len(rendered) - len(skipped), "skipped": len(skipped), "recovery": recovery, "render_failures": render_failures, "manifest": str(manifest_path), "details": details, "generation_stats": generation_stats}


def _tracked_worktree_clean(repo_root: Path) -> bool:
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(repo_root), text=True, capture_output=True, check=True)
    return not result.stdout.strip()


def _fingerprint(path_value: Any) -> Mapping[str, Any]:
    path = Path(str(path_value))
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _resources_lock(config: Mapping[str, Any], repo_root: Path) -> Mapping[str, Any]:
    scenes = load_scene_registry(config["registries"]["scenes_path"], str(repo_root))
    dry = load_dry_audio_registry(config["registries"]["dry_audio_path"], str(repo_root))
    scene_lock = {}
    for scene_id, entry in scenes.items():
        item = {"scene_id": scene_id}
        for key in ("scene_asset", "navmesh", "semantic_info", "stage_config"):
            if entry.get(key):
                item[key] = _fingerprint(entry[key])
        scene_lock[scene_id] = item
    return {"scenes": scene_lock, "dry_audio": {audio_id: {"audio_id": audio_id, "path": entry["path"], "sha256": entry["sha256"]} for audio_id, entry in dry.items()}}


def _storage_summary(storage: DatasetStorage) -> Mapping[str, Any]:
    files, counters = [], {"wav_count": 0, "wav_bytes": 0, "rir_count": 0, "rir_bytes": 0, "manifest_bytes": 0, "metadata_report_log_bytes": 0}
    for path in sorted(storage.root.rglob("*")):
        if not path.is_file() or path.name == "_SUCCESS":
            continue
        relative, size = path.relative_to(storage.root), path.stat().st_size
        if path.suffix == ".wav": counters["wav_count"] += 1; counters["wav_bytes"] += size
        elif path.suffix == ".npz": counters["rir_count"] += 1; counters["rir_bytes"] += size
        elif relative.parts and relative.parts[0] == "manifests": counters["manifest_bytes"] += size
        else: counters["metadata_report_log_bytes"] += size
        files.append({"path": str(relative), "bytes": size})
    counters["measured_bytes_before_storage_summary_finalize"] = sum(row["bytes"] for row in files)
    counters["files"] = files
    return counters


def final_filesystem_bytes(storage: DatasetStorage) -> int:
    """Measure the finalized tree once, without rewriting its summary."""

    return sum(path.stat().st_size for path in storage.root.rglob("*") if path.is_file())


def finalize_dataset(config_path: str, validation: Mapping[str, Any]) -> Dict[str, Any]:
    config = load_resolved_config(config_path)
    repo_root = Path(config["_repo_root"])
    storage = _storage_for_config(config)
    storage.ensure_writable()
    if validation.get("status") != "PASS":
        raise RuntimeError("cannot finalize a failed dataset")
    if not _tracked_worktree_clean(repo_root):
        raise StorageError("generation-relevant tracked working tree is not clean")
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    storage.atomic_write_text(storage.path("config_resolved.yaml"), yaml.safe_dump(public_config, sort_keys=True))
    config_sha = hashlib.sha256(storage.path("config_resolved.yaml").read_bytes()).hexdigest()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    identity = {"dataset_id": config["storage"]["dataset_id"], "schema_version": config["experiment"]["schema_version"], "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit, "config_sha256": config_sha, "global_seed": config["experiment"]["global_seed"], "habitat_version": getattr(__import__("habitat_sim"), "__version__", "unknown"), "scene_registry_sha256": sha256_file(repo_root / config["registries"]["scenes_path"]), "dry_audio_registry_sha256": sha256_file(repo_root / config["registries"]["dry_audio_path"])}
    storage.atomic_write_json(storage.path("identity.json"), identity)
    storage.atomic_write_json(storage.path("resources.lock.json"), _resources_lock(config, repo_root))
    storage.atomic_write_json(storage.path("reports", "generation_summary.json"), {"status": "PASS", "viewpoints": validation.get("viewpoint_count", 0), "dataset_id": config["storage"]["dataset_id"]})
    storage.atomic_write_json(storage.path("reports", "validation.json"), validation)
    storage.atomic_write_json(storage.path("reports", "storage_summary.json"), _storage_summary(storage))
    storage.atomic_write_text(storage.path("logs", "validation.log"), "M2.1 validation PASS\n")
    storage.atomic_write_text(storage.path("logs", "failures.jsonl"), "")
    storage.atomic_write_text(storage.success_path, SUCCESS_TEXT)
    return {"dataset_root": str(storage.root), "success": str(storage.success_path), "git_commit": git_commit, "final_filesystem_bytes": final_filesystem_bytes(storage)}


def run_v0(config_path: str, resume: bool = False, overwrite: bool = False) -> Dict[str, Any]:
    from active_audition.cli import plan, precheck
    precheck_result = precheck(config_path)
    plan_result = plan(config_path)
    render_result = render_dataset(config_path, resume=resume, overwrite=overwrite)
    validation_result = validate_dataset(render_result["dataset_root"], load_resolved_config(config_path), require_success=False)
    finalize_result = finalize_dataset(config_path, validation_result)
    return {"precheck": precheck_result, "plan": plan_result, "render": render_result, "validation": validation_result, "finalize": finalize_result}
