"""Pipeline V0 precheck, planning, M2 rendering and validation entry points."""

import argparse
import json
from pathlib import Path

from active_audition.config.loader import load_resolved_config
from active_audition.data.catalog import load_dry_audio_registry, load_scene_registry
from active_audition.data.manifest import write_plan_manifests
from active_audition.data.storage import DatasetStorage
from active_audition.acoustics.precomputed_rir import PrecomputedRIRBackend


def precheck(config_path: str) -> dict:
    config = load_resolved_config(config_path)
    repo_root = Path(config["_repo_root"])
    if config["acoustics"].get("backend") == "soundspaces_precomputed":
        backend = PrecomputedRIRBackend(str(Path(config["_repo_root"]) / config["acoustics"]["rir_root"]), config["navigation"].get("heading_mapping_version", "dataset-native-v1"), str(Path(config["_repo_root"]) / config["acoustics"]["metadata_root"]))
        if not backend.available:
            return {"status": "BLOCKED_ASSET_NOT_AVAILABLE", "rir_root": config["acoustics"]["rir_root"], "backend": "soundspaces_precomputed", "message": "official precomputed RIR root is not available; no download or RLRA fallback was attempted"}
        return {"status": "PASS", "backend": "soundspaces_precomputed", "rir_root": str(backend.root), "scenes": backend.list_scenes(), "headings": {scene: backend.list_headings(scene) for scene in backend.list_scenes()}, "nodes": {scene: backend.list_nodes(scene) for scene in backend.list_scenes()}}
    scenes = load_scene_registry(config["registries"]["scenes_path"], str(repo_root))
    dry_audio = load_dry_audio_registry(config["registries"]["dry_audio_path"], str(repo_root))
    from active_audition.scene.simulator import create_scene_simulator
    from active_audition.navigation.pathfinder import PathFinderAdapter
    from active_audition.scene.episode import fixed_golden_episode
    with create_scene_simulator(config) as context:
        from active_audition.navigation.pathfinder import PathFinderAdapter
        from active_audition.scene.episode import generate_episodes
        from active_audition.navigation.candidates import generate_candidates
        from active_audition.scene.simulator import create_scene_simulator
        pathfinder = PathFinderAdapter(context.pathfinder)
        episode = fixed_golden_episode(config, pathfinder)
        source_geodesic = pathfinder.geodesic_distance(
            episode.listener_initial.base_position_world,
            config["golden"]["source_anchor_base_position_world"],
        )
        pathfinder_loaded = bool(pathfinder.is_loaded)
        source_geodesic = None if source_geodesic is None else float(source_geodesic)
    return {
        "experiment": config["experiment"],
        "scene_ids": sorted(scenes),
        "dry_audio_ids": sorted(dry_audio),
        "golden": config["golden"],
        "acoustics": config["acoustics"],
        "golden_navigation": {
            "pathfinder_loaded": pathfinder_loaded,
            "source_geodesic_m": source_geodesic,
        },
    }


def plan(config_path: str, output_root: str = "") -> dict:
    config = load_resolved_config(config_path)
    if config["acoustics"].get("backend") == "soundspaces_precomputed":
        from active_audition.pipeline.precomputed import plan_precomputed
        return plan_precomputed(config_path)
    repo_root = Path(config["_repo_root"])
    dry_audio = load_dry_audio_registry(config["registries"]["dry_audio_path"], str(repo_root))
    storage = DatasetStorage(output_root) if output_root else DatasetStorage.from_config(str(repo_root), config)
    from active_audition.scene.simulator import create_scene_simulator
    from active_audition.navigation.pathfinder import PathFinderAdapter
    from active_audition.scene.episode import generate_episodes
    from active_audition.navigation.candidates import generate_candidates
    sampling_diagnostics = {
        "accepted_episode_count": 0,
        "total_sampling_attempts": 0,
        "source_too_close_rejections": 0,
        "source_too_far_rejections": 0,
        "unreachable_rejections": 0,
        "identical_anchor_rejections": 0,
        "sampling_failure_count": 0,
    } if config["navigation"]["thresholds_enabled"] else None
    with create_scene_simulator(config) as context:
        pathfinder = PathFinderAdapter(context.pathfinder)
        episodes = generate_episodes(config, pathfinder, dry_audio, sampling_diagnostics=sampling_diagnostics)
        candidates = []
        for episode in episodes:
            candidates.extend(generate_candidates(episode.episode_id, episode.listener_initial, pathfinder, config))
    if config["episode"]["mode"] in ("fixed", "fixed_or_sampled") and not all(candidate.valid for candidate in candidates):
        invalid = [candidate.candidate_id for candidate in candidates if not candidate.valid]
        raise RuntimeError("Golden candidate structural failure: {}".format(invalid))
    paths = write_plan_manifests(str(storage.root), episodes, candidates)
    if sampling_diagnostics is not None:
        sampling_diagnostics["accepted_episode_count"] = len(episodes)
        storage.atomic_write_json(storage.path("logs", "sampling_diagnostics.json"), sampling_diagnostics)
    invalid_reasons = {}
    for candidate in candidates:
        if not candidate.valid:
            invalid_reasons[candidate.invalid_reason] = invalid_reasons.get(candidate.invalid_reason, 0) + 1
    return {
        "episode_id": episodes[0].episode_id if len(episodes) == 1 else None,
        "episode_count": len(episodes),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "valid_candidates": sum(candidate.valid for candidate in candidates),
        "invalid_candidates": sum(not candidate.valid for candidate in candidates),
        "invalid_reason_counts": invalid_reasons,
        "manifest_paths": paths,
        "wav_rendered": False,
        "rir_rendered": False,
        "dataset_finalized": False,
        "sampling_diagnostics": sampling_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m active_audition.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("precheck")
    command.add_argument("--config", required=True)
    plan_command = subparsers.add_parser("plan")
    plan_command.add_argument("--config", required=True)
    plan_command.add_argument("--output-root", default="")
    render_command = subparsers.add_parser("render")
    render_command.add_argument("--config", required=True)
    render_command.add_argument("--resume", action="store_true")
    render_command.add_argument("--overwrite", action="store_true")
    validate_command = subparsers.add_parser("validate")
    validate_command.add_argument("--dataset", required=True)
    validate_command.add_argument("--config", default="configs/active_audition/v0_replica_debug.yaml")
    gate_command = subparsers.add_parser("channel-gate")
    gate_command.add_argument("--config", required=True)
    run_command = subparsers.add_parser("run-v0")
    run_command.add_argument("--config", required=True)
    run_command.add_argument("--resume", action="store_true")
    run_command.add_argument("--overwrite", action="store_true")
    finalize_command = subparsers.add_parser("finalize")
    finalize_command.add_argument("--config", required=True)
    qc_command = subparsers.add_parser("qc")
    qc_command.add_argument("--dataset", required=True)
    qc_command.add_argument("--config", required=True)
    qc_command.add_argument("--run-id", required=True)
    qc_command.add_argument("--topdown", action="store_true")
    qc_command.add_argument("--evidence", default="")
    args = parser.parse_args()
    if args.command == "precheck":
        print(json.dumps(precheck(args.config), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "plan":
        print(
            json.dumps(
                plan(args.config, args.output_root),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "render":
        from active_audition.pipeline.v0 import render_dataset
        print(json.dumps(render_dataset(args.config, args.resume, args.overwrite), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "validate":
        from active_audition.data.validation import validate_dataset
        config = load_resolved_config(args.config)
        result = validate_dataset(args.dataset, config, require_success=False)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if result["status"] != "PASS":
            raise SystemExit(1)
    elif args.command == "channel-gate":
        config = load_resolved_config(args.config)
        from active_audition.scene.episode import fixed_golden_episode
        from active_audition.navigation.pathfinder import PathFinderAdapter
        from active_audition.scene.simulator import create_scene_simulator
        from active_audition.acoustics.rir import run_channel_order_gate
        with create_scene_simulator(config) as context:
            episode = fixed_golden_episode(config, PathFinderAdapter(context.pathfinder))
            result = run_channel_order_gate(
                context,
                episode.listener_initial,
                [-0.422410, 0.531130, 1.841960],
                [1.577590, 0.531130, 1.841960],
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "run-v0":
        from active_audition.pipeline.v0 import run_v0
        print(json.dumps(run_v0(args.config, args.resume, args.overwrite), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "finalize":
        config = load_resolved_config(args.config)
        if config["acoustics"].get("backend") == "soundspaces_precomputed":
            from active_audition.pipeline.precomputed import finalize_precomputed
            from active_audition.data.validation import validate_dataset
            dataset_root = str(Path(config["_repo_root"]) / "datasets/active_audition_v05" / config["storage"]["dataset_id"])
            validation = validate_dataset(dataset_root, config, require_success=False)
            print(json.dumps(finalize_precomputed(args.config, validation), ensure_ascii=False, indent=2, sort_keys=True))
            return
        from active_audition.pipeline.v0 import finalize_dataset
        from active_audition.data.validation import validate_dataset
        dataset_root = str(DatasetStorage.from_config(config["_repo_root"], config).root)
        validation = validate_dataset(dataset_root, config, require_success=False)
        print(json.dumps(finalize_dataset(args.config, validation), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "qc":
        from active_audition.evaluation.qc import run_qc
        print(json.dumps(run_qc(args.dataset, args.config, args.run_id, args.topdown, args.evidence or None), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
