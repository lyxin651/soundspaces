"""M0/M1 command-line precheck and planning entry points."""

import argparse
import json
from pathlib import Path

from active_audition.config.loader import load_resolved_config
from active_audition.data.catalog import load_dry_audio_registry, load_scene_registry
from active_audition.data.manifest import write_plan_manifests
from active_audition.navigation.candidates import generate_candidates
from active_audition.navigation.pathfinder import PathFinderAdapter
from active_audition.scene.episode import fixed_golden_episode
from active_audition.scene.simulator import create_scene_simulator


def precheck(config_path: str) -> dict:
    config = load_resolved_config(config_path)
    repo_root = Path(config["_repo_root"])
    scenes = load_scene_registry(config["registries"]["scenes_path"], str(repo_root))
    dry_audio = load_dry_audio_registry(config["registries"]["dry_audio_path"], str(repo_root))
    with create_scene_simulator(config) as context:
        pathfinder = PathFinderAdapter(context.pathfinder)
        episode = fixed_golden_episode(config, pathfinder)
        source_geodesic = pathfinder.geodesic_distance(
            episode.listener_initial.base_position_world,
            config["golden"]["source_anchor_base_position_world"],
        )
    return {
        "experiment": config["experiment"],
        "scene_ids": sorted(scenes),
        "dry_audio_ids": sorted(dry_audio),
        "golden": config["golden"],
        "acoustics": config["acoustics"],
        "golden_navigation": {
            "pathfinder_loaded": pathfinder.is_loaded,
            "source_geodesic_m": source_geodesic,
        },
    }


def plan(config_path: str, output_root: str = "") -> dict:
    config = load_resolved_config(config_path)
    repo_root = Path(config["_repo_root"])
    if not output_root:
        output_root = str(
            repo_root
            / "datasets/active_audition_v0/aa_v0_replica_debug_001/incomplete"
        )
    with create_scene_simulator(config) as context:
        pathfinder = PathFinderAdapter(context.pathfinder)
        episode = fixed_golden_episode(config, pathfinder)
        candidates = generate_candidates(
            episode.episode_id, episode.listener_initial, pathfinder, config
        )
    if not all(candidate.valid for candidate in candidates):
        invalid = [candidate.candidate_id for candidate in candidates if not candidate.valid]
        raise RuntimeError("Golden candidate structural failure: {}".format(invalid))
    paths = write_plan_manifests(output_root, episode, candidates)
    return {
        "episode_id": episode.episode_id,
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "valid_candidates": sum(candidate.valid for candidate in candidates),
        "manifest_paths": paths,
        "wav_rendered": False,
        "rir_rendered": False,
        "dataset_finalized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m active_audition.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("precheck")
    command.add_argument("--config", required=True)
    plan_command = subparsers.add_parser("plan")
    plan_command.add_argument("--config", required=True)
    plan_command.add_argument("--output-root", default="")
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


if __name__ == "__main__":
    main()
