"""M0 command-line precheck; rendering commands are added in later milestones."""

import argparse
import json
from pathlib import Path

from active_audition.config.loader import load_resolved_config
from active_audition.data.catalog import load_dry_audio_registry, load_scene_registry


def precheck(config_path: str) -> dict:
    config = load_resolved_config(config_path)
    repo_root = Path(config["_repo_root"])
    scenes = load_scene_registry(config["registries"]["scenes_path"], str(repo_root))
    dry_audio = load_dry_audio_registry(config["registries"]["dry_audio_path"], str(repo_root))
    return {
        "experiment": config["experiment"],
        "scene_ids": sorted(scenes),
        "dry_audio_ids": sorted(dry_audio),
        "golden": config["golden"],
        "acoustics": config["acoustics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m active_audition.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("precheck")
    command.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "precheck":
        print(json.dumps(precheck(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
