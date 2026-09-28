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
    v1_command = subparsers.add_parser(
        "v1",
        help="Active-ASR V1.1 A0-A3 gated commands",
    )
    v1_subparsers = v1_command.add_subparsers(dest="v1_command", required=True)
    v1_validate = v1_subparsers.add_parser("validate", aliases=["validate-contract"])
    v1_validate.add_argument("--config", required=True)
    v1_hash = v1_subparsers.add_parser("hash", aliases=["contract-hash"])
    v1_hash.add_argument("--config", required=True)
    v1_metric_validate = v1_subparsers.add_parser("validate-metric")
    v1_metric_validate.add_argument("--config", required=True)
    v1_metric_hash = v1_subparsers.add_parser("hash-metric")
    v1_metric_hash.add_argument("--config", required=True)
    v1_oracle_validate = v1_subparsers.add_parser("validate-oracle")
    v1_oracle_validate.add_argument("--config", required=True)
    v1_oracle_hash = v1_subparsers.add_parser("hash-oracle")
    v1_oracle_hash.add_argument("--config", required=True)
    v1_asr_validate = v1_subparsers.add_parser("validate-asr")
    v1_asr_validate.add_argument("--config", required=True)
    v1_asr_validate.add_argument("--require-frozen", action="store_true")
    v1_asr_hash = v1_subparsers.add_parser("hash-asr")
    v1_asr_hash.add_argument("--config", required=True)
    v1_asr_prepare = v1_subparsers.add_parser("prepare-asr-freeze")
    v1_asr_prepare.add_argument("--config", required=True)
    v1_asr_prepare.add_argument("--sources", default="configs/active_audition/v1/sources.yaml")
    v1_asr_prepare.add_argument("--registry-dir", default="registries/active_asr_a3")
    v1_asr_prepare.add_argument("--dependency-lock", default="registries/active_asr_a3/asr_environment_lock.txt")
    v1_asr_rirs = v1_subparsers.add_parser("prepare-asr-rirs")
    v1_asr_rirs.add_argument("--config", required=True)
    v1_asr_rirs.add_argument("--runtime-config", default="configs/active_audition/v0_replica_debug.yaml")
    v1_asr_rirs.add_argument("--output-dir", default="runs/active_asr_v1/a3_frozen_rirs")
    v1_asr_qualify = v1_subparsers.add_parser("qualify-asr")
    v1_asr_qualify.add_argument("--config", required=True)
    v1_asr_qualify.add_argument("--rir-lock", required=True)
    v1_asr_qualify.add_argument("--output-dir", default="runs/active_asr_v1/a3_qualification")
    v1_g6_attribute = v1_subparsers.add_parser("attribute-g6")
    v1_g6_attribute.add_argument("--config", required=True)
    v1_g6_attribute.add_argument("--metric-contract", default="configs/active_audition/v1/metric_contract.yaml")
    v1_g6_attribute.add_argument("--rir-lock", required=True)
    v1_g6_attribute.add_argument("--g6-artifact", required=True)
    v1_g6_attribute.add_argument("--output-dir", default="runs/active_asr_v1/a3_g6_failure_attribution_v1")
    v1_oracle_qualify = v1_subparsers.add_parser("qualify-oracle-alignment")
    v1_oracle_qualify.add_argument("--config", required=True)
    v1_oracle_qualify.add_argument(
        "--runtime-config",
        default="configs/active_audition/v0_replica_debug.yaml",
        help="Existing legacy live-runtime config used by the native16 qualification",
    )
    v1_oracle_qualify.add_argument(
        "--historical-run-dir",
        default="runs/active_asr_v1/a2_failure_attribution_run3",
        help="Read-only historical v2 evidence source; it is never overwritten",
    )
    v1_oracle_qualify.add_argument("--output-dir", default="runs/active_asr_v1/a2_native16_oracle_alignment")
    v1_qualify = v1_subparsers.add_parser("qualify-physics")
    v1_qualify.add_argument("--metric-contract", required=True)
    v1_qualify.add_argument(
        "--runtime-config",
        default="configs/active_audition/v0_replica_debug.yaml",
        help="Existing legacy live-runtime config used by the A2 technical smoke",
    )
    v1_qualify.add_argument("--scene-id", default="replica.office_0")
    v1_qualify.add_argument("--output-dir", default="runs/active_asr_v1/a2_physics_qualification")
    v1_sample_rate = v1_subparsers.add_parser("calibrate-sample-rate")
    v1_sample_rate.add_argument("--metric-contract", required=True)
    v1_sample_rate.add_argument(
        "--runtime-config",
        default="configs/active_audition/v0_replica_debug.yaml",
        help="Existing legacy live-runtime config used by the controlled shoebox calibration",
    )
    v1_sample_rate.add_argument(
        "--formal-sample-rate-artifact",
        default="runs/active_asr_v1/a2_failure_attribution_run3/sample_rate_ab.json",
        help="Preserved formal v2 sample_rate_ab.json used for read-only offset attribution",
    )
    v1_sample_rate.add_argument("--output-dir", default="runs/active_asr_v1/a2_sample_rate_blocker_calibration")
    v1_audit = v1_subparsers.add_parser("audit-runtime")
    v1_audit.add_argument("--config", required=True, help="Active-ASR V1.1 A0 contract")
    v1_audit.add_argument(
        "--runtime-config",
        default="configs/active_audition/v0_replica_debug.yaml",
        help="Existing legacy live-runtime config used by create_scene_simulator",
    )
    v1_audit.add_argument("--scene-id", default="replica.office_0")
    v1_audit.add_argument("--output-dir", default="runs/active_asr_v1/a1_runtime_audit")
    v1_materials_audit = v1_subparsers.add_parser("audit-real-scene-materials")
    v1_materials_audit.add_argument(
        "--runtime-config", default="configs/active_audition/v0_replica_debug.yaml"
    )
    v1_materials_audit.add_argument("--scene-id", default="replica.office_0")
    v1_materials_audit.add_argument(
        "--output-dir", default="runs/active_asr_v1/a3_real_scene_domain_attribution_v1"
    )
    v1_freeze_real_scene = v1_subparsers.add_parser("freeze-real-scene-manifest")
    v1_freeze_real_scene.add_argument(
        "--runtime-config", default="configs/active_audition/v0_replica_debug.yaml"
    )
    v1_freeze_real_scene.add_argument("--scene-id", default="replica.office_0")
    v1_freeze_real_scene.add_argument(
        "--manifest", default="registries/active_asr_a3/replica_office0_domain_diagnostic_manifest_v1.json"
    )
    v1_real_scene = v1_subparsers.add_parser("attribute-real-scene-g6")
    v1_real_scene.add_argument("--config", required=True)
    v1_real_scene.add_argument(
        "--metric-contract", default="configs/active_audition/v1/metric_contract.yaml"
    )
    v1_real_scene.add_argument(
        "--manifest", default="registries/active_asr_a3/replica_office0_domain_diagnostic_manifest_v1.json"
    )
    v1_real_scene.add_argument(
        "--materials-audit", default="runs/active_asr_v1/a3_g6_failure_attribution_v1/real_scene_materials_audit/real_scene_materials_audit.json"
    )
    v1_real_scene.add_argument(
        "--rir-lock", required=True,
        help="Renderer-only Replica RIR lock produced in the ss environment",
    )
    v1_real_scene.add_argument(
        "--runtime-config", default="configs/active_audition/v0_replica_debug.yaml"
    )
    v1_real_scene.add_argument(
        "--output-dir", default="runs/active_asr_v1/a3_real_scene_domain_attribution_v1"
    )
    v1_render_real_scene = v1_subparsers.add_parser("render-real-scene-rirs")
    v1_render_real_scene.add_argument("--config", required=True)
    v1_render_real_scene.add_argument(
        "--manifest", default="registries/active_asr_a3/replica_office0_domain_diagnostic_manifest_v1.json"
    )
    v1_render_real_scene.add_argument("--materials-audit", required=True)
    v1_render_real_scene.add_argument(
        "--runtime-config", default="configs/active_audition/v0_replica_debug.yaml"
    )
    v1_render_real_scene.add_argument("--output-dir", required=True)
    v1_revise_real_scene = v1_subparsers.add_parser("revise-real-scene-g6-summary")
    v1_revise_real_scene.add_argument("--output-dir", required=True)
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
    if args.command == "v1":
        if args.v1_command in ("validate-asr", "hash-asr", "prepare-asr-freeze", "prepare-asr-rirs", "qualify-asr", "attribute-g6"):
            from active_audition.asr.contract import asr_contract_sha256, load_asr_contract

            if args.v1_command == "qualify-asr":
                from active_audition.asr.run_qualification import run_a3_qualification

                result = run_a3_qualification(
                    args.config,
                    args.rir_lock,
                    args.output_dir,
                )
            elif args.v1_command == "attribute-g6":
                from active_audition.asr.g6_failure_attribution import run_g6_failure_attribution

                result = run_g6_failure_attribution(
                    args.config,
                    args.metric_contract,
                    args.rir_lock,
                    args.g6_artifact,
                    args.output_dir,
                )
            elif args.v1_command == "prepare-asr-rirs":
                from active_audition.asr.rir_bridge import render_a3_qualification_rirs

                result = render_a3_qualification_rirs(
                    args.config,
                    args.output_dir,
                    args.runtime_config,
                )
            elif args.v1_command == "prepare-asr-freeze":
                from active_audition.asr.qualification import prepare_a3_freeze_material

                result = prepare_a3_freeze_material(
                    args.config,
                    args.sources,
                    args.registry_dir,
                    args.dependency_lock,
                )
            else:
                contract = load_asr_contract(
                    args.config,
                    require_frozen=bool(getattr(args, "require_frozen", False)),
                )
                result = {
                    "status": "PASS",
                    "gate": contract["contract"]["gate"],
                    "state": contract["contract"]["state"],
                    "schema_version": contract["contract"]["version"],
                    "asr_contract_sha256": asr_contract_sha256(contract),
                }
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return
        if args.v1_command in ("audit-real-scene-materials", "freeze-real-scene-manifest", "render-real-scene-rirs", "attribute-real-scene-g6", "revise-real-scene-g6-summary"):
            from active_audition.asr.real_scene_domain_attribution import (
                freeze_domain_manifest,
                render_real_scene_rirs,
                revise_real_scene_domain_attribution_summary,
                run_materials_audit,
                run_real_scene_domain_attribution,
            )

            if args.v1_command == "audit-real-scene-materials":
                result = run_materials_audit(args.output_dir, args.runtime_config, args.scene_id)
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return
            if args.v1_command == "freeze-real-scene-manifest":
                result = freeze_domain_manifest(args.manifest, args.runtime_config, args.scene_id)
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return
            if args.v1_command == "render-real-scene-rirs":
                result = render_real_scene_rirs(
                    args.config,
                    args.manifest,
                    args.materials_audit,
                    args.output_dir,
                    args.runtime_config,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return
            if args.v1_command == "revise-real-scene-g6-summary":
                result = revise_real_scene_domain_attribution_summary(args.output_dir)
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return
            result = run_real_scene_domain_attribution(
                args.config,
                args.metric_contract,
                args.manifest,
                args.materials_audit,
                args.rir_lock,
                args.output_dir,
                args.runtime_config,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return
        if args.v1_command in ("validate-oracle", "hash-oracle", "qualify-oracle-alignment"):
            from active_audition.receiver.oracle_alignment import (
                load_oracle_contract,
                oracle_contract_sha256,
                run_oracle_alignment,
            )

            if args.v1_command == "qualify-oracle-alignment":
                result = run_oracle_alignment(
                    args.config,
                    args.output_dir,
                    args.runtime_config,
                    args.historical_run_dir,
                )
            else:
                contract = load_oracle_contract(args.config)
                result = {
                    "status": "PASS",
                    "gate": contract["contract"]["gate"],
                    "schema_version": contract["contract"]["version"],
                    "oracle_contract_sha256": oracle_contract_sha256(contract),
                    "parent_metric_contract_sha256": contract["contract"]["parent_metric_contract_sha256"],
                }
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return
        if args.v1_command in ("validate-metric", "hash-metric", "qualify-physics", "calibrate-sample-rate"):
            from active_audition.receiver.qualification import (
                load_metric_contract,
                metric_contract_sha256,
                run_a2_sample_rate_blocker_calibration,
                run_a2_qualification,
            )

            if args.v1_command == "qualify-physics":
                result = run_a2_qualification(
                    args.metric_contract,
                    args.output_dir,
                    args.runtime_config,
                    args.scene_id,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                if result["status"] != "PASS":
                    raise SystemExit(1)
                return
            if args.v1_command == "calibrate-sample-rate":
                result = run_a2_sample_rate_blocker_calibration(
                    args.metric_contract,
                    args.output_dir,
                    args.runtime_config,
                    args.formal_sample_rate_artifact,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return
            metric = load_metric_contract(args.config)
            if args.v1_command == "hash-metric":
                result = {"metric_contract_sha256": metric_contract_sha256(metric)}
            else:
                result = {
                    "status": "PASS",
                    "gate": metric["contract"]["gate"],
                    "schema_version": metric["contract"]["version"],
                    "metric_contract_sha256": metric_contract_sha256(metric),
                }
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return
        if args.v1_command == "audit-runtime":
            from active_audition.receiver.audit import run_runtime_audit

            result = run_runtime_audit(args.config, args.output_dir, args.runtime_config, args.scene_id)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            if result["status"] == "BLOCKED":
                raise SystemExit(1)
            return
        from active_audition.v1.cli import _result

        result = _result(args.config)
        if args.v1_command in ("hash", "contract-hash"):
            result = {"contract_sha256": result["contract_sha256"]}
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "precheck":
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
