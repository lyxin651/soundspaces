#!/usr/bin/env python3
"""Prepare the first O1 exploratory block set without acoustic/ASR results.

This command is intentionally run with the ``ss`` environment because it
performs one deterministic navmesh/runtime probe and source-free sampling.
It selects only technical metadata, then writes an independent O1 manifest;
it never renders RIR, reads WER, or chooses a source from an acoustic result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from active_audition.a4.contract import contract_sha256, load_contract
from active_audition.a4.identity import canonical_json, identity_sha256, stable_id
from active_audition.a4.pose_sampler import annotate_geometry_legality, make_sampler_context, sample_source_free
from active_audition.a4.smoke_preparation import build_smoke_block_payload, engineering_candidate_contract
from active_audition.a4.smoke_manifest import EngineeringSmokeManifest
from active_audition.navigation.pathfinder import PathFinderAdapter
from active_audition.o1.manifest import FORBIDDEN_RESULT_INPUTS, O1ExploratoryManifest


ROOT = Path(__file__).resolve().parents[1]
SCENE = "replica.apartment_2"
POINT_SEED = 20260930
POINT_COUNT = 160
INITIAL_INDICES = (0, 1, 2, 3)
SOURCE_INDICES = ((81, 98), (80, 97), (81, 98), (80, 97))
EXCLUDED_SCENES = frozenset({"replica.apartment_0", "replica.apartment_1", "replica.office_0"})
EXCLUDED_NOISE = frozenset({
    "noise/free-sound/noise-free-sound-0002",
    "noise/free-sound/noise-free-sound-0020",
    "noise/free-sound/noise-free-sound-0048",
    "noise/free-sound/noise-free-sound-0270",
})


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_sha(path: Path) -> tuple[str, int, int]:
    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    value = np.ascontiguousarray(samples, dtype=np.dtype("<f4"))
    if int(rate) != 16000 or value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise RuntimeError("source technical decode gate failed: {}".format(path))
    return hashlib.sha256(value.tobytes()).hexdigest(), int(rate), int(value.size)


def _registry_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_path(relative: str, kind: str) -> Path:
    return ROOT / "data/active_asr_a3/corpora" / ("LibriSpeech" if kind == "speech" else "musan") / relative


def _verify_row(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
    path = _source_path(row["relative_source_path"], kind)
    if not path.is_file():
        raise RuntimeError("selected source is missing: {}".format(path))
    if _sha_file(path) != row["source_file_sha256"]:
        raise RuntimeError("selected source file SHA mismatch: {}".format(path))
    decoded_sha, rate, samples = _decoded_sha(path)
    if decoded_sha != row["decoded_waveform_sha256"] or rate != int(row.get("decode_sample_rate_hz", row.get("sample_rate_hz", 16000))) or samples != int(row["samples"]):
        raise RuntimeError("selected decoded source provenance mismatch: {}".format(path))
    return dict(row)


def _select_speech() -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    rows = _registry_rows(ROOT / "registries/active_asr_a3/librispeech.jsonl")
    a4 = json.loads((ROOT / "configs/active_audition/v1/a4_engineering_smoke_manifest.json").read_text(encoding="utf-8"))
    excluded = {item["speaker_id"] for block in a4["blocks"] for item in block["speech_sources"]}
    v2 = json.loads((ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_speech_manifest.json").read_text(encoding="utf-8"))
    excluded.update(item["speaker_id"] for item in v2["records"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        path = _source_path(row["relative_source_path"], "speech")
        if not (
            row.get("eligible") is True and row.get("complete_utterance") is True
            and row.get("source_sample_rate_hz") == 16000 and row.get("decode_sample_rate_hz") == 16000
            and 4.0 <= float(row["duration_sec"]) <= 15.0 and int(row["reference_word_count"]) >= 10
            and path.is_file() and row["speaker_id"] not in excluded
        ):
            continue
        grouped[row["speaker_id"]].append(row)
    speakers = sorted(speaker for speaker, values in grouped.items() if len(values) >= 4)
    if len(speakers) < 4:
        raise RuntimeError("fewer than four technical O1 speakers are available")
    selected_speakers = speakers[:4]
    selected = []
    for speaker in selected_speakers:
        utterances = sorted(grouped[speaker], key=lambda row: row["utterance_id"])[:4]
        selected.append([_verify_row(row, "speech") for row in utterances])
    return selected, {
        "selection_rule": "lexical first four speakers after frozen technical eligibility and A4/A3v2 speaker exclusion",
        "excluded_speaker_ids": sorted(excluded),
        "selected_speaker_ids": selected_speakers,
        "constraints": {"complete_utterance": True, "duration_sec": [4.0, 15.0], "reference_word_count_min": 10, "sample_rate_hz": 16000},
    }


def _select_noise() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _registry_rows(ROOT / "registries/active_asr_a3/musan_noise.jsonl")
    a4 = json.loads((ROOT / "configs/active_audition/v1/a4_engineering_smoke_manifest.json").read_text(encoding="utf-8"))
    required = max(int(block["noise_plan"]["required_parent_sample_count"]) for block in a4["blocks"])
    eligible = []
    for row in rows:
        path = _source_path(row["relative_source_path"], "noise")
        technical = row.get("technical_audit", {})
        if not (
            row.get("corpus") == "MUSAN" and row.get("subset") == "noise" and row.get("excluded") is False
            and technical.get("readable") is True and technical.get("too_short") is False
            and technical.get("extreme_silence") is False and row.get("sample_rate_hz") == 16000
            and int(row.get("samples", 0)) >= required and path.is_file()
            and row["parent_recording_id"] not in EXCLUDED_NOISE
        ):
            continue
        eligible.append(_verify_row(row, "noise"))
    eligible.sort(key=lambda row: row["parent_recording_id"])
    if len(eligible) < 4:
        raise RuntimeError("fewer than four technical O1 noise parents are available")
    selected = eligible[:4]
    return selected, {
        "selection_rule": "lexical first four MUSAN noise parents after technical eligibility and A4 parent exclusion",
        "excluded_parent_ids": sorted(EXCLUDED_NOISE),
        "required_parent_sample_count": required,
        "required_parent_duration_sec": required / 16000.0,
        "technical_eligible_count": len(eligible),
        "selected_parent_ids": [row["parent_recording_id"] for row in selected],
        "constraints": {"corpus": "MUSAN", "subset": "noise", "excluded": False, "sample_rate_hz": 16000, "no_resampling": True},
    }


def _scene_row() -> dict[str, Any]:
    inventory = json.loads((ROOT / "registries/active_asr_a3_v2/replica_scene_inventory.json").read_text(encoding="utf-8"))
    candidates = [
        row for row in inventory["scenes"]
        if row["scene_id"] not in EXCLUDED_SCENES and row.get("technical_eligibility") is True
    ]
    candidates.sort(key=lambda row: row["scene_id"])
    if not candidates or candidates[0]["scene_id"] != SCENE:
        raise RuntimeError("deterministic O1 scene policy did not select apartment_2")
    row = candidates[0]
    runtime = row["runtime_scene_load_smoke"]
    if runtime["status"] != "PASS" or runtime["effective_sample_rate_hz"] != 16000 or runtime["effective_channel_layout"] != "binaural" or runtime["materials_effective"] is not False:
        raise RuntimeError("O1 scene runtime compatibility gate failed")
    return row


def _scene_case(row: Mapping[str, Any], base: tuple[float, float, float], sensor: tuple[float, float, float], target: tuple[float, float, float], noise: tuple[float, float, float]) -> dict[str, Any]:
    assets = row["assets"]
    return {
        "scene_id": row["scene_id"],
        "scene_asset_sha256": assets["scene_asset"]["sha256"],
        "navmesh_sha256": assets["navmesh"]["sha256"],
        "stage_config_sha256": assets["stage_config"]["sha256"],
        "listener_base_position_world": list(base),
        "listener_sensor_position_world": list(sensor),
        "listener_yaw_deg": 0.0,
        "target_source_case": {"source_position_world": list(target)},
        "noise_source_case": {"source_position_world": list(noise)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/active_asr_v1")
    args = parser.parse_args()
    contract = load_contract(str(ROOT / "configs/active_audition/v1/a4_infrastructure_contract.yaml"), require_frozen=True)
    infrastructure_sha = contract_sha256(contract)
    scene_row = _scene_row()
    speech_blocks, speech_policy = _select_speech()
    noise_rows, noise_policy = _select_noise()
    candidate = engineering_candidate_contract()
    config = __import__("active_audition.config.loader", fromlist=["load_resolved_config"]).load_resolved_config(str(ROOT / "configs/active_audition/v0_replica_debug.yaml"))
    config["_repo_root"] = str(ROOT)
    config["scene"]["ids"] = [SCENE]
    config["acoustics"]["sample_rate_hz"] = 16000
    config["acoustics"]["materials_enabled"] = False
    habitat = ROOT / "data/scene_datasets/replica" / SCENE.split(".", 1)[1] / "habitat"
    scene_override = {"scene_asset": str(habitat / "mesh_semantic.ply"), "navmesh": str(habitat / "mesh_semantic.navmesh")}
    from active_audition.scene.simulator import create_scene_simulator
    with create_scene_simulator(config, scene_id=SCENE, acoustics_overrides={"sampleRate": 16000}, require_navmesh=True, load_semantic_mesh=False, scene_override=scene_override) as context:
        pathfinder = PathFinderAdapter(context.pathfinder)
        rng = np.random.default_rng(POINT_SEED)
        points = [pathfinder.sample_navigable_point(rng) for _ in range(POINT_COUNT)]
        blocks = []
        block_geometry = []
        for index, ((initial_index, (target_index, noise_index)), speech_rows, noise_row) in enumerate(zip(zip(INITIAL_INDICES, SOURCE_INDICES), speech_blocks, noise_rows), 1):
            initial = tuple(float(value) for value in points[initial_index])
            sensor = (initial[0], initial[1] + 1.5, initial[2])
            resources = {
                "scene_asset_sha256": scene_row["assets"]["scene_asset"]["sha256"],
                "navmesh_sha256": scene_row["assets"]["navmesh"]["sha256"],
                "stage_config_sha256": scene_row["assets"]["stage_config"]["sha256"],
            }
            context_record = make_sampler_context(SCENE, resources, initial, initial, "active-asr-a0-listener-sensor-offset-v1", sensor, 0.0, candidate)
            source_free = sample_source_free(context_record, candidate, pathfinder)
            if source_free.opportunity_constrained or len(source_free.selected_positions) != 6 or len(source_free.yaw_plans) != 48:
                raise RuntimeError("O1 block {} did not produce the required 6x8 source-free plan".format(index))
            target = tuple(float(value) for value in points[target_index])
            noise = tuple(float(value) for value in points[noise_index])
            target_sensor = (target[0], target[1] + 1.5, target[2])
            noise_sensor = (noise[0], noise[1] + 1.5, noise[2])
            case = _scene_case(scene_row, initial, sensor, target_sensor, noise_sensor)
            block = dict(build_smoke_block_payload(case, case["target_source_case"], case["noise_source_case"], speech_rows, noise_row, source_free, candidate))
            legal = sum(item["geometry_legality"] == "LEGAL" for item in block["poses"])
            if legal != 48:
                raise RuntimeError("O1 block {} has {} legal poses; deterministic first block set requires 48".format(index, legal))
            blocks.append(block)
            block_geometry.append({
                "block_index": index,
                "initial_point_index": initial_index,
                "target_point_index": target_index,
                "noise_point_index": noise_index,
                "block_id": block["block_record"]["block_id"],
                "geometry_id": block["geometry_record"]["geometry_id"],
                "sampler_context_id": block["sampler_context"]["sampler_context_id"],
                "selected_position_ids": [item["position_id"] for item in block["poses"][:6]],
                "pose_count": len(block["poses"]),
                "legal_pose_count": legal,
            })
    ledger_payload = {
        "schema_version": "active-asr-o1-consumed-scene-ledger-v1",
        "selection_policy": {
            "scene_rule": "lexical first technically eligible scene after A4/formal held-out/prior-ASR exclusions",
            "inputs": ["scene/resource loadability", "navmesh", "source-free local sampler", "source clearance", "runtime compatibility"],
            "forbidden": ["RIR", "energy", "DRR", "ASR", "WER", "Oracle", "movement_benefit"],
        },
        "scene_id": SCENE,
        "scene_resource_identities": {
            "scene_asset_sha256": scene_row["assets"]["scene_asset"]["sha256"],
            "navmesh_sha256": scene_row["assets"]["navmesh"]["sha256"],
            "stage_config_sha256": scene_row["assets"]["stage_config"]["sha256"],
        },
        "scene_inventory_identity": _sha_file(ROOT / "registries/active_asr_a3_v2/replica_scene_inventory.json"),
        "engineering_only": True,
        "o2_exclusion": {"excluded": True, "reason": "consumed by O1 exploratory landscape; never eligible for O2 held-out main set"},
        "block_geometry_selection": block_geometry,
        "speech_selection_policy": speech_policy,
        "noise_selection_policy": noise_policy,
        "selected_speakers": [group[0]["speaker_id"] for group in speech_blocks],
        "selected_noise_parents": [row["parent_recording_id"] for row in noise_rows],
        "result_dependent_selection": False,
    }
    ledger_id = stable_id("o1-consumed-scene-ledger", ledger_payload)
    ledger_payload["ledger_id"] = ledger_id
    ledger_payload["ledger_sha256"] = identity_sha256({key: value for key, value in ledger_payload.items() if key not in ("ledger_id", "ledger_sha256")})
    manifest = O1ExploratoryManifest(
        infrastructure_contract_sha256=infrastructure_sha,
        selection_policy={
            "algorithm": "active-asr-o1-technical-single-scene-selection-v1",
            "scene_rule": "lexical first technically eligible scene after explicit exclusions",
            "point_generation": {"seed": POINT_SEED, "count": POINT_COUNT, "source_free_block_indices": list(INITIAL_INDICES)},
            "source_geometry_rule": "fixed technical point indices before any RIR/ASR/WER result",
            "speech_rule": speech_policy,
            "noise_rule": noise_policy,
            "forbidden_inputs": list(FORBIDDEN_RESULT_INPUTS),
        },
        blocks=tuple(blocks),
        o2_exclusion={"excluded": True, "scene_id": SCENE, "ledger_id": ledger_id, "reason": "O1 exploratory consumption; not held-out O2"},
        consumed_scene_ledger_id=ledger_id,
    )
    run_root = args.output_root / "o1_replica_apartment_2_{}".format(manifest.manifest_sha256[:12])
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "o1_exploratory_manifest.json").write_text(canonical_json(manifest.to_payload()) + "\n", encoding="utf-8")
    (run_root / "o1_consumed_ledger.json").write_text(canonical_json(ledger_payload) + "\n", encoding="utf-8")
    preparation = {
        "schema_version": "active-asr-o1-preparation-v1",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.manifest_sha256,
        "infrastructure_contract_sha256": infrastructure_sha,
        "scene_id": SCENE,
        "blocks": 4,
        "episodes": 16,
        "poses": 192,
        "expected_mixtures": 768,
        "expected_asr_records": 2304,
        "engineering_only": True,
        "o2_excluded": True,
        "result_dependent_selection": False,
        "run_root": str(run_root),
    }
    (run_root / "o1_preparation_summary.json").write_text(canonical_json(preparation) + "\n", encoding="utf-8")
    print(json.dumps(preparation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
