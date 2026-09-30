#!/usr/bin/env python3
"""Build the audited O1 scientific manifest without result-dependent selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from active_audition.a4.identity import canonical_json, identity_sha256, stable_id
from active_audition.a4.smoke_preparation import build_smoke_block_payload
from active_audition.o1.manifest import O1ExploratoryManifest, O1_MANIFEST_SCHEMA_VERSION_V2
from active_audition.o1.noise_audit import O1NoiseParentAuditRecord
from active_audition.o1.replacement_audit import O1FinalizedReplacementAuditBatch
from active_audition.o1.manifest import _sampler_output
from active_audition.a4.pose_sampler import CandidateContract


ROOT = Path(__file__).resolve().parents[1]
OLD_MANIFEST_SHA = "c8c821eae92232421103bc12ecdca4f8c8187b490546dcee0f31807c5d3f5190"
FINALIZED_REPLACEMENT = ROOT / "data/logs/o1_noise_replacement_audit_batch01/o1_noise_audit_records_finalized.json"
FINALIZED_ORIGINAL = ROOT / "data/logs/o1_noise_manual_audit/o1_noise_audit_records_finalized.json"
PARENT_ORDER = (
    "noise/free-sound/noise-free-sound-0041",
    "noise/free-sound/noise-free-sound-0073",
    "noise/free-sound/noise-free-sound-0032",
    "noise/free-sound/noise-free-sound-0042",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _registry_row(parent_id: str) -> dict:
    for row in _rows(ROOT / "registries/active_asr_a3/musan_noise.jsonl"):
        if row["parent_recording_id"] == parent_id:
            return row
    raise RuntimeError("missing MUSAN registry row: {}".format(parent_id))


def _speech_rows(block: dict) -> list[dict]:
    wanted = {item["utterance_id"] for item in block["speech_sources"]}
    rows = {row["utterance_id"]: row for row in _rows(ROOT / "registries/active_asr_a3/librispeech.jsonl")}
    if wanted - set(rows):
        raise RuntimeError("missing frozen speech registry rows")
    return [rows[item["utterance_id"]] for item in block["speech_sources"]]


def _audit_records() -> dict[str, dict]:
    replacement = O1FinalizedReplacementAuditBatch.from_payload(_json(FINALIZED_REPLACEMENT))
    original = _json(FINALIZED_ORIGINAL)
    records = {}
    for payload in original["records"]:
        record = O1NoiseParentAuditRecord.from_payload(payload)
        records[record.parent_recording_id] = record.to_payload()
    for payload in replacement.records:
        record = O1NoiseParentAuditRecord.from_payload(payload)
        records[record.parent_recording_id] = record.to_payload()
    if tuple(PARENT_ORDER) != tuple(records[item]["parent_recording_id"] for item in PARENT_ORDER):
        raise RuntimeError("final O1 audit mapping is incomplete")
    return records


def _rebuild_block(old: dict, new_parent_id: str, audit: dict) -> dict:
    geometry = old["geometry_record"]
    case = {
        "scene_id": geometry["scene_id"],
        "scene_asset_sha256": geometry["scene_resource_identities"]["scene_asset_sha256"],
        "navmesh_sha256": geometry["scene_resource_identities"]["navmesh_sha256"],
        "stage_config_sha256": geometry["scene_resource_identities"]["stage_config_sha256"],
        "listener_base_position_world": geometry["initial_listener_requested_base_xyz"],
        "listener_sensor_position_world": geometry["sensor_xyz"],
        "listener_yaw_deg": geometry["initial_yaw_deg"],
    }
    target_source_case = {"source_position_world": geometry["target_world_pose"]["position_xyz"]}
    noise_source_case = {"source_position_world": geometry["noise_world_pose"]["position_xyz"]}
    candidate = CandidateContract.from_payload(old["candidate_contract"])
    source_free = _sampler_output(old["sampler_output"])
    rebuilt = dict(build_smoke_block_payload(
        case, target_source_case, noise_source_case, _speech_rows(old), _registry_row(new_parent_id), source_free, candidate
    ))
    rebuilt["noise_audit_record"] = audit
    if rebuilt["geometry_record"] != old["geometry_record"]:
        raise RuntimeError("geometry changed while replacing noise parent")
    if rebuilt["candidate_contract"] != old["candidate_contract"]:
        raise RuntimeError("candidate contract changed while replacing noise parent")
    if rebuilt["sampler_context"] != old["sampler_context"] or rebuilt["sampler_output"] != old["sampler_output"]:
        raise RuntimeError("source-free sampler output changed while replacing noise parent")
    if rebuilt["poses"] != old["poses"]:
        raise RuntimeError("pose records changed while replacing noise parent")
    return rebuilt


def _assert_retained_block_unchanged(old: dict, rebuilt: dict, expected_parent_id: str) -> None:
    """Make retained-block provenance an assertion, not a no-op conditional."""

    for field in (
        "block_record", "geometry_record", "candidate_contract", "sampler_context",
        "sampler_output", "poses", "speech_sources", "global_gain",
        "selection_evaluation_policy",
    ):
        if rebuilt[field] != old[field]:
            raise RuntimeError("retained block {} changed: {}".format(expected_parent_id, field))
    if rebuilt["noise_parent"]["parent_recording_id"] != expected_parent_id:
        raise RuntimeError("retained block noise parent changed: {}".format(expected_parent_id))
    return rebuilt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, default=ROOT / "runs/active_asr_v1/o1_replica_apartment_2_c8c821eae922/o1_exploratory_manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/active_asr_v1")
    args = parser.parse_args()
    old = _json(args.old_manifest)
    if old["manifest_sha256"] != OLD_MANIFEST_SHA:
        raise SystemExit("unexpected pre-audit manifest SHA")
    audits = _audit_records()
    parents = PARENT_ORDER
    blocks = []
    for old_block, parent_id in zip(old["blocks"], parents):
        blocks.append(_rebuild_block(old_block, parent_id, audits[parent_id]))
    _assert_retained_block_unchanged(old["blocks"][2], blocks[2], PARENT_ORDER[2])
    _assert_retained_block_unchanged(old["blocks"][3], blocks[3], PARENT_ORDER[3])
    ledger_base = {
        "schema_version": "active-asr-o1-consumed-scene-ledger-v2",
        "supersedes_manifest_sha256": OLD_MANIFEST_SHA,
        "supersedes_ledger_id": old["consumed_scene_ledger_id"],
        "scene_id": "replica.apartment_2",
        "engineering_only": True,
        "o2_exclusion": {"excluded": True, "reason": "consumed by O1 exploratory landscape; never eligible for O2 held-out main set"},
        "block_noise_parent_order": list(parents),
        "replacement_slot_mapping": {
            "slot1": {"parent_recording_id": parents[0], "replaces_parent_recording_id": "noise/free-sound/noise-free-sound-0015"},
            "slot2": {"parent_recording_id": parents[1], "replaces_parent_recording_id": "noise/free-sound/noise-free-sound-0030"},
        },
        "selected_position_ids": [[item["position_id"] for item in block["sampler_output"]["selected_positions"]] for block in blocks],
        "result_dependent_selection": False,
    }
    ledger = dict(ledger_base, ledger_id=stable_id("o1-consumed-scene-ledger", ledger_base))
    ledger["ledger_sha256"] = identity_sha256(ledger_base)
    selection_policy = dict(old["selection_policy"])
    selection_policy["noise_audit_rule"] = "exact four parents bound to finalized user manual audit; all three dimensions PASS"
    selection_policy["replacement_slot_mapping"] = ledger["replacement_slot_mapping"]
    selection_policy["supersedes_manifest_sha256"] = OLD_MANIFEST_SHA
    manifest = O1ExploratoryManifest(
        infrastructure_contract_sha256=old["infrastructure_contract_sha256"],
        selection_policy=selection_policy,
        blocks=tuple(blocks),
        o2_exclusion=dict(old["o2_exclusion"], supersedes_manifest_sha256=OLD_MANIFEST_SHA),
        consumed_scene_ledger_id=ledger["ledger_id"],
        schema_version=O1_MANIFEST_SCHEMA_VERSION_V2,
    )
    run_root = args.output_root / "o1_replica_apartment_2_{}".format(manifest.manifest_sha256[:12])
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "o1_scientific_manifest.json").write_text(canonical_json(manifest.to_payload()) + "\n", encoding="utf-8")
    (run_root / "o1_exploratory_manifest.json").write_text(canonical_json(manifest.to_payload()) + "\n", encoding="utf-8")
    (run_root / "o1_consumed_ledger.json").write_text(canonical_json(ledger) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "active-asr-o1-scientific-preparation-v2",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.manifest_sha256,
        "parent_manifest_sha256": OLD_MANIFEST_SHA,
        "noise_parent_order": list(parents),
        "blocks": 4,
        "episodes": 16,
        "poses": 192,
        "expected_rirs": 384,
        "expected_mixtures": 768,
        "expected_component_snr": 768,
        "expected_asr": 2304,
        "scientific_noise_audit": True,
        "result_dependent_selection": False,
        "run_root": str(run_root),
    }
    (run_root / "o1_scientific_preparation_summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
