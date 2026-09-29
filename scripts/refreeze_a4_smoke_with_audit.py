#!/usr/bin/env python3
"""Re-freeze the A4 engineering smoke with explicit manual noise audits.

This is a metadata-only checkpoint.  It reconstructs the existing frozen
source-free sampler output and geometry records, replaces only the audited
noise parent/segment provenance, and never renders a RIR or consumes an audio,
ASR, WER, or Oracle result.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from active_audition.a4.identity import canonical_json, identity_sha256, stable_id
from active_audition.a4.budget import MotionCost
from active_audition.a4.noise_audit import (
    NOISE_AUDIT_POLICY_IDENTITY,
    NOISE_AUDIT_SCHEMA_VERSION,
    NoiseParentAuditRecord,
)
from active_audition.a4.pose_sampler import (
    CandidateContract,
    ProbeAttemptRecord,
    SampledPositionPlan,
    SampledYawPlan,
    SamplerContext,
    SamplerOutput,
)
from active_audition.a4.smoke_manifest import (
    ENGINEERING_SMOKE_MANIFEST_SCHEMA_VERSION,
    EngineeringSmokeManifest,
)
from active_audition.a4.smoke_preparation import (
    build_engineering_smoke_manifest,
    build_smoke_block_payload,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_MANIFEST = ROOT / "configs/active_audition/v1/a4_engineering_smoke_manifest.json"
CONTRACT_SHA = ROOT / "configs/active_audition/v1/a4_infrastructure_contract.sha256"
SPEECH_REGISTRY = ROOT / "registries/active_asr_a3/librispeech.jsonl"
NOISE_REGISTRY = ROOT / "registries/active_asr_a3/musan_noise.jsonl"
REPLACEMENT_SELECTION = ROOT / "data/logs/a4_musan_replacement_audit/candidate_selection.json"
REPLACEMENT_AUDIT = ROOT / "data/logs/a4_musan_replacement_audit/audit_sheet.json"
MANUAL_AUDIT = ROOT / "data/logs/a4_musan_manual_audit/audit_sheet.json"
OUTPUT_AUDIT = ROOT / "registries/active_asr_a4/a4_engineering_noise_parent_audit_v1.json"
OUTPUT_MANIFEST = OLD_MANIFEST
OUTPUT_CACHE_PLAN = ROOT / "configs/active_audition/v1/a4_engineering_smoke_cache_plan.json"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sampler_output(payload: Mapping[str, Any]) -> SamplerOutput:
    attempts = tuple(ProbeAttemptRecord(**dict(item)) for item in payload["probe_attempts"])
    positions = tuple(SampledPositionPlan(**dict(item)) for item in payload["selected_positions"])
    yaws = tuple(
        SampledYawPlan(
            motion_cost=MotionCost(**dict(item["motion_cost"])),
            **{key: value for key, value in item.items() if key != "motion_cost"},
        )
        for item in payload["yaw_plans"]
    )
    return SamplerOutput(
        schema_version=payload["schema_version"],
        sampler_context_id=payload["sampler_context_id"],
        candidate_contract_identity=payload["candidate_contract_identity"],
        probe_attempts=attempts,
        selected_positions=positions,
        yaw_plans=yaws,
        opportunity_constrained=payload["opportunity_constrained"],
        opportunity_reason=payload["opportunity_reason"],
    )


def _preview_identity(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "relative_preview_path": item["relative_preview_path"],
        "clip_index": item["clip_index"],
        "start_sample": item["start_sample"],
        "end_sample": item["end_sample"],
        "expected_source_slice_float32_sha256": item.get(
            "expected_source_slice_float32_sha256", item.get("expected_slice_float32_sha256")
        ),
        "written_clip_decoded_float32_sha256": item["written_clip_decoded_float32_sha256"],
        "verification_status": item["verification_status"],
    }


def _candidate_batch_identity(selection: Mapping[str, Any]) -> str:
    semantic = {
        "schema_version": selection["schema_version"],
        "technical_eligible_parent_count": selection["technical_eligible_parent_count"],
        "required_parent_sample_count": selection["duration_requirement"]["required_parent_sample_count"],
        "selected_candidates": selection["selected_candidates"],
    }
    return stable_id("a4-noise-candidate-batch", semantic)


def _audit_record(
    row: Mapping[str, Any],
    previews: list[Mapping[str, Any]],
    batch_identity: str,
    batch_sha: str,
    statuses: Mapping[str, str],
    selected: bool,
    reason: str,
    package_name: str,
) -> NoiseParentAuditRecord:
    return NoiseParentAuditRecord(
        schema_version=NOISE_AUDIT_SCHEMA_VERSION,
        audit_policy_identity=NOISE_AUDIT_POLICY_IDENTITY,
        decision_source="user_manual_listening",
        source_candidate_batch_identity=batch_identity,
        source_candidate_batch_sha256=batch_sha,
        parent_recording_id=row["parent_recording_id"],
        relative_source_path=row["relative_source_path"],
        source_file_sha256=row["source_file_sha256"],
        decoded_waveform_sha256=row["decoded_waveform_sha256"],
        sample_rate_hz=row["sample_rate_hz"],
        sample_count=row["samples"],
        review_scope={
            "package": package_name,
            "dimensions": [
                "speech_leakage",
                "strong_reverberation",
                "indoor_localized_source_compatibility",
            ],
            "manual_only": True,
        },
        reviewed_preview_identities=tuple(previews),
        speech_leakage=statuses["speech_leakage"],
        strong_reverberation=statuses["strong_reverberation"],
        indoor_localized_source_compatibility=statuses["indoor_localized_source_compatibility"],
        selected_for_a4_smoke=selected,
        exclusion_reason=reason,
        engineering_only=True,
    )


def _source_case(geometry: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "scene_id": geometry["scene_id"],
        "scene_asset_sha256": geometry["scene_resource_identities"]["scene_asset_sha256"],
        "navmesh_sha256": geometry["scene_resource_identities"]["navmesh_sha256"],
        "stage_config_sha256": geometry["scene_resource_identities"]["stage_config_sha256"],
        "listener_base_position_world": geometry["actual_listener_base_xyz"],
        "listener_sensor_position_world": geometry["sensor_xyz"],
        "listener_yaw_deg": geometry["initial_yaw_deg"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--output-audit", type=Path, default=OUTPUT_AUDIT)
    args = parser.parse_args()

    old = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    contract_sha = CONTRACT_SHA.read_text(encoding="utf-8").strip()
    if contract_sha != "81ae5896c8d0b5348d00cb7314aef1eb2bebb5445b1bdfdb7ff55f4cd4a4abe6":
        raise RuntimeError("Infrastructure Contract SHA changed; refusing re-freeze")
    selection = json.loads(REPLACEMENT_SELECTION.read_text(encoding="utf-8"))
    replacement_sheet = json.loads(REPLACEMENT_AUDIT.read_text(encoding="utf-8"))
    manual_sheet = json.loads(MANUAL_AUDIT.read_text(encoding="utf-8"))
    candidate_identity = _candidate_batch_identity(selection)
    candidate_sha = _sha_file(REPLACEMENT_SELECTION)
    manual_identity = stable_id(
        "a4-manual-noise-audit-batch",
        {"schema_version": manual_sheet["schema_version"], "parents": [item["parent_recording_id"] for item in manual_sheet["parents"]]},
    )
    manual_sha = _sha_file(MANUAL_AUDIT)
    candidate_rows = {item["parent_recording_id"]: item for item in selection["selected_candidates"]}
    replacement_rows = {item["parent_recording_id"]: item for item in replacement_sheet["candidates"]}
    manual_rows = {item["parent_recording_id"]: item for item in manual_sheet["parents"]}
    registry_rows = {item["parent_recording_id"]: item for item in _load_jsonl(NOISE_REGISTRY)}
    audit_specs = (
        ("noise/free-sound/noise-free-sound-0048", manual_rows["noise/free-sound/noise-free-sound-0048"], {
            "speech_leakage": "FLAGGED", "strong_reverberation": "NOT_EVALUATED", "indoor_localized_source_compatibility": "NOT_EVALUATED",
        }, False, "suspected speech-like content in manually reviewed preview; cannot safely rule out speech leakage", [3]),
        ("noise/free-sound/noise-free-sound-0270", manual_rows["noise/free-sound/noise-free-sound-0270"], {
            "speech_leakage": "NOT_EVALUATED", "strong_reverberation": "NOT_EVALUATED", "indoor_localized_source_compatibility": "FLAGGED",
        }, False, "vehicle/road-driving-like recorded noise is incompatible with the current indoor static localized-source approximation", list(range(8))),
        ("noise/free-sound/noise-free-sound-0002", replacement_rows["noise/free-sound/noise-free-sound-0002"], {
            "speech_leakage": "PASS", "strong_reverberation": "PASS", "indoor_localized_source_compatibility": "PASS",
        }, True, "", list(range(6))),
        ("noise/free-sound/noise-free-sound-0020", replacement_rows["noise/free-sound/noise-free-sound-0020"], {
            "speech_leakage": "PASS", "strong_reverberation": "PASS", "indoor_localized_source_compatibility": "PASS",
        }, True, "", list(range(6))),
    )
    audit_records = []
    for parent_id, sheet_row, statuses, selected, reason, preview_indices in audit_specs:
        row = registry_rows[parent_id]
        if row["source_file_sha256"] != sheet_row["source_file_sha256"] or row["decoded_waveform_sha256"] != sheet_row["decoded_waveform_sha256"]:
            raise RuntimeError("audit source identity mismatch for {}".format(parent_id))
        batch_identity, batch_sha, preview_source = (
            (manual_identity, manual_sha, manual_rows[parent_id]["clips"])
            if parent_id in manual_rows
            else (candidate_identity, candidate_sha, replacement_rows[parent_id]["preview_clips"])
        )
        previews = [_preview_identity(preview_source[index]) for index in preview_indices]
        audit_records.append(_audit_record(
            row, previews, batch_identity, batch_sha, statuses, selected, reason,
            "a4_musan_manual_audit" if parent_id in manual_rows else "a4_musan_replacement_audit",
        ))
    audit_by_parent = {record.parent_recording_id: record for record in audit_records}

    speech_rows = {item["utterance_id"]: item for item in _load_jsonl(SPEECH_REGISTRY)}
    candidate = CandidateContract.from_payload(old["blocks"][0]["candidate_contract"])
    new_blocks = []
    for old_block in old["blocks"]:
        geometry = old_block["geometry_record"]
        parent_id = (
            "noise/free-sound/noise-free-sound-0002"
            if geometry["scene_id"] == "replica.apartment_0"
            else "noise/free-sound/noise-free-sound-0020"
        )
        noise_row = registry_rows[parent_id]
        rows = [speech_rows[item["utterance_identity"]["utterance_id"]] for item in old_block["episodes"]]
        source_output = _sampler_output(old_block["sampler_output"])
        context = SamplerContext(**dict(old_block["sampler_context"]))
        if context.candidate_contract_identity != candidate.candidate_contract_identity:
            raise RuntimeError("candidate contract identity changed")
        case = _source_case(geometry)
        target_case = {"source_position_world": geometry["target_world_pose"]["position_xyz"]}
        noise_case = {"source_position_world": geometry["noise_world_pose"]["position_xyz"]}
        block = build_smoke_block_payload(case, target_case, noise_case, rows, noise_row, source_output, candidate)
        old_geometry_id = old_block["geometry_record"]["geometry_id"]
        if block["geometry_record"]["geometry_id"] != old_geometry_id:
            raise RuntimeError("geometry identity changed during re-freeze")
        old_pose_ids = [item["pose_id"] for item in old_block["poses"]]
        new_pose_ids = [item["pose_id"] for item in block["poses"]]
        if old_pose_ids != new_pose_ids:
            raise RuntimeError("source-free geometry-bound pose identities changed")
        audit = audit_by_parent[parent_id]
        block["noise_audit_record"] = dict(audit.to_payload())
        block["noise_audit"].update({
            "audit_record_id": audit.audit_record_id,
            "speech_leakage_audit": {"status": audit.speech_leakage},
            "strong_reverberation_audit": {"status": audit.strong_reverberation},
            "indoor_localized_source_compatibility_audit": {"status": audit.indoor_localized_source_compatibility},
            "selected_for_a4_smoke": True,
        })
        new_blocks.append(block)

    smoke = build_engineering_smoke_manifest(contract_sha, new_blocks, schema_version=ENGINEERING_SMOKE_MANIFEST_SCHEMA_VERSION)
    validated = EngineeringSmokeManifest.from_payload(smoke.to_payload())
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(canonical_json(validated.to_payload()) + "\n", encoding="utf-8")

    audit_payload = {
        "schema_version": "active-asr-a4-noise-parent-audit-batch-v1",
        "audit_policy_identity": NOISE_AUDIT_POLICY_IDENTITY,
        "decision_source": "user_manual_listening",
        "records": [record.to_payload() for record in audit_records],
        "audit_batch_id": stable_id("a4-noise-audit-batch", {"records": [record.identity_payload() for record in audit_records]}),
        "audit_batch_sha256": identity_sha256({"records": [record.identity_payload() for record in audit_records]}),
    }
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(canonical_json(audit_payload) + "\n", encoding="utf-8")

    plan = {
        "schema_version": "active-asr-a4-engineering-smoke-cache-plan-v1",
        "plan_status": "METADATA_ONLY_NOT_CACHE_EXPECTED_MANIFEST",
        "infrastructure_contract_sha256": contract_sha,
        "engineering_smoke_manifest_id": smoke.manifest_id,
        "engineering_smoke_manifest_sha256": smoke.manifest_sha256,
        "expected_counts": {"rir": 192, "mixture": 384, "asr": 1152},
        "dependency_structure": {
            "rir": "2 blocks x 48 poses x 2 source roles",
            "mixture": "2 blocks x 4 episodes x 48 poses",
            "asr": "384 mixtures x 3 frontends",
        },
        "production_keys": "MUST_BE_GENERATED_FROM_ACTUAL_UPSTREAM_ARTIFACTS",
        "forbidden_placeholder_keys": True,
    }
    OUTPUT_CACHE_PLAN.write_text(canonical_json(plan) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest_id": smoke.manifest_id,
        "manifest_sha256": smoke.manifest_sha256,
        "audit_batch_id": audit_payload["audit_batch_id"],
        "audit_records": [record.audit_record_id for record in audit_records],
        "block_ids": [block["block_record"]["block_id"] for block in new_blocks],
        "geometry_ids": [block["geometry_record"]["geometry_id"] for block in new_blocks],
        "pose_counts": [len(block["poses"]) for block in new_blocks],
        "noise_parent_ids": [block["block_record"]["noise_parent_id"] for block in new_blocks],
        "cache_plan": str(OUTPUT_CACHE_PLAN),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
