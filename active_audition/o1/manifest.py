"""Strict metadata manifest for the first exploratory O1 landscape.

The manifest is deliberately separate from the frozen A4 Engineering Smoke
Manifest.  It reuses A4's immutable geometry, sampler, episode, pose, and
noise-plan records, but allows an exploratory multi-block scene inventory.
No ASR or acoustic result is accepted by this schema.
"""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.budget import MotionCost
from active_audition.a4.identity import identity_sha256, stable_id, validate_sha256
from active_audition.a4.noise_segments import NoiseSegmentPlan
from active_audition.a4.pose_sampler import (
    CandidateContract,
    ProbeAttemptRecord,
    SampledPositionPlan,
    SampledYawPlan,
    SamplerContext,
    SamplerOutput,
)
from active_audition.a4.records import BlockRecord, EpisodeRecord, GeometryRecord, PoseRecord
from active_audition.o1.noise_audit import O1NoiseParentAuditRecord, O1NoiseAuditError


O1_MANIFEST_SCHEMA_VERSION = "active-asr-o1-exploratory-manifest-v1"
O1_MANIFEST_SCHEMA_VERSION_V2 = "active-asr-o1-exploratory-manifest-v2"
O1_MANIFEST_STATE = "FROZEN_EXPLORATORY"
O1_MANIFEST_PRE_ASR_STATUS = "PRE_ASR_PENDING_NOISE_AUDIT"
O1_MANIFEST_SCIENTIFIC_STATUS = "SCIENTIFIC_NOISE_AUDIT_READY"
EXPECTED_FRONTENDS = ("mean_lr", "fixed_L", "fixed_R")
FORBIDDEN_RESULT_INPUTS = (
    "RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "Oracle", "movement_benefit"
)
_RESULT_KEYS = frozenset({"wer", "cer", "hypothesis", "score", "drr", "energy", "rir", "oracle", "movement_benefit"})


class O1ManifestError(ValueError):
    pass


def _exact(value: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise O1ManifestError("{} must be a mapping".format(path))
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise O1ManifestError("{} fields invalid: unknown={}, missing={}".format(path, unknown, missing))


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise O1ManifestError("{} must be a non-empty string".format(path))
    return value


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise O1ManifestError(str(exc)) from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _reject_result_fields(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _RESULT_KEYS:
                raise O1ManifestError("{} contains result-dependent field {!r}".format(path, key))
            _reject_result_fields(item, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_result_fields(item, "{}[{}]".format(path, index))


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


def _validate_block(block: Mapping[str, Any], index: int, require_noise_audit: bool = False) -> None:
    path = "manifest.blocks[{}]".format(index)
    required = (
        "block_record", "candidate_contract", "episodes", "geometry_record", "global_gain",
        "noise_audit", "noise_parent", "noise_plan", "poses", "sampler_context",
        "sampler_output", "selection_evaluation_policy", "speech_sources",
    )
    if require_noise_audit:
        required = required + ("noise_audit_record",)
    _exact(block, required, path)
    geometry = GeometryRecord.from_payload(block["geometry_record"])
    record = BlockRecord.from_payload(block["block_record"])
    candidate = CandidateContract.from_payload(block["candidate_contract"])
    context = SamplerContext(**dict(block["sampler_context"]))
    if record.geometry_id != geometry.geometry_id or context.candidate_contract_identity != candidate.candidate_contract_identity:
        raise O1ManifestError("{} geometry/candidate binding mismatch".format(path))
    sampler = _sampler_output(block["sampler_output"])
    if sampler.canonical_bytes() != json.dumps(_plain(block["sampler_output"]), sort_keys=True, separators=(",", ":")).encode():
        raise O1ManifestError("{} sampler output is not canonical".format(path))
    if sampler.sampler_context_id != context.sampler_context_id or sampler.candidate_contract_identity != candidate.candidate_contract_identity:
        raise O1ManifestError("{} sampler binding mismatch".format(path))
    if len(sampler.probe_attempts) != 64 or len(sampler.selected_positions) != 6 or len(sampler.yaw_plans) != 48:
        raise O1ManifestError("{} must contain 64 probes, 6 positions, and 48 yaw plans".format(path))
    if sampler.opportunity_constrained is not False:
        raise O1ManifestError("{} opportunity_constrained must be false".format(path))
    plan = NoiseSegmentPlan.from_payload(block["noise_plan"])
    if plan.plan_id != record.noise_segment_plan_identity or plan.noise_parent_id != record.noise_parent_id:
        raise O1ManifestError("{} noise plan/block binding mismatch".format(path))
    episodes = tuple(EpisodeRecord.from_payload(item) for item in block["episodes"])
    if len(episodes) != 4 or tuple((item.role, item.fixed_dry_noise_segment_identity["role_index"]) for item in episodes) != (
        ("selection", 0), ("selection", 1), ("evaluation", 0), ("evaluation", 1)
    ):
        raise O1ManifestError("{} episode slots are not selection[0..1], evaluation[0..1]".format(path))
    if any(item.block_id != record.block_id for item in episodes):
        raise O1ManifestError("{} episode/block binding mismatch".format(path))
    poses = tuple(PoseRecord.from_payload(item) for item in block["poses"])
    if len(poses) != 48 or any(item.geometry_id != geometry.geometry_id for item in poses):
        raise O1ManifestError("{} must contain 48 geometry-bound poses".format(path))
    if not sampler.yaw_plans or not all(pose.geometry_legality in ("LEGAL", "ILLEGAL") for pose in poses):
        raise O1ManifestError("{} pose legality records are invalid".format(path))
    speech = block["speech_sources"]
    if not isinstance(speech, (list, tuple)) or len(speech) != 4 or len({item["speaker_id"] for item in speech}) != 1:
        raise O1ManifestError("{} must contain four utterances from one speaker".format(path))
    if block["selection_evaluation_policy"] != {
        "selection": "calibration_and_later_pose_choice_eligible",
        "evaluation": "never_changes_pose_choice",
    }:
        raise O1ManifestError("{} selection/evaluation policy is invalid".format(path))
    _reject_result_fields(block["selection_evaluation_policy"], path + ".selection_evaluation_policy")
    if require_noise_audit:
        try:
            audit = O1NoiseParentAuditRecord.from_payload(block["noise_audit_record"])
        except O1NoiseAuditError as exc:
            raise O1ManifestError("{} noise audit is invalid: {}".format(path, exc)) from exc
        if audit.parent_recording_id != block["noise_parent"]["parent_recording_id"]:
            raise O1ManifestError("{} noise audit parent binding mismatch".format(path))
        if not audit.selected_for_o1_scientific_use:
            raise O1ManifestError("{} noise audit is not scientifically eligible".format(path))


@dataclass(frozen=True)
class O1ExploratoryManifest:
    infrastructure_contract_sha256: str
    selection_policy: Mapping[str, Any]
    blocks: Tuple[Mapping[str, Any], ...]
    o2_exclusion: Mapping[str, Any]
    consumed_scene_ledger_id: str
    expected_frontends: Tuple[str, ...] = EXPECTED_FRONTENDS
    forbidden_result_dependent_selection: Tuple[str, ...] = FORBIDDEN_RESULT_INPUTS
    exploratory_only: bool = True
    state: str = O1_MANIFEST_STATE
    schema_version: str = O1_MANIFEST_SCHEMA_VERSION
    manifest_id: str = ""
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        _sha(self.infrastructure_contract_sha256, "manifest.infrastructure_contract_sha256")
        if self.schema_version not in (O1_MANIFEST_SCHEMA_VERSION, O1_MANIFEST_SCHEMA_VERSION_V2) or self.state != O1_MANIFEST_STATE:
            raise O1ManifestError("manifest schema/state is invalid")
        if self.exploratory_only is not True or tuple(self.expected_frontends) != EXPECTED_FRONTENDS:
            raise O1ManifestError("exploratory_only/frontends are invalid")
        if tuple(self.forbidden_result_dependent_selection) != FORBIDDEN_RESULT_INPUTS:
            raise O1ManifestError("forbidden result-dependent selection list is invalid")
        if len(self.blocks) != 4 or len({block["block_record"]["block_id"] for block in self.blocks}) != 4:
            raise O1ManifestError("O1 first landscape requires four distinct blocks")
        if len({block["geometry_record"]["scene_id"] for block in self.blocks}) != 1:
            raise O1ManifestError("first landscape must use one scene")
        require_noise_audit = self.schema_version == O1_MANIFEST_SCHEMA_VERSION_V2
        for index, block in enumerate(self.blocks):
            _validate_block(block, index, require_noise_audit=require_noise_audit)
        if not isinstance(self.selection_policy, Mapping) or not self.selection_policy:
            raise O1ManifestError("selection_policy must be non-empty")
        if not isinstance(self.o2_exclusion, Mapping) or self.o2_exclusion.get("excluded") is not True:
            raise O1ManifestError("O2 exclusion must be explicit")
        _string(self.consumed_scene_ledger_id, "manifest.consumed_scene_ledger_id")
        _reject_result_fields(self.selection_policy, "manifest.selection_policy")
        _reject_result_fields(self.o2_exclusion, "manifest.o2_exclusion")
        expected_id = stable_id("o1-exploratory-manifest", self.identity_payload())
        if self.manifest_id and self.manifest_id != expected_id:
            raise O1ManifestError("manifest_id does not match semantic payload")
        expected_sha = identity_sha256(self.identity_payload())
        if self.manifest_sha256 and self.manifest_sha256 != expected_sha:
            raise O1ManifestError("manifest_sha256 does not match semantic payload")
        object.__setattr__(self, "manifest_id", expected_id)
        object.__setattr__(self, "manifest_sha256", expected_sha)
        object.__setattr__(self, "blocks", tuple(_freeze(block) for block in self.blocks))
        object.__setattr__(self, "selection_policy", _freeze(self.selection_policy))
        object.__setattr__(self, "o2_exclusion", _freeze(self.o2_exclusion))

    @property
    def scientific_use_status(self) -> str:
        return O1_MANIFEST_SCIENTIFIC_STATUS if self.schema_version == O1_MANIFEST_SCHEMA_VERSION_V2 else O1_MANIFEST_PRE_ASR_STATUS

    def require_scientific_noise_audit(self) -> None:
        if self.schema_version != O1_MANIFEST_SCHEMA_VERSION_V2:
            raise O1ManifestError("pre-audit O1 manifest is {}".format(O1_MANIFEST_PRE_ASR_STATUS))
        if self.scientific_use_status != O1_MANIFEST_SCIENTIFIC_STATUS:
            raise O1ManifestError("O1 manifest is not scientifically audit-ready")

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "exploratory_only": self.exploratory_only,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "selection_policy": _plain(self.selection_policy),
            "blocks": [_plain(block) for block in self.blocks],
            "expected_frontends": list(self.expected_frontends),
            "forbidden_result_dependent_selection": list(self.forbidden_result_dependent_selection),
            "o2_exclusion": _plain(self.o2_exclusion),
            "consumed_scene_ledger_id": self.consumed_scene_ledger_id,
        }

    def to_payload(self) -> Mapping[str, Any]:
        return dict(self.identity_payload(), manifest_id=self.manifest_id, manifest_sha256=self.manifest_sha256)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1ExploratoryManifest":
        fields = (
            "schema_version", "state", "exploratory_only", "infrastructure_contract_sha256", "selection_policy",
            "blocks", "expected_frontends", "forbidden_result_dependent_selection", "o2_exclusion",
            "consumed_scene_ledger_id", "manifest_id", "manifest_sha256",
        )
        _exact(payload, fields, "o1_exploratory_manifest")
        return cls(
            infrastructure_contract_sha256=payload["infrastructure_contract_sha256"],
            selection_policy=payload["selection_policy"], blocks=tuple(payload["blocks"]),
            o2_exclusion=payload["o2_exclusion"], consumed_scene_ledger_id=payload["consumed_scene_ledger_id"],
            expected_frontends=tuple(payload["expected_frontends"]),
            forbidden_result_dependent_selection=tuple(payload["forbidden_result_dependent_selection"]),
            exploratory_only=payload["exploratory_only"], state=payload["state"],
            schema_version=payload["schema_version"], manifest_id=payload["manifest_id"],
            manifest_sha256=payload["manifest_sha256"],
        )


__all__ = [
    "O1ExploratoryManifest", "O1ManifestError", "O1_MANIFEST_PRE_ASR_STATUS",
    "O1_MANIFEST_SCHEMA_VERSION", "O1_MANIFEST_SCHEMA_VERSION_V2", "O1_MANIFEST_SCIENTIFIC_STATUS",
    "O1_MANIFEST_STATE",
]
