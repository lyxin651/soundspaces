"""Strict, metadata-only A4 engineering-smoke manifest.

This module freezes the inputs for the later A4 smoke without rendering a
RIR, building a mixture, decoding ASR, or selecting anything from a result.
The manifest is deliberately independent from the Infrastructure Contract:
it binds the final contract SHA, but the contract never embeds this manifest.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.budget import MotionCost
from active_audition.a4.identity import canonical_json_bytes, identity_sha256, stable_id, validate_sha256
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


ENGINEERING_SMOKE_MANIFEST_SCHEMA_VERSION = "active-asr-a4-engineering-smoke-manifest-v1"
ENGINEERING_SMOKE_MANIFEST_STATE = "FROZEN"
EXPECTED_FRONTENDS = ("mean_lr", "fixed_L", "fixed_R")
FORBIDDEN_SELECTION_INPUTS = (
    "RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "hypothesis", "Oracle", "movement_benefit",
)
BLOCK_FIELDS = (
    "block_record", "geometry_record", "candidate_contract", "sampler_context",
    "sampler_output", "poses", "noise_parent", "noise_plan", "episodes",
    "speech_sources", "noise_audit", "global_gain", "selection_evaluation_policy",
)
TOP_FIELDS = (
    "schema_version", "state", "engineering_only", "infrastructure_contract_sha256",
    "selection_policy", "blocks", "expected_frontends", "forbidden_result_dependent_selection",
    "o2_exclusion", "o1_reuse_classification_policy", "manifest_id", "manifest_sha256",
)
_RESULT_KEYS = frozenset({"wer", "cer", "hypothesis", "score", "drr", "energy", "rir", "oracle", "movement_benefit"})


class EngineeringSmokeManifestError(ValueError):
    """Raised when a frozen engineering manifest is malformed or tampered."""


def _exact(value: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise EngineeringSmokeManifestError("{} must be a mapping".format(path))
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise EngineeringSmokeManifestError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise EngineeringSmokeManifestError("{} must be a non-empty string".format(path))
    return value


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise EngineeringSmokeManifestError(str(exc)) from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _reject_result_fields(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _RESULT_KEYS:
                raise EngineeringSmokeManifestError("{} contains forbidden result field {!r}".format(path, key))
            _reject_result_fields(item, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_result_fields(item, "{}[{}]".format(path, index))


def _validate_block(payload: Mapping[str, Any], index: int) -> None:
    path = "manifest.blocks[{}]".format(index)
    _exact(payload, BLOCK_FIELDS, path)
    geometry = GeometryRecord.from_payload(payload["geometry_record"])
    block = BlockRecord.from_payload(payload["block_record"])
    candidate = CandidateContract.from_payload(payload["candidate_contract"])
    context = SamplerContext(**dict(payload["sampler_context"]))
    if geometry.geometry_id != block.geometry_id:
        raise EngineeringSmokeManifestError("{} geometry/block binding mismatch".format(path))
    if geometry.candidate_contract_identity != candidate.candidate_contract_identity:
        raise EngineeringSmokeManifestError("{} candidate contract binding mismatch".format(path))
    if context.candidate_contract_identity != candidate.candidate_contract_identity:
        raise EngineeringSmokeManifestError("{} sampler context binding mismatch".format(path))
    sampler_payload = payload["sampler_output"]
    if not isinstance(sampler_payload, Mapping):
        raise EngineeringSmokeManifestError("{}.sampler_output must be a mapping".format(path))
    try:
        attempts = tuple(ProbeAttemptRecord(**dict(item)) for item in sampler_payload["probe_attempts"])
        positions = tuple(SampledPositionPlan(**dict(item)) for item in sampler_payload["selected_positions"])
        yaws = tuple(
            SampledYawPlan(
                motion_cost=MotionCost(**dict(item["motion_cost"])),
                **{key: value for key, value in item.items() if key != "motion_cost"},
            )
            for item in sampler_payload["yaw_plans"]
        )
        sampler_output = SamplerOutput(
            schema_version=sampler_payload["schema_version"],
            sampler_context_id=sampler_payload["sampler_context_id"],
            candidate_contract_identity=sampler_payload["candidate_contract_identity"],
            probe_attempts=attempts,
            selected_positions=positions,
            yaw_plans=yaws,
            opportunity_constrained=sampler_payload["opportunity_constrained"],
            opportunity_reason=sampler_payload["opportunity_reason"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EngineeringSmokeManifestError("{}.sampler_output is inconsistent: {}".format(path, exc)) from exc
    if sampler_output.sampler_context_id != context.sampler_context_id or sampler_output.candidate_contract_identity != candidate.candidate_contract_identity:
        raise EngineeringSmokeManifestError("{} sampler output binding mismatch".format(path))
    if sampler_output.canonical_bytes() != canonical_json_bytes(sampler_payload):
        raise EngineeringSmokeManifestError("{} sampler output serialization is not canonical".format(path))
    plan = NoiseSegmentPlan.from_payload(payload["noise_plan"])
    if plan.plan_id != block.noise_segment_plan_identity:
        raise EngineeringSmokeManifestError("{} noise plan/block binding mismatch".format(path))
    if plan.noise_parent_id != block.noise_parent_id:
        raise EngineeringSmokeManifestError("{} noise parent/block binding mismatch".format(path))
    episodes_payload = payload["episodes"]
    if not isinstance(episodes_payload, (list, tuple)) or len(episodes_payload) != 4:
        raise EngineeringSmokeManifestError("{} must contain exactly four episodes".format(path))
    episodes = tuple(EpisodeRecord.from_payload(item) for item in episodes_payload)
    expected_slots = (("selection", 0), ("selection", 1), ("evaluation", 0), ("evaluation", 1))
    if tuple((episode.role, episode.fixed_dry_noise_segment_identity["role_index"]) for episode in episodes) != expected_slots:
        raise EngineeringSmokeManifestError("{} episode order is not fixed selection/evaluation order".format(path))
    if any(episode.block_id != block.block_id for episode in episodes):
        raise EngineeringSmokeManifestError("{} episode/block binding mismatch".format(path))
    if tuple(episode.utterance_identity["utterance_id"] for episode in episodes) != (
        *block.selection_utterance_ids, *block.evaluation_utterance_ids
    ):
        raise EngineeringSmokeManifestError("{} utterance assignments do not match block plan".format(path))
    poses = payload["poses"]
    if not isinstance(poses, (list, tuple)) or any(not isinstance(item, Mapping) for item in poses):
        raise EngineeringSmokeManifestError("{}.poses must be a list of pose payloads".format(path))
    for pose in poses:
        PoseRecord.from_payload(pose)
        if pose["geometry_id"] != geometry.geometry_id:
            raise EngineeringSmokeManifestError("{} pose/geometry binding mismatch".format(path))
    if not isinstance(payload["speech_sources"], (list, tuple)) or len(payload["speech_sources"]) != 4:
        raise EngineeringSmokeManifestError("{} must contain four speech sources".format(path))
    if len({row["speaker_id"] for row in payload["speech_sources"]}) != 1:
        raise EngineeringSmokeManifestError("{} speech sources must use one speaker".format(path))
    for row in payload["speech_sources"]:
        _sha(row["source_file_sha256"], path + ".speech_sources.source_file_sha256")
        _sha(row["decoded_waveform_sha256"], path + ".speech_sources.decoded_waveform_sha256")
        if row["engineering_only"] is not True:
            raise EngineeringSmokeManifestError("{} speech source is not engineering-only".format(path))
    if payload["noise_audit"]["engineering_only"] is not True:
        raise EngineeringSmokeManifestError("{} noise audit is not engineering-only".format(path))
    if payload["selection_evaluation_policy"] != {
        "selection": "calibration_and_later_pose_choice_eligible",
        "evaluation": "never_changes_pose_choice",
    }:
        raise EngineeringSmokeManifestError("{} selection/evaluation policy is invalid".format(path))


@dataclass(frozen=True)
class EngineeringSmokeManifest:
    """The exact two-block metadata freeze for the later A4 smoke."""

    infrastructure_contract_sha256: str
    selection_policy: Mapping[str, Any]
    blocks: Tuple[Mapping[str, Any], ...]
    o2_exclusion: Mapping[str, Any]
    o1_reuse_classification_policy: str
    expected_frontends: Tuple[str, ...] = EXPECTED_FRONTENDS
    forbidden_result_dependent_selection: Tuple[str, ...] = FORBIDDEN_SELECTION_INPUTS
    engineering_only: bool = True
    state: str = ENGINEERING_SMOKE_MANIFEST_STATE
    schema_version: str = ENGINEERING_SMOKE_MANIFEST_SCHEMA_VERSION
    manifest_id: str = ""
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        _sha(self.infrastructure_contract_sha256, "manifest.infrastructure_contract_sha256")
        if self.schema_version != ENGINEERING_SMOKE_MANIFEST_SCHEMA_VERSION or self.state != ENGINEERING_SMOKE_MANIFEST_STATE:
            raise EngineeringSmokeManifestError("manifest schema/state is invalid")
        if self.engineering_only is not True:
            raise EngineeringSmokeManifestError("engineering_only must be true")
        if tuple(self.expected_frontends) != EXPECTED_FRONTENDS:
            raise EngineeringSmokeManifestError("expected_frontends must be mean_lr/fixed_L/fixed_R")
        if tuple(self.forbidden_result_dependent_selection) != FORBIDDEN_SELECTION_INPUTS:
            raise EngineeringSmokeManifestError("forbidden result-dependent selection list is invalid")
        blocks = tuple(self.blocks)
        if len(blocks) != 2:
            raise EngineeringSmokeManifestError("A4 engineering smoke requires exactly two blocks")
        for index, block in enumerate(blocks):
            _validate_block(block, index)
        if len({block["block_record"]["block_id"] for block in blocks}) != 2:
            raise EngineeringSmokeManifestError("block IDs must be distinct")
        if len({block["geometry_record"]["geometry_id"] for block in blocks}) != 2:
            raise EngineeringSmokeManifestError("geometry IDs must be distinct")
        if not isinstance(self.selection_policy, Mapping) or not self.selection_policy:
            raise EngineeringSmokeManifestError("selection_policy must be non-empty")
        if not isinstance(self.o2_exclusion, Mapping) or self.o2_exclusion.get("excluded") is not True:
            raise EngineeringSmokeManifestError("O2 exclusion must be explicit")
        _string(self.o1_reuse_classification_policy, "manifest.o1_reuse_classification_policy")
        _reject_result_fields(self.selection_policy, "manifest.selection_policy")
        _reject_result_fields(self.o2_exclusion, "manifest.o2_exclusion")
        expected_id = stable_id("engineering-smoke-manifest", self.identity_payload())
        if self.manifest_id:
            if self.manifest_id != expected_id:
                raise EngineeringSmokeManifestError("manifest_id does not match semantic payload")
        else:
            object.__setattr__(self, "manifest_id", expected_id)
        expected_sha = identity_sha256(self.identity_payload())
        if self.manifest_sha256:
            _sha(self.manifest_sha256, "manifest.manifest_sha256")
            if self.manifest_sha256 != expected_sha:
                raise EngineeringSmokeManifestError("manifest_sha256 does not match semantic payload")
        else:
            object.__setattr__(self, "manifest_sha256", expected_sha)
        object.__setattr__(self, "blocks", tuple(_freeze(block) for block in blocks))
        object.__setattr__(self, "selection_policy", _freeze(self.selection_policy))
        object.__setattr__(self, "o2_exclusion", _freeze(self.o2_exclusion))

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "engineering_only": self.engineering_only,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "selection_policy": _plain(self.selection_policy),
            "blocks": [_plain(block) for block in self.blocks],
            "expected_frontends": list(self.expected_frontends),
            "forbidden_result_dependent_selection": list(self.forbidden_result_dependent_selection),
            "o2_exclusion": _plain(self.o2_exclusion),
            "o1_reuse_classification_policy": self.o1_reuse_classification_policy,
        }

    def to_payload(self) -> Mapping[str, Any]:
        return dict(self.identity_payload(), manifest_id=self.manifest_id, manifest_sha256=self.manifest_sha256)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EngineeringSmokeManifest":
        _exact(payload, TOP_FIELDS, "engineering_smoke_manifest")
        if not isinstance(payload["blocks"], (list, tuple)):
            raise EngineeringSmokeManifestError("manifest.blocks must be a list")
        return cls(
            infrastructure_contract_sha256=payload["infrastructure_contract_sha256"],
            selection_policy=payload["selection_policy"],
            blocks=tuple(payload["blocks"]),
            o2_exclusion=payload["o2_exclusion"],
            o1_reuse_classification_policy=payload["o1_reuse_classification_policy"],
            expected_frontends=tuple(payload["expected_frontends"]),
            forbidden_result_dependent_selection=tuple(payload["forbidden_result_dependent_selection"]),
            engineering_only=payload["engineering_only"],
            state=payload["state"],
            schema_version=payload["schema_version"],
            manifest_id=payload["manifest_id"],
            manifest_sha256=payload["manifest_sha256"],
        )


__all__ = [
    "ENGINEERING_SMOKE_MANIFEST_SCHEMA_VERSION",
    "ENGINEERING_SMOKE_MANIFEST_STATE",
    "EngineeringSmokeManifest",
    "EngineeringSmokeManifestError",
    "EXPECTED_FRONTENDS",
    "FORBIDDEN_SELECTION_INPUTS",
]
