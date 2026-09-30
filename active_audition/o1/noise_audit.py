"""Strict O1 manual noise-parent audit records.

This authority is intentionally separate from the frozen A3 MUSAN registry.
It records human listening decisions only; technical registry eligibility is
not treated as a content audit decision.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.identity import identity_sha256, stable_id, validate_sha256


O1_NOISE_AUDIT_SCHEMA_VERSION = "active-asr-o1-noise-parent-audit-v1"
O1_NOISE_AUDIT_POLICY_IDENTITY = "active-asr-o1-manual-noise-audit-policy-v1"
O1_NOISE_AUDIT_DECISION_SOURCE = "user_manual_listening"
O1_NOISE_AUDIT_STATUSES = ("PASS", "FLAGGED", "UNCERTAIN", "PENDING_USER", "NOT_EVALUATED")
O1_NOISE_AUDIT_FIELDS = (
    "speech_leakage",
    "strong_reverberation",
    "indoor_localized_source_compatibility",
)


class O1NoiseAuditError(ValueError):
    pass


def _exact(value: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise O1NoiseAuditError("{} must be a mapping".format(path))
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise O1NoiseAuditError("{} fields invalid: unknown={}, missing={}".format(path, unknown, missing))


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise O1NoiseAuditError("{} must be a non-empty string".format(path))
    return value


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise O1NoiseAuditError(str(exc)) from exc


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise O1NoiseAuditError("{} must be a positive integer".format(path))
    return int(value)


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise O1NoiseAuditError("{} must be a non-negative integer".format(path))
    return int(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _preview(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    fields = (
        "relative_preview_path", "clip_index", "center_fraction", "start_sample", "end_sample",
        "start_sec", "end_sec", "clip_sample_count", "expected_source_slice_float32_sha256",
        "written_clip_decoded_float32_sha256", "verification_status",
    )
    _exact(value, fields, path)
    _string(value["relative_preview_path"], path + ".relative_preview_path")
    if not isinstance(value["clip_index"], int) or value["clip_index"] <= 0:
        raise O1NoiseAuditError(path + ".clip_index must be positive")
    for field in ("start_sample", "end_sample", "clip_sample_count"):
        _nonnegative_int(value[field], path + "." + field)
    if value["end_sample"] <= value["start_sample"] or value["clip_sample_count"] != value["end_sample"] - value["start_sample"]:
        raise O1NoiseAuditError(path + " sample bounds are inconsistent")
    for field in ("center_fraction", "start_sec", "end_sec"):
        if isinstance(value[field], bool) or not isinstance(value[field], (int, float)):
            raise O1NoiseAuditError(path + ".{} must be numeric".format(field))
    if value["end_sec"] <= value["start_sec"]:
        raise O1NoiseAuditError(path + " seconds are inconsistent")
    _sha(value["expected_source_slice_float32_sha256"], path + ".expected_source_slice_float32_sha256")
    _sha(value["written_clip_decoded_float32_sha256"], path + ".written_clip_decoded_float32_sha256")
    if value["verification_status"] != "SAMPLE_EXACT" or value["expected_source_slice_float32_sha256"] != value["written_clip_decoded_float32_sha256"]:
        raise O1NoiseAuditError(path + " is not sample-exact")
    return _freeze(value)


@dataclass(frozen=True)
class O1NoiseParentAuditRecord:
    schema_version: str
    audit_policy_identity: str
    decision_source: str
    parent_recording_id: str
    relative_source_path: str
    source_file_sha256: str
    decoded_waveform_sha256: str
    sample_rate_hz: int
    sample_count: int
    reviewed_preview_identities: Tuple[Mapping[str, Any], ...]
    review_scope: Mapping[str, Any]
    manual_decision_provenance: Mapping[str, Any]
    speech_leakage: str
    strong_reverberation: str
    indoor_localized_source_compatibility: str
    selected_for_o1_scientific_use: bool
    exclusion_reason: str
    engineering_only: bool
    audit_record_id: str = ""
    audit_record_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != O1_NOISE_AUDIT_SCHEMA_VERSION:
            raise O1NoiseAuditError("unsupported O1 noise audit schema")
        if self.audit_policy_identity != O1_NOISE_AUDIT_POLICY_IDENTITY or self.decision_source != O1_NOISE_AUDIT_DECISION_SOURCE:
            raise O1NoiseAuditError("invalid O1 noise audit policy/provenance")
        for field in ("parent_recording_id", "relative_source_path"):
            _string(getattr(self, field), field)
        for field in ("source_file_sha256", "decoded_waveform_sha256"):
            _sha(getattr(self, field), field)
        if self.sample_rate_hz != 16000:
            raise O1NoiseAuditError("O1 noise audit requires native 16000 Hz")
        _positive_int(self.sample_rate_hz, "sample_rate_hz")
        _positive_int(self.sample_count, "sample_count")
        previews = tuple(_preview(item, "reviewed_preview_identities[{}]".format(index)) for index, item in enumerate(self.reviewed_preview_identities))
        if len(previews) != 6:
            raise O1NoiseAuditError("O1 audit requires exactly six deterministic previews")
        object.__setattr__(self, "reviewed_preview_identities", previews)
        for field in O1_NOISE_AUDIT_FIELDS:
            if getattr(self, field) not in O1_NOISE_AUDIT_STATUSES:
                raise O1NoiseAuditError("{} has invalid status".format(field))
        if not isinstance(self.review_scope, Mapping) or not isinstance(self.manual_decision_provenance, Mapping):
            raise O1NoiseAuditError("audit scope/provenance must be mappings")
        object.__setattr__(self, "review_scope", _freeze(self.review_scope))
        object.__setattr__(self, "manual_decision_provenance", _freeze(self.manual_decision_provenance))
        if not isinstance(self.selected_for_o1_scientific_use, bool) or self.engineering_only is not True:
            raise O1NoiseAuditError("invalid selection/engineering flags")
        all_pass = all(getattr(self, field) == "PASS" for field in O1_NOISE_AUDIT_FIELDS)
        if self.selected_for_o1_scientific_use != all_pass:
            raise O1NoiseAuditError("scientific selection requires exactly three PASS decisions")
        if self.selected_for_o1_scientific_use and self.exclusion_reason:
            raise O1NoiseAuditError("selected parent cannot have exclusion_reason")
        if not self.selected_for_o1_scientific_use and not self.exclusion_reason:
            raise O1NoiseAuditError("unselected parent requires exclusion_reason")
        object.__setattr__(self, "exclusion_reason", _string(self.exclusion_reason, "exclusion_reason") if self.exclusion_reason else "")
        expected_id = stable_id("o1-noise-audit", self.identity_payload())
        expected_sha = identity_sha256(self.identity_payload())
        if self.audit_record_id and self.audit_record_id != expected_id:
            raise O1NoiseAuditError("audit_record_id mismatch")
        if self.audit_record_sha256 and self.audit_record_sha256 != expected_sha:
            raise O1NoiseAuditError("audit_record_sha256 mismatch")
        object.__setattr__(self, "audit_record_id", expected_id)
        object.__setattr__(self, "audit_record_sha256", expected_sha)

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_policy_identity": self.audit_policy_identity,
            "decision_source": self.decision_source,
            "parent_recording_id": self.parent_recording_id,
            "relative_source_path": self.relative_source_path,
            "source_file_sha256": self.source_file_sha256,
            "decoded_waveform_sha256": self.decoded_waveform_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "reviewed_preview_identities": [_plain(item) for item in self.reviewed_preview_identities],
            "review_scope": _plain(self.review_scope),
            "manual_decision_provenance": _plain(self.manual_decision_provenance),
            "speech_leakage": self.speech_leakage,
            "strong_reverberation": self.strong_reverberation,
            "indoor_localized_source_compatibility": self.indoor_localized_source_compatibility,
            "selected_for_o1_scientific_use": self.selected_for_o1_scientific_use,
            "exclusion_reason": self.exclusion_reason,
            "engineering_only": self.engineering_only,
        }

    def to_payload(self) -> Mapping[str, Any]:
        return dict(self.identity_payload(), audit_record_id=self.audit_record_id, audit_record_sha256=self.audit_record_sha256)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1NoiseParentAuditRecord":
        expected = tuple(cls.__dataclass_fields__)
        _exact(payload, expected, "o1_noise_parent_audit_record")
        values = dict(payload)
        values["reviewed_preview_identities"] = tuple(values["reviewed_preview_identities"])
        return cls(**values)


__all__ = [
    "O1_NOISE_AUDIT_FIELDS", "O1_NOISE_AUDIT_POLICY_IDENTITY", "O1_NOISE_AUDIT_SCHEMA_VERSION",
    "O1_NOISE_AUDIT_STATUSES", "O1NoiseAuditError", "O1NoiseParentAuditRecord",
]
