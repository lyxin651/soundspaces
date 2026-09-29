"""Strict user-manual audit records for A4 noise-parent eligibility.

The record is deliberately separate from the frozen A3 MUSAN registry.  It
captures a human decision made before any A4 RIR, mixture, ASR, or WER result
exists.  Timestamps and run provenance are intentionally outside the semantic
identity.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.identity import identity_sha256, stable_id, validate_sha256


NOISE_AUDIT_SCHEMA_VERSION = "active-asr-a4-noise-parent-audit-v1"
NOISE_AUDIT_POLICY_IDENTITY = "active-asr-a4-manual-noise-audit-policy-v1"
NOISE_AUDIT_DECISION_SOURCE = "user_manual_listening"
AUDIT_STATUSES = ("PASS", "FLAGGED", "NOT_EVALUATED")
AUDIT_FIELDS = (
    "speech_leakage",
    "strong_reverberation",
    "indoor_localized_source_compatibility",
)


class NoiseAuditError(ValueError):
    """Raised when an A4 noise audit record is malformed or inconsistent."""


def _exact(value: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise NoiseAuditError("{} must be a mapping".format(path))
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise NoiseAuditError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise NoiseAuditError("{} must be a non-empty string".format(path))
    return value


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise NoiseAuditError(str(exc)) from exc


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NoiseAuditError("{} must be a positive integer".format(path))
    return int(value)


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NoiseAuditError("{} must be a non-negative integer".format(path))
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
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _preview(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    fields = (
        "relative_preview_path",
        "clip_index",
        "start_sample",
        "end_sample",
        "expected_source_slice_float32_sha256",
        "written_clip_decoded_float32_sha256",
        "verification_status",
    )
    _exact(value, fields, path)
    _string(value["relative_preview_path"], path + ".relative_preview_path")
    clip_index = _positive_int(value["clip_index"], path + ".clip_index")
    start = _nonnegative_int(value["start_sample"], path + ".start_sample")
    end = _positive_int(value["end_sample"], path + ".end_sample")
    if end <= start:
        raise NoiseAuditError(path + ".end_sample must exceed start_sample")
    if clip_index <= 0:
        raise NoiseAuditError(path + ".clip_index must be positive")
    _sha(value["expected_source_slice_float32_sha256"], path + ".expected_source_slice_float32_sha256")
    _sha(value["written_clip_decoded_float32_sha256"], path + ".written_clip_decoded_float32_sha256")
    if value["verification_status"] != "SAMPLE_EXACT":
        raise NoiseAuditError(path + ".verification_status must be SAMPLE_EXACT")
    if value["expected_source_slice_float32_sha256"] != value["written_clip_decoded_float32_sha256"]:
        raise NoiseAuditError(path + " preview sample hashes differ")
    return _freeze(value)


@dataclass(frozen=True)
class NoiseParentAuditRecord:
    schema_version: str
    audit_policy_identity: str
    decision_source: str
    source_candidate_batch_identity: str
    source_candidate_batch_sha256: str
    parent_recording_id: str
    relative_source_path: str
    source_file_sha256: str
    decoded_waveform_sha256: str
    sample_rate_hz: int
    sample_count: int
    review_scope: Mapping[str, Any]
    reviewed_preview_identities: Tuple[Mapping[str, Any], ...]
    speech_leakage: str
    strong_reverberation: str
    indoor_localized_source_compatibility: str
    selected_for_a4_smoke: bool
    exclusion_reason: str
    engineering_only: bool
    audit_record_id: str = ""
    audit_record_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != NOISE_AUDIT_SCHEMA_VERSION:
            raise NoiseAuditError("schema_version is not the A4 noise audit schema")
        if self.audit_policy_identity != NOISE_AUDIT_POLICY_IDENTITY:
            raise NoiseAuditError("audit_policy_identity is not the A4 policy")
        if self.decision_source != NOISE_AUDIT_DECISION_SOURCE:
            raise NoiseAuditError("decision_source must be user_manual_listening")
        for field in ("source_candidate_batch_identity", "parent_recording_id", "relative_source_path"):
            _string(getattr(self, field), field)
        _sha(self.source_candidate_batch_sha256, "source_candidate_batch_sha256")
        _sha(self.source_file_sha256, "source_file_sha256")
        _sha(self.decoded_waveform_sha256, "decoded_waveform_sha256")
        _positive_int(self.sample_rate_hz, "sample_rate_hz")
        _positive_int(self.sample_count, "sample_count")
        if self.sample_rate_hz != 16000:
            raise NoiseAuditError("A4 noise audit requires native 16000 Hz")
        if not isinstance(self.review_scope, Mapping) or not self.review_scope:
            raise NoiseAuditError("review_scope must be a non-empty mapping")
        object.__setattr__(self, "review_scope", _freeze(self.review_scope))
        previews = tuple(_preview(item, "reviewed_preview_identities[{}]".format(index)) for index, item in enumerate(self.reviewed_preview_identities))
        if not previews:
            raise NoiseAuditError("at least one reviewed preview is required")
        object.__setattr__(self, "reviewed_preview_identities", previews)
        for field in AUDIT_FIELDS:
            status = getattr(self, field)
            if status not in AUDIT_STATUSES:
                raise NoiseAuditError("{} has invalid status".format(field))
        if not isinstance(self.selected_for_a4_smoke, bool):
            raise NoiseAuditError("selected_for_a4_smoke must be boolean")
        _string(self.exclusion_reason, "exclusion_reason") if not self.selected_for_a4_smoke else None
        if self.selected_for_a4_smoke:
            if any(getattr(self, field) != "PASS" for field in AUDIT_FIELDS):
                raise NoiseAuditError("selected parent requires PASS for all three audit dimensions")
            if self.exclusion_reason:
                raise NoiseAuditError("selected parent cannot have exclusion_reason")
        elif not self.exclusion_reason:
            raise NoiseAuditError("excluded parent requires exclusion_reason")
        if self.engineering_only is not True:
            raise NoiseAuditError("engineering_only must be true")
        expected_id = stable_id("a4-noise-audit", self.identity_payload())
        expected_sha = identity_sha256(self.identity_payload())
        if self.audit_record_id:
            if self.audit_record_id != expected_id:
                raise NoiseAuditError("audit_record_id does not match semantic payload")
        else:
            object.__setattr__(self, "audit_record_id", expected_id)
        if self.audit_record_sha256:
            _sha(self.audit_record_sha256, "audit_record_sha256")
            if self.audit_record_sha256 != expected_sha:
                raise NoiseAuditError("audit_record_sha256 does not match semantic payload")
        else:
            object.__setattr__(self, "audit_record_sha256", expected_sha)

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_policy_identity": self.audit_policy_identity,
            "decision_source": self.decision_source,
            "source_candidate_batch_identity": self.source_candidate_batch_identity,
            "source_candidate_batch_sha256": self.source_candidate_batch_sha256,
            "parent_recording_id": self.parent_recording_id,
            "relative_source_path": self.relative_source_path,
            "source_file_sha256": self.source_file_sha256,
            "decoded_waveform_sha256": self.decoded_waveform_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "review_scope": _plain(self.review_scope),
            "reviewed_preview_identities": [_plain(item) for item in self.reviewed_preview_identities],
            "speech_leakage": self.speech_leakage,
            "strong_reverberation": self.strong_reverberation,
            "indoor_localized_source_compatibility": self.indoor_localized_source_compatibility,
            "selected_for_a4_smoke": self.selected_for_a4_smoke,
            "exclusion_reason": self.exclusion_reason,
            "engineering_only": self.engineering_only,
        }

    def to_payload(self) -> Mapping[str, Any]:
        return dict(self.identity_payload(), audit_record_id=self.audit_record_id, audit_record_sha256=self.audit_record_sha256)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoiseParentAuditRecord":
        expected = tuple(cls.__dataclass_fields__)
        _exact(payload, expected, "noise_parent_audit_record")
        values = dict(payload)
        values["reviewed_preview_identities"] = tuple(values["reviewed_preview_identities"])
        return cls(**values)


__all__ = ["AUDIT_FIELDS", "AUDIT_STATUSES", "NOISE_AUDIT_POLICY_IDENTITY", "NOISE_AUDIT_SCHEMA_VERSION", "NoiseAuditError", "NoiseParentAuditRecord"]
