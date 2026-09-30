"""Strict finalized manual decisions for an O1 replacement batch."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.identity import identity_sha256, stable_id, validate_sha256
from active_audition.o1.noise_audit import O1NoiseParentAuditRecord, O1NoiseAuditError


FINALIZED_REPLACEMENT_AUDIT_SCHEMA = "active-asr-o1-noise-replacement-audit-finalized-v1"
FINALIZED_DECISION_SOURCE = "user_manual_listening"
FINALIZED_POLICY_IDENTITY = "active-asr-o1-manual-noise-audit-policy-v1"


class O1FinalizedReplacementAuditError(ValueError):
    pass


def _exact(value: Mapping[str, Any], fields: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise O1FinalizedReplacementAuditError("{} must be a mapping".format(path))
    unknown = sorted(set(value) - set(fields))
    missing = sorted(set(fields) - set(value))
    if unknown or missing:
        raise O1FinalizedReplacementAuditError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise O1FinalizedReplacementAuditError("{} must be a non-empty string".format(path))
    return value


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise O1FinalizedReplacementAuditError(str(exc)) from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class O1FinalizedReplacementAuditBatch:
    schema_version: str
    audit_policy_identity: str
    decision_source: str
    source_replacement_batch_id: str
    source_replacement_batch_sha256: str
    records: Tuple[Mapping[str, Any], ...]
    selected_replacement_slots: Mapping[str, Mapping[str, str]]
    engineering_only: bool = True
    batch_id: str = ""
    batch_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != FINALIZED_REPLACEMENT_AUDIT_SCHEMA:
            raise O1FinalizedReplacementAuditError("unsupported finalized replacement audit schema")
        if self.audit_policy_identity != FINALIZED_POLICY_IDENTITY:
            raise O1FinalizedReplacementAuditError("invalid audit policy identity")
        if self.decision_source != FINALIZED_DECISION_SOURCE:
            raise O1FinalizedReplacementAuditError("invalid decision source")
        _string(self.source_replacement_batch_id, "source_replacement_batch_id")
        _sha(self.source_replacement_batch_sha256, "source_replacement_batch_sha256")
        if self.engineering_only is not True:
            raise O1FinalizedReplacementAuditError("finalized replacement audit must be engineering-only")
        if len(self.records) != 4:
            raise O1FinalizedReplacementAuditError("exactly four replacement audit records are required")
        parsed = []
        for index, payload in enumerate(self.records):
            try:
                parsed.append(O1NoiseParentAuditRecord.from_payload(payload))
            except O1NoiseAuditError as exc:
                raise O1FinalizedReplacementAuditError("records[{}] invalid: {}".format(index, exc)) from exc
        ids = tuple(record.parent_recording_id for record in parsed)
        if ids != tuple(sorted(ids)) or len(set(ids)) != 4:
            raise O1FinalizedReplacementAuditError("records must be lexical and unique")
        selected = tuple(record.parent_recording_id for record in parsed if record.selected_for_o1_scientific_use)
        if selected != ("noise/free-sound/noise-free-sound-0041", "noise/free-sound/noise-free-sound-0073"):
            raise O1FinalizedReplacementAuditError("replacement selection is not the frozen first-two-PASS result")
        expected_slots = {
            "slot1": {
                "parent_recording_id": "noise/free-sound/noise-free-sound-0041",
                "replaces_parent_recording_id": "noise/free-sound/noise-free-sound-0015",
            },
            "slot2": {
                "parent_recording_id": "noise/free-sound/noise-free-sound-0073",
                "replaces_parent_recording_id": "noise/free-sound/noise-free-sound-0030",
            },
        }
        if _plain(self.selected_replacement_slots) != expected_slots:
            raise O1FinalizedReplacementAuditError("selected replacement slot mapping is invalid")
        object.__setattr__(self, "records", tuple(record.to_payload() for record in parsed))
        identity = self.identity_payload()
        expected_id = stable_id("o1-noise-audit-batch", identity)
        expected_sha = identity_sha256(identity)
        if self.batch_id and self.batch_id != expected_id:
            raise O1FinalizedReplacementAuditError("batch_id mismatch")
        if self.batch_sha256 and self.batch_sha256 != expected_sha:
            raise O1FinalizedReplacementAuditError("batch_sha256 mismatch")
        object.__setattr__(self, "batch_id", expected_id)
        object.__setattr__(self, "batch_sha256", expected_sha)

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_policy_identity": self.audit_policy_identity,
            "decision_source": self.decision_source,
            "source_replacement_batch_id": self.source_replacement_batch_id,
            "source_replacement_batch_sha256": self.source_replacement_batch_sha256,
            "records": [_plain(item) for item in self.records],
            "selected_replacement_slots": _plain(self.selected_replacement_slots),
            "engineering_only": self.engineering_only,
        }

    def to_payload(self) -> Mapping[str, Any]:
        return dict(self.identity_payload(), batch_id=self.batch_id, batch_sha256=self.batch_sha256)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1FinalizedReplacementAuditBatch":
        fields = tuple(cls.__dataclass_fields__)
        _exact(payload, fields, "o1_finalized_replacement_audit_batch")
        values = dict(payload)
        values["records"] = tuple(values["records"])
        return cls(**values)


__all__ = [
    "FINALIZED_DECISION_SOURCE", "FINALIZED_POLICY_IDENTITY",
    "FINALIZED_REPLACEMENT_AUDIT_SCHEMA", "O1FinalizedReplacementAuditBatch",
    "O1FinalizedReplacementAuditError",
]
