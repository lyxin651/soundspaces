"""Strict, result-independent A4 engineering qualification evidence.

The qualification artifact records whether the frozen engineering chain is
technically valid.  It deliberately does not contain pose selection results,
rankings, WER comparisons, or movement-benefit claims.  Per-record ASR
diagnostics are a separate, append-only evidence file.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from active_audition.a4.identity import canonical_json, identity_sha256, stable_id, validate_sha256


QUALIFICATION_SCHEMA_VERSION = "active-asr-a4-engineering-qualification-v1"
QUALIFICATION_ALGORITHM_IDENTITY = "active-asr-a4-g1-g9-technical-qualification-v1"
QUALIFICATION_STATES = (
    "SERVER_RUN_COMPLETE_PENDING_REVIEW",
    "RESOURCE_PROFILE_PENDING_GPU_REPLAY",
)
GATE_NAMES = tuple("G{}".format(index) for index in range(1, 10))
_FORBIDDEN_KEYS = frozenset(
    {
        "oracle",
        "best_pose",
        "pose_rank",
        "ranking",
        "nearest_best",
        "rotate_best",
        "translation_best",
        "stay_vs_move",
        "movement_benefit",
        "bootstrap",
        "minimum_wer_pose",
        "maximum_wer_pose",
    }
)


class QualificationError(ValueError):
    """Raised when qualification evidence is malformed or inconsistent."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _reject_forbidden(value: Any, path: str = "qualification") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_KEYS:
                raise QualificationError("{} contains forbidden result-dependent field {!r}".format(path, key))
            _reject_forbidden(item, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden(item, "{}[{}]".format(path, index))


def _exact(payload: Mapping[str, Any], fields: Sequence[str], path: str) -> None:
    if not isinstance(payload, Mapping):
        raise QualificationError("{} must be a mapping".format(path))
    unknown = sorted(set(payload) - set(fields))
    missing = sorted(set(fields) - set(payload))
    if unknown or missing:
        raise QualificationError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise QualificationError(str(exc)) from exc


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError("{} must be a non-empty string".format(path))
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("{} must be a mapping".format(path))
    return value


@dataclass(frozen=True)
class A4QualificationArtifact:
    """Immutable G1-G9 technical evidence projection."""

    infrastructure_contract_sha256: str
    smoke_manifest_sha256: str
    code_head: str
    gate_records: Mapping[str, Mapping[str, Any]]
    expected_counts: Mapping[str, int]
    valid_counts: Mapping[str, int]
    calibration_evidence: Mapping[str, Any]
    cache_reconciliation_identity: Mapping[str, Any]
    completion_marker_identity: Mapping[str, Any]
    resource_profile_references: Mapping[str, Any]
    asr_diagnostic_references: Mapping[str, Any]
    qualification_state: str
    schema_version: str = QUALIFICATION_SCHEMA_VERSION
    algorithm_identity: str = QUALIFICATION_ALGORITHM_IDENTITY
    artifact_id: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _sha(self.infrastructure_contract_sha256, "qualification.infrastructure_contract_sha256")
        _sha(self.smoke_manifest_sha256, "qualification.smoke_manifest_sha256")
        _string(self.code_head, "qualification.code_head")
        _string(self.schema_version, "qualification.schema_version")
        if self.schema_version != QUALIFICATION_SCHEMA_VERSION:
            raise QualificationError("qualification.schema_version is invalid")
        if self.algorithm_identity != QUALIFICATION_ALGORITHM_IDENTITY:
            raise QualificationError("qualification.algorithm_identity is invalid")
        if self.qualification_state not in QUALIFICATION_STATES:
            raise QualificationError("qualification.qualification_state is invalid")
        if tuple(sorted(self.gate_records)) != GATE_NAMES:
            raise QualificationError("qualification.gate_records must contain exactly G1..G9")
        for gate in GATE_NAMES:
            record = _mapping(self.gate_records[gate], "qualification.gate_records." + gate)
            if record.get("status") != "PASS":
                raise QualificationError("{} must have status PASS".format(gate))
        for name, value in (("expected_counts", self.expected_counts), ("valid_counts", self.valid_counts)):
            mapping = _mapping(value, "qualification." + name)
            for key, count in mapping.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise QualificationError("{}.{} must be a non-negative integer".format(name, key))
        for name, value in (
            ("calibration_evidence", self.calibration_evidence),
            ("cache_reconciliation_identity", self.cache_reconciliation_identity),
            ("completion_marker_identity", self.completion_marker_identity),
            ("resource_profile_references", self.resource_profile_references),
            ("asr_diagnostic_references", self.asr_diagnostic_references),
        ):
            _mapping(value, "qualification." + name)
        _reject_forbidden(self.identity_payload())
        expected_id = stable_id("a4-qualification", self.identity_payload())
        expected_sha = identity_sha256(self.identity_payload())
        if self.artifact_id:
            _string(self.artifact_id, "qualification.artifact_id")
            if self.artifact_id != expected_id:
                raise QualificationError("qualification.artifact_id does not match semantic payload")
        else:
            object.__setattr__(self, "artifact_id", expected_id)
        if self.artifact_sha256:
            _sha(self.artifact_sha256, "qualification.artifact_sha256")
            if self.artifact_sha256 != expected_sha:
                raise QualificationError("qualification.artifact_sha256 does not match semantic payload")
        else:
            object.__setattr__(self, "artifact_sha256", expected_sha)

    def identity_payload(self) -> Dict[str, Any]:
        # code_head is provenance only.  It is intentionally excluded from
        # the semantic qualification identity.
        return {
            "schema_version": self.schema_version,
            "algorithm_identity": self.algorithm_identity,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "smoke_manifest_sha256": self.smoke_manifest_sha256,
            "gate_records": _plain(self.gate_records),
            "expected_counts": _plain(self.expected_counts),
            "valid_counts": _plain(self.valid_counts),
            "calibration_evidence": _plain(self.calibration_evidence),
            "cache_reconciliation_identity": _plain(self.cache_reconciliation_identity),
            "completion_marker_identity": _plain(self.completion_marker_identity),
            "resource_profile_references": _plain(self.resource_profile_references),
            "asr_diagnostic_references": _plain(self.asr_diagnostic_references),
            "qualification_state": self.qualification_state,
        }

    def to_payload(self) -> Dict[str, Any]:
        return dict(
            self.identity_payload(),
            code_head=self.code_head,
            artifact_id=self.artifact_id,
            artifact_sha256=self.artifact_sha256,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "A4QualificationArtifact":
        _exact(
            payload,
            (
                "schema_version", "algorithm_identity", "infrastructure_contract_sha256",
                "smoke_manifest_sha256", "code_head", "gate_records", "expected_counts",
                "valid_counts", "calibration_evidence", "cache_reconciliation_identity",
                "completion_marker_identity", "resource_profile_references",
                "asr_diagnostic_references", "qualification_state", "artifact_id",
                "artifact_sha256",
            ),
            "qualification",
        )
        return cls(**dict(payload))


__all__ = [
    "A4QualificationArtifact",
    "GATE_NAMES",
    "QUALIFICATION_ALGORITHM_IDENTITY",
    "QUALIFICATION_SCHEMA_VERSION",
    "QUALIFICATION_STATES",
    "QualificationError",
]
