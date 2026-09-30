"""Strict deterministic O1 noise replacement candidate batches."""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence, Tuple

from active_audition.a4.identity import identity_sha256, stable_id, validate_sha256


O1_REPLACEMENT_BATCH_SCHEMA_VERSION = "active-asr-o1-noise-replacement-batch-v1"
O1_REPLACEMENT_POLICY_IDENTITY = "active-asr-o1-noise-replacement-technical-lexical-v1"
O1_REPLACEMENT_CANDIDATE_COUNT = 4
_CANDIDATE_FIELDS = (
    "candidate_index", "parent_recording_id", "relative_source_path",
    "source_file_sha256", "decoded_waveform_sha256", "sample_rate_hz",
    "sample_count", "technical_audit",
)
_TECHNICAL_FIELDS = (
    "corpus", "subset", "excluded", "readable", "too_short", "extreme_silence",
    "sample_rate_hz", "source_file_exists", "source_file_sha256_matches",
    "decoded_waveform_sha256_matches", "decoded_mono_finite_float32",
    "sample_count_sufficient",
)


class O1ReplacementError(ValueError):
    pass


def _exact(value: Mapping[str, Any], fields: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise O1ReplacementError("{} must be a mapping".format(path))
    unknown = sorted(set(value) - set(fields))
    missing = sorted(set(fields) - set(value))
    if unknown or missing:
        raise O1ReplacementError("{} fields invalid: unknown={}, missing={}".format(path, unknown, missing))


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise O1ReplacementError("{} must be a non-empty string".format(path))
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise O1ReplacementError("{} must be a positive integer".format(path))
    return int(value)


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise O1ReplacementError(str(exc)) from exc


def _candidate(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    _exact(value, _CANDIDATE_FIELDS, path)
    if value["candidate_index"] <= 0:
        raise O1ReplacementError("{}.candidate_index must be positive".format(path))
    _string(value["parent_recording_id"], path + ".parent_recording_id")
    _string(value["relative_source_path"], path + ".relative_source_path")
    _sha(value["source_file_sha256"], path + ".source_file_sha256")
    _sha(value["decoded_waveform_sha256"], path + ".decoded_waveform_sha256")
    if value["sample_rate_hz"] != 16000:
        raise O1ReplacementError("{} must use native 16000 Hz".format(path))
    _positive_int(value["sample_rate_hz"], path + ".sample_rate_hz")
    _positive_int(value["sample_count"], path + ".sample_count")
    _exact(value["technical_audit"], _TECHNICAL_FIELDS, path + ".technical_audit")
    for field in ("excluded", "readable", "too_short", "extreme_silence", "source_file_exists", "source_file_sha256_matches", "decoded_waveform_sha256_matches", "decoded_mono_finite_float32", "sample_count_sufficient"):
        if not isinstance(value["technical_audit"][field], bool):
            raise O1ReplacementError("{}.technical_audit.{} must be boolean".format(path, field))
    if value["technical_audit"]["corpus"] != "MUSAN" or value["technical_audit"]["subset"] != "noise":
        raise O1ReplacementError("{} is not a MUSAN noise candidate".format(path))
    if any(not value["technical_audit"][field] for field in ("source_file_exists", "source_file_sha256_matches", "decoded_waveform_sha256_matches", "decoded_mono_finite_float32", "sample_count_sufficient")):
        raise O1ReplacementError("{} has failed technical provenance".format(path))
    return dict(value)


@dataclass(frozen=True)
class O1ReplacementCandidateBatch:
    schema_version: str
    selection_policy_identity: str
    registry_relative_path: str
    registry_sha256: str
    required_parent_sample_count: int
    required_parent_duration_sec: float
    excluded_parent_recording_ids: Tuple[str, ...]
    candidate_records: Tuple[Mapping[str, Any], ...]
    engineering_only: bool = True
    batch_id: str = ""
    batch_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != O1_REPLACEMENT_BATCH_SCHEMA_VERSION:
            raise O1ReplacementError("unsupported replacement batch schema")
        if self.selection_policy_identity != O1_REPLACEMENT_POLICY_IDENTITY:
            raise O1ReplacementError("unsupported replacement selection policy")
        _string(self.registry_relative_path, "registry_relative_path")
        _sha(self.registry_sha256, "registry_sha256")
        _positive_int(self.required_parent_sample_count, "required_parent_sample_count")
        if not math.isclose(self.required_parent_duration_sec, self.required_parent_sample_count / 16000.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise O1ReplacementError("required parent duration is not derived from samples")
        excluded = tuple(self.excluded_parent_recording_ids)
        if len(set(excluded)) != len(excluded) or tuple(sorted(excluded)) != excluded:
            raise O1ReplacementError("excluded parent IDs must be unique lexical order")
        if self.engineering_only is not True:
            raise O1ReplacementError("replacement batch must be engineering-only")
        records = tuple(_candidate(item, "candidate_records[{}]".format(index)) for index, item in enumerate(self.candidate_records))
        if len(records) != O1_REPLACEMENT_CANDIDATE_COUNT:
            raise O1ReplacementError("replacement batch must contain exactly four candidates")
        ids = tuple(item["parent_recording_id"] for item in records)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise O1ReplacementError("candidate order must be lexical and unique")
        if tuple(item["candidate_index"] for item in records) != (1, 2, 3, 4):
            raise O1ReplacementError("candidate indices must be 1..4")
        if set(ids) & set(excluded):
            raise O1ReplacementError("excluded/current parent re-entered replacement batch")
        object.__setattr__(self, "excluded_parent_recording_ids", excluded)
        object.__setattr__(self, "candidate_records", records)
        expected_id = stable_id("o1-replacement-batch", self.identity_payload())
        expected_sha = identity_sha256(self.identity_payload())
        if self.batch_id and self.batch_id != expected_id:
            raise O1ReplacementError("batch_id mismatch")
        if self.batch_sha256 and self.batch_sha256 != expected_sha:
            raise O1ReplacementError("batch_sha256 mismatch")
        object.__setattr__(self, "batch_id", expected_id)
        object.__setattr__(self, "batch_sha256", expected_sha)

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_policy_identity": self.selection_policy_identity,
            "registry_relative_path": self.registry_relative_path,
            "registry_sha256": self.registry_sha256,
            "required_parent_sample_count": self.required_parent_sample_count,
            "required_parent_duration_sec": self.required_parent_duration_sec,
            "excluded_parent_recording_ids": list(self.excluded_parent_recording_ids),
            "candidate_records": [dict(item) for item in self.candidate_records],
            "engineering_only": self.engineering_only,
        }

    def to_payload(self) -> Mapping[str, Any]:
        return dict(self.identity_payload(), batch_id=self.batch_id, batch_sha256=self.batch_sha256)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1ReplacementCandidateBatch":
        fields = tuple(cls.__dataclass_fields__)
        _exact(payload, fields, "o1_replacement_candidate_batch")
        values = dict(payload)
        values["excluded_parent_recording_ids"] = tuple(values["excluded_parent_recording_ids"])
        values["candidate_records"] = tuple(values["candidate_records"])
        return cls(**values)


__all__ = [
    "O1_REPLACEMENT_BATCH_SCHEMA_VERSION", "O1_REPLACEMENT_CANDIDATE_COUNT",
    "O1_REPLACEMENT_POLICY_IDENTITY", "O1ReplacementCandidateBatch", "O1ReplacementError",
]
