"""Pure cache-manifest reconciliation and completion-marker semantics.

This module is deliberately limited to resume planning.  It never renders a
RIR, produces a mixture, runs ASR, or chooses a scientific/engineering
fixture.  Cache entry validity is delegated to :class:`CacheStore` so resume
cannot accidentally weaken A4-3A integrity checks.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from active_audition.a4.cache import (
    HIT_VALID,
    INVALID_CORRUPT,
    MISS,
    AsrCacheKey,
    CacheError,
    CacheStore,
    MixtureCacheKey,
    RirCacheKey,
)
from active_audition.a4.identity import (
    canonical_json,
    identity_sha256,
    stable_id,
    validate_sha256,
    validate_stable_id,
)


CACHE_EXPECTED_MANIFEST_SCHEMA_VERSION = "active-asr-a4-cache-expected-manifest-v1"
CACHE_ENTRY_RESULT_SCHEMA_VERSION = "active-asr-a4-cache-entry-result-v1"
CACHE_RECONCILIATION_RECORD_SCHEMA_VERSION = "active-asr-a4-cache-reconciliation-record-v1"
CACHE_COMPLETION_MARKER_SCHEMA_VERSION = "active-asr-a4-cache-completion-marker-v1"
RESUME_ALGORITHM_IDENTITY = "active-asr-a4-deterministic-resume-reconciliation-v1"
RECONCILIATION_ALGORITHM_IDENTITY = "active-asr-a4-cache-reconciliation-v1"
COMPLETION_ALGORITHM_IDENTITY = "active-asr-a4-strict-completion-marker-v1"
COMPLETION_MARKER_FILENAME = "completion.json"
COMPLETION_MARKER_SCOPE_IDENTITY = "manifest_scoped_resume_v1"

LAYER_RIR = "rir"
LAYER_MIXTURE = "mixture"
LAYER_ASR = "asr"
LAYER_ORDER = (LAYER_RIR, LAYER_MIXTURE, LAYER_ASR)

REUSE_VALID = "REUSE_VALID"
REBUILD_MISSING = "REBUILD_MISSING"
REBUILD_CORRUPT = "REBUILD_CORRUPT"
ENTRY_STATUSES = (REUSE_VALID, REBUILD_MISSING, REBUILD_CORRUPT)

COMPLETION_MARKER_ABSENT = "ABSENT"
COMPLETION_MARKER_VALID = "VALID"
STALE_COMPLETION_MARKER = "STALE_COMPLETION_MARKER"
INVALID_COMPLETION_MARKER = "INVALID_COMPLETION_MARKER"
MARKER_STATUSES = (
    COMPLETION_MARKER_ABSENT,
    COMPLETION_MARKER_VALID,
    STALE_COMPLETION_MARKER,
    INVALID_COMPLETION_MARKER,
)


class CacheResumeError(CacheError):
    """Raised for malformed resume manifests, records, or markers."""


def _exact_fields(value: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise CacheResumeError("{} must be a mapping".format(path))
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise CacheResumeError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CacheResumeError("{} must be a non-empty string".format(path))
    return value


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise CacheResumeError(str(exc)) from exc


def _stable(value: Any, namespace: str, path: str) -> str:
    try:
        return validate_stable_id(value, namespace, path)
    except ValueError as exc:
        raise CacheResumeError(str(exc)) from exc


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CacheResumeError("{} must be a non-negative integer".format(path))
    return int(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _mapping_json(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CacheResumeError("{} must be a mapping".format(path))
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise CacheResumeError("{} is not canonicalizable".format(path)) from exc


def _counts(value: Any, path: str) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        raise CacheResumeError("{} must be a mapping".format(path))
    expected = set(LAYER_ORDER)
    if set(value) != expected:
        raise CacheResumeError("{} must contain exactly {}".format(path, sorted(expected)))
    return {
        layer: _nonnegative_int(value[layer], "{}.{}".format(path, layer))
        for layer in LAYER_ORDER
    }


def _key_namespace(layer: str) -> str:
    return {
        LAYER_RIR: "rir-cache-key",
        LAYER_MIXTURE: "mixture-cache-key",
        LAYER_ASR: "asr-cache-key",
    }[layer]


def _key_payload(layer: str, key: Any) -> Mapping[str, Any]:
    if layer == LAYER_RIR and isinstance(key, RirCacheKey):
        return key.to_payload()
    if layer == LAYER_MIXTURE and isinstance(key, MixtureCacheKey):
        return key.to_payload()
    if layer == LAYER_ASR and isinstance(key, AsrCacheKey):
        return key.to_payload()
    raise CacheResumeError("{} key has the wrong layer or schema".format(layer))


def _parse_key(layer: str, payload: Any) -> Any:
    if layer == LAYER_RIR:
        return RirCacheKey.from_payload(payload)
    if layer == LAYER_MIXTURE:
        return MixtureCacheKey.from_payload(payload)
    if layer == LAYER_ASR:
        return AsrCacheKey.from_payload(payload)
    raise CacheResumeError("unknown cache layer {!r}".format(layer))


def _entry_order(entries: Sequence[Tuple[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(
        sorted(
            entries,
            key=lambda item: (LAYER_ORDER.index(item[0]), item[1].cache_key),
        )
    )


@dataclass(frozen=True)
class CacheExpectedManifest:
    """Immutable semantic set of cache entries expected by one run."""

    infrastructure_contract_sha256: str
    expected_rir_keys: Tuple[RirCacheKey, ...] = ()
    expected_mixture_keys: Tuple[MixtureCacheKey, ...] = ()
    expected_asr_keys: Tuple[AsrCacheKey, ...] = ()
    logical_labels: Mapping[str, str] = field(default_factory=dict)
    expected_counts: Mapping[str, int] = field(default_factory=dict)
    manifest_id: str = ""
    schema_version: str = CACHE_EXPECTED_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha(self.infrastructure_contract_sha256, "manifest.infrastructure_contract_sha256")
        if self.schema_version != CACHE_EXPECTED_MANIFEST_SCHEMA_VERSION:
            raise CacheResumeError("manifest.schema_version is invalid")
        normalized = {}
        for layer, field_name in (
            (LAYER_RIR, "expected_rir_keys"),
            (LAYER_MIXTURE, "expected_mixture_keys"),
            (LAYER_ASR, "expected_asr_keys"),
        ):
            values = tuple(getattr(self, field_name))
            if any(_key_payload(layer, key) is None for key in values):
                raise CacheResumeError("manifest.{} contains an invalid key".format(field_name))
            ordered = tuple(sorted(values, key=lambda key: key.cache_key))
            if len({key.cache_key for key in ordered}) != len(ordered):
                raise CacheResumeError("manifest.{} contains duplicate keys".format(field_name))
            normalized[field_name] = ordered
        all_keys = [
            (layer, key.cache_key)
            for layer, field_name in (
                (LAYER_RIR, "expected_rir_keys"),
                (LAYER_MIXTURE, "expected_mixture_keys"),
                (LAYER_ASR, "expected_asr_keys"),
            )
            for key in normalized[field_name]
        ]
        if len({cache_key for _, cache_key in all_keys}) != len(all_keys):
            raise CacheResumeError("manifest contains duplicate cache keys across layers")
        for field_name, values in normalized.items():
            object.__setattr__(self, field_name, values)
        derived_counts = {
            LAYER_RIR: len(normalized["expected_rir_keys"]),
            LAYER_MIXTURE: len(normalized["expected_mixture_keys"]),
            LAYER_ASR: len(normalized["expected_asr_keys"]),
        }
        supplied_counts = derived_counts if not self.expected_counts else _counts(self.expected_counts, "manifest.expected_counts")
        if supplied_counts != derived_counts:
            raise CacheResumeError("manifest.expected_counts do not match expected keys")
        object.__setattr__(self, "expected_counts", MappingProxyType(dict(derived_counts)))
        labels = _mapping_json(self.logical_labels, "manifest.logical_labels")
        all_cache_keys = {cache_key for _, cache_key in all_keys}
        if any(not isinstance(key, str) or key not in all_cache_keys for key in labels):
            raise CacheResumeError("manifest.logical_labels contains an unknown cache key")
        if any(not isinstance(label, str) or not label for label in labels.values()):
            raise CacheResumeError("manifest.logical_labels values must be non-empty strings")
        object.__setattr__(self, "logical_labels", _freeze(labels))
        expected_id = stable_id("cache-manifest", self.identity_payload())
        if self.manifest_id:
            _stable(self.manifest_id, "cache-manifest", "manifest.manifest_id")
            if self.manifest_id != expected_id:
                raise CacheResumeError("manifest.manifest_id does not match semantic expected set")
        else:
            object.__setattr__(self, "manifest_id", expected_id)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "expected_rir_keys": [key.to_payload() for key in self.expected_rir_keys],
            "expected_mixture_keys": [key.to_payload() for key in self.expected_mixture_keys],
            "expected_asr_keys": [key.to_payload() for key in self.expected_asr_keys],
            "expected_counts": dict(self.expected_counts),
        }

    @property
    def manifest_sha256(self) -> str:
        return identity_sha256(self.identity_payload())

    def entries(self) -> Tuple[Tuple[str, Any], ...]:
        return _entry_order(
            tuple((LAYER_RIR, key) for key in self.expected_rir_keys)
            + tuple((LAYER_MIXTURE, key) for key in self.expected_mixture_keys)
            + tuple((LAYER_ASR, key) for key in self.expected_asr_keys)
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "expected_rir_keys": [key.to_payload() for key in self.expected_rir_keys],
            "expected_mixture_keys": [key.to_payload() for key in self.expected_mixture_keys],
            "expected_asr_keys": [key.to_payload() for key in self.expected_asr_keys],
            "expected_counts": dict(self.expected_counts),
            "logical_labels": _thaw(self.logical_labels),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CacheExpectedManifest":
        _exact_fields(
            payload,
            (
                "schema_version", "manifest_id", "infrastructure_contract_sha256",
                "expected_rir_keys", "expected_mixture_keys", "expected_asr_keys",
                "expected_counts", "logical_labels",
            ),
            "cache_expected_manifest",
        )
        keys = {}
        for layer, field_name in (
            (LAYER_RIR, "expected_rir_keys"),
            (LAYER_MIXTURE, "expected_mixture_keys"),
            (LAYER_ASR, "expected_asr_keys"),
        ):
            values = payload[field_name]
            if not isinstance(values, (list, tuple)):
                raise CacheResumeError("manifest.{} must be a list".format(field_name))
            try:
                keys[field_name] = tuple(_parse_key(layer, item) for item in values)
            except (CacheError, TypeError, ValueError) as exc:
                raise CacheResumeError("manifest.{} contains an invalid key".format(field_name)) from exc
        return cls(
            infrastructure_contract_sha256=payload["infrastructure_contract_sha256"],
            expected_rir_keys=keys["expected_rir_keys"],
            expected_mixture_keys=keys["expected_mixture_keys"],
            expected_asr_keys=keys["expected_asr_keys"],
            logical_labels=payload["logical_labels"],
            expected_counts=payload["expected_counts"],
            manifest_id=payload["manifest_id"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class CacheEntryResult:
    layer: str
    cache_key: str
    status: str
    reason: str
    schema_version: str = CACHE_ENTRY_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.layer not in LAYER_ORDER:
            raise CacheResumeError("entry.layer is invalid")
        _stable(self.cache_key, _key_namespace(self.layer), "entry.cache_key")
        if self.status not in ENTRY_STATUSES:
            raise CacheResumeError("entry.status is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise CacheResumeError("entry.reason must be non-empty")
        if self.status == REUSE_VALID and self.reason != HIT_VALID:
            raise CacheResumeError("reused entries must retain HIT_VALID")
        if self.status != REUSE_VALID and self.reason == HIT_VALID:
            raise CacheResumeError("rebuild entries cannot have HIT_VALID reason")
        if self.schema_version != CACHE_ENTRY_RESULT_SCHEMA_VERSION:
            raise CacheResumeError("entry.schema_version is invalid")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": self.layer,
            "cache_key": self.cache_key,
            "status": self.status,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CacheEntryResult":
        _exact_fields(payload, ("schema_version", "layer", "cache_key", "status", "reason"), "cache_entry_result")
        return cls(**dict(payload))


@dataclass(frozen=True)
class CacheReconciliationRecord:
    expected_manifest_id: str
    infrastructure_contract_sha256: str
    entries: Tuple[CacheEntryResult, ...]
    expected_counts: Mapping[str, int]
    reuse_counts: Mapping[str, int]
    rebuild_counts: Mapping[str, int]
    corrupt_counts: Mapping[str, int]
    missing_counts: Mapping[str, int]
    complete: bool
    completion_marker_status: str = COMPLETION_MARKER_ABSENT
    reconciliation_algorithm_identity: str = RECONCILIATION_ALGORITHM_IDENTITY
    schema_version: str = CACHE_RECONCILIATION_RECORD_SCHEMA_VERSION
    record_id: str = ""
    record_sha256: str = ""

    def __post_init__(self) -> None:
        _stable(self.expected_manifest_id, "cache-manifest", "reconciliation.expected_manifest_id")
        _sha(self.infrastructure_contract_sha256, "reconciliation.infrastructure_contract_sha256")
        if self.schema_version != CACHE_RECONCILIATION_RECORD_SCHEMA_VERSION:
            raise CacheResumeError("reconciliation.schema_version is invalid")
        if self.reconciliation_algorithm_identity != RECONCILIATION_ALGORITHM_IDENTITY:
            raise CacheResumeError("reconciliation algorithm identity is invalid")
        if self.completion_marker_status not in MARKER_STATUSES:
            raise CacheResumeError("reconciliation completion marker status is invalid")
        entries = tuple(self.entries)
        if any(not isinstance(entry, CacheEntryResult) for entry in entries):
            raise CacheResumeError("reconciliation.entries contain invalid records")
        expected_order = tuple(
            sorted(entries, key=lambda entry: (LAYER_ORDER.index(entry.layer), entry.cache_key))
        )
        if entries != expected_order:
            raise CacheResumeError("reconciliation.entries are not in deterministic layer/key order")
        if len({(entry.layer, entry.cache_key) for entry in entries}) != len(entries):
            raise CacheResumeError("reconciliation.entries contain duplicates")
        object.__setattr__(self, "entries", entries)
        expected = _counts(self.expected_counts, "reconciliation.expected_counts")
        actual_expected = {layer: sum(entry.layer == layer for entry in entries) for layer in LAYER_ORDER}
        if expected != actual_expected:
            raise CacheResumeError("reconciliation.expected_counts do not match entries")
        derived = {
            "reuse_counts": {layer: 0 for layer in LAYER_ORDER},
            "rebuild_counts": {layer: 0 for layer in LAYER_ORDER},
            "corrupt_counts": {layer: 0 for layer in LAYER_ORDER},
            "missing_counts": {layer: 0 for layer in LAYER_ORDER},
        }
        for entry in entries:
            if entry.status == REUSE_VALID:
                derived["reuse_counts"][entry.layer] += 1
            else:
                derived["rebuild_counts"][entry.layer] += 1
                if entry.status == REBUILD_CORRUPT:
                    derived["corrupt_counts"][entry.layer] += 1
                else:
                    derived["missing_counts"][entry.layer] += 1
        for name, value in derived.items():
            supplied = _counts(getattr(self, name), "reconciliation." + name)
            if supplied != value:
                raise CacheResumeError("reconciliation.{} do not match entries".format(name))
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, "expected_counts", MappingProxyType(dict(expected)))
        # Entry completeness is authoritative.  Marker currency is only
        # metadata about the last completion attempt; a stale/invalid marker
        # must never deadlock a currently valid cache set.
        expected_complete = all(value == 0 for value in self.rebuild_counts.values())
        if not isinstance(self.complete, bool) or self.complete != expected_complete:
            raise CacheResumeError("reconciliation.complete is inconsistent with entries/marker")
        expected_id = stable_id("cache-reconciliation", self.identity_payload())
        expected_sha = identity_sha256(self.identity_payload())
        if self.record_id:
            _stable(self.record_id, "cache-reconciliation", "reconciliation.record_id")
            if self.record_id != expected_id:
                raise CacheResumeError("reconciliation.record_id does not match entries")
        else:
            object.__setattr__(self, "record_id", expected_id)
        if self.record_sha256:
            _sha(self.record_sha256, "reconciliation.record_sha256")
            if self.record_sha256 != expected_sha:
                raise CacheResumeError("reconciliation.record_sha256 does not match entries")
        else:
            object.__setattr__(self, "record_sha256", expected_sha)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expected_manifest_id": self.expected_manifest_id,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "reconciliation_algorithm_identity": self.reconciliation_algorithm_identity,
            "entries": [entry.to_payload() for entry in self.entries],
            "expected_counts": dict(self.expected_counts),
            "reuse_counts": dict(self.reuse_counts),
            "rebuild_counts": dict(self.rebuild_counts),
            "corrupt_counts": dict(self.corrupt_counts),
            "missing_counts": dict(self.missing_counts),
        }

    @property
    def reuse_keys(self) -> Tuple[str, ...]:
        return tuple(entry.cache_key for entry in self.entries if entry.status == REUSE_VALID)

    @property
    def rebuild_keys(self) -> Tuple[str, ...]:
        return tuple(entry.cache_key for entry in self.entries if entry.status != REUSE_VALID)

    @property
    def reasons_by_cache_key(self) -> Mapping[str, str]:
        return MappingProxyType({entry.cache_key: entry.reason for entry in self.entries})

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "record_sha256": self.record_sha256,
            "expected_manifest_id": self.expected_manifest_id,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "reconciliation_algorithm_identity": self.reconciliation_algorithm_identity,
            "entries": [entry.to_payload() for entry in self.entries],
            "expected_counts": dict(self.expected_counts),
            "reuse_counts": dict(self.reuse_counts),
            "rebuild_counts": dict(self.rebuild_counts),
            "corrupt_counts": dict(self.corrupt_counts),
            "missing_counts": dict(self.missing_counts),
            "complete": self.complete,
            "completion_marker_status": self.completion_marker_status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CacheReconciliationRecord":
        _exact_fields(
            payload,
            (
                "schema_version", "record_id", "record_sha256", "expected_manifest_id",
                "infrastructure_contract_sha256", "reconciliation_algorithm_identity", "entries",
                "expected_counts", "reuse_counts", "rebuild_counts", "corrupt_counts",
                "missing_counts", "complete", "completion_marker_status",
            ),
            "cache_reconciliation_record",
        )
        if not isinstance(payload["entries"], (list, tuple)):
            raise CacheResumeError("reconciliation.entries must be a list")
        return cls(
            expected_manifest_id=payload["expected_manifest_id"],
            infrastructure_contract_sha256=payload["infrastructure_contract_sha256"],
            entries=tuple(CacheEntryResult.from_payload(item) for item in payload["entries"]),
            expected_counts=payload["expected_counts"],
            reuse_counts=payload["reuse_counts"],
            rebuild_counts=payload["rebuild_counts"],
            corrupt_counts=payload["corrupt_counts"],
            missing_counts=payload["missing_counts"],
            complete=payload["complete"],
            completion_marker_status=payload["completion_marker_status"],
            reconciliation_algorithm_identity=payload["reconciliation_algorithm_identity"],
            schema_version=payload["schema_version"],
            record_id=payload["record_id"],
            record_sha256=payload["record_sha256"],
        )


@dataclass(frozen=True)
class CacheCompletionMarker:
    expected_manifest_id: str
    infrastructure_contract_sha256: str
    reconciliation_record_id: str
    reconciliation_record_sha256: str
    expected_counts: Mapping[str, int]
    completion_algorithm_identity: str = COMPLETION_ALGORITHM_IDENTITY
    schema_version: str = CACHE_COMPLETION_MARKER_SCHEMA_VERSION
    marker_id: str = ""

    def __post_init__(self) -> None:
        _stable(self.expected_manifest_id, "cache-manifest", "marker.expected_manifest_id")
        _sha(self.infrastructure_contract_sha256, "marker.infrastructure_contract_sha256")
        _stable(self.reconciliation_record_id, "cache-reconciliation", "marker.reconciliation_record_id")
        _sha(self.reconciliation_record_sha256, "marker.reconciliation_record_sha256")
        if self.schema_version != CACHE_COMPLETION_MARKER_SCHEMA_VERSION:
            raise CacheResumeError("marker.schema_version is invalid")
        if self.completion_algorithm_identity != COMPLETION_ALGORITHM_IDENTITY:
            raise CacheResumeError("marker completion algorithm identity is invalid")
        counts = _counts(self.expected_counts, "marker.expected_counts")
        object.__setattr__(self, "expected_counts", MappingProxyType(dict(counts)))
        expected_id = stable_id("cache-completion-marker", self.identity_payload())
        if self.marker_id:
            _stable(self.marker_id, "cache-completion-marker", "marker.marker_id")
            if self.marker_id != expected_id:
                raise CacheResumeError("marker.marker_id does not match semantic payload")
        else:
            object.__setattr__(self, "marker_id", expected_id)

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expected_manifest_id": self.expected_manifest_id,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "reconciliation_record_id": self.reconciliation_record_id,
            "reconciliation_record_sha256": self.reconciliation_record_sha256,
            "expected_counts": dict(self.expected_counts),
            "completion_algorithm_identity": self.completion_algorithm_identity,
        }

    @classmethod
    def from_reconciliation(cls, record: CacheReconciliationRecord) -> "CacheCompletionMarker":
        if not record.complete:
            raise CacheResumeError("cannot write a completion marker for an incomplete reconciliation")
        return cls(
            expected_manifest_id=record.expected_manifest_id,
            infrastructure_contract_sha256=record.infrastructure_contract_sha256,
            reconciliation_record_id=record.record_id,
            reconciliation_record_sha256=record.record_sha256,
            expected_counts=record.expected_counts,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "marker_id": self.marker_id,
            "expected_manifest_id": self.expected_manifest_id,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "reconciliation_record_id": self.reconciliation_record_id,
            "reconciliation_record_sha256": self.reconciliation_record_sha256,
            "expected_counts": dict(self.expected_counts),
            "completion_algorithm_identity": self.completion_algorithm_identity,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CacheCompletionMarker":
        _exact_fields(
            payload,
            (
                "schema_version", "marker_id", "expected_manifest_id",
                "infrastructure_contract_sha256", "reconciliation_record_id",
                "reconciliation_record_sha256", "expected_counts", "completion_algorithm_identity",
            ),
            "cache_completion_marker",
        )
        return cls(**dict(payload))


def completion_marker_path(store: CacheStore, manifest: CacheExpectedManifest) -> Path:
    """Return the manifest-scoped marker path for one expected cache set."""

    if not isinstance(store, CacheStore):
        raise CacheResumeError("store must be CacheStore")
    if not isinstance(manifest, CacheExpectedManifest):
        raise CacheResumeError("manifest must be CacheExpectedManifest")
    return store.root / "resume" / manifest.manifest_id / COMPLETION_MARKER_FILENAME


def _load_marker(store: CacheStore, manifest: CacheExpectedManifest) -> Tuple[str, Optional[CacheCompletionMarker]]:
    path = completion_marker_path(store, manifest)
    if not path.exists():
        return COMPLETION_MARKER_ABSENT, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return COMPLETION_MARKER_VALID, CacheCompletionMarker.from_payload(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, CacheError, TypeError, ValueError):
        return INVALID_COMPLETION_MARKER, None


def reconcile_cache_manifest(
    manifest: CacheExpectedManifest,
    store: CacheStore,
) -> CacheReconciliationRecord:
    """Revalidate every expected entry and return a deterministic rebuild plan."""

    if not isinstance(manifest, CacheExpectedManifest):
        raise CacheResumeError("manifest must be CacheExpectedManifest")
    if not isinstance(store, CacheStore):
        raise CacheResumeError("store must be CacheStore")
    entries = []
    for layer, key in manifest.entries():
        result = getattr(store, "read_{}".format(layer))(key)
        if result.status == HIT_VALID:
            status = REUSE_VALID
            reason = HIT_VALID
        elif result.status == MISS:
            status = REBUILD_MISSING
            reason = result.reason
        elif result.status == INVALID_CORRUPT:
            status = REBUILD_CORRUPT
            reason = result.reason
        else:
            raise CacheResumeError("unknown cache read status {!r}".format(result.status))
        entries.append(CacheEntryResult(layer, key.cache_key, status, reason))
    entries = tuple(entries)
    expected_counts = dict(manifest.expected_counts)
    reuse_counts = {layer: sum(entry.layer == layer and entry.status == REUSE_VALID for entry in entries) for layer in LAYER_ORDER}
    rebuild_counts = {layer: sum(entry.layer == layer and entry.status != REUSE_VALID for entry in entries) for layer in LAYER_ORDER}
    corrupt_counts = {layer: sum(entry.layer == layer and entry.status == REBUILD_CORRUPT for entry in entries) for layer in LAYER_ORDER}
    missing_counts = {layer: sum(entry.layer == layer and entry.status == REBUILD_MISSING for entry in entries) for layer in LAYER_ORDER}
    entry_complete = all(entry.status == REUSE_VALID for entry in entries)
    provisional = CacheReconciliationRecord(
        expected_manifest_id=manifest.manifest_id,
        infrastructure_contract_sha256=manifest.infrastructure_contract_sha256,
        entries=entries,
        expected_counts=expected_counts,
        reuse_counts=reuse_counts,
        rebuild_counts=rebuild_counts,
        corrupt_counts=corrupt_counts,
        missing_counts=missing_counts,
        complete=entry_complete,
        completion_marker_status=COMPLETION_MARKER_ABSENT,
    )
    marker_status, marker = _load_marker(store, manifest)
    if marker_status == INVALID_COMPLETION_MARKER:
        marker_status = INVALID_COMPLETION_MARKER
    elif marker is not None and (
        marker.expected_manifest_id != manifest.manifest_id
        or marker.infrastructure_contract_sha256 != manifest.infrastructure_contract_sha256
        or marker.reconciliation_record_id != provisional.record_id
        or marker.reconciliation_record_sha256 != provisional.record_sha256
        or dict(marker.expected_counts) != expected_counts
    ):
        marker_status = STALE_COMPLETION_MARKER
    elif marker is not None:
        marker_status = COMPLETION_MARKER_VALID
    complete = entry_complete
    return CacheReconciliationRecord(
        expected_manifest_id=manifest.manifest_id,
        infrastructure_contract_sha256=manifest.infrastructure_contract_sha256,
        entries=entries,
        expected_counts=expected_counts,
        reuse_counts=reuse_counts,
        rebuild_counts=rebuild_counts,
        corrupt_counts=corrupt_counts,
        missing_counts=missing_counts,
        complete=complete,
        completion_marker_status=marker_status,
    )


def write_completion_marker(
    store: CacheStore,
    manifest: CacheExpectedManifest,
) -> CacheCompletionMarker:
    """Freshly revalidate and atomically write the current manifest marker.

    The caller cannot reuse an earlier reconciliation record: this closes the
    reconcile-then-corrupt-before-write race and permits stale marker repair.
    """

    if not isinstance(store, CacheStore):
        raise CacheResumeError("store must be CacheStore")
    if not isinstance(manifest, CacheExpectedManifest):
        raise CacheResumeError("manifest must be CacheExpectedManifest")
    reconciliation = reconcile_cache_manifest(manifest, store)
    if not reconciliation.complete:
        raise CacheResumeError("cannot write a completion marker for an incomplete reconciliation")
    marker = CacheCompletionMarker.from_reconciliation(reconciliation)
    store.storage.atomic_write_json(completion_marker_path(store, manifest), marker.to_payload())
    return marker


def read_completion_marker(
    store: CacheStore,
    manifest: CacheExpectedManifest,
) -> Optional[CacheCompletionMarker]:
    """Read a marker strictly; malformed existing markers fail closed."""

    if not isinstance(store, CacheStore):
        raise CacheResumeError("store must be CacheStore")
    if not isinstance(manifest, CacheExpectedManifest):
        raise CacheResumeError("manifest must be CacheExpectedManifest")
    status, marker = _load_marker(store, manifest)
    if status == INVALID_COMPLETION_MARKER:
        raise CacheResumeError("completion marker is invalid")
    return marker


__all__ = [
    "CACHE_COMPLETION_MARKER_SCHEMA_VERSION",
    "CACHE_ENTRY_RESULT_SCHEMA_VERSION",
    "CACHE_EXPECTED_MANIFEST_SCHEMA_VERSION",
    "CACHE_RECONCILIATION_RECORD_SCHEMA_VERSION",
    "COMPLETION_ALGORITHM_IDENTITY",
    "COMPLETION_MARKER_ABSENT",
    "COMPLETION_MARKER_FILENAME",
    "COMPLETION_MARKER_SCOPE_IDENTITY",
    "COMPLETION_MARKER_VALID",
    "CacheCompletionMarker",
    "CacheEntryResult",
    "CacheExpectedManifest",
    "CacheReconciliationRecord",
    "CacheResumeError",
    "completion_marker_path",
    "INVALID_COMPLETION_MARKER",
    "REBUILD_CORRUPT",
    "REBUILD_MISSING",
    "RECONCILIATION_ALGORITHM_IDENTITY",
    "REUSE_VALID",
    "RESUME_ALGORITHM_IDENTITY",
    "STALE_COMPLETION_MARKER",
    "read_completion_marker",
    "reconcile_cache_manifest",
    "write_completion_marker",
]
