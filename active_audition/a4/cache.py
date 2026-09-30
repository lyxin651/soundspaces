"""Pure content-addressed cache keys, metadata, and integrity primitives.

This module deliberately does not import Habitat, SoundSpaces, SpeechBrain, or
an ASR runner.  It binds cache entries to the semantic inputs that produce
their payloads and keeps run provenance outside those identities.  The store
uses the existing :mod:`active_audition.data.storage` atomic primitives:
payload first, metadata last.
"""

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from active_audition.a4.identity import (
    canonical_json,
    canonical_json_bytes,
    identity_sha256,
    sha256_bytes,
    stable_id,
    validate_sha256,
    validate_stable_id,
)
from active_audition.data.storage import DatasetStorage
from active_audition.a4.noise_segments import SOURCE_TIME_CONVENTION_IDENTITY


CACHE_KEY_SERIALIZATION_IDENTITY = "active-asr-a4-cache-key-canonical-json-v1"
CACHE_METADATA_SCHEMA_VERSION = "active-asr-a4-cache-metadata-v1"
RIR_CACHE_KEY_SCHEMA_VERSION = "active-asr-a4-rir-cache-key-v1"
MIXTURE_CACHE_KEY_SCHEMA_VERSION = "active-asr-a4-mixture-cache-key-v3"
ASR_CACHE_KEY_SCHEMA_VERSION = "active-asr-a4-asr-cache-key-v1"
RIR_CACHE_METADATA_SCHEMA_VERSION = "active-asr-a4-rir-cache-metadata-v1"
MIXTURE_CACHE_METADATA_SCHEMA_VERSION = "active-asr-a4-mixture-cache-metadata-v1"
ASR_CACHE_METADATA_SCHEMA_VERSION = "active-asr-a4-asr-cache-metadata-v2"
ASR_RESULT_SCHEMA_VERSION = "active-asr-a4-asr-result-v1"
CACHE_ALGORITHM_IDENTITY = "active-asr-a4-content-addressed-cache-v1"
CACHE_INTEGRITY_IDENTITY = "active-asr-a4-cache-key-metadata-payload-integrity-v1"
CACHE_ATOMIC_COMMIT_IDENTITY = "active-asr-a4-payload-then-metadata-atomic-commit-v1"
CACHE_KEYING_IDENTITY = "content_addressed_semantic_payload"
NATIVE_SAMPLE_RATE_HZ = 16000
CHANNEL_ORDER = ("L", "R")
A2_ACOUSTIC_CONTRACT_SHA256 = "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c"
A3_ASR_CONTRACT_SHA256 = "70864c814a55db5d184ef8a6835b65cb564c4c90fc86112f8705b7ecf6df1ffe"
NOISE_SOURCE_TIME_IDENTITY = SOURCE_TIME_CONVENTION_IDENTITY
ASR_INPUT_LAYOUT = "mono"

CACHE_LAYER_RIR = "rir"
CACHE_LAYER_MIXTURE = "mixture"
CACHE_LAYER_ASR = "asr"
HIT_VALID = "HIT_VALID"
MISS = "MISS"
INVALID_CORRUPT = "INVALID_CORRUPT"

MISSING_METADATA = "MISSING_METADATA"
MISSING_PAYLOAD = "MISSING_PAYLOAD"
KEY_MISMATCH = "KEY_MISMATCH"
METADATA_SCHEMA_MISMATCH = "METADATA_SCHEMA_MISMATCH"
PAYLOAD_HASH_MISMATCH = "PAYLOAD_HASH_MISMATCH"
SHAPE_MISMATCH = "SHAPE_MISMATCH"
DTYPE_MISMATCH = "DTYPE_MISMATCH"
SEMANTIC_IDENTITY_MISMATCH = "SEMANTIC_IDENTITY_MISMATCH"
NONFINITE_PAYLOAD = "NONFINITE_PAYLOAD"
PAYLOAD_SCHEMA_MISMATCH = "PAYLOAD_SCHEMA_MISMATCH"
PAYLOAD_PARSE_ERROR = "PAYLOAD_PARSE_ERROR"


class CacheError(ValueError):
    """Raised for malformed cache requests or metadata."""


class _MetadataError(CacheError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CacheError("{} must be a non-empty string".format(path))
    return value


def _sha(value: Any, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except ValueError as exc:
        raise CacheError(str(exc)) from exc


def _stable(value: Any, namespace: str, path: str) -> str:
    try:
        return validate_stable_id(value, namespace, path)
    except ValueError as exc:
        raise CacheError(str(exc)) from exc


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CacheError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise CacheError("{} must be finite".format(path))
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheError("{} must be a positive integer".format(path))
    return int(value)


def _exact_fields(value: Mapping[str, Any], expected: Sequence[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise CacheError("{} must be a mapping".format(path))
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise CacheError(
            "{} fields invalid: unknown={}, missing={}".format(path, unknown, missing)
        )


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


_FORBIDDEN_KEY_NAMES = frozenset(
    {
        "timestamp",
        "wall_clock",
        "temp_path",
        "temporary_path",
        "output_path",
        "absolute_output_directory",
        "git_head",
        "wer",
        "cer",
        "hypothesis",
        "score",
        "downstream_result",
        "asr_result",
    }
)


def _reject_nonsemantic_names(value: Any, path: str = "key_payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEY_NAMES:
                raise CacheError("{} contains forbidden non-semantic field {!r}".format(path, key))
            _reject_nonsemantic_names(item, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonsemantic_names(item, "{}[{}]".format(path, index))


def _semantic_json(value: Any, path: str) -> Any:
    """Canonicalize an immutable JSON-compatible semantic value."""

    _reject_nonsemantic_names(value, path)
    try:
        text = canonical_json(value)
        return json.loads(text)
    except (TypeError, ValueError) as exc:
        raise CacheError("{} is not canonicalizable: {}".format(path, exc)) from exc


def _provenance_json(value: Any, path: str) -> Any:
    if not isinstance(value, Mapping):
        raise CacheError("{} must be a mapping".format(path))
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise CacheError("{} is not canonicalizable: {}".format(path, exc)) from exc


def _result_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CacheError("ASR result is not deterministically serializable: {}".format(exc)) from exc


def _numeric_json(value: Any, path: str) -> Any:
    """Canonicalize transform coordinates so int/float spellings agree."""

    if isinstance(value, Mapping):
        return {str(key): _numeric_json(item, "{}.{}".format(path, key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numeric_json(item, "{}[{}]".format(path, index)) for index, item in enumerate(value)]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return _finite(value, path)
    raise CacheError("{} contains unsupported value type".format(path))


def _normalize_yaw(value: Any, path: str) -> float:
    result = _finite(value, path)
    return ((result + 180.0) % 360.0) - 180.0


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise CacheError("{} must be a non-empty mapping".format(path))
    return value


def _key_payload_sha(payload: Mapping[str, Any]) -> str:
    return identity_sha256(payload)


@dataclass(frozen=True)
class RirCacheKey:
    scene_resource_identities: Mapping[str, Any]
    acoustic_contract_identity: str
    renderer_algorithm_identity: str
    materials_policy: str
    source_world_transform: Any
    receiver_sensor_transform: Any
    receiver_yaw_deg: float
    sample_rate_hz: int = NATIVE_SAMPLE_RATE_HZ
    channel_layout: str = "binaural"
    channel_order: Tuple[str, str] = CHANNEL_ORDER
    replicate_identity: str = "replicate-0"
    schema_version: str = RIR_CACHE_KEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _mapping(self.scene_resource_identities, "rir.scene_resource_identities")
        _sha(self.acoustic_contract_identity, "rir.acoustic_contract_identity")
        _string(self.renderer_algorithm_identity, "rir.renderer_algorithm_identity")
        _string(self.materials_policy, "rir.materials_policy")
        _numeric_json(self.source_world_transform, "rir.source_world_transform")
        _numeric_json(self.receiver_sensor_transform, "rir.receiver_sensor_transform")
        _finite(self.receiver_yaw_deg, "rir.receiver_yaw_deg")
        _positive_int(self.sample_rate_hz, "rir.sample_rate_hz")
        _string(self.channel_layout, "rir.channel_layout")
        if len(tuple(self.channel_order)) != 2 or any(not isinstance(item, str) or not item for item in self.channel_order):
            raise CacheError("rir.channel_order must contain two non-empty channel names")
        _string(self.replicate_identity, "rir.replicate_identity")
        if self.schema_version != RIR_CACHE_KEY_SCHEMA_VERSION:
            raise CacheError("rir.schema_version is invalid")
        object.__setattr__(self, "scene_resource_identities", _freeze(_semantic_json(self.scene_resource_identities, "rir.scene_resource_identities")))
        object.__setattr__(self, "source_world_transform", _freeze(_semantic_json(_numeric_json(self.source_world_transform, "rir.source_world_transform"), "rir.source_world_transform")))
        object.__setattr__(self, "receiver_sensor_transform", _freeze(_semantic_json(_numeric_json(self.receiver_sensor_transform, "rir.receiver_sensor_transform"), "rir.receiver_sensor_transform")))
        object.__setattr__(self, "receiver_yaw_deg", _normalize_yaw(self.receiver_yaw_deg, "rir.receiver_yaw_deg"))
        object.__setattr__(self, "channel_order", tuple(self.channel_order))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": CACHE_LAYER_RIR,
            "scene_resource_identities": _thaw(self.scene_resource_identities),
            "acoustic_contract_identity": self.acoustic_contract_identity,
            "renderer_algorithm_identity": self.renderer_algorithm_identity,
            "materials_policy": self.materials_policy,
            "source_world_transform": _thaw(self.source_world_transform),
            "receiver_sensor_transform": _thaw(self.receiver_sensor_transform),
            "receiver_yaw_deg": self.receiver_yaw_deg,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_layout": self.channel_layout,
            "channel_order": list(self.channel_order),
            "replicate_identity": self.replicate_identity,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RirCacheKey":
        _exact_fields(payload, tuple(cls.__dataclass_fields__) + ("layer",), "rir_key")
        if payload["layer"] != CACHE_LAYER_RIR:
            raise CacheError("rir_key.layer is invalid")
        data = dict(payload)
        data.pop("layer")
        data["channel_order"] = tuple(data["channel_order"])
        return cls(**data)

    @property
    def cache_key(self) -> str:
        return stable_id("rir-cache-key", self.to_payload())

    @property
    def key_payload_sha256(self) -> str:
        return _key_payload_sha(self.to_payload())


@dataclass(frozen=True)
class MixtureCacheKey:
    target_rir_cache_key: str
    noise_rir_cache_key: str
    target_component_identity: str
    noise_component_identity: str
    target_dry_waveform_sha256: str
    noise_segment_payload_sha256: str
    noise_segment_identity: str
    noise_source_time_identity: str
    calibration_artifact_identity: str
    global_gain_identity: str
    timeline_identity: str
    mixer_contract_identity: str
    schema_version: str = MIXTURE_CACHE_KEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _stable(self.target_rir_cache_key, "rir-cache-key", "mixture.target_rir_cache_key")
        _stable(self.noise_rir_cache_key, "rir-cache-key", "mixture.noise_rir_cache_key")
        _stable(self.target_component_identity, "target-component", "mixture.target_component_identity")
        _stable(self.noise_component_identity, "noise-component", "mixture.noise_component_identity")
        for name in ("target_dry_waveform_sha256", "noise_segment_payload_sha256"):
            _sha(getattr(self, name), "mixture." + name)
        _stable(self.noise_segment_identity, "noise-segment", "mixture.noise_segment_identity")
        if self.noise_source_time_identity != NOISE_SOURCE_TIME_IDENTITY:
            raise CacheError("mixture.noise_source_time_identity is not the frozen A4 source-time identity")
        _stable(self.calibration_artifact_identity, "calibration", "mixture.calibration_artifact_identity")
        _stable(self.global_gain_identity, "global-gain", "mixture.global_gain_identity")
        _stable(self.timeline_identity, "receiver-timeline", "mixture.timeline_identity")
        _stable(self.mixer_contract_identity, "mixture-contract", "mixture.mixer_contract_identity")
        if self.schema_version != MIXTURE_CACHE_KEY_SCHEMA_VERSION:
            raise CacheError("mixture.schema_version is invalid")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": CACHE_LAYER_MIXTURE,
            "target_rir_cache_key": self.target_rir_cache_key,
            "noise_rir_cache_key": self.noise_rir_cache_key,
            "target_component_identity": self.target_component_identity,
            "noise_component_identity": self.noise_component_identity,
            "target_dry_waveform_sha256": self.target_dry_waveform_sha256,
            "noise_segment_payload_sha256": self.noise_segment_payload_sha256,
            "noise_segment_identity": self.noise_segment_identity,
            "noise_source_time_identity": self.noise_source_time_identity,
            "calibration_artifact_identity": self.calibration_artifact_identity,
            "global_gain_identity": self.global_gain_identity,
            "timeline_identity": self.timeline_identity,
            "mixer_contract_identity": self.mixer_contract_identity,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MixtureCacheKey":
        _exact_fields(payload, tuple(cls.__dataclass_fields__) + ("layer",), "mixture_key")
        if payload["layer"] != CACHE_LAYER_MIXTURE:
            raise CacheError("mixture_key.layer is invalid")
        data = dict(payload)
        data.pop("layer")
        return cls(**data)

    @property
    def cache_key(self) -> str:
        return stable_id("mixture-cache-key", self.to_payload())

    @property
    def key_payload_sha256(self) -> str:
        return _key_payload_sha(self.to_payload())


@dataclass(frozen=True)
class AsrCacheKey:
    mono_payload_sha256: str
    frontend: str
    model_identity: Any
    language_model_identity: Any
    tokenizer_identity: Any
    decoder_identity: Any
    precision_runtime_identity: Any
    asr_contract_identity: str
    sample_rate_hz: int = NATIVE_SAMPLE_RATE_HZ
    mono_dtype: str = "float32"
    schema_version: str = ASR_CACHE_KEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha(self.mono_payload_sha256, "asr.mono_payload_sha256")
        if self.frontend not in ("mean_lr", "fixed_L", "fixed_R"):
            raise CacheError("asr.frontend is not a frozen frontend")
        for name in (
            "model_identity",
            "language_model_identity",
            "tokenizer_identity",
            "decoder_identity",
            "precision_runtime_identity",
        ):
            _semantic_json(getattr(self, name), "asr." + name)
        _sha(self.asr_contract_identity, "asr.asr_contract_identity")
        _positive_int(self.sample_rate_hz, "asr.sample_rate_hz")
        if self.sample_rate_hz != NATIVE_SAMPLE_RATE_HZ:
            raise CacheError("asr.sample_rate_hz must be native16")
        if self.mono_dtype != "float32":
            raise CacheError("asr.mono_dtype must be float32")
        if self.schema_version != ASR_CACHE_KEY_SCHEMA_VERSION:
            raise CacheError("asr.schema_version is invalid")
        for name in (
            "model_identity",
            "language_model_identity",
            "tokenizer_identity",
            "decoder_identity",
            "precision_runtime_identity",
        ):
            object.__setattr__(self, name, _freeze(_semantic_json(getattr(self, name), "asr." + name)))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": CACHE_LAYER_ASR,
            "mono_payload_sha256": self.mono_payload_sha256,
            "frontend": self.frontend,
            "model_identity": _thaw(self.model_identity),
            "language_model_identity": _thaw(self.language_model_identity),
            "tokenizer_identity": _thaw(self.tokenizer_identity),
            "decoder_identity": _thaw(self.decoder_identity),
            "precision_runtime_identity": _thaw(self.precision_runtime_identity),
            "asr_contract_identity": self.asr_contract_identity,
            "sample_rate_hz": self.sample_rate_hz,
            "mono_dtype": self.mono_dtype,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AsrCacheKey":
        _exact_fields(payload, tuple(cls.__dataclass_fields__) + ("layer",), "asr_key")
        if payload["layer"] != CACHE_LAYER_ASR:
            raise CacheError("asr_key.layer is invalid")
        data = dict(payload)
        data.pop("layer")
        return cls(**data)

    @property
    def cache_key(self) -> str:
        return stable_id("asr-cache-key", self.to_payload())

    @property
    def key_payload_sha256(self) -> str:
        return _key_payload_sha(self.to_payload())


def _metadata_identity_common(
    payload: Mapping[str, Any],
    expected_schema: str,
    expected_layer: str,
    key: Any,
    path: str,
    fields: Sequence[str],
) -> None:
    _exact_fields(
        payload,
        fields,
        path,
    )
    if payload["schema_version"] != expected_schema:
        raise CacheError("{}.schema_version is invalid".format(path))
    if payload["layer"] != expected_layer:
        raise CacheError("{}.layer is invalid".format(path))
    if payload["cache_key"] != key.cache_key:
        raise _MetadataError(KEY_MISMATCH, "{}.cache_key does not match request".format(path))
    if payload["key_payload"] != key.to_payload():
        raise _MetadataError(KEY_MISMATCH, "{}.key_payload does not match request".format(path))
    if payload["key_payload_sha256"] != key.key_payload_sha256:
        raise _MetadataError(KEY_MISMATCH, "{}.key_payload_sha256 does not match request".format(path))
    _sha(payload["payload_sha256"], path + ".payload_sha256")
    if not isinstance(payload["provenance"], Mapping):
        raise CacheError("{}.provenance must be a mapping".format(path))


def _metadata_common(
    payload: Mapping[str, Any],
    expected_schema: str,
    expected_layer: str,
    key: Any,
    path: str,
) -> None:
    _metadata_identity_common(
        payload,
        expected_schema,
        expected_layer,
        key,
        path,
        fields=(
            "schema_version",
            "layer",
            "cache_key",
            "key_payload",
            "key_payload_sha256",
            "payload_filename",
            "payload_type",
            "payload_sha256",
            "expected_shape",
            "dtype",
            "sample_rate_hz",
            "channel_order",
            "semantic_identities",
            "provenance",
        ),
    )


def _shape(value: Any, path: str, rank: int) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rank:
        raise CacheError("{} must have rank {}".format(path, rank))
    result = tuple(_positive_int(item, "{}[{}]".format(path, index)) for index, item in enumerate(value))
    return result


def _float32_metadata(payload: Mapping[str, Any], path: str) -> None:
    if payload["dtype"] != "float32":
        raise CacheError("{}.dtype must be float32".format(path))
    if payload["sample_rate_hz"] != NATIVE_SAMPLE_RATE_HZ:
        raise CacheError("{}.sample_rate_hz must be native16".format(path))
    if payload["channel_order"] != list(CHANNEL_ORDER):
        raise CacheError("{}.channel_order must be [L, R]".format(path))


@dataclass(frozen=True)
class RirCacheMetadata:
    key: RirCacheKey
    payload_sha256: str
    expected_shape: Tuple[int, int]
    provenance: Mapping[str, Any]
    dtype: str = "float32"
    sample_rate_hz: int = NATIVE_SAMPLE_RATE_HZ
    channel_order: Tuple[str, str] = CHANNEL_ORDER
    schema_version: str = RIR_CACHE_METADATA_SCHEMA_VERSION
    payload_filename: str = "payload.npy"
    payload_type: str = "numpy_npy"

    def __post_init__(self) -> None:
        _sha(self.payload_sha256, "rir_metadata.payload_sha256")
        object.__setattr__(self, "expected_shape", _shape(self.expected_shape, "rir_metadata.expected_shape", 2))
        if self.dtype != "float32" or self.sample_rate_hz != NATIVE_SAMPLE_RATE_HZ or tuple(self.channel_order) != CHANNEL_ORDER:
            raise CacheError("RIR metadata dtype/sample/channel policy is invalid")
        if self.key.sample_rate_hz != self.sample_rate_hz or tuple(self.key.channel_order) != tuple(self.channel_order):
            raise CacheError("RIR metadata sample/channel policy does not match key")
        if self.schema_version != RIR_CACHE_METADATA_SCHEMA_VERSION or self.payload_filename != "payload.npy" or self.payload_type != "numpy_npy":
            raise CacheError("RIR metadata schema or payload policy is invalid")
        if not isinstance(self.provenance, Mapping):
            raise CacheError("rir_metadata.provenance must be a mapping")
        object.__setattr__(self, "provenance", _freeze(_provenance_json(self.provenance, "rir_metadata.provenance")))

    @property
    def semantic_identities(self) -> Dict[str, Any]:
        payload = self.key.to_payload()
        return {
            "scene_resource_identities": payload["scene_resource_identities"],
            "acoustic_contract_identity": payload["acoustic_contract_identity"],
            "renderer_algorithm_identity": payload["renderer_algorithm_identity"],
            "materials_policy": payload["materials_policy"],
            "replicate_identity": payload["replicate_identity"],
        }

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": CACHE_LAYER_RIR,
            "cache_key": self.key.cache_key,
            "key_payload": self.key.to_payload(),
            "key_payload_sha256": self.key.key_payload_sha256,
            "payload_filename": self.payload_filename,
            "payload_type": self.payload_type,
            "payload_sha256": self.payload_sha256,
            "expected_shape": list(self.expected_shape),
            "dtype": self.dtype,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_order": list(self.channel_order),
            "semantic_identities": self.semantic_identities,
            "provenance": _thaw(self.provenance),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], key: Optional[RirCacheKey] = None) -> "RirCacheMetadata":
        key_payload = payload.get("key_payload") if isinstance(payload, Mapping) else None
        try:
            parsed_key = RirCacheKey.from_payload(key_payload)
        except Exception as exc:
            raise _MetadataError(METADATA_SCHEMA_MISMATCH, "invalid RIR key payload") from exc
        if key is not None and parsed_key.cache_key != key.cache_key:
            raise _MetadataError(KEY_MISMATCH, "RIR metadata key does not match request")
        try:
            _metadata_common(payload, RIR_CACHE_METADATA_SCHEMA_VERSION, CACHE_LAYER_RIR, parsed_key, "rir_metadata")
            _float32_metadata(payload, "rir_metadata")
            shape = _shape(payload["expected_shape"], "rir_metadata.expected_shape", 2)
            if payload["semantic_identities"] != {
                "scene_resource_identities": parsed_key.to_payload()["scene_resource_identities"],
                "acoustic_contract_identity": parsed_key.acoustic_contract_identity,
                "renderer_algorithm_identity": parsed_key.renderer_algorithm_identity,
                "materials_policy": parsed_key.materials_policy,
                "replicate_identity": parsed_key.replicate_identity,
            }:
                raise _MetadataError(SEMANTIC_IDENTITY_MISMATCH, "RIR semantic identities do not match key")
            return cls(
                key=parsed_key,
                payload_sha256=payload["payload_sha256"],
                expected_shape=shape,
                provenance=payload["provenance"],
                dtype=payload["dtype"],
                sample_rate_hz=payload["sample_rate_hz"],
                channel_order=tuple(payload["channel_order"]),
                schema_version=payload["schema_version"],
                payload_filename=payload["payload_filename"],
                payload_type=payload["payload_type"],
            )
        except _MetadataError:
            raise
        except CacheError as exc:
            raise _MetadataError(METADATA_SCHEMA_MISMATCH, str(exc)) from exc


@dataclass(frozen=True)
class MixtureCacheMetadata:
    key: MixtureCacheKey
    payload_sha256: str
    expected_shape: Tuple[int, int]
    provenance: Mapping[str, Any]
    dtype: str = "float32"
    sample_rate_hz: int = NATIVE_SAMPLE_RATE_HZ
    channel_order: Tuple[str, str] = CHANNEL_ORDER
    schema_version: str = MIXTURE_CACHE_METADATA_SCHEMA_VERSION
    payload_filename: str = "payload.npy"
    payload_type: str = "numpy_npy"

    def __post_init__(self) -> None:
        _sha(self.payload_sha256, "mixture_metadata.payload_sha256")
        object.__setattr__(self, "expected_shape", _shape(self.expected_shape, "mixture_metadata.expected_shape", 2))
        if self.dtype != "float32" or self.sample_rate_hz != NATIVE_SAMPLE_RATE_HZ or tuple(self.channel_order) != CHANNEL_ORDER:
            raise CacheError("mixture metadata dtype/sample/channel policy is invalid")
        if self.schema_version != MIXTURE_CACHE_METADATA_SCHEMA_VERSION or self.payload_filename != "payload.npy" or self.payload_type != "numpy_npy":
            raise CacheError("mixture metadata schema or payload policy is invalid")
        object.__setattr__(self, "provenance", _freeze(_provenance_json(self.provenance, "mixture_metadata.provenance")))

    @property
    def semantic_identities(self) -> Dict[str, Any]:
        payload = self.key.to_payload()
        return {name: payload[name] for name in payload if name not in ("schema_version", "layer")}

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": CACHE_LAYER_MIXTURE,
            "cache_key": self.key.cache_key,
            "key_payload": self.key.to_payload(),
            "key_payload_sha256": self.key.key_payload_sha256,
            "payload_filename": self.payload_filename,
            "payload_type": self.payload_type,
            "payload_sha256": self.payload_sha256,
            "expected_shape": list(self.expected_shape),
            "dtype": self.dtype,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_order": list(self.channel_order),
            "semantic_identities": self.semantic_identities,
            "provenance": _thaw(self.provenance),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], key: Optional[MixtureCacheKey] = None) -> "MixtureCacheMetadata":
        try:
            parsed_key = MixtureCacheKey.from_payload(payload.get("key_payload"))
        except Exception as exc:
            raise _MetadataError(METADATA_SCHEMA_MISMATCH, "invalid mixture key payload") from exc
        if key is not None and parsed_key.cache_key != key.cache_key:
            raise _MetadataError(KEY_MISMATCH, "mixture metadata key does not match request")
        try:
            _metadata_common(payload, MIXTURE_CACHE_METADATA_SCHEMA_VERSION, CACHE_LAYER_MIXTURE, parsed_key, "mixture_metadata")
            _float32_metadata(payload, "mixture_metadata")
            shape = _shape(payload["expected_shape"], "mixture_metadata.expected_shape", 2)
            if payload["semantic_identities"] != MixtureCacheMetadata(parsed_key, payload["payload_sha256"], shape, payload["provenance"]).semantic_identities:
                raise _MetadataError(SEMANTIC_IDENTITY_MISMATCH, "mixture semantic identities do not match key")
            return cls(
                key=parsed_key,
                payload_sha256=payload["payload_sha256"],
                expected_shape=shape,
                provenance=payload["provenance"],
                dtype=payload["dtype"],
                sample_rate_hz=payload["sample_rate_hz"],
                channel_order=tuple(payload["channel_order"]),
                schema_version=payload["schema_version"],
                payload_filename=payload["payload_filename"],
                payload_type=payload["payload_type"],
            )
        except _MetadataError:
            raise
        except CacheError as exc:
            raise _MetadataError(METADATA_SCHEMA_MISMATCH, str(exc)) from exc


def _validate_asr_result(value: Any, key: AsrCacheKey) -> Dict[str, Any]:
    expected = (
        "schema_version",
        "hypothesis",
        "score",
        "score_semantics",
        "input_waveform_sha256",
        "frontend",
        "model_identity",
        "decoder_identity",
        "asr_contract_identity",
        "raw_decoder_metadata",
    )
    _exact_fields(value, expected, "asr_result")
    if value["schema_version"] != ASR_RESULT_SCHEMA_VERSION:
        raise CacheError("asr_result.schema_version is invalid")
    if not isinstance(value["hypothesis"], str):
        raise CacheError("asr_result.hypothesis must be a string")
    if value["score"] is not None:
        _finite(value["score"], "asr_result.score")
    _string(value["score_semantics"], "asr_result.score_semantics")
    _sha(value["input_waveform_sha256"], "asr_result.input_waveform_sha256")
    if value["input_waveform_sha256"] != key.mono_payload_sha256:
        raise CacheError("asr_result input waveform hash does not match key")
    if value["frontend"] != key.frontend:
        raise CacheError("asr_result frontend does not match key")
    if _semantic_json(value["model_identity"], "asr_result.model_identity") != _thaw(key.model_identity):
        raise CacheError("asr_result model identity does not match key")
    if _semantic_json(value["decoder_identity"], "asr_result.decoder_identity") != _thaw(key.decoder_identity):
        raise CacheError("asr_result decoder identity does not match key")
    if value["asr_contract_identity"] != key.asr_contract_identity:
        raise CacheError("asr_result contract identity does not match key")
    if not isinstance(value["raw_decoder_metadata"], Mapping):
        raise CacheError("asr_result.raw_decoder_metadata must be a mapping")
    return _thaw(_freeze(dict(value)))


@dataclass(frozen=True)
class AsrCacheMetadata:
    key: AsrCacheKey
    payload_sha256: str
    input_shape: Tuple[int]
    provenance: Mapping[str, Any]
    input_layout: str = ASR_INPUT_LAYOUT
    input_dtype: str = "float32"
    sample_rate_hz: int = NATIVE_SAMPLE_RATE_HZ
    schema_version: str = ASR_CACHE_METADATA_SCHEMA_VERSION
    payload_filename: str = "payload.json"
    payload_type: str = "json_asr_result"

    def __post_init__(self) -> None:
        _sha(self.payload_sha256, "asr_metadata.payload_sha256")
        object.__setattr__(self, "input_shape", _shape(self.input_shape, "asr_metadata.input_shape", 1))
        if self.input_layout != ASR_INPUT_LAYOUT:
            raise CacheError("ASR metadata input_layout must be mono")
        if self.input_dtype != "float32" or self.sample_rate_hz != NATIVE_SAMPLE_RATE_HZ:
            raise CacheError("ASR metadata input dtype/sample policy is invalid")
        if self.key.sample_rate_hz != self.sample_rate_hz or self.key.mono_dtype != self.input_dtype:
            raise CacheError("ASR metadata input policy does not match key")
        if self.schema_version != ASR_CACHE_METADATA_SCHEMA_VERSION or self.payload_filename != "payload.json" or self.payload_type != "json_asr_result":
            raise CacheError("ASR metadata schema or payload policy is invalid")
        object.__setattr__(self, "provenance", _freeze(_provenance_json(self.provenance, "asr_metadata.provenance")))

    @property
    def expected_shape(self) -> Tuple[int]:
        """Compatibility view; the serialized authority is input_shape."""

        return self.input_shape

    @property
    def dtype(self) -> str:
        """Compatibility view; the serialized authority is input_dtype."""

        return self.input_dtype

    @property
    def semantic_identities(self) -> Dict[str, Any]:
        payload = self.key.to_payload()
        return {name: payload[name] for name in payload if name not in ("schema_version", "layer")}

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": CACHE_LAYER_ASR,
            "cache_key": self.key.cache_key,
            "key_payload": self.key.to_payload(),
            "key_payload_sha256": self.key.key_payload_sha256,
            "payload_filename": self.payload_filename,
            "payload_type": self.payload_type,
            "payload_sha256": self.payload_sha256,
            "input_layout": self.input_layout,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "sample_rate_hz": self.sample_rate_hz,
            "semantic_identities": self.semantic_identities,
            "provenance": _thaw(self.provenance),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], key: Optional[AsrCacheKey] = None) -> "AsrCacheMetadata":
        try:
            parsed_key = AsrCacheKey.from_payload(payload.get("key_payload"))
        except Exception as exc:
            raise _MetadataError(METADATA_SCHEMA_MISMATCH, "invalid ASR key payload") from exc
        if key is not None and parsed_key.cache_key != key.cache_key:
            raise _MetadataError(KEY_MISMATCH, "ASR metadata key does not match request")
        try:
            _metadata_identity_common(
                payload,
                ASR_CACHE_METADATA_SCHEMA_VERSION,
                CACHE_LAYER_ASR,
                parsed_key,
                "asr_metadata",
                fields=(
                    "schema_version", "layer", "cache_key", "key_payload", "key_payload_sha256",
                    "payload_filename", "payload_type", "payload_sha256", "input_layout", "input_shape",
                    "input_dtype", "sample_rate_hz", "semantic_identities", "provenance",
                ),
            )
            if payload["input_layout"] != ASR_INPUT_LAYOUT:
                raise CacheError("asr_metadata.input_layout must be mono")
            if payload["input_dtype"] != "float32":
                raise CacheError("asr_metadata.input_dtype must be float32")
            if payload["sample_rate_hz"] != NATIVE_SAMPLE_RATE_HZ:
                raise CacheError("asr_metadata.sample_rate_hz must be native16")
            input_shape = _shape(payload["input_shape"], "asr_metadata.input_shape", 1)
            if payload["semantic_identities"] != {name: parsed_key.to_payload()[name] for name in parsed_key.to_payload() if name not in ("schema_version", "layer")}:
                raise _MetadataError(SEMANTIC_IDENTITY_MISMATCH, "ASR semantic identities do not match key")
            return cls(
                key=parsed_key,
                payload_sha256=payload["payload_sha256"],
                input_shape=input_shape,
                provenance=payload["provenance"],
                input_layout=payload["input_layout"],
                input_dtype=payload["input_dtype"],
                sample_rate_hz=payload["sample_rate_hz"],
                schema_version=payload["schema_version"],
                payload_filename=payload["payload_filename"],
                payload_type=payload["payload_type"],
            )
        except _MetadataError:
            raise
        except CacheError as exc:
            raise _MetadataError(METADATA_SCHEMA_MISMATCH, str(exc)) from exc


@dataclass(frozen=True)
class CacheReadResult:
    status: str
    reason: str
    payload: Any = None
    metadata: Any = None


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _load_npy(payload: bytes) -> np.ndarray:
    buffer = io.BytesIO(payload)
    value = np.load(buffer, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise CacheError("Numpy payload is not an array")
    return value


def _validate_waveform(payload: Any, path: str) -> np.ndarray:
    if not isinstance(payload, np.ndarray):
        raise CacheError("{} must be a numpy array".format(path))
    if payload.dtype != np.dtype(np.float32):
        raise CacheError("{} must be float32".format(path))
    if payload.ndim != 2 or payload.shape[0] <= 0 or payload.shape[1] != 2:
        raise CacheError("{} must have shape (N,2)".format(path))
    if not np.isfinite(payload).all():
        raise CacheError("{} contains NaN/Inf".format(path))
    return np.ascontiguousarray(payload)


def _validate_mono_shape(shape: Sequence[int], path: str) -> Tuple[int]:
    return _shape(shape, path, 1)


class CacheStore:
    """Atomic A4 cache store with fail-closed read semantics."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.storage = DatasetStorage(str(self.root))

    def entry_dir(self, key: Any) -> Path:
        return self.root / key.to_payload()["layer"] / key.cache_key

    def _paths(self, key: Any, payload_filename: str) -> Tuple[Path, Path]:
        directory = self.entry_dir(key)
        return directory / "metadata.json", directory / payload_filename

    def _write(self, key: Any, metadata: Any, payload_bytes: bytes) -> Any:
        metadata_path, payload_path = self._paths(key, metadata.payload_filename)
        self.storage.atomic_write_bytes(payload_path, payload_bytes)
        self.storage.atomic_write_bytes(metadata_path, canonical_json_bytes(metadata.to_payload()))
        return metadata

    def write_rir(self, key: RirCacheKey, payload: Any, provenance: Optional[Mapping[str, Any]] = None) -> RirCacheMetadata:
        array = _validate_waveform(payload, "rir_payload")
        data = _npy_bytes(array)
        metadata = RirCacheMetadata(key, sha256_bytes(data), tuple(array.shape), provenance or {})
        return self._write(key, metadata, data)

    def write_mixture(self, key: MixtureCacheKey, payload: Any, provenance: Optional[Mapping[str, Any]] = None) -> MixtureCacheMetadata:
        array = _validate_waveform(payload, "mixture_payload")
        data = _npy_bytes(array)
        metadata = MixtureCacheMetadata(key, sha256_bytes(data), tuple(array.shape), provenance or {})
        return self._write(key, metadata, data)

    def write_asr(
        self,
        key: AsrCacheKey,
        payload: Mapping[str, Any],
        input_shape: Sequence[int],
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> AsrCacheMetadata:
        normalized = _validate_asr_result(payload, key)
        shape = _validate_mono_shape(input_shape, "asr_input_shape")
        data = _result_json_bytes(normalized)
        metadata = AsrCacheMetadata(key, sha256_bytes(data), shape, provenance or {})
        return self._write(key, metadata, data)

    def _missing_result(self, metadata_path: Path, payload_path: Path) -> Optional[CacheReadResult]:
        metadata_exists = metadata_path.exists()
        payload_exists = payload_path.exists()
        if not metadata_exists and not payload_exists:
            return CacheReadResult(MISS, MISSING_METADATA)
        if not metadata_exists:
            return CacheReadResult(INVALID_CORRUPT, MISSING_METADATA)
        if not payload_exists:
            return CacheReadResult(INVALID_CORRUPT, MISSING_PAYLOAD)
        return None

    def _read(
        self,
        key: Any,
        metadata_type: Any,
        payload_loader: Any,
        payload_validator: Any,
    ) -> CacheReadResult:
        payload_filename = "payload.json" if isinstance(key, AsrCacheKey) else "payload.npy"
        metadata_path, payload_path = self._paths(key, payload_filename)
        missing = self._missing_result(metadata_path, payload_path)
        if missing is not None:
            return missing
        try:
            metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = metadata_type.from_payload(metadata_raw, key)
        except _MetadataError as exc:
            return CacheReadResult(INVALID_CORRUPT, exc.reason)
        except (OSError, UnicodeError, json.JSONDecodeError, CacheError, TypeError, ValueError):
            return CacheReadResult(INVALID_CORRUPT, METADATA_SCHEMA_MISMATCH)
        try:
            payload_bytes = payload_path.read_bytes()
        except OSError:
            return CacheReadResult(INVALID_CORRUPT, MISSING_PAYLOAD)
        if sha256_bytes(payload_bytes) != metadata.payload_sha256:
            return CacheReadResult(INVALID_CORRUPT, PAYLOAD_HASH_MISMATCH)
        try:
            payload = payload_loader(payload_bytes)
            payload_validator(payload, metadata)
        except CacheError as exc:
            message = str(exc)
            if "dtype mismatch" in message:
                reason = DTYPE_MISMATCH
            elif "shape mismatch" in message:
                reason = SHAPE_MISMATCH
            elif "NaN/Inf" in message:
                reason = NONFINITE_PAYLOAD
            else:
                reason = PAYLOAD_SCHEMA_MISMATCH
            return CacheReadResult(INVALID_CORRUPT, reason)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, UnicodeError):
            return CacheReadResult(INVALID_CORRUPT, PAYLOAD_PARSE_ERROR)
        return CacheReadResult(HIT_VALID, "", payload, metadata)

    def read_rir(self, key: RirCacheKey) -> CacheReadResult:
        def validate(payload: Any, metadata: RirCacheMetadata) -> None:
            if not isinstance(payload, np.ndarray):
                raise CacheError("RIR payload is not an array")
            if payload.dtype != np.dtype(np.float32):
                raise CacheError("RIR dtype mismatch")
            if tuple(payload.shape) != metadata.expected_shape or payload.ndim != 2 or payload.shape[1] != 2:
                raise CacheError("RIR shape mismatch")
            if not np.isfinite(payload).all():
                raise CacheError("RIR contains NaN/Inf")

        return self._read(key, RirCacheMetadata, _load_npy, validate)

    def read_mixture(self, key: MixtureCacheKey) -> CacheReadResult:
        def validate(payload: Any, metadata: MixtureCacheMetadata) -> None:
            if not isinstance(payload, np.ndarray):
                raise CacheError("mixture payload is not an array")
            if payload.dtype != np.dtype(np.float32):
                raise CacheError("mixture dtype mismatch")
            if tuple(payload.shape) != metadata.expected_shape or payload.ndim != 2 or payload.shape[1] != 2:
                raise CacheError("mixture shape mismatch")
            if not np.isfinite(payload).all():
                raise CacheError("mixture contains NaN/Inf")

        return self._read(key, MixtureCacheMetadata, _load_npy, validate)

    def read_asr(self, key: AsrCacheKey) -> CacheReadResult:
        def load(payload: bytes) -> Mapping[str, Any]:
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, Mapping):
                raise CacheError("ASR result must be an object")
            return value

        def validate(payload: Any, metadata: AsrCacheMetadata) -> None:
            _validate_asr_result(payload, key)

        return self._read(key, AsrCacheMetadata, load, validate)


__all__ = [
    "A2_ACOUSTIC_CONTRACT_SHA256",
    "A3_ASR_CONTRACT_SHA256",
    "ASR_CACHE_KEY_SCHEMA_VERSION",
    "ASR_CACHE_METADATA_SCHEMA_VERSION",
    "ASR_INPUT_LAYOUT",
    "ASR_RESULT_SCHEMA_VERSION",
    "CACHE_ALGORITHM_IDENTITY",
    "CACHE_ATOMIC_COMMIT_IDENTITY",
    "CACHE_INTEGRITY_IDENTITY",
    "CACHE_KEY_SERIALIZATION_IDENTITY",
    "CACHE_KEYING_IDENTITY",
    "CACHE_METADATA_SCHEMA_VERSION",
    "MIXTURE_CACHE_KEY_SCHEMA_VERSION",
    "CacheError",
    "CacheReadResult",
    "CacheStore",
    "HIT_VALID",
    "INVALID_CORRUPT",
    "KEY_MISMATCH",
    "METADATA_SCHEMA_MISMATCH",
    "MISSING_METADATA",
    "MISSING_PAYLOAD",
    "MISS",
    "MixtureCacheKey",
    "MixtureCacheMetadata",
    "NOISE_SOURCE_TIME_IDENTITY",
    "NONFINITE_PAYLOAD",
    "PAYLOAD_HASH_MISMATCH",
    "PAYLOAD_PARSE_ERROR",
    "PAYLOAD_SCHEMA_MISMATCH",
    "RIR_CACHE_KEY_SCHEMA_VERSION",
    "RIR_CACHE_METADATA_SCHEMA_VERSION",
    "RirCacheKey",
    "RirCacheMetadata",
    "SEMANTIC_IDENTITY_MISMATCH",
    "SHAPE_MISMATCH",
    "DTYPE_MISMATCH",
    "AsrCacheKey",
    "AsrCacheMetadata",
]
