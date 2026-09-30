"""The single canonical identity implementation for the A4 namespace.

The encoding deliberately follows the already frozen A0 ``canonical-json-v1``
convention: UTF-8 JSON, sorted keys, compact separators, and no newline or
filesystem metadata.  This module adds recursive type and finite-number
checks so identities cannot silently depend on Python's permissive JSON
coercions.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_JSON_VERSION = "canonical-json-v1"
RESULT_DEPENDENT_KEYS = frozenset(
    {
        "wer",
        "cer",
        "hypothesis",
        "score",
        "sdiw",
        "run_timestamp",
        "timestamp",
        "created_at",
        "updated_at",
        "raw_decoder_metadata",
        "asr_result",
    }
)


class A4IdentityError(ValueError):
    """Raised when a semantic identity payload cannot be canonicalized."""


def _canonical_value(value: Any, path: str = "payload") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise A4IdentityError("{} contains a non-string key".format(path))
            if key.lower() in RESULT_DEPENDENT_KEYS:
                raise A4IdentityError("{} contains result-dependent field {!r}".format(path, key))
            result[key] = _canonical_value(item, "{}.{}".format(path, key))
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, "{}[{}]".format(path, index)) for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise A4IdentityError("{} must be finite".format(path))
        # JSON has one zero value; normalize negative zero like A0 does.
        return 0.0 if value == 0.0 else value
    raise A4IdentityError("{} contains unsupported value type {}".format(path, type(value).__name__))


def canonical_json(value: Any) -> str:
    """Return strict, deterministic, newline-free canonical JSON text."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical JSON encoded as UTF-8 bytes."""

    return canonical_json(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise A4IdentityError("sha256 input must be bytes")
    return hashlib.sha256(value).hexdigest()


def identity_sha256(payload: Any) -> str:
    """Hash a semantic payload without adding run-dependent metadata."""

    return sha256_bytes(canonical_json_bytes(payload))


def stable_id(namespace: str, payload: Any) -> str:
    """Return a readable ID that retains the complete SHA-256 digest."""

    if not isinstance(namespace, str) or not namespace or not re.match(r"^[a-z][a-z0-9_-]*$", namespace):
        raise A4IdentityError("namespace must be a lower-case stable identifier")
    return "{}-{}".format(namespace, identity_sha256(payload))


def validate_sha256(value: Any, path: str = "sha256") -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise A4IdentityError("{} must be a lower-case SHA256 digest".format(path))
    return value


def validate_stable_id(value: Any, namespace: str, path: str = "id") -> str:
    if not isinstance(value, str) or not value.startswith(namespace + "-"):
        raise A4IdentityError("{} must use namespace {!r}".format(path, namespace))
    validate_sha256(value[len(namespace) + 1 :], path + ".sha256")
    return value


__all__ = [
    "A4IdentityError",
    "CANONICAL_JSON_VERSION",
    "RESULT_DEPENDENT_KEYS",
    "SHA256_RE",
    "canonical_json",
    "canonical_json_bytes",
    "identity_sha256",
    "sha256_bytes",
    "stable_id",
    "validate_sha256",
    "validate_stable_id",
]
