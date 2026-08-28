"""V1 storage paths, provenance files and finalized immutability."""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from active_audition.data.storage import DatasetStorage

from .schema import DATASET_FAMILY, RenderPolicy, SCHEMA_VERSION


class V1StorageError(ValueError):
    """Raised for unsafe or immutable V1 storage operations."""


V1_DATASET_RELATIVE_ROOT = Path("datasets/binaural_foa_clsdoa_v1")
_SAFE_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset_id(dataset_id: str) -> str:
    value = str(dataset_id)
    if not _SAFE_DATASET_ID.fullmatch(value) or value in (".", ".."):
        raise V1StorageError("dataset_id contains unsafe path characters")
    return value


def validate_relative_path(value: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or not str(value) or ".." in path.parts:
        raise V1StorageError("payload path must be relative and traversal-free")
    return path.as_posix()


class V1DatasetStorage:
    """V1 path resolver that reuses V0 atomic-write primitives where possible."""

    def __init__(self, root: str, save_rir: bool = True, require_rir: bool = True):
        self.root = Path(root)
        self._primitives = DatasetStorage(str(self.root))
        self.render_policy = RenderPolicy(save_rir=save_rir, require_rir=require_rir)

    @classmethod
    def from_config(cls, repo_root: str, config: Mapping[str, Any]) -> "V1DatasetStorage":
        dataset_id = validate_dataset_id(config["dataset_id"] if "dataset_id" in config else config["storage"]["dataset_id"])
        policy = RenderPolicy.from_config(config)
        return cls(Path(repo_root) / V1_DATASET_RELATIVE_ROOT / dataset_id, policy.save_rir, policy.require_rir)

    @property
    def success_path(self) -> Path:
        return self.root / "_SUCCESS"

    @property
    def is_finalized(self) -> bool:
        return self.success_path.is_file()

    def ensure_writable(self) -> None:
        if self.is_finalized:
            raise V1StorageError("finalized V1 dataset is immutable: {}".format(self.root))
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure_layout(self) -> None:
        self.ensure_writable()
        for relative in ("manifests", "audio/binaural", "audio/foa", "cache/rir", "reports"):
            self.root.joinpath(relative).mkdir(parents=True, exist_ok=True)

    def manifest_path(self, name: str) -> Path:
        if name not in ("episodes.jsonl", "renders.jsonl"):
            raise V1StorageError("unsupported V1 manifest: {}".format(name))
        return self.root / "manifests" / name

    def payload_path(self, relative: str) -> Path:
        return self.root / validate_relative_path(relative)

    def write_bytes(self, relative: str, payload: bytes) -> Path:
        self.ensure_writable()
        path = self.payload_path(relative)
        self._primitives.atomic_write_bytes(path, payload)
        return path

    def write_json(self, relative: str, value: Any) -> Path:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        return self.write_bytes(relative, payload)

    def write_identity(self, identity: Mapping[str, Any]) -> Path:
        validate_identity(identity)
        return self.write_json("identity.json", identity)

    def write_config_resolved(self, config: Mapping[str, Any]) -> str:
        self.ensure_writable()
        text = yaml.safe_dump(deepcopy(dict(config)), allow_unicode=True, sort_keys=True)
        payload = text.encode("utf-8")
        self._primitives.atomic_write_bytes(self.root / "config_resolved.yaml", payload)
        return sha256_bytes(payload)

    def write_resources_lock(self, resources: Mapping[str, Any]) -> Path:
        validate_resources_lock(resources)
        return self.write_json("resources.lock.json", resources)

    def finalize(self) -> Path:
        if self.is_finalized:
            raise V1StorageError("V1 dataset is already finalized")
        required = ("identity.json", "config_resolved.yaml", "resources.lock.json", "manifests/episodes.jsonl", "manifests/renders.jsonl")
        missing = [item for item in required if not self.root.joinpath(item).is_file()]
        if missing:
            raise V1StorageError("cannot finalize; missing: {}".format(", ".join(missing)))
        self.ensure_writable()
        self._primitives.atomic_write_bytes(self.success_path, b"Dataset finalized / immutable\n")
        return self.success_path


def validate_identity(identity: Mapping[str, Any]) -> None:
    required = ("dataset_id", "dataset_family", "schema_version", "generation_code_commit", "created_at", "config_sha256", "ontology_sha256", "source_registry_sha256", "scene_registry_sha256")
    missing = [key for key in required if key not in identity]
    if missing:
        raise V1StorageError("identity missing fields: {}".format(", ".join(missing)))
    if identity["dataset_family"] != DATASET_FAMILY or identity.get("schema_version") != SCHEMA_VERSION:
        raise V1StorageError("identity does not match V1 family/schema")
    validate_dataset_id(str(identity["dataset_id"]))
    for key in required[5:]:
        if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(identity[key])):
            raise V1StorageError("{} must be a SHA-256 hex digest".format(key))


def validate_resources_lock(resources: Mapping[str, Any]) -> None:
    required = ("soundspaces_commit", "habitat_sim_version", "rlraudio_propagation", "hrtf", "foa_contract_version", "ontology_sha256", "source_registry_sha256", "scene_registry_sha256", "source_split_version", "scene_split_version", "sample_rate_hz", "clip_duration_sec", "indirectRayCount", "sourceRayCount", "materials_mode", "random_seed")
    missing = [key for key in required if key not in resources]
    if missing:
        raise V1StorageError("resources.lock missing fields: {}".format(", ".join(missing)))
    if int(resources["sample_rate_hz"]) != 24000 or float(resources["clip_duration_sec"]) != 5.0:
        raise V1StorageError("resources.lock audio contract mismatch")
    if int(resources["indirectRayCount"]) != 5000 or int(resources["sourceRayCount"]) != 200 or resources["materials_mode"] != "OFF":
        raise V1StorageError("resources.lock acoustic contract mismatch")


def merge_resolved_config(contract: Mapping[str, Any], build: Optional[Mapping[str, Any]] = None, runtime: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    """Deterministically merge contract, build config and runtime overrides."""

    def merge(left: Mapping[str, Any], right: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
        result = deepcopy(dict(left))
        for key, value in (right or {}).items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    return merge(merge(contract, build), runtime)
