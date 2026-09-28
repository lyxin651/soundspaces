"""Strict A3-v2 contract validation and canonical hashing.

This module intentionally keeps the A3-v1 contract loader untouched.  The
v2 contract inherits the frozen model, decoder, environment, source registry,
frontend, and text identities from v1 while adding a held-out realistic-domain
qualification boundary.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from active_audition.asr.contract import asr_contract_sha256, load_asr_contract


A3_V2_VERSION = "active-asr-a3-instrument-v2"
A3_V1_SHA256 = "b972d3ca2354ead8a10d2954a60602896ebbc5204adb7e8692c22f2b340497e2"
A0_SHA256 = "d731393cda3ddb29f0bdf58249f104da59f29d012b976eeb2de1f160e1df8107"
A2_V2_SHA256 = "f1185d2c5fe81091199d5525f224822a3227c700c60bba9ab168e6d9ddc38f83"
A2_V3_SHA256 = "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c"
SHA256_HEX_LENGTH = 64


class A3V2ContractError(ValueError):
    """Raised when the versioned A3-v2 contract is not authoritative."""


def _only(value: Mapping, keys: Iterable[str], path: str) -> None:
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise A3V2ContractError("unknown A3-v2 field(s) at {}: {}".format(path, ", ".join(unknown)))


def _require(value: Mapping, keys: Iterable[str], path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise A3V2ContractError("missing A3-v2 field(s) at {}: {}".format(path, ", ".join(missing)))


def _mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise A3V2ContractError("{} must be a mapping".format(path))
    return value


def _section(root: Mapping, name: str, keys: Iterable[str]) -> Mapping:
    value = _mapping(root.get(name), name)
    _only(value, keys, name)
    _require(value, keys, name)
    return value


def _string(value: Any, path: str, expected: Optional[str] = None) -> str:
    if not isinstance(value, str) or not value:
        raise A3V2ContractError("{} must be a non-empty string".format(path))
    if expected is not None and value != expected:
        raise A3V2ContractError("{} must be {!r}".format(path, expected))
    return value


def _bool(value: Any, path: str, expected: Optional[bool] = None) -> bool:
    if not isinstance(value, bool):
        raise A3V2ContractError("{} must be boolean".format(path))
    if expected is not None and value is not expected:
        raise A3V2ContractError("{} must be {}".format(path, str(expected).lower()))
    return value


def _int(value: Any, path: str, expected: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise A3V2ContractError("{} must be integer".format(path))
    if expected is not None and value != expected:
        raise A3V2ContractError("{} must be {}".format(path, expected))
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise A3V2ContractError("{} must be numeric".format(path))
    value = float(value)
    if not math.isfinite(value):
        raise A3V2ContractError("{} must be finite".format(path))
    return value


def _sha(value: Any, path: str) -> str:
    value = _string(value, path)
    if len(value) != SHA256_HEX_LENGTH or any(ch not in "0123456789abcdef" for ch in value):
        raise A3V2ContractError("{} must be a lowercase SHA256".format(path))
    return value


def _identity(value: Any, path: str) -> Mapping:
    item = _mapping(value, path)
    _only(item, ("path", "sha256"), path)
    _require(item, ("path", "sha256"), path)
    _string(item["path"], path + ".path")
    _sha(item["sha256"], path + ".sha256")
    return item


def _manifest(value: Any, path: str, expected_schema: Optional[str] = None) -> Mapping:
    item = _mapping(value, path)
    _only(item, ("path", "sha256", "schema_version", "records"), path)
    _require(item, ("path", "sha256", "schema_version", "records"), path)
    _string(item["path"], path + ".path")
    _sha(item["sha256"], path + ".sha256")
    _string(item["schema_version"], path + ".schema_version", expected_schema)
    _int(item["records"], path + ".records")
    if item["records"] < 0:
        raise A3V2ContractError(path + ".records must be non-negative")
    return item


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise A3V2ContractError("contract contains non-finite number")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise A3V2ContractError("contract contains unsupported value type")


def canonical_a3_v2_json(contract: Mapping) -> str:
    validate_a3_v2_contract(contract)
    return json.dumps(_canonical(contract), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def a3_v2_contract_sha256(contract: Mapping) -> str:
    return hashlib.sha256(canonical_a3_v2_json(contract).encode("utf-8")).hexdigest()


def _v1_identity_sections(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or Path.cwd()
    path = root / "configs/active_audition/v1/asr_contract.yaml"
    v1 = load_asr_contract(str(path), require_frozen=True)
    if asr_contract_sha256(v1) != A3_V1_SHA256:
        raise A3V2ContractError("A3-v1 contract hash changed; v2 cannot inherit it")
    return {key: v1[key] for key in ("model", "environment", "decoder", "input", "frontends", "text", "sources")}


def validate_a3_v2_contract(contract: Mapping, require_frozen: bool = False, repo_root: Optional[str] = None) -> Mapping:
    root = _mapping(contract, "root")
    top = (
        "contract", "serialization", "parents", "production_acoustics", "materials_policy",
        "historical_v1", "office0", "realistic_domain", "model", "environment", "decoder",
        "input", "frontends", "text", "sources", "qualification", "artifacts",
    )
    _only(root, top, "root")
    _require(root, top, "root")

    meta = _section(root, "contract", ("namespace", "version", "gate", "state"))
    _string(meta["namespace"], "contract.namespace", "active-asr")
    _string(meta["version"], "contract.version", A3_V2_VERSION)
    _string(meta["gate"], "contract.gate", "A3")
    state = _string(meta["state"], "contract.state")
    if state not in ("DRAFT", "FROZEN"):
        raise A3V2ContractError("contract.state must be DRAFT or FROZEN")
    if require_frozen and state != "FROZEN":
        raise A3V2ContractError("A3-v2 qualification requires contract.state=FROZEN")

    serial = _section(root, "serialization", ("version", "hash_algorithm"))
    _string(serial["version"], "serialization.version", "canonical-json-v1")
    _string(serial["hash_algorithm"], "serialization.hash_algorithm", "sha256")

    parents = _section(root, "parents", ("a0_contract_sha256", "a2_v2_sha256", "a2_v3_sha256", "a3_v1_contract_sha256"))
    _sha(parents["a0_contract_sha256"], "parents.a0_contract_sha256")
    _sha(parents["a2_v2_sha256"], "parents.a2_v2_sha256")
    _sha(parents["a2_v3_sha256"], "parents.a2_v3_sha256")
    _sha(parents["a3_v1_contract_sha256"], "parents.a3_v1_contract_sha256")
    if dict(parents) != {
        "a0_contract_sha256": A0_SHA256,
        "a2_v2_sha256": A2_V2_SHA256,
        "a2_v3_sha256": A2_V3_SHA256,
        "a3_v1_contract_sha256": A3_V1_SHA256,
    }:
        raise A3V2ContractError("parent contract identity is not the accepted A0/A2/A3 authority")

    production = _section(root, "production_acoustics", (
        "renderer", "sample_rate_hz", "channel_layout", "channel_order", "convolution",
        "materials", "post_convolution_normalization", "per_rir_normalization",
        "per_utterance_normalization", "primary_frontend", "sensitivity_frontends",
    ))
    _string(production["renderer"], "production_acoustics.renderer", "SoundSpaces2_HabitatSim0.2.2_RLRAudioPropagation")
    _int(production["sample_rate_hz"], "production_acoustics.sample_rate_hz", 16000)
    _string(production["channel_layout"], "production_acoustics.channel_layout", "binaural")
    if production["channel_order"] != ["L", "R"]:
        raise A3V2ContractError("production_acoustics.channel_order must be [L,R]")
    _string(production["convolution"], "production_acoustics.convolution", "full")
    _string(production["materials"], "production_acoustics.materials", "OFF")
    for key in ("post_convolution_normalization", "per_rir_normalization", "per_utterance_normalization"):
        _string(production[key], "production_acoustics." + key, "none")
    _string(production["primary_frontend"], "production_acoustics.primary_frontend", "mean_lr")
    if production["sensitivity_frontends"] != ["fixed_L", "fixed_R"]:
        raise A3V2ContractError("production_acoustics.sensitivity_frontends must be fixed_L/fixed_R")

    materials = _section(root, "materials_policy", ("production", "materials_on", "audit_status", "audit_evidence"))
    _string(materials["production"], "materials_policy.production", "OFF")
    _string(materials["materials_on"], "materials_policy.materials_on", "DEFERRED_UNSUPPORTED_REALISM_EXTENSION")
    _string(materials["audit_status"], "materials_policy.audit_status", "MATERIALS_ON_PROVENANCE_RECOVERED_BUT_RUNTIME_UNSTABLE")
    audit = _mapping(materials["audit_evidence"], "materials_policy.audit_evidence")
    _only(audit, ("provenance", "capability", "report"), "materials_policy.audit_evidence")
    _require(audit, ("provenance", "capability", "report"), "materials_policy.audit_evidence")
    for key in audit:
        _identity(audit[key], "materials_policy.audit_evidence." + key)

    hist = _section(root, "historical_v1", ("g6_status", "g6_wer_fraction", "g6_threshold_fraction", "g6_manifest", "attribution_evidence"))
    _string(hist["g6_status"], "historical_v1.g6_status", "FAIL")
    _number(hist["g6_wer_fraction"], "historical_v1.g6_wer_fraction")
    _number(hist["g6_threshold_fraction"], "historical_v1.g6_threshold_fraction")
    _manifest(hist["g6_manifest"], "historical_v1.g6_manifest")
    attribution = _mapping(hist["attribution_evidence"], "historical_v1.attribution_evidence")
    _only(attribution, ("failure_attribution", "real_scene_attribution"), "historical_v1.attribution_evidence")
    _require(attribution, ("failure_attribution", "real_scene_attribution"), "historical_v1.attribution_evidence")
    for key in attribution:
        _identity(attribution[key], "historical_v1.attribution_evidence." + key)

    office = _section(root, "office0", ("attribution_only", "forbidden_for_v2_acceptance", "manifest", "summary"))
    _bool(office["attribution_only"], "office0.attribution_only", True)
    _bool(office["forbidden_for_v2_acceptance"], "office0.forbidden_for_v2_acceptance", True)
    _identity(office["manifest"], "office0.manifest")
    _identity(office["summary"], "office0.summary")

    realistic = _section(root, "realistic_domain", (
        "scene_inventory", "scene_manifest", "speech_manifest", "selected_scene_count", "cases_per_scene",
        "expected_records", "selection_inputs_forbidden", "technical_scene_eligibility", "visibility_policy",
    ))
    _manifest(realistic["scene_inventory"], "realistic_domain.scene_inventory")
    _manifest(realistic["scene_manifest"], "realistic_domain.scene_manifest")
    _manifest(realistic["speech_manifest"], "realistic_domain.speech_manifest")
    _int(realistic["selected_scene_count"], "realistic_domain.selected_scene_count", 2)
    _int(realistic["cases_per_scene"], "realistic_domain.cases_per_scene", 4)
    _int(realistic["expected_records"], "realistic_domain.expected_records", 192)
    forbidden = ["RIR", "energy", "DRR", "ASR", "WER", "decoder_score", "Oracle"]
    if realistic["selection_inputs_forbidden"] != forbidden:
        raise A3V2ContractError("realistic_domain.selection_inputs_forbidden is not frozen")
    if realistic["technical_scene_eligibility"] != [
        "file_completeness", "scene_load", "pathfinder", "audio_sensor_construction",
        "native16", "binaural", "materials_off",
    ]:
        raise A3V2ContractError("realistic_domain.technical_scene_eligibility is not frozen")
    visibility_policy = _mapping(realistic["visibility_policy"], "realistic_domain.visibility_policy")
    _only(visibility_policy, ("required_status", "method", "navmesh_route_role", "coordinate_convention", "stage_transform_policy"), "realistic_domain.visibility_policy")
    _require(visibility_policy, ("required_status", "method", "navmesh_route_role", "coordinate_convention", "stage_transform_policy"), "realistic_domain.visibility_policy")
    if dict(visibility_policy) != {
        "required_status": "VERIFIED_GEOMETRIC_LOS_STATIC_MESH",
        "method": "STATIC_MESH_MOLLER_TRUMBORE_V1",
        "navmesh_route_role": "navigation_legality_only",
        "coordinate_convention": "replica_ply_xyz_to_habitat_xyz_x_z_neg_y",
        "stage_transform_policy": "stage_node_identity_verified",
    }:
        raise A3V2ContractError("realistic_domain.visibility_policy is not frozen")

    identity_sections = _v1_identity_sections(Path(repo_root) if repo_root else None)
    for key, value in identity_sections.items():
        if root[key] != value:
            raise A3V2ContractError("A3-v2 {} identity differs from frozen A3-v1".format(key))

    qualification = _section(root, "qualification", ("minimum_nonempty_fraction", "wer_max_fraction", "records_expected", "new_wer_allowed_after_freeze"))
    if abs(_number(qualification["minimum_nonempty_fraction"], "qualification.minimum_nonempty_fraction") - 0.80) > 1e-12:
        raise A3V2ContractError("qualification.minimum_nonempty_fraction must remain 0.80")
    if abs(_number(qualification["wer_max_fraction"], "qualification.wer_max_fraction") - 0.50) > 1e-12:
        raise A3V2ContractError("qualification.wer_max_fraction must remain 0.50")
    _int(qualification["records_expected"], "qualification.records_expected", 192)
    _bool(qualification["new_wer_allowed_after_freeze"], "qualification.new_wer_allowed_after_freeze", True)

    artifacts = _section(root, "artifacts", ("model_lock", "model_lock_sha256", "historical_v1_summary", "historical_v1_report"))
    _identity(artifacts["model_lock"], "artifacts.model_lock")
    _sha(artifacts["model_lock_sha256"], "artifacts.model_lock_sha256")
    _identity(artifacts["historical_v1_summary"], "artifacts.historical_v1_summary")
    _identity(artifacts["historical_v1_report"], "artifacts.historical_v1_report")
    return contract


def load_a3_v2_contract(path: str, require_frozen: bool = False, repo_root: Optional[str] = None) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise A3V2ContractError("A3-v2 contract root must be a mapping")
    validate_a3_v2_contract(value, require_frozen=require_frozen, repo_root=repo_root)
    return dict(value)
