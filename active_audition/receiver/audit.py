"""A1 runtime and receiver authority audit.

This module records what the existing live SoundSpaces/Habitat-Sim adapter
requests, what the created runtime exposes through public readback, and what
cannot be established without a later physics qualification.  It deliberately
does not render an RIR, run a channel-order gate, or infer HRTF geometry.
"""

import importlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from active_audition.config.loader import load_resolved_config as load_legacy_config
from active_audition.config.v1 import contract_sha256, load_resolved_config as load_contract
from active_audition.data.catalog import load_scene_registry, sha256_file
from active_audition.data.storage import DatasetStorage


RUNTIME_LOCK_SCHEMA_VERSION = "active-asr-a1-runtime-lock-v1"
RECEIVER_AUDIT_SCHEMA_VERSION = "active-asr-a1-receiver-audit-v1"
UNKNOWN = "UNKNOWN"
NOT_EXPOSED = "NOT_EXPOSED"
NOT_REQUESTED = "NOT_REQUESTED"
KNOWN = "KNOWN"

_ACOUSTICS_FIELDS = (
    "directRayCount",
    "sourceRayCount",
    "indirectRayCount",
    "indirectRayDepth",
    "sourceRayDepth",
    "maxIRLength",
    "directSHOrder",
    "indirectSHOrder",
    "maxDiffractionOrder",
    "frequencyBands",
    "threadCount",
    "unitScale",
    "temporalCoherence",
    "direct",
    "indirect",
    "transmission",
    "diffraction",
    "globalVolume",
    "meshSimplification",
)


class RuntimeAuditError(RuntimeError):
    """Raised when an A1 audit cannot produce authoritative evidence."""


def _field(status: str, value: Any = None, source: str = "") -> Dict[str, Any]:
    result = {"status": status, "value": value}
    if source:
        result["source"] = source
    return result


def _known(value: Any, source: str) -> Dict[str, Any]:
    return _field(KNOWN, value, source)


def _not_exposed(source: str) -> Dict[str, Any]:
    return _field(NOT_EXPOSED, None, source)


def _not_requested(source: str) -> Dict[str, Any]:
    return _field(NOT_REQUESTED, None, source)


def canonical_json(value: Any) -> str:
    """Serialize an audit value deterministically without whitespace."""

    return json.dumps(_canonical_value(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise RuntimeAuditError("audit value must not contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    return value


def _pretty_json(value: Any) -> str:
    return json.dumps(_canonical_value(value), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"


def _enum_label(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _float_list(value: Any) -> List[float]:
    return [float(item) for item in value]


def _quaternion_values(value: Any) -> Dict[str, float]:
    fields = {}
    for name in ("w", "x", "y", "z"):
        component = getattr(value, name, None)
        if component is None:
            scalar = getattr(value, "scalar", None)
            vector = getattr(value, "vector", None)
            if scalar is None or vector is None:
                return {}
            components = [float(item) for item in vector]
            if len(components) != 3:
                return {}
            return {"w": float(scalar), "x": components[0], "y": components[1], "z": components[2]}
        fields[name] = float(component)
    return fields


def _safe_version(distribution_names: Sequence[str]) -> Optional[str]:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _module_record(import_name: str, module: Any = None, distributions: Sequence[str] = ()) -> Dict[str, Any]:
    try:
        module_spec = importlib.util.find_spec(import_name)
    except (ImportError, ValueError):
        module_spec = None
    path = None if module_spec is None else module_spec.origin
    result: Dict[str, Any] = {
        "import_name": import_name,
        "module_path": str(Path(path).resolve()) if path and path not in ("built-in", "frozen") else path,
        "module_exists": module_spec is not None,
        "distribution_version": _safe_version(distributions),
    }
    if module is not None:
        result["module_version"] = getattr(module, "__version__", None)
        result["loaded"] = True
    else:
        result["module_version"] = None
        result["loaded"] = False
    if result["module_path"] and Path(str(result["module_path"])).is_file():
        result["module_sha256"] = sha256_file(Path(str(result["module_path"])))
    else:
        result["module_sha256"] = None
    return result


def _binary_record(component: str, path: Optional[Path], version: Optional[str] = None) -> Dict[str, Any]:
    exists = path is not None and path.is_file()
    return {
        "component": component,
        "path": str(path.resolve()) if path is not None else None,
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "version": version if version is not None else NOT_EXPOSED,
        "version_source": "binary does not expose a stable public version field" if version is None else "runtime",
    }


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_fingerprint(repo_root: Path) -> Dict[str, Any]:
    # This import order is part of the A1 runtime contract.  Do not move the
    # habitat_sim import above quaternion.
    import quaternion

    import habitat_sim
    import habitat_sim._ext.habitat_sim_bindings as habitat_bindings

    binding_path = Path(habitat_bindings.__file__).resolve()
    rlr_path = binding_path.with_name("libRLRAudioPropagation.so")
    modules = [
        _module_record("habitat_sim", habitat_sim, ("habitat-sim", "habitat_sim")),
        _module_record("habitat_sim._ext.habitat_sim_bindings", habitat_bindings, ()),
        _module_record("quaternion", quaternion, ("numpy-quaternion",)),
    ]
    dependency_modules = []
    for import_name, distributions in (
        ("numpy", ("numpy",)),
        ("scipy", ("scipy",)),
        ("yaml", ("PyYAML", "pyyaml")),
    ):
        try:
            module = importlib.import_module(import_name)
        except ImportError:
            module = None
        dependency_modules.append(_module_record(import_name, module, distributions))

    rlr_module_records = []
    for import_name in ("RLRAudioPropagation", "rlr_audio_propagation"):
        rlr_module_records.append(_module_record(import_name, None, ()))

    return {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "git_commit": _git_commit(repo_root),
        "packages": modules,
        "necessary_dependencies": dependency_modules,
        "rlr_audio_propagation_modules": rlr_module_records,
        "binaries": [
            _binary_record("habitat_sim_bindings", binding_path, getattr(habitat_sim, "__version__", None)),
            _binary_record("RLRAudioPropagation", rlr_path),
        ],
    }


def _resource_record(role: str, path: Optional[Path], runtime_evidence: Mapping[str, Any]) -> Dict[str, Any]:
    exists = path is not None and path.is_file()
    return {
        "role": role,
        "path": str(path.resolve()) if path is not None else None,
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "runtime_evidence": dict(runtime_evidence),
    }


def _scene_provenance(runtime_config: Mapping[str, Any], scene_id: str, context: Any) -> Dict[str, Any]:
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    registry_path = Path(runtime_config["registries"]["scenes_path"])
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    registry_path = registry_path.resolve()
    try:
        with registry_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        raw_entry = raw.get("scenes", {}).get(scene_id, {})
    except (OSError, yaml.YAMLError, AttributeError):
        raw_entry = {}
    scenes = load_scene_registry(str(registry_path), str(repo_root))
    scene = scenes[scene_id]
    path_for = lambda key: Path(scene[key]) if scene.get(key) else None
    pathfinder_loaded = bool(context.pathfinder.is_loaded)
    semantic_loaded = context.simulator.semantic_scene is not None
    sim_cfg = getattr(getattr(context.simulator, "config", None), "sim_cfg", None)
    loaded_scene_path = getattr(sim_cfg, "scene_id", None)
    scene_asset_evidence = {
        "status": KNOWN if loaded_scene_path else NOT_EXPOSED,
        "loaded_path": str(Path(loaded_scene_path).resolve()) if loaded_scene_path else None,
        "source": "SimulatorConfiguration.scene_id",
    }
    resources = {
        "scene_asset": _resource_record("scene_asset", path_for("scene_asset"), scene_asset_evidence),
        "navmesh": _resource_record(
            "navmesh",
            path_for("navmesh"),
            {
                "status": KNOWN if pathfinder_loaded else UNKNOWN,
                "loaded": pathfinder_loaded,
                "path": "scene registry path supplied to create_scene_simulator; navmesh path getter is not public",
            },
        ),
        "semantic_info": _resource_record(
            "semantic_info",
            path_for("semantic_info"),
            {
                "status": KNOWN if semantic_loaded else UNKNOWN,
                "loaded": semantic_loaded,
                "path": "scene registry path used by Habitat semantic-scene loading; runtime path getter is not public",
            },
        ),
        "stage_config": _resource_record(
            "stage_config",
            path_for("stage_config"),
            {
                "status": NOT_EXPOSED,
                "loaded": None,
                "path": "registry provenance only; current create_scene_simulator passes scene_asset directly",
            },
        ),
    }
    return {
        "scene_id": scene_id,
        "registry": _resource_record(
            "scene_registry",
            registry_path,
            {"status": KNOWN, "entry_present": bool(raw_entry), "source": "registries/scenes.yaml"},
        ),
        "dataset": scene["dataset"],
        "materials_mode": scene["materials_mode"],
        "resources": resources,
    }


def _compare(requested: Mapping[str, Any], effective: Mapping[str, Any]) -> str:
    if requested.get("status") == NOT_REQUESTED:
        return NOT_REQUESTED
    if effective.get("status") != KNOWN:
        return str(effective.get("status", UNKNOWN))
    if requested.get("status") != KNOWN:
        return str(requested.get("status", UNKNOWN))
    return "PASS" if _comparison_equal(requested.get("value"), effective.get("value")) else "FAIL"


def _comparison_equal(first: Any, second: Any) -> bool:
    if isinstance(first, str) and isinstance(second, str):
        return first.casefold() == second.casefold()
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(_comparison_equal(left, right) for left, right in zip(first, second))
    return first == second


def _receiver_observation(contract: Mapping[str, Any], runtime_config: Mapping[str, Any], context: Any) -> Dict[str, Any]:
    sensor = context.audio_sensor
    spec = sensor.specification()
    acoustics = spec.acousticsConfig
    channel_layout = spec.channelLayout
    node = sensor.node
    requested_position = list(runtime_config["listener"]["sensor_offset_m"])
    requested_sample_rate = int(runtime_config["acoustics"]["sample_rate_hz"])
    requested_materials = bool(runtime_config["acoustics"]["materials_enabled"])
    effective_position = _float_list(spec.position)
    effective_orientation = _float_list(spec.orientation)
    effective_acoustics = {field: getattr(acoustics, field) for field in _ACOUSTICS_FIELDS}
    sample_rate = float(acoustics.sampleRate)
    effective_acoustics["sampleRate"] = int(sample_rate) if sample_rate.is_integer() else sample_rate
    effective_acoustics = {key: value for key, value in effective_acoustics.items()}
    requested_channel_layout = "Binaural"
    effective_channel_layout = _enum_label(channel_layout.type)

    requested = {
        "sensor_type": _known("AUDIO", "AudioSensorSpec -> SensorType.AUDIO in create_scene_simulator"),
        "uuid": _known("audio_sensor", "active_audition.scene.simulator.create_scene_simulator"),
        "channel_layout": _known(requested_channel_layout, "active_audition.scene.simulator.create_scene_simulator"),
        "channel_count": _known(2, "active_audition.scene.simulator.create_scene_simulator"),
        "sample_rate_hz": _known(requested_sample_rate, "runtime V0 config -> create_scene_simulator"),
        "contract_sample_rate_hz": _known(contract["audio"]["render_sample_rate_hz"], "A0 contract audio.render_sample_rate_hz"),
        "contract_channel_layout": _known(contract["audio"]["channel_layout"], "A0 contract audio.channel_layout"),
        "contract_channel_count": _known(contract["audio"]["binaural_shape"][1], "A0 contract audio.binaural_shape"),
        "contract_sensor_offset_m": _known(contract["coordinates"]["listener_sensor_offset_m"], "A0 contract coordinates.listener_sensor_offset_m"),
        "position_m": _known(requested_position, "runtime V0 config -> create_scene_simulator"),
        "orientation": _not_requested("create_scene_simulator does not set AudioSensorSpec.orientation"),
        "channel_order": _known(contract["audio"]["channel_order"], "A0 contract audio.channel_order"),
        "materials_enabled": _known(requested_materials, "runtime V0 config; code also hardcodes enableMaterials=False"),
        "ray_parameters": _not_requested("create_scene_simulator sets no acousticsConfig ray parameters"),
    }
    effective = {
        "sensor_type": _known(_enum_label(spec.sensor_type), "AudioSensor.specification().sensor_type"),
        "uuid": _known(str(spec.uuid), "AudioSensor.specification().uuid"),
        "channel_layout": _known(effective_channel_layout, "AudioSensor.specification().channelLayout.type"),
        "channel_count": _known(int(channel_layout.channelCount), "AudioSensor.specification().channelLayout.channelCount"),
        "sample_rate_hz": _known(effective_acoustics.pop("sampleRate"), "AudioSensor.specification().acousticsConfig.sampleRate"),
        "position_m": _known(effective_position, "AudioSensor.specification().position"),
        "orientation": _known(effective_orientation, "AudioSensor.specification().orientation"),
        "channel_order": _not_exposed("AudioSensor public specification exposes binaural layout/count but no L/R order getter"),
        "materials_enabled": _known(bool(spec.enableMaterials), "AudioSensor.specification().enableMaterials"),
        "ray_parameters": _known(effective_acoustics, "AudioSensor.specification().acousticsConfig"),
    }
    comparisons = {
        key: _compare(requested[key], effective[key])
        for key in ("sensor_type", "uuid", "channel_layout", "channel_count", "sample_rate_hz", "position_m", "orientation", "channel_order", "materials_enabled", "ray_parameters")
    }
    comparisons.update(
        {
            "contract_sample_rate_hz": _compare(requested["contract_sample_rate_hz"], effective["sample_rate_hz"]),
            "contract_channel_layout": _compare(requested["contract_channel_layout"], effective["channel_layout"]),
            "contract_channel_count": _compare(requested["contract_channel_count"], effective["channel_count"]),
            "contract_sensor_offset_m": _compare(requested["contract_sensor_offset_m"], effective["position_m"]),
        }
    )

    requested_coordinate = contract["coordinates"]
    coordinate = {
        "requested": _known(requested_coordinate, "A0 contract coordinates"),
        "effective": _not_exposed("Habitat-Sim runtime object exposes transforms but no coordinate-convention authority getter"),
        "comparison": NOT_EXPOSED,
    }
    receiver_geometry = {
        "hrtf": {
            "requested": _not_requested("A0 does not freeze an HRTF asset or revision"),
            "effective": _not_exposed("AudioSensor exposes setListenerHRTF but no public HRTF getter or loaded-source readback"),
            "comparison": NOT_EXPOSED,
        },
        "ear_spacing_m": {
            "requested": _not_requested("A0 does not freeze physical ear spacing"),
            "effective": _not_exposed("AudioSensorSpec/AudioSensor public API exposes no ear-spacing or receiver-geometry getter"),
            "comparison": NOT_EXPOSED,
        },
        "receiver_geometry": {
            "requested": _not_requested("A0 does not freeze physical receiver geometry"),
            "effective": _not_exposed("AudioSensorSpec/AudioSensor public API exposes no physical receiver geometry getter"),
            "comparison": NOT_EXPOSED,
        },
    }
    node_rotation = _quaternion_values(node.rotation)
    return {
        "audio_sensor_type": type(sensor).__module__ + "." + type(sensor).__name__,
        "requested": requested,
        "effective": effective,
        "comparisons": comparisons,
        "sensor_transform": {
            "spec_position_m": _known(effective_position, "AudioSensor.specification().position"),
            "spec_orientation": _known(effective_orientation, "AudioSensor.specification().orientation"),
            "node_translation_m": _known(_float_list(node.translation), "AudioSensor.node.translation"),
            "node_rotation_quaternion_wxyz": (
                _known(node_rotation, "AudioSensor.node.rotation")
                if node_rotation
                else _not_exposed("runtime quaternion has no public component getter")
            ),
        },
        "acoustics_config_effective": effective["ray_parameters"],
        "coordinate_convention": coordinate,
        "receiver_geometry": receiver_geometry,
    }


def _status_paths(value: Any, path: str = "root") -> Tuple[List[str], List[str], List[str]]:
    unknown: List[str] = []
    not_exposed: List[str] = []
    not_requested: List[str] = []
    if isinstance(value, Mapping):
        status = value.get("status")
        if status == UNKNOWN:
            unknown.append(path)
        elif status == NOT_EXPOSED:
            not_exposed.append(path)
        elif status == NOT_REQUESTED:
            not_requested.append(path)
        for key, child in value.items():
            if key not in ("status", "source"):
                child_unknown, child_not_exposed, child_not_requested = _status_paths(child, "{}.{}".format(path, key))
                unknown.extend(child_unknown)
                not_exposed.extend(child_not_exposed)
                not_requested.extend(child_not_requested)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_unknown, child_not_exposed, child_not_requested = _status_paths(child, "{}[{}]".format(path, index))
            unknown.extend(child_unknown)
            not_exposed.extend(child_not_exposed)
            not_requested.extend(child_not_requested)
    return unknown, not_exposed, not_requested


def _audit_status(receiver: Mapping[str, Any]) -> Tuple[str, List[str], List[str], List[str]]:
    failures = [key for key, value in receiver["comparisons"].items() if value == "FAIL"]
    unknown, not_exposed, not_requested = _status_paths(receiver)
    if failures:
        return "BLOCKED", failures, unknown, not_exposed + not_requested
    if unknown or not_exposed:
        return "PASS_WITH_UNKNOWN", failures, unknown, not_exposed + not_requested
    return "PASS", failures, unknown, not_exposed + not_requested


def validate_runtime_lock(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the stable top-level shape of ``runtime.lock.json``."""

    required = {"schema_version", "gate", "contract", "repository", "inputs", "runtime_fingerprint"}
    if set(document) != required:
        raise RuntimeAuditError("runtime.lock.json schema keys do not match A1 contract")
    if document["schema_version"] != RUNTIME_LOCK_SCHEMA_VERSION or document["gate"] != "A1":
        raise RuntimeAuditError("runtime.lock.json has an invalid A1 schema or gate")
    if not isinstance(document["runtime_fingerprint"], Mapping):
        raise RuntimeAuditError("runtime.lock.json runtime_fingerprint must be a mapping")
    return document


def validate_receiver_audit(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the stable top-level shape and explicit A2 boundary markers."""

    required = {"schema_version", "gate", "status", "contract_sha256", "scene", "receiver", "findings", "artifacts"}
    if set(document) != required:
        raise RuntimeAuditError("receiver_audit.json schema keys do not match A1 contract")
    if document["schema_version"] != RECEIVER_AUDIT_SCHEMA_VERSION or document["gate"] != "A1":
        raise RuntimeAuditError("receiver_audit.json has an invalid A1 schema or gate")
    if document["status"] not in ("PASS", "PASS_WITH_UNKNOWN", "BLOCKED"):
        raise RuntimeAuditError("receiver_audit.json has an invalid audit status")
    findings = document["findings"]
    if findings.get("a2_not_run") is not True or not findings.get("a2_boundary"):
        raise RuntimeAuditError("receiver_audit.json must explicitly preserve the A1/A2 boundary")
    return document


def _load_scene_entry_for_audit(runtime_config: Mapping[str, Any], scene_id: str) -> None:
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    scenes = load_scene_registry(runtime_config["registries"]["scenes_path"], str(repo_root))
    if scene_id not in scenes:
        raise RuntimeAuditError("scene is not registered: {}".format(scene_id))


def run_runtime_audit(
    contract_path: str,
    output_dir: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
    scene_id: str = "replica.office_0",
) -> Dict[str, Any]:
    """Run the A1 audit and write the three deterministic artifacts."""

    contract = load_contract(contract_path)
    contract_digest = contract_sha256(contract)
    runtime_config = load_legacy_config(runtime_config_path)
    repo_root = Path(runtime_config["_repo_root"]).resolve()
    _load_scene_entry_for_audit(runtime_config, scene_id)

    # Importing the existing simulator here keeps the A1 module import-safe for
    # unit tests and reuses the V0/V0.5 live creation path unchanged.
    from active_audition.scene.simulator import create_scene_simulator

    with create_scene_simulator(runtime_config, scene_id=scene_id) as context:
        runtime_fingerprint = _runtime_fingerprint(repo_root)
        scene = _scene_provenance(runtime_config, scene_id, context)
        receiver = _receiver_observation(contract, runtime_config, context)

    runtime_config_file = Path(runtime_config["_config_path"]).resolve()
    registry_file = Path(runtime_config["registries"]["scenes_path"])
    if not registry_file.is_absolute():
        registry_file = repo_root / registry_file
    registry_file = registry_file.resolve()
    runtime_lock = {
        "schema_version": RUNTIME_LOCK_SCHEMA_VERSION,
        "gate": "A1",
        "contract": {
            "schema_version": contract["contract"]["version"],
            "sha256": contract_digest,
        },
        "repository": {
            "root": str(repo_root),
            "git_commit": runtime_fingerprint["git_commit"],
        },
        "inputs": {
            "runtime_config_path": str(runtime_config_file),
            "runtime_config_sha256": sha256_file(runtime_config_file),
            "scene_id": scene_id,
            "scene_registry_path": str(registry_file),
            "scene_registry_sha256": sha256_file(registry_file),
        },
        "runtime_fingerprint": runtime_fingerprint,
    }
    validate_runtime_lock(runtime_lock)

    status, failures, unknown, not_exposed = _audit_status(receiver)
    receiver_audit = {
        "schema_version": RECEIVER_AUDIT_SCHEMA_VERSION,
        "gate": "A1",
        "status": status,
        "contract_sha256": contract_digest,
        "scene": scene,
        "receiver": receiver,
        "findings": {
            "requested_effective_failures": failures,
            "unknown": unknown,
            "not_exposed_or_not_requested": not_exposed,
            "a2_not_run": True,
            "a2_boundary": [
                "ITD/ILD",
                "direct-window",
                "mirror/yaw physics",
                "near/far qualification",
                "repeat-render qualification",
                "16/24 kHz A/B",
                "ray convergence",
                "run_channel_order_gate qualification",
            ],
        },
    }

    output_root = Path(output_dir).resolve()
    storage = DatasetStorage(str(output_root))
    runtime_lock_path = output_root / "runtime.lock.json"
    storage.atomic_write_text(runtime_lock_path, _pretty_json(runtime_lock))
    runtime_lock_digest = sha256_file(runtime_lock_path)
    receiver_audit["artifacts"] = {
        "runtime_lock": {"path": "runtime.lock.json", "sha256": runtime_lock_digest},
    }
    validate_receiver_audit(receiver_audit)
    receiver_audit_path = output_root / "receiver_audit.json"
    storage.atomic_write_text(receiver_audit_path, _pretty_json(receiver_audit))
    receiver_audit_digest = sha256_file(receiver_audit_path)
    markdown = _render_markdown(receiver_audit, runtime_lock_digest, receiver_audit_digest)
    markdown_path = output_root / "receiver_audit.md"
    storage.atomic_write_text(markdown_path, markdown)
    markdown_digest = sha256_file(markdown_path)

    # The JSON is intentionally not rewritten after its own hash is known;
    # self-referential hashes are not stable.  The CLI result is the manifest
    # of exact emitted bytes, while receiver_audit.json carries the lock hash.
    return {
        "status": status,
        "gate": "A1",
        "contract_sha256": contract_digest,
        "output_dir": str(output_root),
        "artifacts": {
            "runtime.lock.json": {"path": str(runtime_lock_path), "sha256": runtime_lock_digest},
            "receiver_audit.json": {"path": str(receiver_audit_path), "sha256": receiver_audit_digest},
            "receiver_audit.md": {"path": str(markdown_path), "sha256": markdown_digest},
        },
        "findings": receiver_audit["findings"],
    }


def _render_markdown(receiver_audit: Mapping[str, Any], runtime_lock_digest: str, receiver_audit_digest: str) -> str:
    receiver = receiver_audit["receiver"]
    comparisons = receiver["comparisons"]
    lines = [
        "# Active-ASR V1.1 A1 Runtime & Receiver Audit",
        "",
        "- Status: `{}`".format(receiver_audit["status"]),
        "- Gate: `A1`",
        "- Contract SHA-256: `{}`".format(receiver_audit["contract_sha256"]),
        "- Runtime lock SHA-256: `{}`".format(runtime_lock_digest),
        "- Receiver audit JSON SHA-256: `{}`".format(receiver_audit_digest),
        "",
        "## Artifacts",
        "",
        "| File | SHA-256 |",
        "| --- | --- |",
        "| `runtime.lock.json` | `{}` |".format(runtime_lock_digest),
        "| `receiver_audit.json` | `{}` |".format(receiver_audit_digest),
        "| `receiver_audit.md` | computed from emitted bytes by CLI |",
        "",
        "## Requested vs effective",
        "",
        "| Field | Requested status/value | Effective status/value | Comparison |",
        "| --- | --- | --- | --- |",
    ]
    effective_key_for = {
        "contract_sample_rate_hz": "sample_rate_hz",
        "contract_channel_layout": "channel_layout",
        "contract_channel_count": "channel_count",
        "contract_sensor_offset_m": "position_m",
    }
    for key in sorted(comparisons):
        requested = receiver["requested"][key]
        effective = receiver["effective"][effective_key_for.get(key, key)]
        lines.append(
            "| `{}` | `{}` / `{}` | `{}` / `{}` | `{}` |".format(
                key,
                requested["status"],
                json.dumps(requested.get("value"), ensure_ascii=False, sort_keys=True),
                effective["status"],
                json.dumps(effective.get("value"), ensure_ascii=False, sort_keys=True),
                comparisons[key],
            )
        )
    lines.extend(
        [
            "",
            "## Receiver geometry and HRTF authority",
            "",
            "Physical HRTF source, ear spacing, and receiver geometry are not inferred. Each unavailable field is recorded as `NOT_EXPOSED`; these are not A2 qualification results.",
            "",
            "```json",
            json.dumps(receiver["receiver_geometry"], ensure_ascii=False, sort_keys=True, indent=2),
            "```",
            "",
            "## Scene/resource provenance",
            "",
            "```json",
            json.dumps(receiver_audit["scene"], ensure_ascii=False, sort_keys=True, indent=2),
            "```",
            "",
            "## A2 boundary",
            "",
            "No RIR render, channel-order gate, ITD/ILD, direct-window, mirror/yaw, near/far, repeat-render, sample-rate A/B, or ray-convergence qualification was run.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "NOT_EXPOSED",
    "NOT_REQUESTED",
    "RECEIVER_AUDIT_SCHEMA_VERSION",
    "RUNTIME_LOCK_SCHEMA_VERSION",
    "RuntimeAuditError",
    "canonical_json",
    "run_runtime_audit",
    "validate_receiver_audit",
    "validate_runtime_lock",
]
