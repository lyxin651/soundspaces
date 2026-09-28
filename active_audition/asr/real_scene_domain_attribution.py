"""A3-G6 Replica domain attribution and materials capability diagnostics.

This module is deliberately downstream of the frozen A3 instrument.  It has
two separate pre-ASR phases (materials audit and geometry-only manifest
freeze) and a later diagnostic phase.  The latter never changes a frozen
contract, RIR, utterance set, model, decoder, or production frontend.
"""

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.asr.contract import asr_contract_sha256, load_asr_contract
from active_audition.asr.frontends import apply_frontend
from active_audition.asr.g6_failure_attribution import (
    _array_sha,
    _make_rir_variant,
    _paired_deltas,
    _rir_diagnostic,
    _run_asr,
    _stats,
)
from active_audition.asr.qualification import canonical_json_bytes, file_sha256
from active_audition.asr.run_qualification import A3RunError, _decode_speech, _load_manifest
from active_audition.asr.speechbrain_adapter import SpeechBrainASRAdapter
from active_audition.config.loader import load_resolved_config
from active_audition.data.catalog import load_scene_registry
from active_audition.data.storage import DatasetStorage
from active_audition.receiver.audit import _runtime_fingerprint
from active_audition.receiver.qualification import load_metric_contract, metric_contract_sha256
from active_audition.scene.pose import listener_sensor_position, relative_azimuth_deg
from active_audition.types import ListenerPose


MATERIAL_AUDIT_SCHEMA_VERSION = "active-asr-a3-real-scene-materials-audit-v1"
DOMAIN_MANIFEST_SCHEMA_VERSION = "active-asr-a3-replica-office0-domain-manifest-v1"
DOMAIN_OUTPUT_SCHEMA_VERSION = "active-asr-a3-replica-office0-domain-attribution-v1"
EXPECTED_A3_CONTRACT_SHA = "b972d3ca2354ead8a10d2954a60602896ebbc5204adb7e8692c22f2b340497e2"
EXPECTED_A0_SHA = "d731393cda3ddb29f0bdf58249f104da59f29d012b976eeb2de1f160e1df8107"
EXPECTED_A2_V3_SHA = "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c"
SCENE_ID = "replica.office_0"
SAMPLE_RATE_HZ = 16000
FRONTENDS = ("mean_lr", "fixed_L", "fixed_R")
CASE_IDS = (
    "R1_near_front_like",
    "R2_moderate_front_like",
    "R3_far_front_like",
    "R4_moderate_off_axis",
    "R5_far_off_axis",
    "R6_distinct_los",
)
MATERIALS_NOT_AVAILABLE = "MATERIALS_ON_NOT_AVAILABLE"


class RealSceneAttributionError(A3RunError):
    """Raised when immutable A3 inputs or diagnostic provenance is invalid."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> Dict[str, str]:
    DatasetStorage(str(path.parent)).atomic_write_bytes(path, canonical_json_bytes(value))
    return {"path": str(path), "sha256": file_sha256(path)}


def _write_text(path: Path, text: str) -> Dict[str, str]:
    DatasetStorage(str(path.parent)).atomic_write_text(path, text)
    return {"path": str(path), "sha256": file_sha256(path)}


def _relative_or_absolute(repo: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _resource(repo: Path, path_value: Optional[str]) -> Dict[str, Any]:
    if not path_value:
        return {"status": "NOT_REQUESTED", "path": None, "exists": False, "sha256": None}
    path = Path(path_value)
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    exists = path.is_file()
    return {
        "status": "KNOWN",
        "path": _relative_or_absolute(repo, path),
        "exists": bool(exists),
        "sha256": file_sha256(path) if exists else None,
    }


def _load_raw_scene_entry(runtime_config: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    repo = Path(runtime_config["_repo_root"]).resolve()
    registry = Path(runtime_config["registries"]["scenes_path"])
    if not registry.is_absolute():
        registry = repo / registry
    raw = json.loads("{}")
    try:
        import yaml

        raw = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as error:
        raise RealSceneAttributionError("cannot read scene registry: {}".format(registry)) from error
    entry = raw.get("scenes", {}).get(scene_id)
    if not isinstance(entry, Mapping):
        raise RealSceneAttributionError("scene is not present in registry: {}".format(scene_id))
    return entry


def _effective_acoustics(context: Any) -> Dict[str, Any]:
    spec = context.audio_sensor.specification()
    acoustics = spec.acousticsConfig
    fields = (
        "directRayCount", "sourceRayCount", "indirectRayCount", "indirectRayDepth",
        "sourceRayDepth", "maxIRLength", "directSHOrder", "indirectSHOrder",
        "maxDiffractionOrder", "frequencyBands", "threadCount", "unitScale",
        "temporalCoherence", "direct", "indirect", "transmission", "diffraction",
        "globalVolume", "meshSimplification",
    )
    result = {name: getattr(acoustics, name) for name in fields}
    rate = float(acoustics.sampleRate)
    result["sampleRate"] = int(rate) if rate.is_integer() else rate
    return result


def run_materials_audit(
    output_dir: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
    scene_id: str = SCENE_ID,
) -> Dict[str, Any]:
    """Audit materials capability without rendering ASR or acoustic results."""

    import habitat_sim  # quaternion is imported by simulator before this module reaches runtime creation

    runtime_config = load_resolved_config(runtime_config_path)
    repo = Path(runtime_config["_repo_root"]).resolve()
    scenes = load_scene_registry(runtime_config["registries"]["scenes_path"], str(repo))
    if scene_id not in scenes:
        raise RealSceneAttributionError("scene is not registered: {}".format(scene_id))
    scene = scenes[scene_id]
    raw_entry = _load_raw_scene_entry(runtime_config, scene_id)
    runtime_config_file = Path(runtime_config["_config_path"]).resolve()
    registry_file = Path(runtime_config["registries"]["scenes_path"])
    if not registry_file.is_absolute():
        registry_file = repo / registry_file
    registry_file = registry_file.resolve()

    resource_provenance = {
        role: _resource(repo, scene.get(key))
        for role, key in (
            ("scene_asset", "scene_asset"),
            ("navmesh", "navmesh"),
            ("semantic_info", "semantic_info"),
            ("stage_config", "stage_config"),
        )
    }
    code_path = repo / "active_audition/scene/simulator.py"
    audit: Dict[str, Any] = {
        "schema_version": MATERIAL_AUDIT_SCHEMA_VERSION,
        "gate": "A3",
        "scope": "materials_capability_audit_only_no_ASR_no_WER_no_production_change",
        "scene_id": scene_id,
        "registry": {
            "path": _relative_or_absolute(repo, registry_file),
            "sha256": file_sha256(registry_file),
            "entry": dict(raw_entry),
        },
        "runtime_config": {
            "path": _relative_or_absolute(repo, runtime_config_file),
            "sha256": file_sha256(runtime_config_file),
        },
        "resources": resource_provenance,
        "code_provenance": {
            "simulator_path": _relative_or_absolute(repo, code_path),
            "simulator_sha256": file_sha256(code_path),
            "materials_assignment": "create_scene_simulator sets AudioSensorSpec.enableMaterials=False",
        },
        "requested": {
            "config_materials_enabled": {
                "status": "KNOWN",
                "value": bool(runtime_config["acoustics"]["materials_enabled"]),
                "source": "runtime config acoustics.materials_enabled",
            },
            "registry_materials_mode": {
                "status": "KNOWN",
                "value": scene["materials_mode"],
                "source": "scene registry materials_mode",
            },
        },
        "runtime_probe": {
            "habitat_sim_version": getattr(habitat_sim, "__version__", None),
            "audio_spec_enableMaterials_binding": {
                "status": "KNOWN",
                "value": hasattr(habitat_sim.AudioSensorSpec(), "enableMaterials"),
                "source": "habitat_sim.AudioSensorSpec public binding",
            },
        },
        "materials_capability": {
            "status": "BLOCKED",
            "materials_on_status": "NOT_AVAILABLE",
            "reason": (
                "The production simulator path hardcodes enableMaterials=False; the scene registry has no "
                "material configuration/mapping identity and the current public path has no qualification-only "
                "materials-on option. Binding presence alone is not evidence that Replica material mapping is consumed."
            ),
            "material_configuration": {
                "status": "NOT_AVAILABLE",
                "path": None,
                "sha256": None,
                "source": "no material config path in scene registry or runtime config",
            },
            "material_mapping": {
                "status": "NOT_AVAILABLE",
                "value": None,
                "source": "no public runtime/material mapping readback",
            },
            "effective_material_aware_render": {
                "status": "NOT_AVAILABLE",
                "value": None,
                "source": "materials-on runtime was not established",
            },
        },
        "runtime_fingerprint": _runtime_fingerprint(repo),
    }
    with create_scene_simulator(runtime_config, scene_id=scene_id) as context:
        spec = context.audio_sensor.specification()
        audit["effective"] = {
            "audio_sensor_enableMaterials": {
                "status": "KNOWN",
                "value": bool(spec.enableMaterials),
                "source": "AudioSensor.specification().enableMaterials",
            },
            "pathfinder": {
                "status": "KNOWN",
                "value": "LOADED" if context.pathfinder.is_loaded else "NOT_LOADED",
                "source": "PathFinder.is_loaded",
            },
            "semantic_scene": {
                "status": "KNOWN",
                "value": "LOADED" if context.simulator.semantic_scene is not None else "NOT_LOADED",
                "source": "Simulator.semantic_scene",
            },
            "acoustics": _effective_acoustics(context),
        }
    audit["comparison"] = {
        "config_materials_enabled_vs_effective": "PASS" if audit["requested"]["config_materials_enabled"]["value"] == audit["effective"]["audio_sensor_enableMaterials"]["value"] else "FAIL",
        "registry_materials_mode_vs_effective": "PASS" if scene["materials_mode"] == "off" and not audit["effective"]["audio_sensor_enableMaterials"]["value"] else "FAIL",
        "materials_on_capability": "BLOCKED",
    }
    audit["status"] = "BLOCKED"
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = _write_json(output / "real_scene_materials_audit.json", audit)
    lines = [
        "# Replica office_0 Materials Capability Audit",
        "",
        "- Status: **BLOCKED / MATERIALS_ON_NOT_AVAILABLE**",
        "- Scope: materials capability only; no ASR/WER/acoustic ranking was run.",
        "- Habitat-Sim: `{}`".format(audit["runtime_probe"]["habitat_sim_version"]),
        "- Requested config materials: `{}`; effective AudioSensorSpec materials: `{}`.".format(
            audit["requested"]["config_materials_enabled"]["value"],
            audit["effective"]["audio_sensor_enableMaterials"]["value"],
        ),
        "- Replica registry materials mode: `{}`.".format(scene["materials_mode"]),
        "- Material config/mapping: **NOT_AVAILABLE**; binding field existence is not treated as material consumption evidence.",
        "- Production simulator source SHA256: `{}`.".format(audit["code_provenance"]["simulator_sha256"]),
        "- JSON SHA256: `{}`.".format(identity["sha256"]),
        "",
        "Materials-ON diagnostics are intentionally not run. Reviewer decision is required for a production-materials runtime path.",
        "",
    ]
    report_identity = _write_text(output / "real_scene_materials_audit.md", "\n".join(lines))
    return {"status": audit["status"], "output_dir": str(output), "artifacts": {"json": identity, "report": report_identity}}


def _grid_navigable_points(context: Any, grid_step_m: float = 0.20) -> List[Tuple[float, float, float]]:
    """Enumerate snapped navmesh points in a fixed lexicographic grid."""

    raw = context.pathfinder
    lower, upper = raw.get_bounds()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    y = float(lower[1] + 0.25)
    values: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    xs = np.arange(lower[0] + 0.10, upper[0] - 0.10 + grid_step_m / 2.0, grid_step_m)
    zs = np.arange(lower[2] + 0.10, upper[2] - 0.10 + grid_step_m / 2.0, grid_step_m)
    for x in xs:
        for z in zs:
            point = raw.snap_point(np.asarray([x, y, z], dtype=np.float32))
            if point is None:
                continue
            array = np.asarray(point, dtype=np.float64)
            key = tuple(int(round(item * 1000.0)) for item in array)
            values[key] = tuple(float(item) for item in array)
    return [values[key] for key in sorted(values)]


def _category_score(category: str, distance: float, azimuth: float) -> Optional[float]:
    absolute_angle = abs(float(azimuth))
    targets = {
        "near_front_like": (1.25, 0.0, 0.75, 1.65, 25.0),
        "moderate_front_like": (2.20, 0.0, 1.65, 2.85, 25.0),
        "far_front_like": (3.35, 0.0, 2.85, 5.50, 25.0),
        "moderate_off_axis": (2.20, 55.0, 1.65, 2.85, 80.0),
        "far_off_axis": (3.35, 55.0, 2.85, 5.50, 80.0),
        "distinct_los": (2.60, 90.0, 1.00, 5.50, 180.0),
    }
    target_distance, target_angle, minimum, maximum, maximum_angle = targets[category]
    if not (minimum <= distance <= maximum and absolute_angle <= maximum_angle):
        return None
    if category.endswith("front_like") and absolute_angle > 25.0:
        return None
    if category.endswith("off_axis") and not (30.0 <= absolute_angle <= 80.0):
        return None
    if category == "distinct_los" and absolute_angle < 80.0:
        return None
    return abs(distance - target_distance) + 0.02 * abs(absolute_angle - target_angle)


def _select_geometry_cases(
    context: Any,
    runtime_config: Mapping[str, Any],
    scene: Mapping[str, Any],
    grid_step_m: float = 0.20,
) -> List[Dict[str, Any]]:
    from active_audition.navigation.pathfinder import PathFinderAdapter

    pathfinder = PathFinderAdapter(context.pathfinder)
    points = _grid_navigable_points(context, grid_step_m)
    if len(points) < 2:
        raise RealSceneAttributionError("office_0 navmesh grid did not produce enough points")
    offset = np.asarray(runtime_config["listener"]["sensor_offset_m"], dtype=np.float64)
    source_height = float(runtime_config["source"]["height_m"])
    candidates: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {name: [] for name in (
        "near_front_like", "moderate_front_like", "far_front_like", "moderate_off_axis", "far_off_axis", "distinct_los"
    )}
    # Geometry-only selection: no acoustic renderer, ASR adapter, WER, or RIR
    # values are imported or consulted in this loop.
    for listener_base in points:
        listener_sensor = np.asarray(listener_base, dtype=np.float64) + offset
        for source_base in points:
            if listener_base == source_base:
                continue
            source_world = np.asarray(source_base, dtype=np.float64) + np.asarray([0.0, source_height, 0.0])
            distance = float(np.linalg.norm(source_world - listener_sensor))
            azimuth = float(relative_azimuth_deg(source_world, listener_sensor, 0.0))
            route = pathfinder.shortest_path(listener_base, source_base)
            if not route.found or route.geodesic_distance_m is None:
                continue
            floor_distance = float(np.linalg.norm(np.asarray(source_base) - np.asarray(listener_base)))
            route_delta = abs(float(route.geodesic_distance_m) - floor_distance)
            if route_delta > 0.10:
                continue
            common = {
                "listener_base_position_world": list(listener_base),
                "listener_sensor_position_world": [float(item) for item in listener_sensor],
                "source_anchor_base_position_world": list(source_base),
                "source_position_world": [float(item) for item in source_world],
                "listener_yaw_deg": 0.0,
                "distance_m": distance,
                "relative_azimuth_deg": azimuth,
                "navmesh_geodesic_distance_m": float(route.geodesic_distance_m),
                "navmesh_floor_euclidean_distance_m": floor_distance,
                "navmesh_route_delta_m": route_delta,
                "los_status": "GEOMETRIC_NAVMESH_LOS",
                "source_anchor_navmesh_legal": bool(pathfinder.is_navigable(source_base)),
                "listener_base_navmesh_legal": bool(pathfinder.is_navigable(listener_base)),
            }
            for category in candidates:
                score = _category_score(category, distance, azimuth)
                if score is not None and common["source_anchor_navmesh_legal"] and common["listener_base_navmesh_legal"]:
                    candidates[category].append((float(score), common))
    selected: List[Dict[str, Any]] = []
    used = set()
    for index, category in enumerate(candidates):
        ordered = sorted(
            candidates[category],
            key=lambda item: (
                item[0],
                tuple(round(value, 6) for value in item[1]["listener_base_position_world"]),
                tuple(round(value, 6) for value in item[1]["source_anchor_base_position_world"]),
            ),
        )
        choice = None
        for score, item in ordered:
            identity = (tuple(item["listener_base_position_world"]), tuple(item["source_anchor_base_position_world"]))
            if identity not in used:
                choice = (score, item)
                used.add(identity)
                break
        if choice is None:
            raise RealSceneAttributionError("no geometry-only LOS pair satisfies category {}".format(category))
        score, item = choice
        item = dict(item)
        item["case_id"] = CASE_IDS[index]
        item["category"] = category
        item["geometry_selection_score"] = float(score)
        selected.append(item)
    return selected


def validate_domain_manifest(document: Mapping[str, Any], repo_root: Optional[str] = None) -> Mapping[str, Any]:
    required = {
        "schema_version", "gate", "purpose", "scene_id", "selection_policy", "scene_assets",
        "runtime_config", "runtime_geometry_probe", "geometry_cases", "asr_not_run_before_commit",
    }
    if set(document) != required:
        raise RealSceneAttributionError("domain manifest keys do not match strict schema")
    if document["schema_version"] != DOMAIN_MANIFEST_SCHEMA_VERSION or document["gate"] != "A3":
        raise RealSceneAttributionError("invalid domain manifest schema/gate")
    if document["scene_id"] != SCENE_ID or len(document["geometry_cases"]) != 6:
        raise RealSceneAttributionError("domain manifest must contain six office_0 cases")
    if document["asr_not_run_before_commit"] is not True:
        raise RealSceneAttributionError("manifest must record pre-ASR commit boundary")
    if set(item["case_id"] for item in document["geometry_cases"]) != set(CASE_IDS):
        raise RealSceneAttributionError("domain manifest case IDs are incomplete")
    for item in document["geometry_cases"]:
        keys = {
            "case_id", "category", "listener_base_position_world", "listener_sensor_position_world",
            "source_anchor_base_position_world", "source_position_world", "listener_yaw_deg",
            "distance_m", "relative_azimuth_deg", "navmesh_geodesic_distance_m",
            "navmesh_floor_euclidean_distance_m", "navmesh_route_delta_m", "los_status",
            "source_anchor_navmesh_legal", "listener_base_navmesh_legal", "geometry_selection_score",
        }
        if set(item) != keys:
            raise RealSceneAttributionError("unknown/missing geometry case field(s): {}".format(item.get("case_id")))
        if item["los_status"] != "GEOMETRIC_NAVMESH_LOS" or not item["source_anchor_navmesh_legal"] or not item["listener_base_navmesh_legal"]:
            raise RealSceneAttributionError("domain manifest contains non-LOS or illegal pair")
    if repo_root is not None:
        repo = Path(repo_root).resolve()
        for identity in document["scene_assets"].values():
            path = repo / identity["path"]
            if not path.is_file() or file_sha256(path) != identity["sha256"]:
                raise RealSceneAttributionError("domain manifest scene asset identity mismatch: {}".format(path))
        config = repo / document["runtime_config"]["path"]
        if not config.is_file() or file_sha256(config) != document["runtime_config"]["sha256"]:
            raise RealSceneAttributionError("domain manifest runtime config identity mismatch")
    return document


def load_domain_manifest(path: str, repo_root: Optional[str] = None) -> Mapping[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise RealSceneAttributionError("domain manifest missing: {}".format(manifest_path))
    return validate_domain_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), repo_root)


def freeze_domain_manifest(
    manifest_path: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
    scene_id: str = SCENE_ID,
) -> Dict[str, Any]:
    """Create the tracked geometry-only manifest.  No ASR code is touched."""

    runtime_config = load_resolved_config(runtime_config_path)
    repo = Path(runtime_config["_repo_root"]).resolve()
    scenes = load_scene_registry(runtime_config["registries"]["scenes_path"], str(repo))
    if scene_id != SCENE_ID or scene_id not in scenes:
        raise RealSceneAttributionError("only replica.office_0 is authorized for this diagnostic")
    scene = scenes[scene_id]
    registry_file = Path(runtime_config["registries"]["scenes_path"])
    if not registry_file.is_absolute():
        registry_file = repo / registry_file
    registry_file = registry_file.resolve()
    with create_scene_simulator(runtime_config, scene_id=scene_id, require_navmesh=True, load_semantic_mesh=True) as context:
        cases = _select_geometry_cases(context, runtime_config, scene)
        bounds = context.pathfinder.get_bounds()
        navmesh_loaded = bool(context.pathfinder.is_loaded)
        semantic_loaded = context.simulator.semantic_scene is not None
    manifest = {
        "schema_version": DOMAIN_MANIFEST_SCHEMA_VERSION,
        "gate": "A3",
        "purpose": "Replica office_0 geometry-only domain diagnostic; not G6 replacement and not WER-selected",
        "scene_id": scene_id,
        "selection_policy": {
            "method": "fixed_lexicographic_navmesh_grid_then_category_distance_angle_score",
            "grid_step_m": 0.20,
            "navmesh_route_delta_max_m": 0.10,
            "selection_inputs": ["scene geometry", "navmesh legality", "distance", "relative azimuth", "navmesh route"],
            "forbidden_inputs": ["RIR", "energy", "ASR", "WER", "noise", "Oracle"],
            "listener_yaw_deg": 0.0,
        },
        "scene_assets": {
            "scene_registry": {"path": _relative_or_absolute(repo, registry_file), "sha256": file_sha256(registry_file)},
            "scene_asset": _resource(repo, scene["scene_asset"]),
            "navmesh": _resource(repo, scene["navmesh"]),
            "semantic_info": _resource(repo, scene["semantic_info"]),
            "stage_config": _resource(repo, scene["stage_config"]),
        },
        "runtime_config": {
            "path": _relative_or_absolute(repo, Path(runtime_config["_config_path"])),
            "sha256": file_sha256(Path(runtime_config["_config_path"])),
        },
        "runtime_geometry_probe": {
            "pathfinder_loaded": navmesh_loaded,
            "semantic_scene_loaded": semantic_loaded,
            "bounds_world": [np.asarray(bounds[0], dtype=float).tolist(), np.asarray(bounds[1], dtype=float).tolist()],
            "sample_rate_hz": SAMPLE_RATE_HZ,
        },
        "geometry_cases": cases,
        "asr_not_run_before_commit": True,
    }
    validate_domain_manifest(manifest, str(repo))
    path = Path(manifest_path)
    if not path.is_absolute():
        path = repo / path
    identity = _write_json(path.resolve(), manifest)
    return {"status": "PASS", "manifest": identity, "cases": cases, "asr_not_run": True}


def _save_array(path: Path, value: np.ndarray) -> Dict[str, str]:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value, dtype=np.float32), allow_pickle=False)
    DatasetStorage(str(path.parent)).atomic_write_bytes(path, buffer.getvalue())
    return {"path": str(path), "sha256": file_sha256(path)}


def render_real_scene_rirs(
    contract_path: str,
    manifest_path: str,
    materials_audit_path: str,
    output_dir: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
) -> Dict[str, Any]:
    """Render only Replica materials-OFF RIRs in the legacy ``ss`` env.

    This phase must not import SpeechBrain.  The resulting lock is consumed by
    ``run_real_scene_domain_attribution`` in the independent ASR environment.
    """

    import quaternion  # noqa: F401  # import-order authority for Habitat-Sim
    from active_audition.acoustics.rir import render_native_rir
    from active_audition.receiver.audit import _runtime_fingerprint
    from active_audition.scene.simulator import create_scene_simulator

    repo = Path.cwd().resolve()
    output = Path(output_dir).resolve()
    _ensure_clean_output(output)
    contract = load_asr_contract(str(repo / contract_path) if not Path(contract_path).is_absolute() else contract_path, require_frozen=True)
    contract_sha = asr_contract_sha256(contract)
    if contract_sha != EXPECTED_A3_CONTRACT_SHA:
        raise RealSceneAttributionError("frozen A3 contract SHA mismatch")
    manifest = load_domain_manifest(str(repo / manifest_path) if not Path(manifest_path).is_absolute() else manifest_path, str(repo))
    audit_path = Path(materials_audit_path)
    if not audit_path.is_absolute():
        audit_path = repo / audit_path
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != MATERIAL_AUDIT_SCHEMA_VERSION or audit.get("status") != "BLOCKED":
        raise RealSceneAttributionError("materials audit status does not authorize materials-OFF render")
    runtime_config = load_resolved_config(runtime_config_path)
    runtime_fp = _runtime_fingerprint(repo)
    runtime_sha = _sha_bytes(canonical_json_bytes(runtime_fp))
    records = []
    with create_scene_simulator(runtime_config, scene_id=SCENE_ID, require_navmesh=True, load_semantic_mesh=True) as context:
        effective = _effective_acoustics(context)
        for case in manifest["geometry_cases"]:
            pose = ListenerPose(
                base_position_world=tuple(case["listener_base_position_world"]),
                sensor_position_world=tuple(case["listener_sensor_position_world"]),
                yaw_deg=float(case["listener_yaw_deg"]),
            )
            rir = np.asarray(render_native_rir(context, case["source_position_world"], pose), dtype=np.float32)
            if rir.ndim != 2 or rir.shape[1] != 2 or not np.isfinite(rir).all():
                raise RealSceneAttributionError("invalid Replica RIR for {}".format(case["case_id"]))
            filename = "{}__16000hz.npy".format(case["case_id"])
            identity = _save_array(output / filename, rir)
            records.append({
                "case_id": case["case_id"],
                "relative_path": filename,
                "file_sha256": identity["sha256"],
                "array_sha256": _array_sha(rir),
                "shape": list(rir.shape),
                "dtype": "float32",
                "channel_order": ["L", "R"],
                "effective_acoustics": effective,
                "case": case,
            })
    lock = {
        "schema_version": "active-asr-a3-replica-office0-rir-lock-v1",
        "gate": "A3",
        "purpose": "Replica office_0 materials-OFF G6 domain diagnostic RIR handoff",
        "materials": "off",
        "asr_not_run": True,
        "contract_sha256": contract_sha,
        "manifest": {"path": _relative_or_absolute(repo, Path(manifest_path)), "sha256": file_sha256(Path(manifest_path))},
        "materials_audit": {"path": _relative_or_absolute(repo, audit_path), "sha256": file_sha256(audit_path)},
        "runtime_config": {"path": _relative_or_absolute(repo, Path(runtime_config["_config_path"])), "sha256": file_sha256(Path(runtime_config["_config_path"]))},
        "runtime_fingerprint": runtime_fp,
        "runtime_sha256": runtime_sha,
        "records": sorted(records, key=lambda item: item["case_id"]),
    }
    lock_identity = _write_json(output / "replica_office0_materials_off_rir_lock.json", lock)
    return {"status": "PASS", "output_dir": str(output), "records": len(records), "rir_lock": lock_identity, "runtime_sha256": runtime_sha}


def _load_rir_lock(repo: Path, path: str, manifest_sha256: str, expected_case_ids: Sequence[str], contract_sha: str) -> Tuple[Mapping[str, Any], Dict[str, np.ndarray]]:
    lock_path = Path(path)
    if not lock_path.is_absolute():
        lock_path = repo / lock_path
    if not lock_path.is_file():
        raise RealSceneAttributionError("Replica RIR lock missing: {}".format(lock_path))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {"schema_version", "gate", "purpose", "materials", "asr_not_run", "contract_sha256", "manifest", "materials_audit", "runtime_config", "runtime_fingerprint", "runtime_sha256", "records"}
    if set(lock) != required or lock["schema_version"] != "active-asr-a3-replica-office0-rir-lock-v1":
        raise RealSceneAttributionError("Replica RIR lock schema mismatch")
    if lock["contract_sha256"] != contract_sha or lock["materials"] != "off" or lock["asr_not_run"] is not True:
        raise RealSceneAttributionError("Replica RIR lock contract/material state mismatch")
    if lock["manifest"]["sha256"] != manifest_sha256:
        raise RealSceneAttributionError("Replica RIR lock manifest hash mismatch")
    arrays: Dict[str, np.ndarray] = {}
    expected_ids = set(expected_case_ids)
    if {record["case_id"] for record in lock["records"]} != expected_ids:
        raise RealSceneAttributionError("Replica RIR lock case set mismatch")
    for record in lock["records"]:
        array_path = lock_path.parent / record["relative_path"]
        if not array_path.is_file() or file_sha256(array_path) != record["file_sha256"]:
            raise RealSceneAttributionError("Replica RIR file identity mismatch: {}".format(array_path))
        array = np.asarray(np.load(str(array_path), allow_pickle=False), dtype=np.float32)
        if list(array.shape) != record["shape"] or _array_sha(array) != record["array_sha256"]:
            raise RealSceneAttributionError("Replica RIR array identity mismatch: {}".format(array_path))
        arrays[record["case_id"]] = array
    return lock, arrays


def _simple_aggregate(value: Mapping[str, Any]) -> Dict[str, Any]:
    aggregate = value["aggregate"]
    return {key: aggregate[key] for key in ("S", "D", "I", "N", "WER", "CER")}


def _augment_rows(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], binaural: Sequence[np.ndarray], dry: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
    by_id = {str(record.get("speech", record)["utterance_id"]): (record, value, dry_value) for record, value, dry_value in zip(records, binaural, dry)}
    result = []
    for row in rows:
        record, value, dry_value = by_id[str(row["utterance_id"])]
        result_row = dict(row)
        result_row["dry_waveform"] = _stats(dry_value)
        result_row["binaural_waveform"] = {
            "shape": list(value.shape),
            "channel_order": ["L", "R"],
            "L": _stats(value[:, 0]),
            "R": _stats(value[:, 1]),
            "mean_lr": _stats(apply_frontend(value, "mean_lr")),
            "sha256": _array_sha(value),
        }
        result.append(result_row)
    return result


def _run_frontend_baselines(
    adapter: SpeechBrainASRAdapter,
    records: Sequence[Mapping[str, Any]],
    dry: Sequence[np.ndarray],
    binaural: Sequence[np.ndarray],
    case_id: str,
    rir_sha: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"production_frontend": "mean_lr", "frontends": {}}
    for frontend in FRONTENDS:
        waveforms = [apply_frontend(value, frontend) for value in binaural]
        asr = _run_asr(
            adapter,
            records,
            waveforms,
            frontend,
            [_stats(value) for value in waveforms],
            {"case_id": case_id, "rir_sha256": rir_sha, "normalization": "none", "convolution_time_axis": "full"},
        )
        asr["rows"] = _augment_rows(asr["rows"], records, binaural, dry)
        result["frontends"][frontend] = asr
    return result


def _ensure_clean_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise RealSceneAttributionError("refusing to overwrite non-empty output directory: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)


def run_real_scene_domain_attribution(
    contract_path: str,
    metric_contract_path: str,
    manifest_path: str,
    materials_audit_path: str,
    rir_lock_path: str,
    output_dir: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
) -> Dict[str, Any]:
    """Run Replica materials-OFF attribution after the manifest commit."""

    repo = Path.cwd().resolve()
    output = Path(output_dir).resolve()
    _ensure_clean_output(output)
    contract = load_asr_contract(str(repo / contract_path) if not Path(contract_path).is_absolute() else contract_path, require_frozen=True)
    contract_sha = asr_contract_sha256(contract)
    if contract_sha != EXPECTED_A3_CONTRACT_SHA:
        raise RealSceneAttributionError("frozen A3 contract SHA mismatch")
    if contract["parents"]["a0_contract_sha256"] != EXPECTED_A0_SHA or contract["parents"]["a2_oracle_v3_sha256"] != EXPECTED_A2_V3_SHA:
        raise RealSceneAttributionError("A0/A2 frozen parent identity mismatch")
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = repo / manifest_file
    manifest = load_domain_manifest(str(manifest_file), str(repo))
    audit_path = Path(materials_audit_path)
    if not audit_path.is_absolute():
        audit_path = repo / audit_path
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != MATERIAL_AUDIT_SCHEMA_VERSION or audit.get("status") != "BLOCKED":
        raise RealSceneAttributionError("materials audit identity/status does not authorize materials-OFF diagnostic")
    metric_contract = load_metric_contract(str(repo / metric_contract_path) if not Path(metric_contract_path).is_absolute() else metric_contract_path)
    metric_sha = metric_contract_sha256(metric_contract)
    clean_manifest = _load_manifest(repo, contract, "clean")
    sources = json.loads("{}")
    import yaml

    sources = yaml.safe_load((repo / contract["sources"]["config_path"]).read_text(encoding="utf-8"))
    source_root = repo / sources["librispeech"]["root"]
    dry = [_decode_speech(repo, source_root, record)[0] for record in clean_manifest["records"]]
    lock, rir_by_case = _load_rir_lock(
        repo,
        rir_lock_path,
        file_sha256(manifest_file),
        [case["case_id"] for case in manifest["geometry_cases"]],
        contract_sha,
    )
    runtime_fp = lock["runtime_fingerprint"]
    runtime_sha = lock["runtime_sha256"]
    effective_by_case = {record["case_id"]: record["effective_acoustics"] for record in lock["records"]}

    acoustic_cases: Dict[str, Any] = {}
    asr_cases: Dict[str, Any] = {}
    reverb_cases: Dict[str, Any] = {}
    adapter = SpeechBrainASRAdapter(contract)
    for case in manifest["geometry_cases"]:
        case_id = case["case_id"]
        rir = rir_by_case[case_id]
        rir_sha = _array_sha(rir)
        acoustic_cases[case_id] = {
            "case": case,
            "materials": "off",
            "effective_acoustics": effective_by_case[case_id],
            "runtime_sha256": runtime_sha,
            "resource_hashes": manifest["scene_assets"],
            "rir": _rir_diagnostic(rir, metric_contract),
        }
        binaural = [convolve_binaural(value, rir) for value in dry]
        asr_cases[case_id] = _run_frontend_baselines(adapter, clean_manifest["records"], dry, binaural, case_id, rir_sha)

        direct_window = acoustic_cases[case_id]["rir"]["direct_window"]
        if direct_window.get("applicability") != "APPLICABLE":
            raise RealSceneAttributionError("frozen direct window is not applicable for {}".format(case_id))
        start = int(direct_window["start_sample"])
        variants = (
            ("direct_only", start, int(direct_window["end_sample_exclusive"])),
            ("direct_plus_50ms", start, start + int(round(0.050 * SAMPLE_RATE_HZ))),
            ("direct_plus_100ms", start, start + int(round(0.100 * SAMPLE_RATE_HZ))),
            ("direct_plus_200ms", start, start + int(round(0.200 * SAMPLE_RATE_HZ))),
            ("full_rir", 0, rir.shape[0]),
        )
        variant_result: Dict[str, Any] = {"normalization": "none", "time_axis_preserved": True, "variants": {}}
        for name, variant_start, variant_end in variants:
            variant = rir if name == "full_rir" else _make_rir_variant(rir, variant_start, variant_end)
            waves = [apply_frontend(convolve_binaural(value, variant), "mean_lr") for value in dry]
            result = _run_asr(
                adapter,
                clean_manifest["records"],
                waves,
                "mean_lr",
                [_stats(value) for value in waves],
                {
                    "case_id": case_id,
                    "variant": name,
                    "rir_variant_start_sample": int(variant_start),
                    "rir_variant_end_sample_exclusive": min(int(variant_end), int(rir.shape[0])),
                    "variant_rir_sha256": _array_sha(variant),
                    "normalization": "none",
                },
            )
            variant_result["variants"][name] = result
        full_rows = variant_result["variants"]["full_rir"]["rows"]
        for item in variant_result["variants"].values():
            item["paired_deltas_to_full_rir"] = _paired_deltas(full_rows, item["rows"])
        reverb_cases[case_id] = {"direct_window": direct_window, **variant_result}

    # Tail intervention reuses already computed full-RIR convolutions only for
    # the waveform prefix; each crop is decoded independently.
    tail_cases: Dict[str, Any] = {}
    for case in manifest["geometry_cases"]:
        case_id = case["case_id"]
        rir = rir_by_case[case_id]
        variants: Dict[str, Any] = {}
        full_waves = [apply_frontend(convolve_binaural(value, rir), "mean_lr") for value in dry]
        options = (("through_dry_end", 0.0), ("dry_end_plus_0.25s", 0.25), ("dry_end_plus_0.5s", 0.5), ("dry_end_plus_1.0s", 1.0), ("full_convolution", None))
        for name, extra in options:
            cropped = []
            prefix_checks = []
            for source, full in zip(dry, full_waves):
                if extra is None:
                    value = full
                else:
                    value = full[: min(full.size, source.size + int(round(extra * SAMPLE_RATE_HZ)))]
                cropped.append(value)
                prefix_checks.append({
                    "full_speech_time_prefix_sha256": _array_sha(full[: source.size]),
                    "cropped_speech_time_prefix_sha256": _array_sha(value[: source.size]),
                    "speech_time_prefix_unchanged": bool(np.array_equal(full[: source.size], value[: source.size])),
                })
            result = _run_asr(
                adapter,
                clean_manifest["records"],
                cropped,
                "mean_lr",
                [_stats(value) for value in cropped],
                {"case_id": case_id, "variant": name, "extra_tail_sec": extra, "normalization": "none"},
            )
            result["prefix_checks"] = prefix_checks
            variants[name] = result
        full_rows = variants["full_convolution"]["rows"]
        for item in variants.values():
            item["paired_deltas_to_full_convolution"] = _paired_deltas(full_rows, item["rows"])
        tail_cases[case_id] = {"full_rir_convolution_preserved_before_crop": True, "variants": variants}

    input_identities = {
        "a3_contract": {"path": contract_path, "sha256": contract_sha},
        "metric_contract": {"path": metric_contract_path, "sha256": metric_sha},
        "clean_manifest": {"path": contract["qualification"]["manifests"]["clean"]["path"], "sha256": contract["qualification"]["manifests"]["clean"]["sha256"]},
        "domain_manifest": {"path": _relative_or_absolute(repo, Path(manifest_path)), "sha256": file_sha256(Path(manifest_path))},
        "materials_audit": {"path": _relative_or_absolute(repo, audit_path), "sha256": file_sha256(audit_path)},
        "runtime_config": {
            "path": _relative_or_absolute(repo, repo / runtime_config_path),
            "sha256": file_sha256(repo / runtime_config_path),
        },
        "scene_registry": {
            "path": lock["records"][0]["case"].get("scene_registry_path", "registries/scenes.yaml"),
            "sha256": file_sha256(repo / "registries/scenes.yaml"),
        },
        "rir_lock": {"path": _relative_or_absolute(repo, Path(rir_lock_path)), "sha256": file_sha256(Path(rir_lock_path))},
    }
    artifacts: Dict[str, Any] = {
        "replica_office0_materials_off_acoustics.json": {
            "schema_version": DOMAIN_OUTPUT_SCHEMA_VERSION,
            "materials": "off",
            "input_identities": input_identities,
            "runtime_fingerprint": runtime_fp,
            "cases": acoustic_cases,
        },
        "replica_office0_materials_off_asr.json": {
            "schema_version": DOMAIN_OUTPUT_SCHEMA_VERSION,
            "materials": "off",
            "production_frontend": "mean_lr",
            "sensitivity_frontends": ["fixed_L", "fixed_R"],
            "input_identities": input_identities,
            "cases": asr_cases,
        },
        "replica_office0_materials_off_reverb_intervention.json": {
            "schema_version": DOMAIN_OUTPUT_SCHEMA_VERSION,
            "materials": "off",
            "input_identities": input_identities,
            "cases": reverb_cases,
        },
        "replica_office0_materials_off_tail_intervention.json": {
            "schema_version": DOMAIN_OUTPUT_SCHEMA_VERSION,
            "materials": "off",
            "input_identities": input_identities,
            "cases": tail_cases,
        },
    }
    identities = {name: _write_json(output / name, value) for name, value in artifacts.items()}
    # The user-requested three materials-off outputs are retained as distinct
    # artifacts; tail diagnostics have a separate explicit artifact.
    original_dir = repo / "runs/active_asr_v1/a3_g6_failure_attribution_v1"
    original_paths = {
        name: original_dir / name
        for name in ("g6_acoustic_diagnostics.json", "g6_reverb_intervention.json", "g6_24utterance_replication.json", "g6_failure_attribution_summary.json")
    }
    original = {name: {"path": _relative_or_absolute(repo, path), "sha256": file_sha256(path)} for name, path in original_paths.items() if path.is_file()}
    original_acoustic = json.loads(original_paths["g6_acoustic_diagnostics.json"].read_text(encoding="utf-8"))
    original_reverb = json.loads(original_paths["g6_reverb_intervention.json"].read_text(encoding="utf-8"))
    original_replication = json.loads(original_paths["g6_24utterance_replication.json"].read_text(encoding="utf-8"))
    shoebox = {}
    for case in ("front_near", "front_far", "side_left_far"):
        old_acoustic = original_acoustic["cases"][case]
        old_rir = old_acoustic["channels"]
        old_reverb = original_reverb["cases"][case]
        old_rep = original_replication["cases"][case]["mean_lr"]
        shoebox[case] = {
            "acoustic": {
                "length_sec": old_acoustic["length_sec"],
                "direct_window": old_acoustic["direct_window"],
                "mean_drr_proxy_db": float(np.mean([old_rir[side]["drr_proxy_direct_over_late_db"] for side in ("L", "R") if old_rir[side]["drr_proxy_direct_over_late_db"] is not None])),
                "L_total_energy": old_rir["L"]["sum_square_energy"],
                "R_total_energy": old_rir["R"]["sum_square_energy"],
                "L_late_energy": old_rir["L"]["late_energy_after_50ms_from_direct_start"],
                "R_late_energy": old_rir["R"]["late_energy_after_50ms_from_direct_start"],
            },
            "asr_mean_lr_24": _simple_aggregate(old_rep),
            "reverb_mean_lr_24": {name: _simple_aggregate(value) for name, value in old_reverb["variants"].items()},
        }
    real = {}
    for case in manifest["geometry_cases"]:
        case_id = case["case_id"]
        rir_item = acoustic_cases[case_id]["rir"]
        drr = [rir_item["channels"][side]["drr_proxy_direct_over_late_db"] for side in ("L", "R") if rir_item["channels"][side]["drr_proxy_direct_over_late_db"] is not None]
        full = reverb_cases[case_id]["variants"]["full_rir"]
        direct = reverb_cases[case_id]["variants"]["direct_only"]
        real[case_id] = {
            "category": case["category"],
            "distance_m": case["distance_m"],
            "relative_azimuth_deg": case["relative_azimuth_deg"],
            "acoustic": {
                "mean_drr_proxy_db": float(np.mean(drr)) if drr else None,
                "length_sec": rir_item["length_sec"],
                "L_total_energy": rir_item["channels"]["L"]["sum_square_energy"],
                "R_total_energy": rir_item["channels"]["R"]["sum_square_energy"],
                "L_late_energy": rir_item["channels"]["L"]["late_energy_after_50ms_from_direct_start"],
                "R_late_energy": rir_item["channels"]["R"]["late_energy_after_50ms_from_direct_start"],
            },
            "asr_mean_lr_24": _simple_aggregate(asr_cases[case_id]["frontends"]["mean_lr"]),
            "reverb_mean_lr_24": {
                "full_rir": _simple_aggregate(full),
                "direct_only": _simple_aggregate(direct),
                "direct_plus_50ms": _simple_aggregate(reverb_cases[case_id]["variants"]["direct_plus_50ms"]),
                "direct_plus_100ms": _simple_aggregate(reverb_cases[case_id]["variants"]["direct_plus_100ms"]),
                "direct_plus_200ms": _simple_aggregate(reverb_cases[case_id]["variants"]["direct_plus_200ms"]),
            },
        }

    all_shoebox_drr = [shoebox[case]["acoustic"]["mean_drr_proxy_db"] for case in shoebox]
    all_real_drr = [real[case]["acoustic"]["mean_drr_proxy_db"] for case in real if real[case]["acoustic"]["mean_drr_proxy_db"] is not None]
    shoebox_far_wer = float(np.mean([shoebox[case]["asr_mean_lr_24"]["WER"] for case in ("front_far", "side_left_far")]))
    real_wer = float(np.mean([item["asr_mean_lr_24"]["WER"] for item in real.values()]))
    q1_reproduced = bool(all_real_drr and min(all_real_drr) <= max(all_shoebox_drr) + 3.0)
    q2_recovered = bool(real_wer <= 0.50)
    summary = {
        "schema_version": DOMAIN_OUTPUT_SCHEMA_VERSION,
        "status": "A3_OPEN_BLOCKED_UNCHANGED",
        "gate_change": False,
        "contract_sha256": contract_sha,
        "metric_contract_sha256": metric_sha,
        "materials_audit": {"status": audit["status"], "materials_on": "NOT_AVAILABLE"},
        "input_identities": input_identities,
        "historical_g6_artifacts_read_only": original,
        "shoebox_frozen_comparison": shoebox,
        "replica_materials_off": real,
        "questions": {
            "Q1_shoebox_far_extreme_reverberation_reproduced_in_replica": {
                "answer": "YES" if q1_reproduced else "NO_OR_NOT_REPRODUCED",
                "observable_evidence": {"shoebox_drr_proxy_db": all_shoebox_drr, "replica_drr_proxy_db": all_real_drr},
                "rule": "compare raw direct-over-late proxy distributions; no normalization",
            },
            "Q2_replica_geometry_alone_recovers_ASR": {
                "answer": "RECOVERED" if q2_recovered else "NOT_RECOVERED",
                "observable_evidence": {"replica_materials_off_mean_wer": real_wer, "per_case": {key: value["asr_mean_lr_24"] for key, value in real.items()}},
                "rule": "diagnostic comparison only; does not replace frozen G6 denominator",
            },
            "Q3_materials_ON_changes_difficulty": {
                "answer": "BLOCKED_MATERIALS_ON_NOT_AVAILABLE",
                "observable_evidence": {"materials_audit_status": audit["status"], "materials_on_artifacts": []},
            },
            "Q4_failure_attribution": {
                "answer": "MIXED" if (not q1_reproduced and not q2_recovered) else ("REALISTIC_REVERB_DOMAIN_MISMATCH" if q1_reproduced and not q2_recovered else "SHOEBOX_FIXTURE_MISMATCH"),
                "observable_evidence": {
                    "shoebox_far_mean_wer": shoebox_far_wer,
                    "replica_off_mean_wer": real_wer,
                    "shoebox_far_drr_proxy_db": all_shoebox_drr,
                    "replica_drr_proxy_db": all_real_drr,
                    "materials_effect": "NOT_ASSESSED",
                },
            },
        },
        "mechanism_attribution": {
            "LEVEL_EFFECT": {"status": "PRESERVED_PRIOR_EVIDENCE_NOT_REINVESTIGATED", "evidence": "prior frozen failure-attribution artifacts; no new level experiment"},
            "REVERBERATION_EFFECT": {"status": "OBSERVED_IN_NEW_REAL_SCENE_DIAGNOSTIC", "evidence": {case: value["reverb_mean_lr_24"] for case, value in real.items()}},
            "FRONTEND_EFFECT": {"status": "OBSERVED_SENSITIVITY_ONLY", "evidence": {case: {front: _simple_aggregate(asr_cases[case]["frontends"][front]) for front in FRONTENDS} for case in asr_cases}},
            "TAIL_LENGTH_EFFECT": {"status": "OBSERVED_IN_NEW_REAL_SCENE_DIAGNOSTIC", "evidence": {case: {name: _simple_aggregate(value) for name, value in item["variants"].items()} for case, item in tail_cases.items()}},
            "SAMPLE_SELECTION_EFFECT": {"status": "FIXED_24_UTTERANCE_MANIFEST_USED", "evidence": {"utterances": len(clean_manifest["records"]), "no_WER_selection": True}},
        },
        "artifacts": identities,
    }
    summary_identity = _write_json(output / "real_scene_domain_attribution_summary.json", summary)
    identities["real_scene_domain_attribution_summary.json"] = summary_identity
    report_lines = [
        "# A3-G6 Real-Scene Domain Attribution (diagnostic only)",
        "",
        "- Gate remains **A3 IMPLEMENTED / SERVER_RUN_FAIL / OPEN_BLOCKED**; A4 remains CLOSED.",
        "- No frozen contract, threshold, utterance, RIR, model, decoder, or production frontend was changed.",
        "- A3 contract SHA: `{}`; metric contract SHA: `{}`.".format(contract_sha, metric_sha),
        "- Materials capability: **BLOCKED / MATERIALS_ON_NOT_AVAILABLE**; no materials-ON diagnostic was run.",
        "",
        "## Materials audit",
        "",
        "The current simulator hardcodes `AudioSensorSpec.enableMaterials=False`; the registry contains no material mapping/config identity. Binding-field presence is not treated as proof that Replica materials are consumed.",
        "",
        "## Geometry-only manifest",
        "",
        "Six office_0 pairs were selected from deterministic navmesh/grid geometry only. ASR, WER, RIR, energy, noise, and Oracle values were forbidden selection inputs. The tracked manifest was required to exist before this ASR run.",
        "",
        "| case | category | distance (m) | relative azimuth (deg) | mean drr proxy (dB) | mean_lr WER | direct-only WER | full WER |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in manifest["geometry_cases"]:
        item = real[case["case_id"]]
        report_lines.append("| {} | {} | {:.3f} | {:.2f} | {} | {:.4f}% | {:.4f}% | {:.4f}% |".format(
            case["case_id"], case["category"], case["distance_m"], case["relative_azimuth_deg"],
            "N/A" if item["acoustic"]["mean_drr_proxy_db"] is None else "{:.3f}".format(item["acoustic"]["mean_drr_proxy_db"]),
            100.0 * item["reverb_mean_lr_24"]["full_rir"]["WER"],
            100.0 * item["reverb_mean_lr_24"]["direct_only"]["WER"],
            100.0 * item["reverb_mean_lr_24"]["full_rir"]["WER"],
        ))
    report_lines += [
        "",
        "## Attribution questions",
        "",
        "- Q1 shoebox far reverberation reproduced in Replica: **{}**.".format(summary["questions"]["Q1_shoebox_far_extreme_reverberation_reproduced_in_replica"]["answer"]),
        "- Q2 Replica geometry alone recovers ASR: **{}**.".format(summary["questions"]["Q2_replica_geometry_alone_recovers_ASR"]["answer"]),
        "- Q3 materials effect: **BLOCKED_MATERIALS_ON_NOT_AVAILABLE**.",
        "- Q4 current attribution: **{}** (materials effect unassessed).".format(summary["questions"]["Q4_failure_attribution"]["answer"]),
        "",
        "All per-utterance S/D/I/N/WER/CER rows, frontend sensitivity, full/direct/50/100/200 ms interventions, and tail-prefix checks are retained in the JSON artifacts. Historical G6/failure-attribution files are read-only and hash-bound.",
        "",
        "## Artifact identities",
    ]
    for name, identity in identities.items():
        report_lines.append("- `{}`: `{}`".format(name, identity["sha256"]))
    report_lines.append("")
    report_identity = _write_text(output / "real_scene_domain_attribution_report.md", "\n".join(report_lines))
    identities["real_scene_domain_attribution_report.md"] = report_identity
    return {"status": summary["status"], "output_dir": str(output), "summary_sha256": summary_identity["sha256"], "artifacts": identities, "questions": summary["questions"]}
