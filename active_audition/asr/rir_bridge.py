"""A3-only bridge from frozen A2 controlled RIR cases to ASR waveforms.

This module renders only the pre-registered A2 shoebox cases.  It does not
generate listener candidates, mix localized sources, or calibrate an initial
pose SNR.
"""

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from active_audition.asr.contract import asr_contract_sha256, load_asr_contract
from active_audition.asr.qualification import canonical_json_bytes, file_sha256, validate_manifest
from active_audition.config.loader import load_resolved_config
from active_audition.data.storage import DatasetStorage


RIR_LOCK_SCHEMA_VERSION = "active-asr-a3-rir-lock-v1"


class A3RIRBridgeError(ValueError):
    """Raised when a frozen A2 RIR case cannot be reproduced for A3."""


def _array_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _save_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value, dtype=np.float32), allow_pickle=False)
    DatasetStorage(str(path.parent)).atomic_write_bytes(path, buffer.getvalue())


def _load_bound_manifest(repo: Path, contract: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    identity = contract["qualification"]["manifests"][name]
    path = repo / identity["path"]
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        raise A3RIRBridgeError("frozen {} manifest identity mismatch".format(name))
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value, name)
    if len(value["records"]) != int(identity["records"]):
        raise A3RIRBridgeError("frozen {} manifest record count mismatch".format(name))
    return value


def _effective_acoustics(context: Any) -> Dict[str, Any]:
    from active_audition.receiver.audit import _ACOUSTICS_FIELDS

    acoustics = context.audio_sensor.specification().acousticsConfig
    result = {field: getattr(acoustics, field) for field in _ACOUSTICS_FIELDS}
    rate = float(acoustics.sampleRate)
    result["sampleRate"] = int(rate) if rate.is_integer() else rate
    return result


def render_a3_qualification_rirs(
    contract_path: str,
    output_dir: str,
    runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml",
) -> Dict[str, Any]:
    """Render exact frozen SS2/Q4 cases in the legacy ``ss`` environment."""

    # quaternion remains imported before habitat_sim because simulator.py is
    # imported lazily only after this point through create_scene_simulator.
    import quaternion  # noqa: F401
    from active_audition.acoustics.rir import render_native_rir
    from active_audition.receiver.audit import _runtime_fingerprint
    from active_audition.receiver.geometry import load_geometry_registry
    from active_audition.scene.simulator import create_scene_simulator
    from active_audition.types import ListenerPose

    repo = Path(".").resolve()
    contract = load_asr_contract(contract_path, require_frozen=True)
    ss2_manifest = _load_bound_manifest(repo, contract, "ss2_domain")
    q4_manifest = _load_bound_manifest(repo, contract, "q4")
    geometry_registry = load_geometry_registry(
        "registries/active_asr_a2/qualification_geometry.yaml", str(repo)
    )
    geometry = next(item for item in geometry_registry["geometries"] if item["id"] == "a2_symmetric_shoebox_v1")
    runtime_config = load_resolved_config(runtime_config_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    unique_cases: Dict[str, Mapping[str, Any]] = {}
    for manifest in (ss2_manifest, q4_manifest):
        for record in manifest["records"]:
            case = record["rir_case"]
            unique_cases[str(case["case_id"])] = case
    rates_by_case = {case_id: {16000, 24000} for case_id in unique_cases}
    runtime_fingerprint = _runtime_fingerprint(repo)
    runtime_sha = hashlib.sha256(canonical_json_bytes(runtime_fingerprint)).hexdigest()
    records = []
    for sample_rate in (16000, 24000):
        config = copy.deepcopy(runtime_config)
        config["acoustics"] = dict(runtime_config["acoustics"])
        config["acoustics"]["sample_rate_hz"] = sample_rate
        with create_scene_simulator(
            config,
            scene_id=geometry["id"],
            require_navmesh=False,
            load_semantic_mesh=False,
            scene_override={"scene_asset": geometry["mesh_path"]},
        ) as context:
            effective = _effective_acoustics(context)
            for case_id, case in sorted(unique_cases.items()):
                if sample_rate not in rates_by_case[case_id]:
                    continue
                expected = dict(case["expected_effective_acoustics"])
                expected["sampleRate"] = sample_rate
                if effective != expected:
                    raise A3RIRBridgeError(
                        "requested/effective acoustics mismatch for {} Hz: expected={} effective={}".format(
                            sample_rate, expected, effective
                        )
                    )
                receiver_sensor = np.asarray(case["receiver_sensor_position_world"], dtype=np.float64)
                offset = np.asarray(config["listener"]["sensor_offset_m"], dtype=np.float64)
                base = tuple((receiver_sensor - offset).tolist())
                pose = ListenerPose(
                    base_position_world=base,
                    sensor_position_world=tuple(receiver_sensor.tolist()),
                    yaw_deg=float(case["listener_yaw_deg"]),
                )
                rir = np.asarray(
                    render_native_rir(context, case["source_position_world"], pose),
                    dtype=np.float32,
                )
                filename = "{}__{}hz.npy".format(case_id, sample_rate)
                path = output / filename
                _save_npy(path, rir)
                records.append({
                    "case_id": case_id,
                    "sample_rate_hz": sample_rate,
                    "relative_path": filename,
                    "file_sha256": file_sha256(path),
                    "array_sha256": _array_sha256(rir),
                    "shape": list(rir.shape),
                    "dtype": "float32",
                    "channel_order": ["L", "R"],
                    "geometry_id": case["geometry_id"],
                    "mesh_sha256": case["mesh_sha256"],
                    "geometry_registry_sha256": case["geometry_registry_sha256"],
                    "receiver_sensor_position_world": case["receiver_sensor_position_world"],
                    "listener_yaw_deg": case["listener_yaw_deg"],
                    "source_position_world": case["source_position_world"],
                    "distance_m": case["distance_m"],
                    "relative_azimuth_deg": case["relative_azimuth_deg"],
                    "line_of_sight": case["line_of_sight"],
                    "effective_acoustics": effective,
                    "runtime_sha256": runtime_sha,
                })
    lock = {
        "schema_version": RIR_LOCK_SCHEMA_VERSION,
        "gate": "A3",
        "purpose": "ASR_instrument_and_source_qualification_only",
        "a4_features_present": False,
        "asr_contract_sha256": asr_contract_sha256(contract),
        "runtime_config": {
            "path": runtime_config_path,
            "sha256": file_sha256(repo / runtime_config_path),
        },
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_sha256": runtime_sha,
        "geometry_registry_sha256": geometry_registry["registry_sha256"],
        "records": sorted(records, key=lambda item: (item["case_id"], item["sample_rate_hz"])),
    }
    lock_path = output / "rir_lock.json"
    DatasetStorage(str(output)).atomic_write_bytes(lock_path, canonical_json_bytes(lock))
    return {
        "status": "PASS",
        "records": len(records),
        "rir_lock": {"path": str(lock_path), "sha256": file_sha256(lock_path)},
        "runtime_sha256": runtime_sha,
        "asr_contract_sha256": asr_contract_sha256(contract),
    }


def load_rir_lock(path: str, expected_contract_sha256: str) -> Tuple[Mapping[str, Any], Path]:
    lock_path = Path(path).resolve()
    value = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "gate", "purpose", "a4_features_present", "asr_contract_sha256",
        "runtime_config", "runtime_fingerprint", "runtime_sha256", "geometry_registry_sha256", "records",
    }:
        raise A3RIRBridgeError("RIR lock schema fields are invalid")
    if value["schema_version"] != RIR_LOCK_SCHEMA_VERSION or value["gate"] != "A3" or value["a4_features_present"] is not False:
        raise A3RIRBridgeError("RIR lock identity is invalid")
    if value["asr_contract_sha256"] != expected_contract_sha256:
        raise A3RIRBridgeError("RIR lock is bound to a different A3 contract")
    for record in value["records"]:
        array_path = lock_path.parent / record["relative_path"]
        if not array_path.is_file() or file_sha256(array_path) != record["file_sha256"]:
            raise A3RIRBridgeError("RIR cache file identity mismatch: {}".format(array_path))
        array = np.load(str(array_path), allow_pickle=False)
        if _array_sha256(array) != record["array_sha256"]:
            raise A3RIRBridgeError("RIR array identity mismatch: {}".format(array_path))
    return value, lock_path
