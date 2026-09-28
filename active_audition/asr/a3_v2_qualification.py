"""Active-ASR A3-v2 Phase-B qualification.

This module is intentionally downstream of the frozen A3-v2 Phase-A inputs.
It has two explicit entry points:

* :func:`render_a3_v2_realistic_rirs` runs only the native16 renderer in the
  legacy ``ss`` environment and writes a new RIR lock.
* :func:`run_a3_v2_qualification` consumes that lock in the independent ASR
  environment and evaluates G5--G8.

No selection, normalization, model change, decoder tuning, or A4 mechanism is
implemented here.  Historical A3-v1 inputs used by the deterministic G7/Q4
fixtures are read-only provenance and are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.acoustics.resampling import resample_array
from active_audition.asr.a3_v2_freeze import (
    validate_scene_inventory,
    validate_scene_manifest,
    validate_speech_manifest,
)
from active_audition.asr.contract import asr_contract_sha256, load_asr_contract
from active_audition.asr.contract_v2 import (
    A3_V1_SHA256,
    a3_v2_contract_sha256,
    load_a3_v2_contract,
)
from active_audition.asr.frontends import apply_frontend
from active_audition.asr.qualification import (
    add_noise_at_snr,
    canonical_json_bytes,
    deterministic_broadband_noise,
    file_sha256,
    waveform_identity,
)
from active_audition.asr.rir_bridge import load_rir_lock
from active_audition.asr.run_qualification import (
    A3RunError,
    _decode_speech,
    _evaluate,
    _rir_index,
    _transcribe_many,
)
from active_audition.asr.speechbrain_adapter import SpeechBrainASRAdapter
from active_audition.config.loader import load_resolved_config
from active_audition.data.storage import DatasetStorage
from active_audition.evaluation.asr_metrics import aggregate_error_counts
from active_audition.receiver.audit import _runtime_fingerprint
from active_audition.types import ListenerPose


V2_CONTRACT_SHA = "70864c814a55db5d184ef8a6835b65cb564c4c90fc86112f8705b7ecf6df1ffe"
INVENTORY_SHA = "062535e44c66e61c4339b1e791f319ec7d15b320dfe38feaba93f94c6eba3f29"
SCENE_MANIFEST_SHA = "18cc31b13fb7b6410ad7eb548874fd56b69e1562e95c5e898b64b15efba0f477"
SPEECH_MANIFEST_SHA = "9f8fc5e4a62a9b2aee9af0b250517cfb85250fdb24b599d8c0dd818794755c99"
V1_CLEAN_MANIFEST_SHA = "3c3fe2a84fd88bdaa878d4f0b0094b29eebe074fc9b81c7e6d55a284fb1ff63c"
V1_SNR_MANIFEST_SHA = "afe772ace63ddd109af065a65a378529a6347d1fe8dacd4ea79be1cf45286dcb"
V1_Q4_MANIFEST_SHA = "b5a812a150a2828fe25ad1b9b2da40f14f4ac14eba69406508f2726c2e52d5b4"
V1_RIR_LOCK_SHA = "bb71cc51123dbbb418910c8e055a326a2113c663e35d22856438575b4d7cc65e"
SAMPLE_RATE_HZ = 16000
FRONTENDS = ("mean_lr", "fixed_L", "fixed_R")
RIR_LOCK_SCHEMA = "active-asr-a3-v2-realistic-rir-lock-v1"
QUALIFICATION_SCHEMA = "active-asr-a3-v2-qualification-v1"


class A3V2QualificationError(A3RunError):
    """Raised when a frozen Phase-A input or Phase-B technical result is invalid."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha(value: Any) -> str:
    return _sha_bytes(np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes(order="C"))


def _write_json(path: Path, value: Any) -> Dict[str, str]:
    DatasetStorage(str(path.parent)).atomic_write_bytes(path, canonical_json_bytes(value))
    return {"path": str(path), "sha256": file_sha256(path)}


def _identity(repo: Path, path: Path) -> Dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise A3V2QualificationError("required artifact is missing: {}".format(resolved))
    return {"path": str(resolved.relative_to(repo)), "sha256": file_sha256(resolved)}


def _load_json_bound(repo: Path, identity: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    path = repo / str(identity["path"])
    if not path.is_file() or file_sha256(path) != str(identity["sha256"]):
        raise A3V2QualificationError("{} hash mismatch: {}".format(label, path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise A3V2QualificationError("{} must be a JSON object".format(label))
    return value


def _load_v2_inputs(repo: Path, contract_path: str) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str]:
    contract = load_a3_v2_contract(str(repo / contract_path) if not Path(contract_path).is_absolute() else contract_path, require_frozen=True, repo_root=str(repo))
    contract_sha = a3_v2_contract_sha256(contract)
    if contract_sha != V2_CONTRACT_SHA:
        raise A3V2QualificationError("frozen A3-v2 contract SHA mismatch: {}".format(contract_sha))
    realistic = contract["realistic_domain"]
    inventory = _load_json_bound(repo, realistic["scene_inventory"], "scene inventory")
    scene_manifest = _load_json_bound(repo, realistic["scene_manifest"], "scene manifest")
    speech_manifest = _load_json_bound(repo, realistic["speech_manifest"], "speech manifest")
    if realistic["scene_inventory"]["sha256"] != INVENTORY_SHA or file_sha256(repo / realistic["scene_inventory"]["path"]) != INVENTORY_SHA:
        raise A3V2QualificationError("frozen inventory SHA mismatch")
    if realistic["scene_manifest"]["sha256"] != SCENE_MANIFEST_SHA or file_sha256(repo / realistic["scene_manifest"]["path"]) != SCENE_MANIFEST_SHA:
        raise A3V2QualificationError("frozen scene manifest SHA mismatch")
    if realistic["speech_manifest"]["sha256"] != SPEECH_MANIFEST_SHA or file_sha256(repo / realistic["speech_manifest"]["path"]) != SPEECH_MANIFEST_SHA:
        raise A3V2QualificationError("frozen speech manifest SHA mismatch")
    validate_scene_inventory(inventory)
    validate_scene_manifest(scene_manifest)
    validate_speech_manifest(speech_manifest)
    if len(speech_manifest["records"]) != 24:
        raise A3V2QualificationError("v2 speech manifest must contain 24 records")
    return contract, inventory, scene_manifest, speech_manifest, contract_sha


def _sensor_effective(context: Any) -> Dict[str, Any]:
    spec = context.audio_sensor.specification()
    acoustics = spec.acousticsConfig
    layout = spec.channelLayout
    layout_type = str(getattr(layout, "type", ""))
    if hasattr(layout.type, "name"):
        layout_type = str(layout.type.name)
    rate = float(acoustics.sampleRate)
    return {
        "sample_rate_hz": int(rate) if rate.is_integer() else rate,
        "channel_layout": layout_type,
        "channel_count": int(layout.channelCount),
        "materials": bool(spec.enableMaterials),
        "sensor_position": [float(item) for item in spec.position],
    }


def _onset(channel: np.ndarray) -> Dict[str, Any]:
    value = np.asarray(channel, dtype=np.float32)
    peak = float(np.max(np.abs(value)))
    threshold = peak * 0.10
    indices = np.flatnonzero(np.abs(value) >= threshold) if threshold > 0.0 else np.asarray([], dtype=np.int64)
    sample = int(indices[0]) if indices.size else None
    return {"sample": sample, "time_sec": None if sample is None else float(sample / SAMPLE_RATE_HZ), "threshold_fraction_of_peak": 0.10, "threshold_abs": threshold}


def _rir_diagnostics(rir: np.ndarray) -> Dict[str, Any]:
    value = np.asarray(rir, dtype=np.float32)
    rows = {}
    onsets = []
    for index, channel_name in enumerate(("L", "R")):
        channel = value[:, index]
        square = np.square(channel.astype(np.float64))
        onset = _onset(channel)
        onsets.append(onset["sample"] if onset["sample"] is not None else 0)
        start = onset["sample"] if onset["sample"] is not None else 0
        early_end = min(value.shape[0], start + int(round(0.050 * SAMPLE_RATE_HZ)))
        direct_end = min(value.shape[0], start + int(round(0.010 * SAMPLE_RATE_HZ)))
        direct_energy = float(np.sum(square[start:direct_end]))
        late_energy = float(np.sum(square[early_end:]))
        edc = np.cumsum(square[::-1])[::-1]
        rows[channel_name] = {
            "onset": onset,
            "peak_abs": float(np.max(np.abs(channel))),
            "rms": float(np.sqrt(np.mean(square))),
            "sum_square_energy": float(np.sum(square)),
            "direct_energy_onset_plus_10ms": direct_energy,
            "early_energy_onset_plus_50ms": float(np.sum(square[start:early_end])),
            "late_energy_after_onset_plus_50ms": late_energy,
            "drr_proxy_direct_over_late_db": None if late_energy <= 0.0 else float(10.0 * math.log10(max(direct_energy, 1e-300) / late_energy)),
            "schroeder_edc": {"definition": "reverse cumulative sum(x^2), raw unnormalized", "sample_step": max(1, SAMPLE_RATE_HZ // 1000), "sha256": _sha_bytes(np.ascontiguousarray(edc, dtype="<f8").tobytes(order="C"))},
        }
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "length_samples": int(value.shape[0]),
        "length_sec": float(value.shape[0] / SAMPLE_RATE_HZ),
        "channel_order": ["L", "R"],
        "channels": rows,
        "diagnostic_direct_definition": "10ms from each channel's 10%-peak onset; not a hard G6 metric gate",
        "no_normalization": True,
    }


def _scene_case_lookup(scene_manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    cases = []
    for scene in scene_manifest["scenes"]:
        cases.extend(scene["cases"])
    if len(cases) != 8:
        raise A3V2QualificationError("scene manifest must contain exactly 8 cases")
    return sorted(cases, key=lambda item: str(item["case_id"]))


def _verify_inventory_scene(repo: Path, inventory: Mapping[str, Any], case: Mapping[str, Any]) -> Dict[str, str]:
    scene = next((row for row in inventory["scenes"] if row["scene_id"] == case["scene_id"]), None)
    if not isinstance(scene, Mapping):
        raise A3V2QualificationError("scene is absent from frozen inventory: {}".format(case["scene_id"]))
    asset = (repo / str(scene["assets"]["scene_asset"]["path"])).resolve()
    navmesh = (repo / str(scene["assets"]["navmesh"]["path"])).resolve()
    if not asset.is_file() or file_sha256(asset) != case["scene_asset_sha256"]:
        raise A3V2QualificationError("scene asset provenance mismatch: {}".format(asset))
    if not navmesh.is_file() or file_sha256(navmesh) != case["navmesh_sha256"]:
        raise A3V2QualificationError("navmesh provenance mismatch: {}".format(navmesh))
    return {"scene_asset": str(asset), "navmesh": str(navmesh)}


def render_a3_v2_realistic_rirs(contract_path: str, output_dir: str, runtime_config_path: str = "configs/active_audition/v0_replica_debug.yaml") -> Dict[str, Any]:
    """Render the frozen 8 native16 Replica cases twice in ``ss``."""

    # This import order is intentional: simulator.py imports quaternion before
    # habitat_sim, and no Habitat import is allowed to precede it here.
    import quaternion  # noqa: F401
    from active_audition.acoustics.rir import render_native_rir
    from active_audition.scene.simulator import create_scene_simulator

    repo = Path.cwd().resolve()
    contract, inventory, scene_manifest, _speech_manifest, contract_sha = _load_v2_inputs(repo, contract_path)
    runtime_config = load_resolved_config(runtime_config_path)
    cases = _scene_case_lookup(scene_manifest)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rir_dir = output / "realistic_rirs"
    rir_dir.mkdir(parents=True, exist_ok=True)
    runtime_fingerprint = _runtime_fingerprint(repo)
    runtime_sha = _sha_bytes(canonical_json_bytes(runtime_fingerprint))
    records = []
    for case in cases:
        scene_paths = _verify_inventory_scene(repo, inventory, case)
        repeats = []
        for repeat_id in (1, 2):
            with create_scene_simulator(
                runtime_config,
                scene_id=case["scene_id"],
                require_navmesh=True,
                load_semantic_mesh=True,
                scene_override=scene_paths,
            ) as context:
                effective = _sensor_effective(context)
                if effective["sample_rate_hz"] != 16000 or effective["channel_count"] != 2 or "Binaural" not in effective["channel_layout"] or effective["materials"] is not False:
                    raise A3V2QualificationError("effective AudioSensor state failed for {}: {}".format(case["case_id"], effective))
                offset = np.asarray(runtime_config["listener"]["sensor_offset_m"], dtype=np.float64)
                base = np.asarray(case["listener_base_position_world"], dtype=np.float64)
                sensor = np.asarray(case["listener_sensor_position_world"], dtype=np.float64)
                if not np.allclose(base + offset, sensor, atol=1e-5, rtol=0.0):
                    raise A3V2QualificationError("frozen base/sensor pose mismatch for {}".format(case["case_id"]))
                pose = ListenerPose(tuple(base.tolist()), tuple(sensor.tolist()), float(case["listener_yaw_deg"]))
                rir = np.asarray(render_native_rir(context, case["source_position_world"], pose), dtype=np.float32)
                if rir.ndim != 2 or rir.shape[1] != 2 or rir.shape[0] <= 0 or not np.isfinite(rir).all():
                    raise A3V2QualificationError("invalid native16 canonical RIR for {}".format(case["case_id"]))
                path = rir_dir / (str(case["case_id"]) + "__repeat{} .npy".format(repeat_id)).replace(" ", "")
                np.save(str(path), rir, allow_pickle=False)
                repeats.append({
                    "repeat_id": repeat_id,
                    "relative_path": str(path.relative_to(output)),
                    "file_sha256": file_sha256(path),
                    "array_sha256": _array_sha(rir),
                    "shape": list(rir.shape),
                    "canonical_channel_order": ["L", "R"],
                    "requested_sensor": {"sample_rate_hz": 16000, "channel_layout": "binaural", "channel_count": 2, "materials": False, "position": [float(x) for x in sensor]},
                    "effective_sensor": effective,
                    "diagnostics": _rir_diagnostics(rir),
                })
        first = np.load(str(output / repeats[0]["relative_path"]), allow_pickle=False)
        second = np.load(str(output / repeats[1]["relative_path"]), allow_pickle=False)
        diff = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
        repeatability = {
            "shape_equal": bool(first.shape == second.shape),
            "array_sha_equal": repeats[0]["array_sha256"] == repeats[1]["array_sha256"],
            "max_abs_difference": float(np.max(np.abs(diff))),
            "rms_difference": float(np.sqrt(np.mean(np.square(diff)))),
            "finite_both": bool(np.isfinite(first).all() and np.isfinite(second).all()),
            "status": "PASS" if first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all() else "FAIL",
        }
        records.append({"case_id": case["case_id"], "scene_id": case["scene_id"], "category": case["category"], "distance_m": case["distance_m"], "relative_azimuth_deg": case["relative_azimuth_deg"], "line_of_sight": case["los_status"], "source_position_world": case["source_position_world"], "listener_base_position_world": case["listener_base_position_world"], "listener_sensor_position_world": case["listener_sensor_position_world"], "listener_yaw_deg": case["listener_yaw_deg"], "scene_asset_sha256": case["scene_asset_sha256"], "navmesh_sha256": case["navmesh_sha256"], "repeats": repeats, "repeatability": repeatability})
    lock = {
        "schema_version": RIR_LOCK_SCHEMA,
        "gate": "A3",
        "purpose": "A3-v2 held-out realistic Replica native16 materials-OFF qualification",
        "a4_features_present": False,
        "a3_v2_contract_sha256": contract_sha,
        "scene_manifest": {"path": contract["realistic_domain"]["scene_manifest"]["path"], "sha256": SCENE_MANIFEST_SHA},
        "speech_manifest": {"path": contract["realistic_domain"]["speech_manifest"]["path"], "sha256": SPEECH_MANIFEST_SHA},
        "runtime_config": {"path": str(Path(runtime_config_path).as_posix()), "sha256": file_sha256(repo / runtime_config_path)},
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_sha256": runtime_sha,
        "records": records,
    }
    lock_path = output / "realistic_rirs" / "a3_v2_realistic_rir_lock.json"
    lock_identity = _write_json(lock_path, lock)
    return {"status": "PASS", "cases": len(records), "repeats_per_case": 2, "lock": lock_identity, "runtime_sha256": runtime_sha, "a3_v2_contract_sha256": contract_sha}


def _load_speech_records(repo: Path, source_root: Path, records: Sequence[Mapping[str, Any]]) -> List[Tuple[Mapping[str, Any], np.ndarray]]:
    result = []
    for record in records:
        result.append((record, _decode_speech(repo, source_root, record)[0]))
    return result


def _evaluate_frontends(adapter: SpeechBrainASRAdapter, records: Sequence[Mapping[str, Any]], binaurals: Sequence[np.ndarray], context: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for frontend in FRONTENDS:
        mono = [apply_frontend(value, frontend) for value in binaurals]
        outputs = _transcribe_many(adapter, mono, frontend)
        rows, aggregate = _evaluate(records, outputs)
        for row, waveform in zip(rows, mono):
            row.update(context)
            row["frontend"] = frontend
            row["waveform_sha256"] = waveform_identity(waveform)
            row["normalization"] = "none"
        result[frontend] = {"aggregate": aggregate, "rows": rows}
    return result


def _v1_adapter_and_sources(repo: Path, v2_contract: Mapping[str, Any]) -> Tuple[SpeechBrainASRAdapter, Mapping[str, Any], Mapping[str, Any], str]:
    v1_path = repo / "configs/active_audition/v1/asr_contract.yaml"
    v1 = load_asr_contract(str(v1_path), require_frozen=True)
    if asr_contract_sha256(v1) != A3_V1_SHA256:
        raise A3V2QualificationError("A3-v1 identity changed")
    for key in ("model", "environment", "decoder", "input", "frontends", "text", "sources"):
        if v1[key] != v2_contract[key]:
            raise A3V2QualificationError("v2 inherited identity differs from v1: {}".format(key))
    sources = json.loads(json.dumps(v1["sources"]))
    source_config = repo / v1["sources"]["config_path"]
    import yaml
    source_config_value = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    return SpeechBrainASRAdapter(v1), sources, source_config_value, asr_contract_sha256(v1)


def _manifest(repo: Path, path: str, expected_sha: str) -> Mapping[str, Any]:
    value_path = repo / path
    if not value_path.is_file() or file_sha256(value_path) != expected_sha:
        raise A3V2QualificationError("historical manifest integrity failure: {}".format(path))
    value = json.loads(value_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise A3V2QualificationError("manifest is not an object: {}".format(path))
    return value


def _load_v2_rirs(repo: Path, path: str, contract_sha: str, expected_cases: Sequence[str]) -> Tuple[Mapping[str, Any], Dict[str, np.ndarray], str]:
    lock_path = repo / path
    if not lock_path.is_file():
        raise A3V2QualificationError("v2 RIR lock missing: {}".format(path))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != RIR_LOCK_SCHEMA or lock.get("a3_v2_contract_sha256") != contract_sha:
        raise A3V2QualificationError("v2 RIR lock identity mismatch")
    result: Dict[str, np.ndarray] = {}
    for record in lock.get("records", []):
        if len(record.get("repeats", [])) != 2:
            raise A3V2QualificationError("v2 RIR lock repeat count is not 2")
        repeat = record["repeats"][0]
        # B1 records paths relative to the qualification output root, while
        # the lock itself lives one directory below that root.
        array_path = lock_path.parent.parent / repeat["relative_path"]
        if not array_path.is_file() or file_sha256(array_path) != repeat["file_sha256"]:
            raise A3V2QualificationError("v2 RIR array file identity mismatch")
        array = np.load(str(array_path), allow_pickle=False)
        if _array_sha(array) != repeat["array_sha256"] or array.shape[1] != 2 or not np.isfinite(array).all():
            raise A3V2QualificationError("v2 RIR array invalid")
        result[str(record["case_id"])] = np.asarray(array, dtype=np.float32)
    if sorted(result) != sorted(expected_cases):
        raise A3V2QualificationError("v2 RIR lock cases do not match frozen scene manifest")
    return lock, result, file_sha256(lock_path)


def _g5(repo: Path, source_root: Path, adapter: SpeechBrainASRAdapter, v1_contract: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = _manifest(repo, v1_contract["qualification"]["manifests"]["clean"]["path"], V1_CLEAN_MANIFEST_SHA)
    records = list(manifest["records"])
    loaded = _load_speech_records(repo, source_root, records)
    waveforms = [item[1] for item in loaded]
    repeats = [_transcribe_many(adapter, waveforms, "clean_mono") for _ in range(3)]
    rows, aggregate = _evaluate(records, repeats[0])
    consistency = [{"utterance_id": record["utterance_id"], "hypotheses": [run[index].hypothesis for run in repeats], "consistent": len({run[index].hypothesis for run in repeats}) == 1} for index, record in enumerate(records)]
    return {"schema_version": "active-asr-a3-v2-clean-sanity-v1", "status": "PASS" if aggregate["WER"] <= 0.10 and all(row["consistent"] for row in consistency) else "FAIL", "planned": len(records), "valid": len(rows), "aggregate": aggregate, "threshold": {"wer_max_fraction": 0.10}, "repeat_consistency": consistency, "rows": rows, "manifest_sha256": V1_CLEAN_MANIFEST_SHA, "complete_utterance": True}


def _g6(repo: Path, source_root: Path, adapter: SpeechBrainASRAdapter, speech_manifest: Mapping[str, Any], rirs: Mapping[str, np.ndarray], scene_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    records = list(speech_manifest["records"])
    cases = _scene_case_lookup(scene_manifest)
    speech_loaded = _load_speech_records(repo, source_root, records)
    by_id = {str(record["utterance_id"]): waveform for record, waveform in speech_loaded}
    rows_by_frontend: Dict[str, List[Dict[str, Any]]] = {name: [] for name in FRONTENDS}
    metadata = {str(case["case_id"]): case for case in cases}
    ordered = [(case, record, by_id[str(record["utterance_id"])]) for case in cases for record in records]
    binaurals = [convolve_binaural(waveform, rirs[str(case["case_id"])]) for case, _record, waveform in ordered]
    repeated_records = [dict(record) for _case, record, _waveform in ordered]
    contexts = [{"record_id": "{}__{}".format(record["utterance_id"], case["case_id"]), "case_id": case["case_id"], "scene_id": case["scene_id"], "category": case["category"], "rir_sha256": _array_sha(rirs[str(case["case_id"])]), "binaural_waveform_sha256": _array_sha(waveform), "convolution_time_axis": "full"} for (case, record, _waveform), waveform in zip(ordered, binaurals)]
    for frontend in FRONTENDS:
        mono = [apply_frontend(value, frontend) for value in binaurals]
        outputs = _transcribe_many(adapter, mono, frontend)
        rows, aggregate = _evaluate(repeated_records, outputs)
        for row, context, value in zip(rows, contexts, mono):
            row.update(context)
            row["frontend"] = frontend
            row["mono_waveform_sha256"] = waveform_identity(value)
            row["normalization"] = "none"
            row["valid_technical_record"] = bool(np.isfinite(value).all() and value.size > 0)
        rows_by_frontend[frontend] = rows
    production = rows_by_frontend["mean_lr"]
    aggregate = aggregate_error_counts([row["metric"] for row in production])
    nonempty = sum(bool(row["hypothesis"].strip()) for row in production)
    by_case: Dict[str, Any] = {}
    by_category: Dict[str, Any] = {}
    for field in ("case_id", "category", "scene_id"):
        destination = by_case if field == "case_id" else by_category if field == "category" else {}
        if field == "scene_id":
            destination = {}
        for value in sorted({str(row[field]) for row in production}):
            group = [row for row in production if str(row[field]) == value]
            destination[value] = {"records": len(group), "aggregate": aggregate_error_counts([row["metric"] for row in group])}
        if field == "scene_id":
            scene_aggregates = destination
    return {"schema_version": "active-asr-a3-v2-realistic-domain-sanity-v1", "status": "PASS" if len(production) == 192 and nonempty / 192.0 >= 0.80 and aggregate["WER"] <= 0.50 and all(row["valid_technical_record"] for row in production) else "FAIL", "planned": 192, "valid": sum(row["valid_technical_record"] for row in production), "nonempty_hypotheses": nonempty, "nonempty_fraction": nonempty / 192.0, "aggregate": aggregate, "thresholds": {"minimum_nonempty_fraction": 0.80, "wer_max_fraction": 0.50}, "production_frontend": "mean_lr", "scene_aggregates": scene_aggregates, "case_aggregates": by_case, "category_aggregates": by_category, "frontends": {name: {"role": "hard_gate" if name == "mean_lr" else "fixed_sensitivity_diagnostic", "aggregate": aggregate_error_counts([row["metric"] for row in rows_by_frontend[name]]), "rows": rows_by_frontend[name]} for name in FRONTENDS}, "rows": production}


def _g7(repo: Path, source_root: Path, adapter: SpeechBrainASRAdapter, v1_contract: Mapping[str, Any], old_rirs: Mapping[Tuple[str, int], np.ndarray]) -> Dict[str, Any]:
    manifest_path = v1_contract["qualification"]["manifests"]["snr_sensitivity"]["path"]
    manifest = _manifest(repo, manifest_path, V1_SNR_MANIFEST_SHA)
    records = list(manifest["records"])
    sources = v1_contract["sources"]
    source_root_value = repo / str(sources["config_path"])
    import yaml
    source_config = yaml.safe_load(source_root_value.read_text(encoding="utf-8"))
    source_root = repo / source_config["librispeech"]["root"]
    speech_loaded = [_decode_speech(repo, source_root, record["speech"])[0] for record in records]
    by_condition: Dict[str, List[np.ndarray]] = {"reverb_only": [], "10": [], "0": [], "-10": []}
    refs = [record["speech"] for record in records]
    for record, dry in zip(records, speech_loaded):
        case = str(record["rir_case"]["case_id"])
        target = apply_frontend(convolve_binaural(dry, old_rirs[(case, 16000)]), "mean_lr")
        noise = deterministic_broadband_noise(target.size, int(record["noise_seed"]))
        by_condition["reverb_only"].append(target)
        for level in (10, 0, -10):
            by_condition[str(level)].append(add_noise_at_snr(target, noise, float(level))[0])
    results = {}
    for condition, values in by_condition.items():
        outputs = _transcribe_many(adapter, values, "mean_lr")
        rows, aggregate = _evaluate(refs, outputs)
        results[condition] = {"aggregate": aggregate, "rows": rows}
    high, low = results["10"]["aggregate"], results["-10"]["aggregate"]
    high_edits = high["S"] + high["D"] + high["I"]
    low_edits = low["S"] + low["D"] + low["I"]
    delta_pp = (low["WER"] - high["WER"]) * 100.0
    passed = low["WER"] > high["WER"] and delta_pp >= 10.0 and low_edits > high_edits
    return {"schema_version": "active-asr-a3-v2-snr-sensitivity-v1", "status": "PASS" if passed else "FAIL", "fixture": "frozen_A3_v1_deterministic_broadband_mechanism", "manifest_sha256": V1_SNR_MANIFEST_SHA, "conditions": results, "low_minus_high_wer_percentage_points": delta_pp, "high_edit_count": high_edits, "low_edit_count": low_edits, "hard_gate": {"low_wer_greater_than_high": low["WER"] > high["WER"], "minimum_delta_percentage_points": 10.0, "delta_pass": delta_pp >= 10.0, "low_edit_count_greater_than_high": low_edits > high_edits}}


def _g8(repo: Path, source_root: Path, adapter: SpeechBrainASRAdapter, v1_contract: Mapping[str, Any], old_rirs: Mapping[Tuple[str, int], np.ndarray]) -> Dict[str, Any]:
    manifest = _manifest(repo, v1_contract["qualification"]["manifests"]["q4"]["path"], V1_Q4_MANIFEST_SHA)
    records = list(manifest["records"])
    refs = [record["speech"] for record in records]
    path_a, path_b, meta = [], [], []
    for record in records:
        dry16 = _decode_speech(repo, source_root, record["speech"])[0]
        case = str(record["rir_case"]["case_id"])
        a = convolve_binaural(dry16, old_rirs[(case, 16000)])
        dry24 = resample_array(dry16, 16000, 24000)
        b = resample_array(convolve_binaural(dry24, old_rirs[(case, 24000)]), 24000, 16000)
        for frontend in FRONTENDS:
            aa, bb = apply_frontend(a, frontend), apply_frontend(b, frontend)
            path_a.append(aa); path_b.append(bb)
            meta.append({"record_id": record["record_id"], "utterance_id": record["speech"]["utterance_id"], "case_id": case, "frontend": frontend, "path_a_waveform_sha256": waveform_identity(aa), "path_b_waveform_sha256": waveform_identity(bb), "separate_normalization": False})
    outputs_a = _transcribe_many(adapter, path_a, "q4_A")
    outputs_b = _transcribe_many(adapter, path_b, "q4_B")
    rows_a, aggregate_a = _evaluate(refs * 3, outputs_a)
    rows_b, aggregate_b = _evaluate(refs * 3, outputs_b)
    paired = []
    for item, left, right in zip(meta, rows_a, rows_b):
        paired.append({**item, "path_a": left, "path_b": right, "hypothesis_equal": left["hypothesis"] == right["hypothesis"]})
    return {"schema_version": "active-asr-a3-v2-q4-sample-rate-bridge-v1", "status": "Q4_EVIDENCE_COMPLETE_REVIEW_REQUIRED", "decision_authority": "reviewer", "separate_normalization": False, "path_a": {"description": "native_SS2_16khz", "aggregate": aggregate_a}, "path_b": {"description": "native_SS2_24khz_resample_poly_to_16khz", "aggregate": aggregate_b}, "paired": paired, "provenance": {"manifest_sha256": V1_Q4_MANIFEST_SHA, "rir_lock_sha256": V1_RIR_LOCK_SHA}}


def run_a3_v2_qualification(contract_path: str, rir_lock_path: str, output_dir: str, legacy_rir_lock_path: str = "runs/active_asr_v1/a3_frozen_rirs_07a7769/rir_lock.json") -> Dict[str, Any]:
    """Run G5--G8 after the v2 RIR lock has passed technical validation."""

    repo = Path.cwd().resolve()
    contract, _inventory, scene_manifest, speech_manifest, contract_sha = _load_v2_inputs(repo, contract_path)
    lock, rirs, lock_sha = _load_v2_rirs(repo, rir_lock_path, contract_sha, [str(item["case_id"]) for item in _scene_case_lookup(scene_manifest)])
    adapter, _sources, source_config, v1_sha = _v1_adapter_and_sources(repo, contract)
    source_root = repo / source_config["librispeech"]["root"]
    v1_contract = load_asr_contract(str(repo / "configs/active_audition/v1/asr_contract.yaml"), require_frozen=True)
    old_lock, old_lock_path = load_rir_lock(str(repo / legacy_rir_lock_path), v1_sha)
    old_rirs = _rir_index(old_lock, old_lock_path)
    if file_sha256(old_lock_path) != V1_RIR_LOCK_SHA:
        raise A3V2QualificationError("historical v1 RIR lock SHA mismatch")
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    g5 = _g5(repo, source_root, adapter, v1_contract)
    g6 = _g6(repo, source_root, adapter, speech_manifest, rirs, scene_manifest)
    g7 = _g7(repo, source_root, adapter, v1_contract, old_rirs)
    g8 = _g8(repo, source_root, adapter, v1_contract, old_rirs)
    model_lock = _load_json_bound(repo, contract["artifacts"]["model_lock"], "model lock")
    model_lock_result = {"schema_version": "active-asr-a3-v2-model-lock-check-v1", "status": "PASS", "a3_v2_contract_sha256": contract_sha, "a3_v1_contract_sha256": v1_sha, "model_lock_sha256": file_sha256(repo / contract["artifacts"]["model_lock"]["path"]), "model_repo_id": model_lock.get("repo_id"), "resolved_revision": model_lock.get("resolved_revision"), "environment": contract["environment"]}
    gates = {"G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS", "G5": g5["status"], "G6": g6["status"], "G7": g7["status"], "G8": g8["status"]}
    hard_pass = all(gates[key] == "PASS" for key in ("G1", "G2", "G3", "G4", "G5", "G6", "G7"))
    overall = "A3_V2_GOAL_EVIDENCE_COMPLETE / READY_FOR_REVIEW" if hard_pass else "A3_V2_SERVER_RUN_FAIL / OPEN_BLOCKED"
    artifacts = {
        "a3_v2_model_lock_check.json": model_lock_result,
        "a3_v2_clean_asr_qualification.json": g5,
        "a3_v2_realistic_domain_sanity.json": g6,
        "a3_v2_snr_sensitivity.json": g7,
        "a3_v2_q4_sample_rate_bridge.json": g8,
    }
    identities = {}
    for name, value in artifacts.items():
        identities[name] = _write_json(output / name, value)
    summary = {"schema_version": QUALIFICATION_SCHEMA, "gate": "A3", "status": overall, "a3_closed": False, "a4_opened": False, "a3_v2_contract_sha256": contract_sha, "rir_lock_sha256": lock_sha, "historical_v1_rir_lock_sha256": V1_RIR_LOCK_SHA, "acceptance_matrix": gates, "artifacts": identities, "blocker": None if hard_pass else "ONE_OR_MORE_A3_V2_G1_TO_G7_HARD_GATES_FAILED"}
    report_lines = ["# Active-ASR V1.1 A3-v2 Phase-B Qualification", "", "- Overall: **{}**".format(overall), "- A3-v2 contract: `{}`".format(contract_sha), "- RIR lock: `{}`".format(lock_sha), "", "| Gate | Status |", "|---|---|"] + ["| {} | {} |".format(key, value) for key, value in gates.items()] + ["", "G8 remains `Q4_EVIDENCE_COMPLETE_REVIEW_REQUIRED`; reviewer decision is required. A3 is not CLOSED and A4 remains CLOSED.", ""]
    report_path = output / "a3_v2_report.md"
    DatasetStorage(str(output)).atomic_write_text(report_path, "\n".join(report_lines))
    identities["a3_v2_report.md"] = {"path": "a3_v2_report.md", "sha256": file_sha256(report_path)}
    summary["artifacts"] = identities
    summary_identity = _write_json(output / "a3_v2_summary.json", summary)
    summary["summary_artifact"] = summary_identity
    return summary
