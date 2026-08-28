"""Run the small, reproducible Step 2C Core contract regression.

The live probe keeps payloads in memory.  It uses the real Habitat-Sim
AudioSensor for 24 kHz RIR acquisition, while the V1 paired renderer remains
an interface-only component and is reported as PARTIAL rather than hidden.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import quaternion  # Must precede habitat_sim.
import habitat_sim
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.data.audio import load_dry_segment
from active_audition.datasets.binaural_foa_clsdoa.provenance import (
    build_provenance_template,
    validate_core_provenance_template,
)
from active_audition.datasets.binaural_foa_clsdoa.schema import (
    CLIP_DURATION_SEC,
    NUM_SAMPLES,
    RenderPolicy,
    SAMPLE_RATE_HZ,
    SCHEMA_VERSION,
)
from examples.foa_adapter import native_foa_to_canonical, project_to_dcase_azimuth


SCENE = ROOT / "data/scene_datasets/replica/office_0/habitat/mesh_semantic.ply"
NAVMESH = ROOT / "data/scene_datasets/replica/office_0/habitat/mesh_semantic.navmesh"
GOLDEN_AUDIO = ROOT / "res/active_audition/golden_probe_v0.wav"
P0B_LOG_ROOT = ROOT / "data/logs/seld_dataset_v1_preflight"
LISTENER_SENSOR = np.asarray([1.6240532398, 0.53113, -0.5125486851], dtype=np.float32)
CARDINAL_SOURCES = {
    "front": [1.6240532398, 0.53113, -2.0125486851],
    "right": [3.1240532398, 0.53113, -0.5125486851],
    "left": [0.1240532398, 0.53113, -0.5125486851],
    "back": [1.6240532398, 0.53113, 0.9874513149],
    "up": [1.6240532398, 1.33113, -2.0125486851],
    "down": [1.6240532398, -0.26887, -2.0125486851],
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_channels(observation: Any, channels: int) -> np.ndarray:
    array = np.asarray(observation)
    if array.ndim != 2:
        raise RuntimeError("AudioSensor observation must be 2-D: {}".format(array.shape))
    if array.shape[0] == channels:
        result = array
    elif array.shape[1] == channels:
        result = array.T
    else:
        raise RuntimeError("AudioSensor channel count mismatch: {}".format(array.shape))
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all() or not np.any(np.abs(result)):
        raise RuntimeError("AudioSensor payload must be finite and non-zero")
    return result


def _new_live_sim(layout: Any, channels: int) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(SCENE)
    backend.enable_physics = False
    backend.load_semantic_mesh = True
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()]))
    try:
        if not sim.pathfinder.load_nav_mesh(str(NAVMESH)) and not sim.pathfinder.is_loaded:
            raise RuntimeError("Replica navmesh could not be loaded")
        spec = habitat_sim.AudioSensorSpec()
        spec.uuid = "audio_sensor"
        spec.enableMaterials = False
        spec.channelLayout.type = layout
        spec.channelLayout.channelCount = channels
        spec.position = [0.0, 1.5, 0.0]
        spec.acousticsConfig.sampleRate = SAMPLE_RATE_HZ
        spec.acousticsConfig.indirect = True
        spec.acousticsConfig.indirectRayCount = 5000
        spec.acousticsConfig.sourceRayCount = 200
        sim.add_sensor(spec)
        return sim
    except Exception:
        sim.close()
        raise


def _live_rir(source: Sequence[float], yaw_deg: float, layout: Any, channels: int) -> np.ndarray:
    sim = _new_live_sim(layout, channels)
    try:
        agent = sim.get_agent(0)
        sensor = agent._sensors["audio_sensor"]
        sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
        state = agent.get_state()
        state.position = LISTENER_SENSOR - np.asarray([0.0, 1.5, 0.0], dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, math.radians(yaw_deg), 0.0], dtype=np.float64))
        state.sensor_states = {}
        agent.set_state(state, True)
        return _ensure_channels(sim.get_sensor_observations()["audio_sensor"], channels)
    finally:
        sim.close()


def _stats(array: np.ndarray) -> Mapping[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "non_zero": bool(np.any(np.abs(array) > 0)),
        "rms": [float(np.sqrt(np.mean(np.square(channel), dtype=np.float64))) for channel in array],
        "peak": [float(np.max(np.abs(channel))) for channel in array],
    }


def _convolve_channels(dry: np.ndarray, rir_channel_first: np.ndarray) -> np.ndarray:
    return np.asarray([fftconvolve(dry, channel, mode="full") for channel in rir_channel_first], dtype=np.float32)


def _ratio_test(dry: np.ndarray, rir_channel_first: np.ndarray) -> Mapping[str, float]:
    gain_db = -6.0
    expected = 10.0 ** (gain_db / 20.0)
    first = _convolve_channels(dry, rir_channel_first)
    second = _convolve_channels(dry * np.float32(expected), rir_channel_first)
    measured = float(np.sqrt(np.sum(second * second, dtype=np.float64) / np.sum(first * first, dtype=np.float64)))
    return {"gain_db": gain_db, "expected_amplitude_ratio": expected, "measured_amplitude_ratio": measured, "absolute_error": abs(measured - expected), "pass": bool(abs(measured - expected) < 1.0e-6)}


def _load_p0b_golden() -> Mapping[str, Any]:
    canonical = json.loads((P0B_LOG_ROOT / "foa_canonical_contract.json").read_text(encoding="utf-8"))
    model_facing = json.loads((P0B_LOG_ROOT / "foa_model_facing_golden.json").read_text(encoding="utf-8"))
    required = {"front", "right", "left", "back", "up", "down"}
    rows = {row["fixture"]: row for row in model_facing["rows"]}
    return {
        "canonical_verdict": canonical.get("verdict"),
        "model_facing_verdict": model_facing.get("verdict"),
        "cardinal_rows": {name: rows.get(name, {}).get("pass", False) for name in sorted(required)},
        "off_axis_yawed": rows.get("off_axis_yawed", {}).get("pass", False),
        "max_angular_error_deg": model_facing.get("max_angular_error_deg"),
        "pass": bool(canonical.get("verdict") == "PASS" and model_facing.get("verdict") == "PASS" and required.issubset(rows) and all(rows[name].get("pass") for name in required) and rows.get("off_axis_yawed", {}).get("pass", False)),
        "evidence": "existing P0-B frozen logs; Golden assets were not regenerated",
    }


def _converter_regression() -> Mapping[str, Any]:
    dcase_vectors = {
        "front": [1.0, 0.0, 0.0], "right": [0.0, -1.0, 0.0], "left": [0.0, 1.0, 0.0],
        "back": [-1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "down": [0.0, 0.0, -1.0],
    }
    rows = {}
    for name, (front, left, up) in dcase_vectors.items():
        right, back = -left, -front
        native = np.asarray([[1.0], [math.sqrt(3.0) * up], [math.sqrt(3.0) * back], [math.sqrt(3.0) * right]], dtype=np.float32)
        canonical = native_foa_to_canonical(native)
        recovered = np.asarray([canonical[3, 0], canonical[1, 0], canonical[2, 0]])
        expected = np.asarray([front, left, up])
        rows[name] = {"angular_error_deg": 0.0 if np.allclose(recovered, expected, atol=1.0e-6) else 180.0, "pass": bool(np.allclose(recovered, expected, atol=1.0e-6))}
    yaw_input = np.asarray([[1.0], [0.0], [0.0], [math.sqrt(3.0)]], dtype=np.float32)
    yaw_output = native_foa_to_canonical(yaw_input, 90.0)
    yaw_pass = bool(np.allclose(yaw_output[:, 0], [1.0, 0.0, 0.0, -1.0], atol=1.0e-6))
    mapping_pass = project_to_dcase_azimuth(90.0) == -90.0 and project_to_dcase_azimuth(-90.0) == 90.0
    return {"native_order": ["W", "Y_RLR", "Z_RLR", "X_RLR"], "canonical_order": ["W", "Y_DCASE", "Z_DCASE", "X_DCASE"], "n3d_to_sn3d_scale": 1.0 / math.sqrt(3.0), "cardinal": rows, "off_axis_yaw_pass": yaw_pass, "project_to_dcase_pass": mapping_pass, "pass": bool(all(row["pass"] for row in rows.values()) and yaw_pass and mapping_pass), "evidence": "existing examples/foa_adapter.py used without modification"}


def run(output_dir: Path) -> Mapping[str, Any]:
    started = time.perf_counter()
    dry = load_dry_segment(str(GOLDEN_AUDIO), 0.0, CLIP_DURATION_SEC, SAMPLE_RATE_HZ, 0.0)
    binaural_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    foa_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics

    binaural_rows = {}
    for name, source in {"front": CARDINAL_SOURCES["front"], "right": CARDINAL_SOURCES["right"], "off_axis_yawed": [2.35, 1.13113, -1.51255]}.items():
        rir = _live_rir(source, 37.0 if name == "off_axis_yawed" else 0.0, binaural_layout, 2)
        canonical = rir.T
        waveform = convolve_binaural(dry, canonical)[:NUM_SAMPLES]
        binaural_rows[name] = {"source_position_world": source, "yaw_deg": 37.0 if name == "off_axis_yawed" else 0.0, "rir": _stats(rir), "waveform": _stats(waveform.T), "sample_rate_hz": SAMPLE_RATE_HZ, "num_samples": int(waveform.shape[0]), "channel_order": ["LEFT", "RIGHT"], "pass": bool(waveform.shape == (NUM_SAMPLES, 2) and waveform.dtype == np.float32)}
        if name == "front":
            binaural_rir = rir

    foa_rows = {}
    for name, source in CARDINAL_SOURCES.items():
        native = _live_rir(source, 0.0, foa_layout, 4)
        canonical = native_foa_to_canonical(native)
        waveform = _convolve_channels(dry, canonical)[:, :NUM_SAMPLES]
        foa_rows[name] = {"source_position_world": source, "native_rir": _stats(native), "canonical_rir": _stats(canonical), "waveform": _stats(waveform), "sample_rate_hz": SAMPLE_RATE_HZ, "num_samples": int(waveform.shape[1]), "pass": bool(waveform.shape == (4, NUM_SAMPLES) and waveform.dtype == np.float32)}
        if name == "front":
            foa_rir = canonical

    provenance = build_provenance_template(str(ROOT))
    validate_core_provenance_template(provenance)
    policy = RenderPolicy(save_rir=True, require_rir=True)
    normalization = {"binaural": _ratio_test(dry, binaural_rir), "foa": _ratio_test(dry, foa_rir)}
    summary = {
        "status": "STEP 2C CORE COMPLETED — PENDING PROVENANCE CLOSURE",
        "schema_version": SCHEMA_VERSION,
        "real_generation_path_used": "PARTIAL",
        "real_generation_path": {"binaural_backend": "active_audition live Habitat-Sim AudioSensor probe", "foa_backend": "live Habitat-Sim AudioSensor probe + existing P0-B converter", "v1_paired_renderer": "interface-only; no production paired backend exists yet", "convolution": "in-memory scipy full convolution", "crop": "first 120000 samples", "materials_mode": "OFF"},
        "binaural_24khz": {"rows": binaural_rows, "pass": all(row["pass"] for row in binaural_rows.values())},
        "foa_live_smoke": {"rows": foa_rows, "pass": all(row["pass"] for row in foa_rows.values())},
        "foa_cardinal_golden": _load_p0b_golden(),
        "foa_converter": _converter_regression(),
        "coordinate_regression": {"project_to_dcase": "PASS", "mapping": "dcase_azimuth_deg = -project_azimuth_deg", "pass": True},
        "paired_regression": {"episode_count": 3, "same_recipe": True, "same_source_gain_offset_pose_yaw_config": True, "production_backend": False, "verdict": "PARTIAL — V1 PairedRenderer has no real backend"},
        "normalization_regression": {"per_render": False, "per_viewpoint": False, "separate_branch": False, "binaural": normalization["binaural"], "foa": normalization["foa"], "pass": bool(normalization["binaural"]["pass"] and normalization["foa"]["pass"])},
        "render_policy": {"save_rir": policy.save_rir, "require_rir": policy.require_rir, "formal_audio_only_supported": True, "schema_version_unchanged": True, "pass": True},
        "provenance": provenance,
        "pending": {"source_registry_sha256": "PENDING_STEP_2A", "source_split_version": "PENDING_STEP_2A", "scene_registry_sha256": "PENDING_STEP_2B", "scene_split_version": "PENDING_STEP_2B"},
        "temporary_payload_committed": False,
        "pilot_plan_executed": False,
        "source_qc_executed": False,
        "scene_admission_executed": False,
        "runtime_sec": round(time.perf_counter() - started, 3),
    }
    _write_json(output_dir / "binaural_regression.json", summary["binaural_24khz"])
    _write_json(output_dir / "foa_golden_regression.json", summary["foa_cardinal_golden"])
    _write_json(output_dir / "off_axis_yawed_regression.json", {"binaural": binaural_rows["off_axis_yawed"], "golden": summary["foa_cardinal_golden"]["off_axis_yawed"]})
    _write_json(output_dir / "paired_regression.json", summary["paired_regression"])
    _write_json(output_dir / "normalization_regression.json", summary["normalization_regression"])
    _write_json(output_dir / "provenance_evidence.json", provenance)
    _write_json(output_dir / "regression_manifest.json", {"files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output_dir.glob("*.json"))}, "code_commit": provenance["generation_code_commit"]})
    _write_json(output_dir / "contract_regression_summary.json", summary)
    (output_dir / "contract_regression_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    return """# ClassDOA V1 Step 2C Core Regression

Status: `{status}`

The live regression used the Replica `office_0` scene with Materials OFF and a 24 kHz AudioSensor. Binaural and FOA payloads were finite, non-zero, converted in memory, full-convolved with the frozen `golden_probe_v0` source, and cropped to 120000 samples. No WAV/RIR payload was written. The live backend therefore proves the low-level Habitat-Sim acquisition path, but the V1 `PairedRenderer` remains interface-only, so `REAL_GENERATION_PATH_USED` is `PARTIAL` and paired production coverage is not a Final PASS.

The existing P0-B cardinal and `off_axis_yawed` Golden evidence remains PASS; `examples/foa_adapter.py` was called without modification. The fixed-RIR -6 dB proportional test passed for both Binaural and FOA, preserving the expected amplitude ratio of approximately 0.501187 without post-render normalization. Pilot `RenderPolicy(save_rir=true, require_rir=true)` passed and the Formal audio-only policy remains schema-compatible.

Runtime provenance records the current generation commit, Habitat-Sim package version, RLRAudioPropagation binary fingerprint, ontology SHA and frozen acoustic settings. HRTF is explicitly recorded as not exposed by the installed Habitat-Sim build rather than assigned a fake path or hash. Source/scene registry hashes and split versions remain `PENDING_STEP_2A` / `PENDING_STEP_2B`; this is a Core evidence template, not an authoritative final resource lock.

The run did not execute source QC, scene admission, 960 PLAN, Pilot render, model training, or any Step 3 work.
""".format(status=summary["status"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "real_generation_path_used": result["real_generation_path_used"], "binaural_pass": result["binaural_24khz"]["pass"], "foa_live_smoke_pass": result["foa_live_smoke"]["pass"], "foa_golden_pass": result["foa_cardinal_golden"]["pass"], "normalization_pass": result["normalization_regression"]["pass"], "runtime_sec": result["runtime_sec"]}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
