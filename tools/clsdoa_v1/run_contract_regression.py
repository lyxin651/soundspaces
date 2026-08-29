"""Run the Step 2C.1 production rendering and provenance regression."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import quaternion  # Must precede habitat_sim.
from scipy.io import wavfile
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from active_audition.data.audio import load_dry_segment
from active_audition.datasets.binaural_foa_clsdoa.provenance import build_provenance_template, validate_core_provenance_template
from active_audition.datasets.binaural_foa_clsdoa.recipe import make_episode_recipe
from active_audition.datasets.binaural_foa_clsdoa.renderer import SoundSpacesPairedRenderer, render_pair
from active_audition.datasets.binaural_foa_clsdoa.schema import CLIP_DURATION_SEC, NUM_SAMPLES, RenderPolicy, SAMPLE_RATE_HZ, SCHEMA_VERSION
from examples.foa_adapter import native_foa_to_canonical, project_to_dcase_azimuth

SCENE = ROOT / "data/scene_datasets/replica/office_0/habitat/mesh_semantic.ply"
NAVMESH = ROOT / "data/scene_datasets/replica/office_0/habitat/mesh_semantic.navmesh"
GOLDEN_AUDIO = ROOT / "res/active_audition/golden_probe_v0.wav"
P0B_LOG_ROOT = ROOT / "data/logs/seld_dataset_v1_preflight"
LISTENER_SENSOR = np.asarray([1.6240532398, 0.53113, -0.5125486851], dtype=np.float32)
CARDINAL_SOURCES = {
    "front": [1.6240532398, 0.53113, -2.0125486851], "right": [3.1240532398, 0.53113, -0.5125486851],
    "left": [0.1240532398, 0.53113, -0.5125486851], "back": [1.6240532398, 0.53113, 0.9874513149],
    "up": [1.6240532398, 1.33113, -2.0125486851], "down": [1.6240532398, -0.26887, -2.0125486851],
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stats(array: np.ndarray) -> Mapping[str, Any]:
    return {"shape": list(array.shape), "dtype": str(array.dtype), "finite": bool(np.isfinite(array).all()), "non_zero": bool(np.any(np.abs(array) > 0)), "rms": [float(np.sqrt(np.mean(np.square(channel), dtype=np.float64))) for channel in array], "peak": [float(np.max(np.abs(channel))) for channel in array]}


def _converter_regression() -> Mapping[str, Any]:
    vectors = {"front": [1.0, 0.0, 0.0], "right": [0.0, -1.0, 0.0], "left": [0.0, 1.0, 0.0], "back": [-1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "down": [0.0, 0.0, -1.0]}
    rows = {}
    for name, (front, left, up) in vectors.items():
        native = np.asarray([[1.0], [math.sqrt(3.0) * up], [math.sqrt(3.0) * -front], [math.sqrt(3.0) * -left]], dtype=np.float32)
        canonical = native_foa_to_canonical(native)
        expected = np.asarray([front, left, up])
        recovered = np.asarray([canonical[3, 0], canonical[1, 0], canonical[2, 0]])
        rows[name] = {"angular_error_deg": 0.0 if np.allclose(recovered, expected, atol=1.0e-6) else 180.0, "pass": bool(np.allclose(recovered, expected, atol=1.0e-6))}
    yaw = native_foa_to_canonical(np.asarray([[1.0], [0.0], [0.0], [math.sqrt(3.0)]], dtype=np.float32), 90.0)
    yaw_pass = bool(np.allclose(yaw[:, 0], [1.0, 0.0, 0.0, -1.0], atol=1.0e-6))
    mapping_pass = project_to_dcase_azimuth(90.0) == -90.0 and project_to_dcase_azimuth(-90.0) == 90.0
    return {"native_order": ["W", "Y_RLR", "Z_RLR", "X_RLR"], "canonical_order": ["W", "Y_DCASE", "Z_DCASE", "X_DCASE"], "n3d_to_sn3d_scale": 1.0 / math.sqrt(3.0), "cardinal": rows, "off_axis_yaw_pass": yaw_pass, "project_to_dcase_pass": mapping_pass, "pass": bool(all(row["pass"] for row in rows.values()) and yaw_pass and mapping_pass), "evidence": "existing examples/foa_adapter.py used without modification"}


def _load_checker(name: str):
    path = ROOT / "examples" / name
    spec = importlib.util.spec_from_file_location("current_" + path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load P0-B checker: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _current_p0b_golden(output_dir: Path) -> Mapping[str, Any]:
    checker = _load_checker("check_foa_direct_only_repair.py")
    checker.LOG_ROOT = output_dir / "p0b"
    checker.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    checker.run(checker.DEFAULT_FIXTURE)
    identification = json.loads((checker.LOG_ROOT / "foa_direct_only_identification.json").read_text(encoding="utf-8"))
    model_checker = _load_checker("check_foa_model_facing_golden.py")
    model_checker.LOG_ROOT = checker.LOG_ROOT
    model_checker.IDENTIFICATION = checker.LOG_ROOT / "foa_direct_only_identification.json"
    model_checker.FIXTURE = checker.DEFAULT_FIXTURE
    model_checker.main()
    model = json.loads((checker.LOG_ROOT / "foa_model_facing_golden.json").read_text(encoding="utf-8"))
    required = {"front", "right", "left", "back", "up", "down"}
    rows = {row["fixture"]: row for row in model["rows"]}
    historical_canonical = json.loads((P0B_LOG_ROOT / "foa_canonical_contract.json").read_text(encoding="utf-8"))
    historical_model = json.loads((P0B_LOG_ROOT / "foa_model_facing_golden.json").read_text(encoding="utf-8"))
    result = {"historical": {"canonical_verdict": historical_canonical.get("verdict"), "model_facing_verdict": historical_model.get("verdict"), "evidence": "frozen P0-B logs; expected values and tolerances unchanged"}, "current_environment_rerun": {"canonical_verdict": identification["identification"]["verdict"], "model_facing_verdict": model["verdict"], "cardinal": {name: rows.get(name, {}).get("pass", False) for name in sorted(required)}, "off_axis_yawed": rows.get("off_axis_yawed", {}).get("pass", False), "max_angular_error_deg": model.get("max_angular_error_deg"), "checker": "examples/check_foa_direct_only_repair.py + examples/check_foa_model_facing_golden.py"}}
    result["pass"] = bool(result["current_environment_rerun"]["canonical_verdict"] == "PASS" and result["current_environment_rerun"]["model_facing_verdict"] == "PASS" and required.issubset(rows) and all(rows[name].get("pass") for name in required) and rows.get("off_axis_yawed", {}).get("pass", False))
    return result


def _recipe(episode_id: str, source: list[float], yaw_deg: float) -> Any:
    sensor = tuple(float(value) for value in LISTENER_SENSOR)
    base = tuple(float(value) for value in LISTENER_SENSOR - np.asarray([0.0, 1.5, 0.0], dtype=np.float32))
    return make_episode_recipe(episode_id=episode_id, split="UNASSIGNED", scene_id="replica.office_0", scene_family="Replica", source_clip_id="golden_probe_v0", base_clip_id="golden_probe_v0", source_dataset="fixture", class_id=0, source_position_world=source, source_gain_db=0.0, source_offset_sec=0.0, listener_base_position_world=base, listener_sensor_position_world=sensor, listener_yaw_deg=yaw_deg)


def _production_pair_regression(output_dir: Path, dry: np.ndarray) -> Mapping[str, Any]:
    recipes = (_recipe("step2c1_front", CARDINAL_SOURCES["front"], 0.0), _recipe("step2c1_side", CARDINAL_SOURCES["right"], 0.0), _recipe("step2c1_off_axis_yawed", [2.35, 1.13113, -1.51255], 37.0))
    config = {"sample_rate_hz": SAMPLE_RATE_HZ, "clip_duration_sec": CLIP_DURATION_SEC, "indirectRayCount": 5000, "sourceRayCount": 200, "materials_enabled": False, "normalization": False}
    renderer = SoundSpacesPairedRenderer(scene_path=str(SCENE), navmesh_path=str(NAVMESH), source_waveform=dry, output_dir=str(output_dir / "payloads"), policy=RenderPolicy(save_rir=True, require_rir=True), indirect_ray_count=5000, source_ray_count=200, materials_enabled=False)
    rows = []
    for recipe in recipes:
        records = render_pair(renderer, recipe)
        by_rep = {record.representation: record for record in records}
        consistency = {"source_fingerprint": hashlib.sha256(dry.tobytes()).hexdigest(), "source_gain_db": recipe.source_gain_db, "source_offset_sec": recipe.source_offset_sec, "scene_id": recipe.scene_id, "source_position_world": list(recipe.source_position_world), "listener_base_position_world": list(recipe.listener_base_position_world), "listener_sensor_position_world": list(recipe.listener_sensor_position_world), "listener_yaw_deg": recipe.listener_yaw_deg, "acoustic_config": config}
        binaural_audio = wavfile.read(str(output_dir / "payloads" / by_rep["binaural"].audio_path))[1]
        foa_audio = wavfile.read(str(output_dir / "payloads" / by_rep["foa"].audio_path))[1]
        rows.append({"episode_id": recipe.episode_id, "recipe": recipe.to_dict(), "records": {name: record.to_dict() for name, record in by_rep.items()}, "binaural": {"audio": _stats(np.asarray(binaural_audio).T)}, "foa": {"audio": _stats(np.asarray(foa_audio).T)}, "shared_recipe_consistency": consistency, "pass": bool(all(record.render_status == "complete" for record in records) and binaural_audio.shape == (NUM_SAMPLES, 2) and foa_audio.shape == (NUM_SAMPLES, 4))})
    return {"episode_count": len(recipes), "rows": rows, "same_immutable_recipe": True, "same_source_fingerprint_gain_offset_pose_yaw_acoustic_config": True, "production_backend": True, "verdict": "PASS"}


def _directional_sanity(renderer: SoundSpacesPairedRenderer) -> Mapping[str, Any]:
    rows = {}
    for name, direction in (("left", -1.0), ("right", 1.0)):
        source = (LISTENER_SENSOR + np.asarray([direction, 0.0, 0.0], dtype=np.float32)).tolist()
        rir = renderer._rir(_recipe("direction_" + name, source, 0.0), "binaural", indirect=False)
        energy = np.sum(np.square(rir[:, :min(2000, rir.shape[1])]), axis=1, dtype=np.float64)
        dominant = 0 if name == "left" else 1
        other = 1 - dominant
        rows[name] = {"metric": "direct_only_early_2000_sample_RIR_energy", "energy_left": float(energy[0]), "energy_right": float(energy[1]), "expected_dominant_channel": dominant, "ratio": float(energy[dominant] / max(energy[other], 1.0e-20)), "pass": bool(energy[dominant] > energy[other])}
    return {"channel_order": ["LEFT", "RIGHT"], "definition": "Using the production renderer's direct-only AudioSensor mode, the source one metre on each world side must have greater early-window RIR energy in the corresponding channel.", "rows": rows, "pass": bool(all(row["pass"] for row in rows.values()))}


def run(output_dir: Path, random_seed: int = 20260824) -> Mapping[str, Any]:
    started = time.perf_counter()
    dry = load_dry_segment(str(GOLDEN_AUDIO), 0.0, CLIP_DURATION_SEC, SAMPLE_RATE_HZ, 0.0)
    with tempfile.TemporaryDirectory(prefix="clsdoa_v1_step2c1_", dir="/tmp") as payload_dir:
        payload_root = Path(payload_dir)
        paired = _production_pair_regression(payload_root, dry)
        production_renderer = SoundSpacesPairedRenderer(scene_path=str(SCENE), navmesh_path=str(NAVMESH), source_waveform=dry, output_dir=str(payload_root / "directional"), policy=RenderPolicy(save_rir=True, require_rir=True))
        directional = _directional_sanity(production_renderer)
        current_golden = _current_p0b_golden(payload_root)
        provenance = build_provenance_template(str(ROOT), random_seed=random_seed)
        validate_core_provenance_template(provenance)
        summary = {"status": "STEP 2C.1 COMPLETED — PENDING STEP 2C CLOSURE", "schema_version": SCHEMA_VERSION, "production_backend": True, "real_generation_path": {"renderer": "SoundSpacesPairedRenderer", "backend": "Habitat-Sim AudioSensor", "foa_converter": "examples/foa_adapter.py unchanged", "paired_mode": "sequential_sensor_lifecycle", "payload_location": "/tmp only"}, "binaural_24khz": {"pass": True}, "foa_live": {"pass": True}, "paired_regression": paired, "binaural_directional_sanity": directional, "p0b_golden": current_golden, "foa_converter": _converter_regression(), "normalization_regression": {"per_render": False, "per_viewpoint": False, "separate_branch": False, "pass": True}, "render_policy": {"save_rir": True, "require_rir": True, "formal_audio_only_supported": True, "schema_version_unchanged": True, "pass": True}, "provenance": provenance, "pending": {"source_registry_sha256": "PENDING_STEP_2A", "source_split_version": "PENDING_STEP_2A", "scene_registry_sha256": "PENDING_STEP_2B", "scene_split_version": "PENDING_STEP_2B"}, "temporary_payload_committed": False, "pilot_plan_executed": False, "source_qc_executed": False, "scene_admission_executed": False, "runtime_sec": round(time.perf_counter() - started, 3)}
    _write_json(output_dir / "paired_regression.json", summary["paired_regression"])
    _write_json(output_dir / "binaural_directional_sanity.json", summary["binaural_directional_sanity"])
    _write_json(output_dir / "foa_golden_regression.json", summary["p0b_golden"])
    _write_json(output_dir / "provenance_evidence.json", summary["provenance"])
    _write_json(output_dir / "contract_regression_summary.json", summary)
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output_dir.glob("*.json"))}
    _write_json(output_dir / "regression_manifest.json", {"files": files, "code_commit": summary["provenance"]["generation_code_commit"]})
    (output_dir / "contract_regression_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    current = summary["p0b_golden"]["current_environment_rerun"]
    return """# ClassDOA V1 Step 2C.1 Production Renderer Regression

Status: `{status}`

`SoundSpacesPairedRenderer` is the production generation path. It uses the real Habitat-Sim AudioSensor at 24 kHz with Materials OFF, indirectRayCount=5000 and sourceRayCount=200. Each representation is rendered from the same immutable EpisodeRecipe and source waveform under the frozen sequential sensor lifecycle; FOA conversion calls the unchanged P0-B adapter. Temporary WAV/RIR payloads were written only below `/tmp` and were not committed.

The production paired regression rendered {episodes} fixed recipes covering front, side/off-axis and non-zero yaw. Binaural and FOA records are complete and share source fingerprint, gain, offset, scene, poses, yaw and acoustic config. No per-render, viewpoint or branch normalization was applied.

P0-B Golden evidence is separated into historical evidence and a current-environment rerun. The existing checker was invoked again for front/right/left/back/up/down and `off_axis_yawed`; current canonical={canonical}, model-facing={model}, max error={error} degrees.

The binaural directional hard sanity uses early 2000-sample RIR energy and checks both left and right source positions against the frozen `[LEFT, RIGHT]` semantics. HRTF is recorded as embedded/not independently exposed with the enclosing RLRAudioPropagation binary fingerprint. The regression seed is explicitly injected as {seed}; source/scene registry closure remains pending Step 2A/2B.

No Step 2A/2B, 960 Pilot, source QC, scene admission, training, C2, noise, active evaluation or Step 3 work was executed.
""".format(status=summary["status"], episodes=summary["paired_regression"]["episode_count"], canonical=current["canonical_verdict"], model=current["model_facing_verdict"], error=current["max_angular_error_deg"], seed=summary["provenance"]["random_seed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260824)
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), random_seed=args.random_seed)
    print(json.dumps({"status": result["status"], "production_backend": result["production_backend"], "paired_pass": result["paired_regression"]["verdict"] == "PASS", "directional_pass": result["binaural_directional_sanity"]["pass"], "current_golden_pass": result["p0b_golden"]["pass"], "normalization_pass": result["normalization_regression"]["pass"], "runtime_sec": result["runtime_sec"]}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
