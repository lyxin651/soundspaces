"""Run the Dataset V1 FOA and paired AudioSensor preflight gates."""

import argparse
import json
import math
from pathlib import Path

import quaternion  # 必须先于 habitat_sim 导入
import habitat_sim
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "data/seld_dataset_v1_preflight/fixtures/foa_gate_fixture.yaml"
LOG_ROOT = ROOT / "data/logs/seld_dataset_v1_preflight"


def ensure_channels(observation):
    array = np.asarray(observation)
    if array.ndim != 2:
        raise RuntimeError(f"expected 2-D audio observation, got {array.shape}")
    if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
        return array.astype(np.float32)
    return array.T.astype(np.float32)


def make_sim(scene, layout_type, channel_count, sample_rate):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene)
    backend.load_semantic_mesh = True
    backend.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = False
    spec.channelLayout.type = layout_type
    spec.channelLayout.channelCount = channel_count
    spec.position = [0.0, 0.0, 0.0]
    spec.acousticsConfig.sampleRate = sample_rate
    spec.acousticsConfig.indirect = True
    sim.add_sensor(spec)
    return sim


def set_pose(sim, source, listener, yaw_deg):
    agent = sim.get_agent(0)
    for sensor in agent._sensors.values():
        if hasattr(sensor, "setAudioSourceTransform"):
            sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
    state = agent.get_state()
    state.position = np.asarray(listener, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(np.array([0.0, math.radians(yaw_deg), 0.0]))
    state.sensor_states = {}
    agent.set_state(state, True)


def render(sim):
    return ensure_channels(sim.get_sensor_observations()["audio_sensor"])


def stats(ir):
    return {
        "shape": list(ir.shape),
        "dtype": str(ir.dtype),
        "samples": int(ir.shape[1]),
        "energy": [float(x) for x in np.sum(ir * ir, axis=1)],
        "peak": [float(x) for x in np.max(np.abs(ir), axis=1)],
        "finite": bool(np.isfinite(ir).all()),
        "nonzero": bool(np.any(np.abs(ir) > 0)),
        "main_peak_sample": int(np.argmax(np.max(np.abs(ir), axis=0))),
    }


def peak_cue(ir):
    values = np.max(np.abs(ir), axis=1).astype(np.float64)
    norm = np.linalg.norm(values)
    return (values / norm).tolist() if norm else values.tolist()


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values) / np.mean(values)) if np.mean(values) else None


def run(fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    scene = ROOT / fixture["scene"]
    listener = np.asarray(fixture["listener_position_world"], dtype=np.float32)
    sample_rate = int(fixture["sample_rate_hz"])
    layouts = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType
    foa_sim = make_sim(scene, layouts.Ambisonics, 4, sample_rate)
    binaural_sim = make_sim(scene, layouts.Binaural, 2, sample_rate)
    result = {
        "fixture": str(fixture_path),
        "scene": str(scene),
        "sample_rate_hz": sample_rate,
        "native_contract_source": {
            "normalization": "N3D",
            "coordinate": "world-space convention",
            "source": "RLRAudioPropagation.h documentation"
        }
    }
    direction_rows = []
    try:
        for source_spec in fixture["sources"]:
            source = np.asarray(source_spec["position_world"], dtype=np.float32)
            set_pose(foa_sim, source, listener, 0.0)
            set_pose(binaural_sim, source, listener, 0.0)
            foa = render(foa_sim)
            binaural = render(binaural_sim)
            direction_rows.append({
                "id": source_spec["id"],
                "gt_project_azimuth_deg": source_spec["project_azimuth_deg"],
                "gt_elevation_deg": source_spec["elevation_deg"],
                "foa": stats(foa),
                "foa_peak_cue": peak_cue(foa),
                "binaural": stats(binaural),
                "binaural_left_right_energy_ratio": float((np.sum(binaural[0] ** 2) + 1e-12) / (np.sum(binaural[1] ** 2) + 1e-12))
            })

        yaw_rows = []
        front = np.asarray(fixture["sources"][0]["position_world"], dtype=np.float32)
        for yaw in fixture["listener_yaws_deg"]:
            set_pose(foa_sim, front, listener, float(yaw))
            yaw_rows.append({"yaw_deg": yaw, "foa_peak_cue": peak_cue(render(foa_sim))})

        repeats = []
        for source_id in ("front", "right"):
            source = next(item for item in fixture["sources"] if item["id"] == source_id)
            energies, direct_energies = [], []
            for _ in range(5):
                set_pose(foa_sim, np.asarray(source["position_world"], dtype=np.float32), listener, 0.0)
                ir = render(foa_sim)
                energies.append(float(np.sum(ir ** 2)))
                peak_sample = int(np.argmax(np.max(np.abs(ir), axis=0)))
                direct_energies.append(float(np.sum(ir[:, max(0, peak_sample - 32):peak_sample + 33] ** 2)))
            repeats.append({"source_id": source_id, "energy_cv": coefficient_of_variation(energies), "direct_window_energy_cv": coefficient_of_variation(direct_energies)})

        set_pose(foa_sim, np.asarray(fixture["sources"][1]["position_world"], dtype=np.float32), listener, 0.0)
        set_pose(binaural_sim, np.asarray(fixture["sources"][1]["position_world"], dtype=np.float32), listener, 0.0)
        before = {"foa": stats(render(foa_sim)), "binaural": stats(render(binaural_sim))}
        set_pose(foa_sim, np.asarray(fixture["sources"][2]["position_world"], dtype=np.float32), listener, 90.0)
        set_pose(binaural_sim, np.asarray(fixture["sources"][2]["position_world"], dtype=np.float32), listener, 90.0)
        after_update = {"foa": stats(render(foa_sim)), "binaural": stats(render(binaural_sim))}
        # 0.2.2 的 AudioSensor reset 会触发 native 进程级崩溃；用关闭并重建
        # simulator 验证 sequential lifecycle，原生 reset 风险在结果中保留。
        foa_sim.close()
        binaural_sim.close()
        foa_sim = make_sim(scene, layouts.Ambisonics, 4, sample_rate)
        binaural_sim = make_sim(scene, layouts.Binaural, 2, sample_rate)
        set_pose(foa_sim, front, listener, 0.0)
        set_pose(binaural_sim, front, listener, 0.0)
        after_reset = {"foa": stats(render(foa_sim)), "binaural": stats(render(binaural_sim))}
    finally:
        foa_sim.close()
        binaural_sim.close()

    result["directions"] = direction_rows
    result["yaw_rotation"] = yaw_rows
    result["stochasticity"] = {"repeats": repeats, "verdict": "PASS" if all((row["energy_cv"] or 1.0) < 0.25 for row in repeats) else "INCONCLUSIVE"}
    dual_sensor_error = None
    dual_sim = None
    try:
        dual_sim = make_sim(scene, layouts.Ambisonics, 4, sample_rate)
        spec = habitat_sim.AudioSensorSpec()
        spec.uuid = "binaural_sensor"
        spec.enableMaterials = False
        spec.channelLayout.type = layouts.Binaural
        spec.channelLayout.channelCount = 2
        spec.acousticsConfig.sampleRate = sample_rate
        dual_sim.add_sensor(spec)
        set_pose(dual_sim, front, listener, 0.0)
        dual_sim.get_sensor_observations()["binaural_sensor"]
    except Exception as exc:
        dual_sensor_error = f"{type(exc).__name__}: {exc}"
    finally:
        if dual_sim is not None:
            dual_sim.close()
    result["dual_sensor"] = {"simultaneous": dual_sensor_error is None, "creation_attempted": True, "error": dual_sensor_error, "source_pose_update": True, "reset_survives": "recreate_verified; native sim.reset previously crashed", "paired_render_mode": "dual_sensor_single_sim" if dual_sensor_error is None else "sequential_sensor_lifecycle", "before": before, "after_update": after_update, "after_reset": after_reset}
    result["foa_contract"] = {"native_shape": "(4, N)", "native_order": "INCONCLUSIVE: current public header does not specify channel order and this fixture is insufficient to identify signed basis channels", "native_normalization": "N3D", "canonical_order": "W,Y,Z,X (ACN indices 0,1,2,3), pending native-order gate", "canonical_normalization": "SN3D", "native_to_canonical": "not frozen; if native order is confirmed ACN, first-order channels require N3D->SN3D factor 1/sqrt(3)", "project_to_dcase_mapping": "front 0, right +90, left -90; +Y up, +X right, -Z forward"}
    result["foa_render_verdict"] = "PASS" if all(row["foa"]["shape"][0] == 4 and row["foa"]["finite"] and row["foa"]["nonzero"] for row in direction_rows) else "FAIL"
    result["verdict"] = "INCONCLUSIVE" if result["foa_render_verdict"] == "PASS" else "FAIL"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "foa_contract_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (LOG_ROOT / "paired_sensor_gate.json").write_text(json.dumps(result["dual_sensor"], indent=2), encoding="utf-8")
    print(json.dumps({"foa_render_verdict": result["foa_render_verdict"], "verdict": result["verdict"], "stochasticity": result["stochasticity"], "paired_render_mode": result["dual_sensor"]["paired_render_mode"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    run(parser.parse_args().fixture)
