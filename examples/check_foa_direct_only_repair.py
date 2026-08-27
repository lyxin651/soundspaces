"""Identify native FOA channels from direct-only signed impulse responses."""

import argparse
import json
import math
import time
from pathlib import Path

import quaternion  # 必须先于 habitat_sim 导入
import habitat_sim
import numpy as np

from foa_adapter import measure_shared_direct_coefficients


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data/logs/seld_dataset_v1_preflight"
DEFAULT_FIXTURE = ROOT / "data/seld_dataset_v1_preflight/fixtures/foa_direct_only_fixture.yaml"


def make_sim(scene, sample_rate, indirect=True, indirect_rays=5000):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene)
    backend.load_semantic_mesh = True
    backend.enable_physics = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()]))
    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = False
    spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
    spec.channelLayout.channelCount = 4
    spec.position = [0.0, 0.0, 0.0]
    spec.acousticsConfig.sampleRate = sample_rate
    spec.acousticsConfig.direct = True
    spec.acousticsConfig.indirect = indirect
    spec.acousticsConfig.diffraction = bool(indirect)
    spec.acousticsConfig.transmission = bool(indirect)
    spec.acousticsConfig.indirectRayCount = indirect_rays
    sim.add_sensor(spec)
    return sim


def set_pose(sim, source, listener, yaw_deg):
    sensor = sim.get_agent(0)._sensors["audio_sensor"]
    sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
    state = sim.get_agent(0).get_state()
    state.position = np.asarray(listener, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(np.array([0.0, math.radians(yaw_deg), 0.0]))
    state.sensor_states = {}
    sim.get_agent(0).set_state(state, True)


def render(sim):
    value = np.asarray(sim.get_sensor_observations()["audio_sensor"])
    if value.shape[0] != 4:
        value = value.T
    return value.astype(np.float32)


def direct_measurement(ir):
    measured = measure_shared_direct_coefficients(ir)
    return {"peak_sample": measured["direct_sample"], "signed_peak": measured["signed_coefficients"].tolist(), "window_energy": measured["window_energy"].tolist()}


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values) / np.mean(values)) if np.mean(values) else None


def identify_channels(rows):
    import itertools

    directions = np.asarray([item["direction_xyz"] for item in rows], dtype=np.float64)
    measurements = np.asarray([item["measurement"]["signed_peak"] for item in rows], dtype=np.float64)
    hypotheses = []
    for w_channel in range(4):
        directional_channels = [channel for channel in range(4) if channel != w_channel]
        for permutation in itertools.permutations(directional_channels):
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                errors = []
                for direction, measurement in zip(directions, measurements):
                    if abs(measurement[w_channel]) <= 1e-8:
                        continue
                    observed = measurement / measurement[w_channel]
                    predicted = np.zeros(4, dtype=np.float64)
                    predicted[w_channel] = 1.0
                    for axis, channel, sign in zip(range(3), permutation, signs):
                        predicted[channel] = np.sqrt(3.0) * sign * direction[axis]
                    errors.append(float(np.mean((observed - predicted) ** 2)))
                hypotheses.append({"order": ["W" if i == w_channel else None for i in range(4)], "w_channel": w_channel, "permutation": list(permutation), "signs": list(signs), "error": float(np.mean(errors)) if errors else float("inf")})
    hypotheses.sort(key=lambda item: item["error"])
    best, second = hypotheses[0], hypotheses[1]
    names = ["X", "Y", "Z"]
    def format_order(hypothesis):
        order = [None] * 4
        order[hypothesis["w_channel"]] = "W"
        for axis, channel in enumerate(hypothesis["permutation"]):
            order[channel] = names[axis]
        return order

    native_order = format_order(best)
    return {"best_order": native_order, "best_sign": ["+" if sign > 0 else "-" for sign in best["signs"]], "best_error": best["error"], "second_best_order": format_order(second), "second_best_sign": ["+" if sign > 0 else "-" for sign in second["signs"]], "second_best_error": second["error"], "confidence_margin": second["error"] - best["error"], "hypothesis_count": len(hypotheses), "verdict": "PASS" if native_order == ["W", "Y", "Z", "X"] and best["error"] < second["error"] else "INCONCLUSIVE"}


def doa_error_deg(native_ir, expected):
    measurement = direct_measurement(native_ir)["signed_peak"]
    w, y, z, x = [float(v) for v in measurement]
    inferred = np.asarray([x, y, z], dtype=np.float64) / max(abs(w) * np.sqrt(3.0), 1e-12)
    inferred /= max(np.linalg.norm(inferred), 1e-12)
    expected = np.asarray(expected, dtype=np.float64)
    expected /= max(np.linalg.norm(expected), 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(inferred, expected), -1.0, 1.0))))


def run_stochasticity(scene, fixture, listener, sr, indirect, rays, output_name):
    rows = []
    for item_id in ("front", "right"):
        item = next(x for x in fixture["sources"] if x["id"] == item_id)
        sim = make_sim(scene, sr, indirect=indirect, indirect_rays=rays)
        energies, levels, doa = [], [], []
        started = time.perf_counter()
        try:
            # 固定 pose 后连续读取 observation，避免把 native context 重建
            # 误差混入声学 Monte-Carlo 方差，也避免重复更新触发资源膨胀。
            set_pose(sim, item["position_world"], listener, item["listener_yaw_deg"])
            for _ in range(10):
                ir = render(sim)
                measure = direct_measurement(ir)
                energies.append(float(np.sum(ir * ir)))
                levels.append(10.0 * np.log10(max(np.sum(np.asarray(measure["window_energy"])), 1e-20)))
                doa.append(doa_error_deg(ir, item["direction_xyz"]))
        finally:
            sim.close()
        rows.append({"id": item_id, "repeats": 10, "direct_level_std_db": float(np.std(levels)), "doa_std_deg": float(np.std(doa)), "mean_doa_error_deg": float(np.mean(doa)), "full_energy_std_db": float(np.std(10.0 * np.log10(np.maximum(energies, 1e-20)))), "energy_cv": coefficient_of_variation(energies), "sec_per_render": (time.perf_counter() - started) / 10.0})
    output = {"condition": "full_acoustic" if indirect else "direct_only", "indirectRayCount": rays, "sourceRayCount": 200, "rows": rows}
    (LOG_ROOT / output_name).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def run(fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    scene = ROOT / fixture["scene"]
    listener = np.asarray(fixture["listener_position_world"], dtype=np.float32)
    sr = int(fixture["sample_rate_hz"])
    result = {"fixture": str(fixture_path), "sample_rate_hz": sr, "configuration": {"direct": True, "indirect": False, "diffraction": False, "transmission": False, "materials": False}}
    sim = make_sim(scene, sr, indirect=False)
    rows = []
    try:
        for item in fixture["sources"]:
            item_listener = np.asarray(item.get("listener_position_world", fixture["listener_position_world"]), dtype=np.float32)
            set_pose(sim, item["position_world"], item_listener, item["listener_yaw_deg"])
            ir = render(sim)
            measure = direct_measurement(ir)
            rows.append({"id": item["id"], "direction_xyz": item["direction_xyz"], "shape": list(ir.shape), "dtype": str(ir.dtype), "finite": bool(np.isfinite(ir).all()), "nonzero": bool(np.any(ir)), "measurement": measure})
    finally:
        sim.close()

    # Repeat direct-only poses to separate deterministic direct path from full-tail variance.
    repeat_rows = []
    for item_id in ("front", "right"):
        item = next(x for x in fixture["sources"] if x["id"] == item_id)
        repeat_sim = make_sim(scene, sr, indirect=False)
        measurements = []
        started = time.perf_counter()
        try:
            for _ in range(10):
                set_pose(repeat_sim, item["position_world"], listener, item["listener_yaw_deg"])
                measurements.append(direct_measurement(render(repeat_sim)))
        finally:
            repeat_sim.close()
        signed = np.asarray([x["signed_peak"] for x in measurements], dtype=np.float64)
        levels = 10.0 * np.log10(np.maximum(np.sum(signed * signed, axis=1), 1e-20))
        repeat_rows.append({"id": item_id, "repeats": 10, "peak_sample_std": float(np.std(np.asarray([x["peak_sample"] for x in measurements], dtype=np.float64), axis=0).max()), "direct_level_std_db": float(np.std(levels)), "direct_energy_cv": coefficient_of_variation(np.sum(signed * signed, axis=1)), "sec_per_render": (time.perf_counter() - started) / 10.0})

    result["directions"] = rows
    result["direct_only_repeats"] = repeat_rows
    result["identification"] = identify_channels(rows)
    result["render_verdict"] = "PASS" if all(row["shape"][0] == 4 and row["finite"] and row["nonzero"] for row in rows) else "FAIL"
    direct_result = {"condition": "direct_only", "indirectRayCount": 0, "sourceRayCount": 200, "rows": repeat_rows}
    (LOG_ROOT / "stochasticity_direct_only.json").write_text(json.dumps(direct_result, indent=2), encoding="utf-8")
    # 先落盘 direct-only 与 mapping，避免 full acoustic native failure 丢失已完成证据。
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "foa_direct_only_identification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    full_result = run_stochasticity(scene, fixture, listener, sr, indirect=True, rays=5000, output_name="stochasticity_full.json")
    (LOG_ROOT / "stochasticity_direct_vs_full.json").write_text(json.dumps({"direct_only": direct_result, "full_acoustic": full_result}, indent=2), encoding="utf-8")
    (LOG_ROOT / "foa_direct_only_identification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["identification"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    run(parser.parse_args().fixture)
