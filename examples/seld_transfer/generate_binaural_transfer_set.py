"""生成 SoundSpaces 双耳 SELD 迁移测试集。

必须在 sound-spaces 根目录、ss 环境下运行。AudioSensor 输出 IR；本脚本
使用同一个 dry segment 与每个 IR 卷积，再统一重采样到 24 kHz/5 s。
"""

import argparse
import csv
import json
import math
from pathlib import Path

import quaternion  # 必须先于 habitat_sim 导入
import habitat_sim
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly


CLASS_MAP = {
    0: "Female speech", 1: "Male speech", 2: "Clapping", 3: "Telephone",
    4: "Laughter", 5: "Domestic sounds", 6: "Footsteps", 7: "Door",
    8: "Music", 9: "Musical instrument", 10: "Water tap", 11: "Bell", 12: "Knock",
}
TARGET_RELATIVE_AZIMUTHS = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]


def ensure_samples_channels(signal):
    signal = np.asarray(signal)
    if signal.ndim == 2 and signal.shape[0] == 2 and signal.shape[1] != 2:
        signal = signal.T
    if signal.ndim == 1:
        signal = signal[:, None]
    return signal


def read_mono(path, target_rate):
    sample_rate, signal = wavfile.read(str(path))
    signal = np.asarray(signal)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
    else:
        signal = signal.astype(np.float32)
    if sample_rate != target_rate:
        signal = resample_poly(signal, target_rate, sample_rate).astype(np.float32)
    return signal, sample_rate


def pad_or_crop(signal, samples):
    result = np.zeros(samples, dtype=np.float32)
    result[: min(samples, signal.shape[0])] = signal[:samples]
    return result


def fold_azimuth(azimuth):
    if azimuth > 90.0:
        return 180.0 - azimuth
    if azimuth < -90.0:
        return -180.0 - azimuth
    return azimuth


def relative_azimuth(source, listener, yaw_deg):
    delta = np.asarray(source, dtype=np.float64) - np.asarray(listener, dtype=np.float64)
    delta[1] = 0.0
    yaw = math.radians(yaw_deg)
    # Habitat forward at yaw=0 is -Z; inverse yaw maps world delta to listener frame.
    local_x = math.cos(yaw) * delta[0] - math.sin(yaw) * delta[2]
    local_z = math.sin(yaw) * delta[0] + math.cos(yaw) * delta[2]
    return math.degrees(math.atan2(local_x, -local_z))


def make_sim(scene, source, agent_position, yaw, materials, materials_json, sample_rate):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    sensor_spec = habitat_sim.AudioSensorSpec()
    sensor_spec.uuid = "audio_sensor"
    sensor_spec.enableMaterials = materials
    sensor_spec.channelLayout.type = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    sensor_spec.channelLayout.channelCount = 2
    sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_spec.acousticsConfig.sampleRate = sample_rate
    sensor_spec.acousticsConfig.indirect = True
    sim.add_sensor(sensor_spec)
    sensor = sim.get_agent(0)._sensors["audio_sensor"]
    if materials:
        sensor.setAudioMaterialsJSON(materials_json)
    set_pose(sim, source, agent_position, yaw)
    return sim


def set_pose(sim, source, agent_position, yaw):
    sensor = sim.get_agent(0)._sensors["audio_sensor"]
    sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
    state = sim.get_agent(0).get_state()
    state.position = np.asarray(agent_position, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(np.array([0.0, math.radians(yaw), 0.0]))
    state.sensor_states = {}
    sim.get_agent(0).set_state(state, True)


def render_ir(sim):
    ir = ensure_samples_channels(sim.get_sensor_observations()["audio_sensor"])
    if ir.shape[1] != 2:
        raise RuntimeError(f"expected binaural IR, got shape={ir.shape}")
    return ir.astype(np.float32)


def write_audio(path, sample_rate, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sample_rate, audio.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="data/scene_datasets/replica_compat/office_0/habitat/mesh_semantic.ply")
    parser.add_argument("--dry-file", default="data/sounds/telephone.wav")
    parser.add_argument("--materials-json", default="data/replica_material_config.json")
    parser.add_argument("--materials", action="store_true")
    parser.add_argument("--output-root", default="data/experiments/seld_transfer_gate_20260824")
    parser.add_argument("--source", type=float, nargs=3, default=[0.1810176075, -0.9688690305, 2.44338727])
    parser.add_argument("--agent", type=float, nargs=3, default=[1.6240532398, -0.9688690305, -0.5125486851])
    parser.add_argument("--position", type=float, nargs=3, action="append", default=[])
    parser.add_argument("--dry-start-s", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(args.output_root)
    original_dir = root / "wav_original"
    model_dir = root / "wav_24k_5s"
    manifest_path = root / "manifest.csv"
    manifest_json_path = root / "manifest.json"
    render_rate = 16000
    model_rate = 24000
    duration_s = 5.0
    source = np.asarray(args.source, dtype=np.float32)
    agent = np.asarray(args.agent, dtype=np.float32)
    positions = [agent] + [np.asarray(value, dtype=np.float32) for value in args.position]
    if not args.position:
        positions.append(np.asarray([1.3441598415, -0.9688690305, 0.7292646170], dtype=np.float32))

    dry, original_dry_rate = read_mono(Path(args.dry_file), render_rate)
    start = int(round(args.dry_start_s * render_rate))
    dry = dry[start:]
    dry = pad_or_crop(dry, int(duration_s * render_rate))
    rows = []
    scene_name = Path(args.scene).parts[-3] if len(Path(args.scene).parts) >= 3 else Path(args.scene).stem
    sim = make_sim(args.scene, source + np.array([0.0, 1.5, 0.0]), agent, 0.0, args.materials, args.materials_json, render_rate)
    try:
        for listener_index, (sweep, listener) in enumerate(
            [("yaw", agent)] + [("position", pos) for pos in positions]
        ):
            listener_audio = listener + np.array([0.0, 1.5, 0.0])
            base_azimuth = relative_azimuth(
                source + np.array([0.0, 1.5, 0.0]), listener_audio, 0.0
            )
            target_azimuths = TARGET_RELATIVE_AZIMUTHS if sweep == "yaw" else [base_azimuth]
            for index, target_azimuth in enumerate(target_azimuths):
                # For this pose convention, relative azimuth changes as base+yaw.
                yaw = target_azimuth - base_azimuth
                sample_id = f"{scene_name}_telephone_{sweep}_{listener_index:02d}_{index:03d}"
                set_pose(sim, source + np.array([0.0, 1.5, 0.0]), listener, yaw)
                ir = render_ir(sim)
                convolved = np.stack([
                    fftconvolve(dry, ir[:, channel], mode="full")
                    for channel in range(2)
                ], axis=1).astype(np.float32)
                original_path = original_dir / f"{sample_id}.wav"
                write_audio(original_path, render_rate, convolved)
                model_audio = resample_poly(convolved, model_rate, render_rate).astype(np.float32)
                model_audio = model_audio[: int(duration_s * model_rate)]
                if model_audio.shape[0] < int(duration_s * model_rate):
                    model_audio = np.pad(model_audio, ((0, int(duration_s * model_rate) - model_audio.shape[0]), (0, 0)))
                peak = float(np.max(np.abs(model_audio)))
                gain = min(1.0, 0.98 / peak) if peak > 0.98 else 1.0
                model_audio *= gain
                model_path = model_dir / f"{sample_id}.wav"
                write_audio(model_path, model_rate, model_audio)
                full_azimuth = relative_azimuth(source + np.array([0.0, 1.5, 0.0]), listener_audio, yaw)
                distance = float(np.linalg.norm((source + np.array([0.0, 1.5, 0.0])) - listener_audio))
                rows.append({
                    "sample_id": sample_id, "scene": scene_name, "dry_file": args.dry_file,
                    "class_id": 3, "class_name": CLASS_MAP[3], "sweep": sweep,
                    "source_world_xyz": json.dumps([float(x) for x in source + [0.0, 1.5, 0.0]]),
                    "agent_world_xyz": json.dumps([float(x) for x in listener]),
                    "listener_world_xyz": json.dumps([float(x) for x in listener_audio]),
                    "agent_yaw_deg": yaw, "gt_azimuth_full_deg": full_azimuth,
                    "gt_azimuth_folded_deg": fold_azimuth(full_azimuth),
                    "source_listener_distance_m": distance, "materials_on": args.materials,
                    "rir_shape": json.dumps(list(ir.shape)), "render_sample_rate": render_rate,
                    "output_sample_rate": model_rate, "clip_seconds": duration_s,
                    "input_shape": json.dumps(list(model_audio.shape)), "input_rms": float(np.sqrt(np.mean(model_audio ** 2))),
                    "input_peak": float(np.max(np.abs(model_audio))), "input_clipping_count": int(np.sum(np.abs(model_audio) > 1.0)),
                    "peak_guard_gain": gain, "wav_original": str(original_path), "wav_path": str(model_path),
                })
                print(f"rendered {len(rows):02d}: {sample_id} az={full_azimuth:.1f} ir={ir.shape}")
    finally:
        sim.close()

    root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_json_path.write_text(json.dumps({"class_map": CLASS_MAP, "rows": rows}, indent=2), encoding="utf-8")
    print(f"manifest rows: {len(rows)} -> {manifest_path}")


if __name__ == "__main__":
    main()
