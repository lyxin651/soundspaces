import csv
import argparse
from pathlib import Path

import quaternion
import habitat_sim
import habitat_sim.sim
import numpy as np
from scipy.io import wavfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render binaural audio from a given source pose and listener pose in SoundSpaces."
    )

    parser.add_argument(
        "--scene",
        type=str,
        default="data/scene_datasets/habitat-test-scenes/apartment_1.glb",
        help="Path to the scene .glb file.",
    )

    parser.add_argument(
        "--source",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Source position in world coordinates: x y z.",
    )

    parser.add_argument(
        "--agent",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Agent base position in world coordinates: x y z.",
    )

    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="Agent yaw angle in degrees. Rotation is around the vertical Y axis.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output wav path.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path to append one row of render statistics.",
    )

    return parser.parse_args()


def ensure_audio_shape(obs):
    """
    Convert audio observation to shape (samples, channels) if needed.
    Some Habitat-Sim versions return (channels, samples), e.g. (2, N).
    scipy.io.wavfile.write expects (samples, channels), e.g. (N, 2).
    """
    obs = np.array(obs)

    if obs.ndim == 2 and obs.shape[0] == 2 and obs.shape[1] != 2:
        obs = obs.T

    return obs


def compute_energy_stats(wav):
    """
    Compute simple energy statistics for mono or binaural audio.
    """
    if wav.ndim == 1:
        total_energy = float(np.mean(wav ** 2))
        return {
            "channels": 1,
            "left_energy": None,
            "right_energy": None,
            "total_energy": total_energy,
            "lr_ratio": None,
        }

    left_energy = float(np.mean(wav[:, 0] ** 2))
    right_energy = float(np.mean(wav[:, 1] ** 2))
    total_energy = float(np.mean(wav ** 2))
    lr_ratio = left_energy / (right_energy + 1e-12)

    return {
        "channels": wav.shape[1],
        "left_energy": left_energy,
        "right_energy": right_energy,
        "total_energy": total_energy,
        "lr_ratio": lr_ratio,
    }


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_position = np.array(args.source, dtype=np.float32)
    agent_position = np.array(args.agent, dtype=np.float32)

    # The audio sensor is placed 1.5m above the agent base position.
    sensor_relative_position = np.array([0.0, 1.5, 0.0], dtype=np.float32)
    listener_position = agent_position + sensor_relative_position

    distance = float(np.linalg.norm(source_position - listener_position))

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = args.scene
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"

    # Minimal working setting for habitat-test-scenes.
    audio_sensor_spec.enableMaterials = False

    # Binaural audio: left ear + right ear.
    audio_sensor_spec.channelLayout.type = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    audio_sensor_spec.channelLayout.channelCount = 2

    audio_sensor_spec.position = sensor_relative_position.tolist()
    audio_sensor_spec.acousticsConfig.sampleRate = args.sample_rate
    audio_sensor_spec.acousticsConfig.indirect = True

    sim.add_sensor(audio_sensor_spec)

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(source_position)

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = agent_position

    # Set yaw around vertical Y axis.
    state.rotation = quaternion.from_rotation_vector(
        np.array([0.0, np.deg2rad(args.yaw), 0.0])
    )

    state.sensor_states = {}
    agent.set_state(state, True)

    obs = sim.get_sensor_observations()["audio_sensor"]
    wav = ensure_audio_shape(obs)

    wavfile.write(str(output_path), args.sample_rate, wav)

    stats = compute_energy_stats(wav)

    print("=== Render Audio From Pose ===")
    print(f"scene: {args.scene}")
    print(f"source position: {source_position.tolist()}")
    print(f"agent base position: {agent_position.tolist()}")
    print(f"listener/audio sensor position: {listener_position.tolist()}")
    print(f"yaw degrees: {args.yaw}")
    print(f"source-listener distance: {distance:.4f} m")
    print(f"output wav: {output_path}")
    print(f"wav shape: {wav.shape}")
    print(f"sample rate: {args.sample_rate}")
    print(f"channels: {stats['channels']}")
    print(f"total energy: {stats['total_energy']:.8e}")

    if stats["channels"] == 2:
        print(f"left energy: {stats['left_energy']:.8e}")
        print(f"right energy: {stats['right_energy']:.8e}")
        print(f"L/R: {stats['lr_ratio']:.4f}")

    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "scene": args.scene,
            "source_x": float(source_position[0]),
            "source_y": float(source_position[1]),
            "source_z": float(source_position[2]),
            "agent_x": float(agent_position[0]),
            "agent_y": float(agent_position[1]),
            "agent_z": float(agent_position[2]),
            "listener_x": float(listener_position[0]),
            "listener_y": float(listener_position[1]),
            "listener_z": float(listener_position[2]),
            "yaw": float(args.yaw),
            "distance": distance,
            "wav_shape": str(wav.shape),
            "sample_rate": args.sample_rate,
            "channels": stats["channels"],
            "left_energy": stats["left_energy"],
            "right_energy": stats["right_energy"],
            "total_energy": stats["total_energy"],
            "lr_ratio": stats["lr_ratio"],
            "output_wav": str(output_path),
        }

        file_exists = csv_path.exists()

        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    sim.close()


if __name__ == "__main__":
    main()
