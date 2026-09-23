#!/usr/bin/env python3
"""M1-M3 discrete navigation toward a sound source.

The script intentionally keeps the policy independent from ground-truth
distance.  Distance and shortest paths are used only for sampling and
evaluation.  Run from the sound-spaces repository root.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import quaternion  # Must be imported before habitat_sim.
import habitat_sim
import habitat_sim.sim
import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.stats import spearmanr


DEFAULT_SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
DEFAULT_SOURCE = (-8.56, 1.5, 0.50)
SENSOR_OFFSET = np.array([0.0, 1.5, 0.0], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the SoundSpaces M1-M3 sound-directed navigation loop."
    )
    parser.add_argument(
        "--stage", choices=("mapping", "audit", "m1", "m2", "m3", "all"), required=True
    )
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--source", type=float, nargs=3, default=DEFAULT_SOURCE)
    parser.add_argument("--materials-json", default="data/mp3d_material_config.json")
    parser.add_argument("--enable-materials", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--source-audio", default="data/sounds/telephone.wav")
    parser.add_argument("--output-root", default="data/nav_experiments")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--steps", "--m1-steps", dest="m1_steps", type=int, default=20)
    parser.add_argument("--m2-starts", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--step-time", type=float, default=0.5)
    parser.add_argument("--forward-step", type=float, default=0.25)
    parser.add_argument("--turn-angle", type=float, default=10.0)
    parser.add_argument("--indirect-ray-count", type=int, default=200)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--lr-threshold", type=float, default=1.5)
    parser.add_argument("--policy-version", choices=("v0", "v2"), default="v0")
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--stop-confirmation-frames", type=int, default=3)
    parser.add_argument("--turn-hold-steps", type=int, default=3)
    parser.add_argument("--stop-threshold", type=float, default=None)
    parser.add_argument("--goal-radius", type=float, default=0.35)
    parser.add_argument("--waypoint-radius", type=float, default=0.45)
    parser.add_argument("--position-epsilon", type=float, default=1e-4)
    parser.add_argument("--min-start-distance", type=float, default=2.0)
    parser.add_argument("--calibration-min", type=float, default=0.8)
    parser.add_argument("--calibration-max", type=float, default=1.2)
    parser.add_argument("--calibration-points", type=int, default=5)
    parser.add_argument("--calibration-yaws", type=int, default=4)
    parser.add_argument("--stuck-window", type=int, default=10)
    parser.add_argument("--max-sample-attempts", type=int, default=10000)
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    ensure_dir(Path(path).parent)
    with Path(path).open("w") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    ensure_dir(Path(path).parent)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(path, x, ys, labels, xlabel, ylabel, title):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; CSV output was kept without a plot")
        return False
    ensure_dir(Path(path).parent)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for values, label in zip(ys, labels):
        axis.plot(x, values, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    if labels:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


def ensure_audio(path, sample_rate):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"source audio not found: {path}; download the official sound package "
            "or pass --source-audio to a 16 kHz mono WAV"
        )
    rate, signal = wavfile.read(str(path))
    signal = np.asarray(signal)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float32)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal / max(peak, 1.0)
    else:
        signal = signal / max(peak, 1.0)
    if rate != sample_rate:
        signal = librosa.resample(signal, orig_sr=rate, target_sr=sample_rate)
        rate = sample_rate
    if rate != sample_rate or signal.size == 0:
        raise ValueError("source audio must contain a non-empty signal")
    return signal.astype(np.float32), rate


def normalize_ir(observation):
    ir = np.asarray(observation)
    if ir.ndim != 2:
        raise ValueError(f"IR must be rank 2, got shape {ir.shape}")
    if ir.shape[0] == 2 and ir.shape[1] != 2:
        ir = ir.T
    if ir.shape[1] != 2:
        raise ValueError(f"IR must be binaural with shape (samples, 2), got {ir.shape}")
    if not np.isfinite(ir).all():
        raise ValueError("IR contains non-finite values")
    return ir.astype(np.float32, copy=False)


def ir_stats(ir):
    left = float(np.mean(ir[:, 0] ** 2))
    right = float(np.mean(ir[:, 1] ** 2))
    return {
        "left_energy": left,
        "right_energy": right,
        "total_energy": float(np.mean(ir ** 2)),
        "lr_ratio": left / (right + 1e-12),
        "ir_samples": int(ir.shape[0]),
        "ir_channels": int(ir.shape[1]),
    }


def listener_position(agent_position):
    return np.asarray(agent_position, dtype=np.float32) + SENSOR_OFFSET


def listener_distance(agent_position, source_position):
    return float(np.linalg.norm(listener_position(agent_position) - source_position))


def make_sim(args):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = args.scene
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.action_space = {
        "stop": habitat_sim.ActionSpec("stop"),
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=args.forward_step)
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=args.turn_angle)
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=args.turn_angle)
        ),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    sensor_spec = habitat_sim.AudioSensorSpec()
    sensor_spec.uuid = "audio_sensor"
    sensor_spec.enableMaterials = bool(args.enable_materials)
    sensor_spec.channelLayout.type = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    sensor_spec.channelLayout.channelCount = 2
    sensor_spec.position = SENSOR_OFFSET.tolist()
    sensor_spec.acousticsConfig.sampleRate = args.sample_rate
    sensor_spec.acousticsConfig.indirect = True
    sensor_spec.acousticsConfig.indirectRayCount = args.indirect_ray_count
    sensor_spec.acousticsConfig.threadCount = args.thread_count
    sim.add_sensor(sensor_spec)
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    if args.enable_materials:
        audio_sensor.setAudioMaterialsJSON(args.materials_json)
    audio_sensor.setAudioSourceTransform(np.asarray(args.source, dtype=np.float32))
    return sim


def set_agent_pose(agent, position, yaw_degrees):
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(
        np.array([0.0, math.radians(yaw_degrees), 0.0], dtype=np.float64)
    )
    state.sensor_states = {}
    agent.set_state(state, True)


def current_forward(agent):
    rotation = agent.get_state().rotation
    return quaternion.as_rotation_matrix(rotation).dot(np.array([0.0, 0.0, -1.0]))


def current_yaw_degrees(agent):
    forward = current_forward(agent)
    return float(math.degrees(math.atan2(-forward[0], -forward[2])))


def signed_angle_delta_degrees(after, before):
    return float((after - before + 180.0) % 360.0 - 180.0)


def shortest_path(pathfinder, start, end):
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(end, dtype=np.float32)
    found = pathfinder.find_path(path)
    distance = float(path.geodesic_distance)
    points = [np.asarray(point, dtype=np.float32) for point in path.points]
    valid = bool(found and np.isfinite(distance) and len(points) >= 2)
    return valid, distance, points


def source_navigation_point(pathfinder, source):
    snapped = np.asarray(pathfinder.snap_point(source), dtype=np.float32)
    valid = snapped.shape == (3,) and np.isfinite(snapped).all()
    if valid:
        valid = bool(pathfinder.is_navigable(snapped))
    return snapped, valid


def resolve_source(sim, requested_source, rng, args):
    requested_source = np.asarray(requested_source, dtype=np.float32)
    snapped, valid = source_navigation_point(sim.pathfinder, requested_source)
    if valid:
        return requested_source, snapped, False
    sampled_points = []
    for _ in range(max(args.calibration_points * 100, 500)):
        point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
        if np.isfinite(point).all():
            sampled_points.append(point)
    if not sampled_points:
        raise RuntimeError("fixed source was not navigable and fallback sampling returned NaN")
    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    order = rng.permutation(len(sampled_points))
    fallback_floor = None
    source_height = float(SENSOR_OFFSET[1])
    source_height_adjusted = False
    for index in order:
        candidate = sampled_points[index]
        candidate_source = np.array([candidate[0], SENSOR_OFFSET[1], candidate[2]])
        distances = np.linalg.norm(
            listener_position(sampled_points) - candidate_source, axis=1
        )
        ring_count = int(
            np.sum((distances >= args.calibration_min) & (distances <= args.calibration_max))
        )
        if ring_count >= args.calibration_points:
            fallback_floor = candidate
            break
    if fallback_floor is None:
        for index in order:
            candidate = sampled_points[index]
            candidate_source = np.array([candidate[0], candidate[1] + SENSOR_OFFSET[1], candidate[2]])
            distances = np.linalg.norm(
                listener_position(sampled_points) - candidate_source, axis=1
            )
            ring_count = int(
                np.sum((distances >= args.calibration_min) & (distances <= args.calibration_max))
            )
            if ring_count >= args.calibration_points:
                fallback_floor = candidate
                source_height = float(candidate[1] + SENSOR_OFFSET[1])
                source_height_adjusted = True
                break
    if fallback_floor is None:
        raise RuntimeError("no fallback source had enough 3D calibration-ring points")
    # The test scene has floors below y=0.  Place a fallback source 1.5 m
    # above its floor so the <1 m listener-distance criterion is attainable.
    source_height = float(fallback_floor[1] + SENSOR_OFFSET[1])
    source_height_adjusted = not math.isclose(source_height, SENSOR_OFFSET[1], abs_tol=0.05)
    effective_source = np.array(
        [fallback_floor[0], source_height, fallback_floor[2]], dtype=np.float32
    )
    sim.get_agent(0)._sensors["audio_sensor"].setAudioSourceTransform(effective_source)
    # Keep the original sampled ground point.  Snapping the y=1.5 source can
    # select a different disconnected floor region in this test navmesh.
    args.source_height_adjusted = source_height_adjusted
    return effective_source, fallback_floor, True


def sample_start(pathfinder, source_goal, source_position, rng, args):
    for _ in range(args.max_sample_attempts):
        candidate = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if not np.isfinite(candidate).all():
            continue
        valid, distance, points = shortest_path(pathfinder, candidate, source_goal)
        direct_distance = listener_distance(candidate, source_position)
        if valid and distance >= args.min_start_distance and direct_distance >= args.min_start_distance:
            return candidate, distance, points
    raise RuntimeError(
        "could not sample a reachable start with shortest distance >= "
        f"{args.min_start_distance} m after {args.max_sample_attempts} attempts"
    )


def sample_calibration_points(pathfinder, source_goal, source_position, rng, args):
    points = []
    seen = set()
    for _ in range(args.max_sample_attempts):
        candidate = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if not np.isfinite(candidate).all():
            continue
        distance = listener_distance(candidate, source_position)
        if not (args.calibration_min <= distance <= args.calibration_max):
            continue
        valid, _, _ = shortest_path(pathfinder, candidate, source_goal)
        if not valid:
            continue
        key = tuple(np.round(candidate, 3))
        if key not in seen:
            seen.add(key)
            points.append(candidate)
        if len(points) >= args.calibration_points:
            break
    if len(points) < args.calibration_points:
        raise RuntimeError(
            f"only found {len(points)} calibration points in "
            f"[{args.calibration_min}, {args.calibration_max}] m; need {args.calibration_points}"
        )
    return points


def render_ir(sim, agent, action=None):
    before = np.asarray(agent.get_state().position, dtype=np.float32).copy()
    started = time.perf_counter()
    acted = True
    if action is not None and action != "stop":
        acted = agent.act(action)
    observations = sim.get_sensor_observations()
    elapsed = time.perf_counter() - started
    after = np.asarray(agent.get_state().position, dtype=np.float32).copy()
    ir = normalize_ir(observations["audio_sensor"])
    return ir, elapsed, bool(acted), before, after


def convolve_with_rir(source_signal, rir, sample_rate, sample_index):
    output_samples = sample_rate
    rir_samples = rir.shape[0]
    source_samples = source_signal.shape[0]
    index = int(sample_index) % source_samples
    if index - rir_samples < 0:
        sound_segment = source_signal[: index + output_samples]
        convolved = [fftconvolve(sound_segment, rir[:, channel]) for channel in range(2)]
        audio = np.asarray(
            [channel[index : index + output_samples] for channel in convolved], dtype=np.float32
        )
    else:
        indices = (np.arange(index - rir_samples + 1, index + output_samples) % source_samples)
        sound_segment = source_signal[indices]
        audio = np.asarray(
            [fftconvolve(sound_segment, rir[:, channel], mode="valid") for channel in range(2)],
            dtype=np.float32,
        )
    if audio.shape[1] < output_samples:
        audio = np.pad(audio, ((0, 0), (0, output_samples - audio.shape[1])))
    return audio[:, :output_samples]


def audio_features(audio, sample_rate):
    stats = ir_stats(audio.T)
    mel = []
    for channel in range(2):
        values = librosa.feature.melspectrogram(
            y=audio[channel],
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            power=2.0,
        )
        mel.append(np.log10(values + 1e-10).astype(np.float32))
    return stats, np.asarray(mel, dtype=np.float32)


def save_observation(output_dir, episode_id, step, audio, mel):
    ensure_dir(output_dir / "audio")
    ensure_dir(output_dir / "mel")
    np.save(output_dir / "audio" / f"episode_{episode_id:03d}_step_{step:04d}.npy", audio)
    np.save(output_dir / "mel" / f"episode_{episode_id:03d}_step_{step:04d}.npy", mel)


def m1(args, sim, source_goal, source_goal_valid, rng, output_dir):
    if not source_goal_valid:
        raise RuntimeError("source snap_point is invalid; M1 cannot start")
    agent = sim.get_agent(0)
    start, shortest_distance, _ = sample_start(
        sim.pathfinder, source_goal, np.asarray(args.source), rng, args
    )
    yaw = float(rng.uniform(0.0, 360.0))
    set_agent_pose(agent, start, yaw)
    rows = []
    stalled_forwards = 0
    steps = min(args.m1_steps, args.max_steps)
    for step in range(steps):
        action = "turn_left" if stalled_forwards >= 3 else "move_forward"
        ir, elapsed, acted, before, after = render_ir(sim, agent, action)
        moved = float(np.linalg.norm(after - before)) > args.position_epsilon
        stats = ir_stats(ir)
        rows.append(
            {
                "episode_id": 0,
                "step": step,
                "action": action,
                "acted": acted,
                "moved": moved,
                "position_x": float(after[0]),
                "position_y": float(after[1]),
                "position_z": float(after[2]),
                "render_seconds": elapsed,
                "shortest_start_distance": shortest_distance,
                **stats,
            }
        )
        if action == "move_forward":
            stalled_forwards = stalled_forwards + 1 if not moved else 0
        else:
            stalled_forwards = 0
    write_rows(output_dir / "steps.csv", rows)
    times = np.asarray([row["render_seconds"] for row in rows])
    summary = {
        "steps": len(rows),
        "render_seconds_mean": float(times.mean()),
        "render_seconds_min": float(times.min()),
        "render_seconds_max": float(times.max()),
        "moved_steps": int(sum(row["moved"] for row in rows)),
        "source_snap_point": source_goal,
        "source_snap_valid": source_goal_valid,
        "forward_step_m": args.forward_step,
        "turn_angle_degrees": args.turn_angle,
        "indirect_ray_count": args.indirect_ray_count,
    }
    write_json(output_dir / "summary.json", summary)
    print("M1 summary:", json.dumps(summary, indent=2, default=json_default))
    return summary


def navigate_to_waypoints(agent, path_points, args):
    waypoint_index = 1
    while waypoint_index < len(path_points) - 1:
        distance = np.linalg.norm(
            np.asarray(agent.get_state().position)[[0, 2]]
            - path_points[waypoint_index][[0, 2]]
        )
        if distance <= args.waypoint_radius:
            waypoint_index += 1
        else:
            break
    if waypoint_index >= len(path_points):
        return "stop", waypoint_index
    target = path_points[waypoint_index]
    position = np.asarray(agent.get_state().position)
    desired = target - position
    desired[1] = 0.0
    desired_norm = np.linalg.norm(desired)
    if desired_norm <= args.waypoint_radius:
        return "move_forward", waypoint_index
    desired = desired / desired_norm
    forward = current_forward(agent)
    forward[1] = 0.0
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    cross_y = forward[2] * desired[0] - forward[0] * desired[2]
    dot = float(np.clip(np.dot(forward, desired), -1.0, 1.0))
    angle = math.atan2(cross_y, dot)
    if abs(angle) > math.radians(args.turn_angle * 0.45):
        return ("turn_left" if angle > 0 else "turn_right"), waypoint_index
    return "move_forward", waypoint_index


def route_episode(
    sim,
    agent,
    source_signal,
    source_position,
    source_goal,
    path_points,
    args,
    episode_id,
    output_dir,
    initial_yaw,
):
    set_agent_pose(agent, path_points[0], initial_yaw)
    try:
        follower = sim.make_greedy_follower(
            0, args.goal_radius, stop_key=0, forward_key=1, left_key=2, right_key=3
        )
    except TypeError:
        follower = sim.make_greedy_follower(0, args.goal_radius)
    rows = []
    sample_index = 0
    stuck_steps = 0
    total_path = 0.0
    terminal_reason = "timeout"
    for step in range(args.max_steps):
        follower_action = follower.next_action_along(np.asarray(source_goal, dtype=np.float32))
        action_id = int(follower_action)
        action = {
            0: "stop",
            1: "move_forward",
            2: "turn_left",
            3: "turn_right",
        }.get(action_id)
        if action is None:
            raise RuntimeError(f"greedy follower returned unknown action id {action_id}")
        ir, elapsed, acted, before, after = render_ir(sim, agent, action)
        audio = convolve_with_rir(source_signal, ir, args.sample_rate, sample_index)
        features, mel = audio_features(audio, args.sample_rate)
        if episode_id < 3:
            save_observation(output_dir, episode_id, step, audio, mel)
        moved_distance = float(np.linalg.norm(after - before))
        total_path += moved_distance
        moved = moved_distance > args.position_epsilon
        distance = listener_distance(after, source_position)
        rows.append(
            {
                "episode_id": episode_id,
                "step": step,
                "action": action,
                "acted": acted,
                "moved": moved,
                "position_x": float(after[0]),
                "position_y": float(after[1]),
                "position_z": float(after[2]),
                "distance_to_source": distance,
                "render_seconds": elapsed,
                **features,
            }
        )
        sample_index += int(args.sample_rate * args.step_time)

        if action == "stop":
            terminal_reason = "waypoint_goal"
            break

        if action == "move_forward":
            stuck_steps = stuck_steps + 1 if not moved else 0
        if stuck_steps >= args.stuck_window:
            terminal_reason = "stuck"
            break

    if len(rows) >= args.max_steps:
        terminal_reason = "timeout"
    final_position = np.asarray(agent.get_state().position, dtype=np.float32)
    final_distance = listener_distance(final_position, source_position)
    return rows, {
        "terminal_reason": terminal_reason,
        "final_distance": final_distance,
        "actual_path_length": total_path,
        "steps": len(rows),
    }


def m2(args, sim, source_goal, source_goal_valid, rng, output_dir):
    if not source_goal_valid:
        raise RuntimeError("source snap_point is invalid; M2 cannot start")
    source_signal, source_rate = ensure_audio(args.source_audio, args.sample_rate)
    agent = sim.get_agent(0)
    all_rows = []
    episode_summaries = []
    for episode_id in range(args.m2_starts):
        start, shortest_distance, path_points = sample_start(
            sim.pathfinder, source_goal, np.asarray(args.source), rng, args
        )
        rows, summary = route_episode(
            sim,
            agent,
            source_signal,
            np.asarray(args.source, dtype=np.float32),
            source_goal,
            path_points,
            args,
            episode_id,
            output_dir,
            float(rng.choice([0.0, 90.0, 180.0, 270.0])),
        )
        for row in rows:
            row["shortest_start_distance"] = shortest_distance
        all_rows.extend(rows)
        summary.update({"episode_id": episode_id, "shortest_start_distance": shortest_distance})
        episode_summaries.append(summary)

    calibration_points = sample_calibration_points(
        sim.pathfinder, source_goal, np.asarray(args.source), rng, args
    )
    calibration_rows = []
    calibration_energies = []
    yaws = np.linspace(0.0, 360.0, args.calibration_yaws, endpoint=False)
    for point_id, point in enumerate(calibration_points):
        for yaw_id, yaw in enumerate(yaws):
            set_agent_pose(agent, point, float(yaw))
            ir, _, _, _, _ = render_ir(sim, agent, None)
            audio = convolve_with_rir(source_signal, ir, args.sample_rate, 0)
            features, _ = audio_features(audio, args.sample_rate)
            calibration_energies.append(features["total_energy"])
            calibration_rows.append(
                {
                    "point_id": point_id,
                    "yaw_id": yaw_id,
                    "yaw_degrees": float(yaw),
                    "position_x": float(point[0]),
                    "position_y": float(point[1]),
                    "position_z": float(point[2]),
                    "distance_to_source": listener_distance(point, np.asarray(args.source)),
                    **features,
                }
            )
    write_rows(output_dir / "observation_steps.csv", all_rows)
    write_rows(output_dir / "calibration_ring.csv", calibration_rows)
    stop_threshold = (
        float(args.stop_threshold)
        if args.stop_threshold is not None
        else float(np.median(calibration_energies))
    )
    distances = np.asarray([row["distance_to_source"] for row in all_rows], dtype=float)
    energies = np.asarray([row["total_energy"] for row in all_rows], dtype=float)
    correlation = float(spearmanr(-distances, energies).statistic) if len(all_rows) >= 3 else float("nan")
    log_lr_values = np.log(
        np.asarray([row["lr_ratio"] for row in all_rows], dtype=float) + 1e-12
    )
    lr_variation = float(np.std(log_lr_values)) if len(log_lr_values) else float("nan")
    sanity_pass = bool(
        np.isfinite(correlation)
        and correlation >= 0.1
        and np.isfinite(lr_variation)
        and lr_variation >= 0.1
    )
    summary = {
        "source_audio": args.source_audio,
        "source_audio_sample_rate": source_rate,
        "m2_episode_summaries": episode_summaries,
        "calibration_count": len(calibration_energies),
        "calibration_point_count": len(calibration_points),
        "calibration_yaw_count": len(yaws),
        "calibration_energy_min": float(np.min(calibration_energies)),
        "calibration_energy_p25": float(np.percentile(calibration_energies, 25)),
        "calibration_energy_median": float(np.median(calibration_energies)),
        "calibration_energy_p75": float(np.percentile(calibration_energies, 75)),
        "calibration_energy_max": float(np.max(calibration_energies)),
        "stop_threshold": stop_threshold,
        "distance_energy_spearman": correlation,
        "log_lr_std": lr_variation,
        "sanity_pass": sanity_pass,
        "sanity_definition": "inverse-distance/energy Spearman >= 0.1 and log(L/R) std >= 0.1",
    }
    write_json(output_dir / "summary.json", summary)
    maybe_plot(
        output_dir / "energy_vs_step.png",
        np.arange(len(all_rows)),
        [energies],
        ["total energy"],
        "observation step",
        "total energy",
        "M2 total energy",
    )
    print("M2 summary:", json.dumps(summary, indent=2, default=json_default))
    if not sanity_pass:
        raise RuntimeError(
            "M2 sanity check failed: inverse-distance/energy Spearman correlation "
            f"was {correlation:.4f}; inspect observation_steps.csv before M3"
        )
    return summary


def classify_failure(rows, terminal_reason, stop_threshold):
    if terminal_reason == "stuck":
        return "stuck"
    if terminal_reason == "timeout":
        return "timeout"
    if not rows:
        return "no_observation"
    decision_energy = rows[-1].get("decision_total_energy", rows[-1]["total_energy"])
    if decision_energy >= stop_threshold:
        return "threshold_false_positive"
    return "navigation_or_occlusion"


def mapping_validation(args, sim, source_goal, source_goal_valid, rng, output_dir):
    if not source_goal_valid:
        raise RuntimeError("source snap_point is invalid; mapping validation cannot start")
    source_signal, _ = ensure_audio(args.source_audio, args.sample_rate)
    point = sample_calibration_points(
        sim.pathfinder, source_goal, np.asarray(args.source), rng, args
    )[0]
    source_delta = np.asarray(args.source, dtype=np.float32) - point
    base_yaw = math.degrees(math.atan2(-source_delta[0], -source_delta[2]))
    agent = sim.get_agent(0)
    rows = []
    for action in ("baseline", "turn_left", "turn_right"):
        set_agent_pose(agent, point, base_yaw)
        applied_action = None if action == "baseline" else action
        ir, elapsed, acted, before, after = render_ir(sim, agent, applied_action)
        audio = convolve_with_rir(source_signal, ir, args.sample_rate, 0)
        features, _ = audio_features(audio, args.sample_rate)
        rows.append(
            {
                "action": action,
                "acted": acted,
                "yaw_before_degrees": base_yaw,
                "yaw_after_degrees": current_yaw_degrees(agent),
                "yaw_delta_degrees": signed_angle_delta_degrees(
                    current_yaw_degrees(agent), base_yaw
                ),
                "render_seconds": elapsed,
                **features,
            }
        )
    baseline = next(row for row in rows if row["action"] == "baseline")
    left = next(row for row in rows if row["action"] == "turn_left")
    right = next(row for row in rows if row["action"] == "turn_right")
    baseline_log_lr = math.log(baseline["lr_ratio"] + 1e-12)
    left_log_lr_delta = math.log(left["lr_ratio"] + 1e-12) - baseline_log_lr
    right_log_lr_delta = math.log(right["lr_ratio"] + 1e-12) - baseline_log_lr
    summary = {
        "position": point,
        "base_yaw_degrees": base_yaw,
        "left_yaw_delta_degrees": left["yaw_delta_degrees"],
        "right_yaw_delta_degrees": right["yaw_delta_degrees"],
        "left_log_lr_delta": left_log_lr_delta,
        "right_log_lr_delta": right_log_lr_delta,
        "yaw_action_mapping_pass": left["yaw_delta_degrees"] > 0 and right["yaw_delta_degrees"] < 0,
        "auditory_mapping_pass": left_log_lr_delta > 0 and right_log_lr_delta < 0,
        "interpretation": (
            "The agent faces the source at baseline; in this SoundSpaces binaural output, turn_left "
            "increases L/R and turn_right decreases L/R. The M3 policy uses the corresponding "
            "left-ear-high -> turn_left and right-ear-high -> turn_right convention."
        ),
    }
    write_rows(output_dir / "actions.csv", rows)
    write_json(output_dir / "summary.json", summary)
    print("mapping summary:", json.dumps(summary, indent=2, default=json_default))
    return summary


def audit_repeated_endpoints(output_root, policy_version="v0"):
    evaluation_dir = Path(output_root) / (
        "m3_evaluation_v2" if policy_version == "v2" else "m3_evaluation"
    )
    episodes_path = evaluation_dir / "episodes.csv"
    if not episodes_path.exists():
        raise FileNotFoundError(f"missing {policy_version} evaluation: {episodes_path}")
    with episodes_path.open() as handle:
        episodes = list(csv.DictReader(handle))
    rows = []
    for episode in episodes:
        trace_path = evaluation_dir / "episodes" / f"episode_{int(episode['episode_id']):03d}.csv"
        with trace_path.open() as handle:
            trace = list(csv.DictReader(handle))
        final = trace[-1]
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "failure_reason": episode["failure_reason"],
                "final_distance": episode["final_distance"],
                "final_x": final["position_x"],
                "final_y": final["position_y"],
                "final_z": final["position_z"],
            }
        )
    groups = {}
    for row in rows:
        groups.setdefault(round(float(row["final_distance"]), 9), []).append(row)
    repeated = [group for group in groups.values() if len(group) > 1]
    write_rows(evaluation_dir / "repeated_endpoints.csv", rows)
    summary = {
        "episode_count": len(rows),
        "repeated_distance_groups": len(repeated),
        "repeated_groups": repeated,
        "coordinate_unique_count": len(
            {(row["final_x"], row["final_y"], row["final_z"]) for row in rows}
        ),
    }
    write_json(evaluation_dir / "repeated_endpoints_summary.json", summary)
    print("endpoint audit:", json.dumps(summary, indent=2, default=json_default))
    return summary


def m3(args, sim, source_goal, source_goal_valid, rng, output_dir):
    if not source_goal_valid:
        raise RuntimeError("source snap_point is invalid; M3 cannot start")
    m2_summary_path = Path(args.output_root) / "m2_observation_check" / "summary.json"
    if not m2_summary_path.exists() and args.stop_threshold is None:
        raise RuntimeError("M3 needs M2 summary.json for stop-threshold calibration")
    m2_summary = json.loads(m2_summary_path.read_text()) if m2_summary_path.exists() else {}
    if m2_summary and not m2_summary.get("sanity_pass", False):
        raise RuntimeError("M3 is blocked because the M2 auditory sanity check did not pass")
    stop_threshold = (
        float(args.stop_threshold)
        if args.stop_threshold is not None
        else float(m2_summary["stop_threshold"])
    )
    is_v2 = args.policy_version == "v2"
    source_signal, _ = ensure_audio(args.source_audio, args.sample_rate)
    agent = sim.get_agent(0)
    episode_rows = []
    all_trace_rows = []
    attempts = 0
    while len(episode_rows) < args.episodes and attempts < args.max_sample_attempts:
        attempts += 1
        try:
            start, shortest_distance, path_points = sample_start(
                sim.pathfinder, source_goal, np.asarray(args.source), rng, args
            )
        except RuntimeError:
            break
        set_agent_pose(agent, start, float(rng.uniform(0.0, 360.0)))
        rows = []
        sample_index = 0
        total_path = 0.0
        no_motion_steps = 0
        force_turn = False
        ema_left = None
        ema_right = None
        ema_total = None
        stop_streak = 0
        held_turn = None
        turn_hold_remaining = 0
        terminal_reason = "timeout"
        for step in range(args.max_steps):
            ir, elapsed, acted, before, after = render_ir(sim, agent, None)
            audio = convolve_with_rir(source_signal, ir, args.sample_rate, sample_index)
            features, _ = audio_features(audio, args.sample_rate)
            distance = listener_distance(after, np.asarray(args.source))
            row_extra = {}
            if is_v2:
                if ema_total is None:
                    ema_left = features["left_energy"]
                    ema_right = features["right_energy"]
                    ema_total = features["total_energy"]
                else:
                    alpha = args.ema_alpha
                    ema_left = alpha * features["left_energy"] + (1.0 - alpha) * ema_left
                    ema_right = alpha * features["right_energy"] + (1.0 - alpha) * ema_right
                    ema_total = alpha * features["total_energy"] + (1.0 - alpha) * ema_total
                decision_lr = ema_left / (ema_right + 1e-12)
                decision_total = ema_total
                row_extra = {
                    "ema_left_energy": ema_left,
                    "ema_right_energy": ema_right,
                    "ema_total_energy": ema_total,
                    "ema_lr_ratio": decision_lr,
                    "stop_streak": stop_streak,
                }
            else:
                decision_lr = features["lr_ratio"]
                decision_total = features["total_energy"]
            action = "stop"
            if is_v2:
                stop_streak = stop_streak + 1 if decision_total >= stop_threshold else 0
                should_stop = stop_streak >= args.stop_confirmation_frames
            else:
                should_stop = decision_total >= stop_threshold
            if should_stop:
                terminal_reason = "stop"
            else:
                if force_turn:
                    action = "turn_left"
                    held_turn = None
                    turn_hold_remaining = 0
                elif is_v2 and turn_hold_remaining > 0 and held_turn is not None:
                    action = held_turn
                    turn_hold_remaining -= 1
                else:
                    log_lr = math.log(decision_lr + 1e-12)
                    if log_lr > math.log(args.lr_threshold):
                        action = "turn_left" if decision_lr >= 1.0 else "turn_right"
                    elif log_lr < -math.log(args.lr_threshold):
                        action = "turn_right" if decision_lr <= 1.0 else "turn_left"
                    else:
                        action = "move_forward"
                    if is_v2 and action in ("turn_left", "turn_right"):
                        held_turn = action
                        turn_hold_remaining = max(args.turn_hold_steps - 1, 0)

            if action == "stop":
                row_extra["stop_streak"] = stop_streak
                rows.append(
                    {
                        "episode_id": len(episode_rows),
                        "step": step,
                        "action": action,
                        "acted": acted,
                        "moved": False,
                        "position_x": float(after[0]),
                        "position_y": float(after[1]),
                        "position_z": float(after[2]),
                        "distance_to_source": distance,
                        "render_seconds": elapsed,
                        "decision_total_energy": decision_total,
                        **row_extra,
                        **features,
                    }
                )
                break

            started = time.perf_counter()
            acted = bool(agent.act(action))
            moved_position = np.asarray(agent.get_state().position, dtype=np.float32)
            action_elapsed = time.perf_counter() - started
            moved_distance = float(np.linalg.norm(moved_position - after))
            moved = moved_distance > args.position_epsilon
            total_path += moved_distance
            no_motion_steps = no_motion_steps + 1 if not moved else 0
            force_turn = action == "move_forward" and not moved
            if no_motion_steps >= args.stuck_window:
                terminal_reason = "stuck"
            rows.append(
                {
                    "episode_id": len(episode_rows),
                    "step": step,
                    "action": action,
                    "acted": acted,
                    "moved": moved,
                    "position_x": float(moved_position[0]),
                    "position_y": float(moved_position[1]),
                    "position_z": float(moved_position[2]),
                    "distance_to_source": distance,
                    "render_seconds": elapsed + action_elapsed,
                    "decision_total_energy": decision_total,
                    **row_extra,
                    **features,
                }
            )
            sample_index += int(args.sample_rate * args.step_time)
            if terminal_reason == "stuck":
                break
        else:
            terminal_reason = "timeout"

        final_position = np.asarray(agent.get_state().position, dtype=np.float32)
        final_distance = listener_distance(final_position, np.asarray(args.source))
        success = terminal_reason == "stop" and final_distance < 1.0
        failure_reason = "success" if success else classify_failure(rows, terminal_reason, stop_threshold)
        summary = {
            "episode_id": len(episode_rows),
            "success": success,
            "failure_reason": failure_reason,
            "terminal_reason": terminal_reason,
            "steps": len(rows),
            "start_x": float(start[0]),
            "start_y": float(start[1]),
            "start_z": float(start[2]),
            "shortest_path_length": shortest_distance,
            "actual_path_length": total_path,
            "path_efficiency": total_path / shortest_distance if shortest_distance > 0 else float("nan"),
            "final_distance": final_distance,
        }
        episode_rows.append(summary)
        all_trace_rows.extend(rows)
        write_rows(output_dir / "episodes" / f"episode_{summary['episode_id']:03d}.csv", rows)
        maybe_plot(
            output_dir / "episodes" / f"episode_{summary['episode_id']:03d}_curves.png",
            np.arange(len(rows)),
            [
                [row["total_energy"] for row in rows],
                [row["lr_ratio"] for row in rows],
            ],
            ["total energy", "L/R ratio"],
            "step",
            "value",
            f"M3 episode {summary['episode_id']}",
        )
        maybe_plot(
            output_dir / "episodes" / f"episode_{summary['episode_id']:03d}_trajectory.png",
            np.arange(len(rows)),
            [
                [row["position_x"] for row in rows],
                [row["position_z"] for row in rows],
            ],
            ["x", "z"],
            "step",
            "position",
            f"M3 episode {summary['episode_id']} trajectory",
        )
    write_rows(output_dir / "episodes.csv", episode_rows)
    write_rows(output_dir / "all_traces.csv", all_trace_rows)
    failures = [row for row in episode_rows if not row["success"]]
    summary = {
        "episodes": len(episode_rows),
        "successes": int(sum(row["success"] for row in episode_rows)),
        "success_rate": float(sum(row["success"] for row in episode_rows) / max(len(episode_rows), 1)),
        "average_steps": float(np.mean([row["steps"] for row in episode_rows])) if episode_rows else float("nan"),
        "average_path_efficiency": float(
            np.nanmean([row["path_efficiency"] for row in episode_rows])
        ) if episode_rows else float("nan"),
        "failure_counts": {
            reason: sum(row["failure_reason"] == reason for row in failures)
            for reason in sorted(set(row["failure_reason"] for row in failures))
        },
        "stop_threshold": stop_threshold,
        "policy_version": args.policy_version,
        "ema_alpha": args.ema_alpha if is_v2 else None,
        "stop_confirmation_frames": args.stop_confirmation_frames if is_v2 else 1,
        "turn_hold_steps": args.turn_hold_steps if is_v2 else 0,
        "stuck_window": args.stuck_window,
        "max_steps": args.max_steps,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "failure_episodes.json", failures[:3])
    print("M3 summary:", json.dumps(summary, indent=2, default=json_default))
    return summary


def main():
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    if args.stage == "audit":
        audit_repeated_endpoints(output_root, args.policy_version)
        return
    rng = np.random.default_rng(args.seed)
    source_position = np.asarray(args.source, dtype=np.float32)
    sim = None
    try:
        sim = make_sim(args)
        requested_source = source_position.copy()
        source_position, source_goal, source_goal_valid = resolve_source(
            sim, source_position, rng, args
        )
        args.source = source_position.tolist()
        args.requested_source = requested_source.tolist()
        args.source_fallback = bool(not np.allclose(requested_source, source_position))
        if not hasattr(args, "source_height_adjusted"):
            args.source_height_adjusted = False
        config_stage = (
            f"{args.stage}_{args.policy_version}"
            if args.stage == "m3"
            else args.stage
        )
        write_json(
            output_root / f"run_config_{config_stage}.json",
            {
                "stage": args.stage,
                "scene": args.scene,
                "requested_source": args.requested_source,
                "effective_source": args.source,
                "source_fallback": args.source_fallback,
                "source_height_adjusted": args.source_height_adjusted,
                "sample_rate": args.sample_rate,
                "step_time": args.step_time,
                "forward_step": args.forward_step,
                "turn_angle": args.turn_angle,
                "indirect_ray_count": args.indirect_ray_count,
                "materials_enabled": args.enable_materials,
                "materials_json": args.materials_json,
                "policy_version": args.policy_version,
                "ema_alpha": args.ema_alpha,
                "stop_confirmation_frames": args.stop_confirmation_frames,
                "turn_hold_steps": args.turn_hold_steps,
            },
        )
        print("scene:", args.scene)
        print("requested source:", requested_source.tolist())
        print("effective source:", source_position.tolist())
        print("source snap_point:", source_goal.tolist())
        print("source snap valid:", source_goal_valid)
        print("pathfinder loaded:", sim.pathfinder.is_loaded)
        print("action space:", list(sim.get_agent(0).agent_config.action_space.keys()))
        print("forward step (m):", args.forward_step)
        print("turn angle (degrees):", args.turn_angle)
        print("step time (s):", args.step_time)
        print("materials enabled:", args.enable_materials)
        print("indirect ray count:", args.indirect_ray_count)
        if args.stage == "mapping":
            mapping_validation(args, sim, source_goal, source_goal_valid, rng, output_root / "mapping_validation")
            return
        if args.stage in ("m1", "all"):
            m1(args, sim, source_goal, source_goal_valid, rng, output_root / "m1_timing")
        if args.stage in ("m2", "all"):
            m2(args, sim, source_goal, source_goal_valid, rng, output_root / "m2_observation_check")
        if args.stage in ("m3", "all"):
            m3_dir = output_root / (
                "m3_evaluation_v2" if args.policy_version == "v2" else "m3_evaluation"
            )
            m3(args, sim, source_goal, source_goal_valid, rng, m3_dir)
    finally:
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
