"""Stress paired binaural and FOA sensors without calling Simulator.reset()."""

import json
import resource
from pathlib import Path

import quaternion  # 必须先于 habitat_sim 导入
import habitat_sim
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data/logs/seld_dataset_v1_preflight"
SCENE = ROOT / "data/scene_datasets/replica/office_0/habitat/mesh_semantic.ply"
SAMPLE_RATE = 24000
LISTENER = np.asarray([0.8, 0.5311309695, 2.4433872699], dtype=np.float32)
POSES = [
    (np.asarray([0.1810176075, 0.5311309695, 2.44338727], dtype=np.float32), 0.0),
    (np.asarray([1.4189824164, 0.5311309695, 2.44338727], dtype=np.float32), 0.0),
]


def make_sim():
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(SCENE)
    backend.load_semantic_mesh = True
    backend.enable_physics = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()]))
    layouts = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType
    for uuid, layout, count in (("audio_sensor", layouts.Binaural, 2), ("foa_sensor", layouts.Ambisonics, 4)):
        spec = habitat_sim.AudioSensorSpec()
        spec.uuid = uuid
        spec.enableMaterials = False
        spec.channelLayout.type = layout
        spec.channelLayout.channelCount = count
        spec.acousticsConfig.sampleRate = SAMPLE_RATE
        spec.acousticsConfig.direct = True
        spec.acousticsConfig.indirect = True
        sim.add_sensor(spec)
    return sim


def set_pose(sim, source, yaw):
    agent = sim.get_agent(0)
    for sensor in agent._sensors.values():
        sensor.setAudioSourceTransform(source)
    state = agent.get_state()
    state.position = LISTENER
    state.rotation = quaternion.from_rotation_vector(np.array([0.0, np.deg2rad(yaw), 0.0]))
    state.sensor_states = {}
    agent.set_state(state, True)


def channels(observation, count):
    value = np.asarray(observation)
    if value.shape[0] != count:
        value = value.T
    return value


def main():
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sim = make_sim()
    checks = []
    try:
        for index in range(20):
            source, yaw = POSES[index % len(POSES)]
            set_pose(sim, source, yaw)
            observations = sim.get_sensor_observations()
            binaural = channels(observations["audio_sensor"], 2)
            foa = channels(observations["foa_sensor"], 4)
            checks.append({"index": index, "binaural_shape": list(binaural.shape), "foa_shape": list(foa.shape), "finite": bool(np.isfinite(binaural).all() and np.isfinite(foa).all()), "nonzero": bool(np.any(binaural) and np.any(foa))})
    finally:
        sim.close()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"poses_tested": len(checks), "reset_called": False, "crash": False, "rss_before_kb": int(rss_before), "rss_after_kb": int(rss_after), "max_rss_kb": int(max(rss_before, rss_after)), "checks": checks, "paired_render_mode": "dual_sensor_single_sim_no_reset" if len(checks) == 20 and all(x["finite"] and x["nonzero"] and x["binaural_shape"][0] == 2 and x["foa_shape"][0] == 4 and x["foa_shape"][1] > 4 for x in checks) else "sequential_sensor_lifecycle", "verdict": "PASS" if len(checks) == 20 and all(x["finite"] and x["nonzero"] and x["binaural_shape"][0] == 2 and x["foa_shape"][0] == 4 and x["foa_shape"][1] > 4 for x in checks) else "FAIL"}
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "dual_sensor_stress.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("poses_tested", "reset_called", "rss_before_kb", "rss_after_kb", "paired_render_mode", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
