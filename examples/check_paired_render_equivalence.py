"""Compare standalone and dual-sensor outputs at one deterministic pose."""

import json
from pathlib import Path

import quaternion  # 必须先于 habitat_sim 导入
import habitat_sim
import numpy as np

from check_dual_sensor_no_reset import LISTENER, POSES, SAMPLE_RATE, channels


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data/logs/seld_dataset_v1_preflight"
SCENE = ROOT / "data/scene_datasets/replica/office_0/habitat/mesh_semantic.ply"


def make_sim(layout, count, uuid="audio_sensor"):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(SCENE)
    backend.load_semantic_mesh = True
    backend.enable_physics = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()]))
    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = uuid
    spec.enableMaterials = False
    spec.channelLayout.type = layout
    spec.channelLayout.channelCount = count
    spec.acousticsConfig.sampleRate = SAMPLE_RATE
    spec.acousticsConfig.direct = True
    spec.acousticsConfig.indirect = False
    spec.acousticsConfig.diffraction = False
    spec.acousticsConfig.transmission = False
    sim.add_sensor(spec)
    return sim


def set_pose(sim, source):
    agent = sim.get_agent(0)
    for sensor in agent._sensors.values():
        sensor.setAudioSourceTransform(source)
    state = agent.get_state()
    state.position = LISTENER
    state.rotation = quaternion.from_rotation_vector(np.zeros(3))
    state.sensor_states = {}
    agent.set_state(state, True)


def read(sim, uuid, count):
    value = np.asarray(sim.get_sensor_observations()[uuid])
    if value.shape[0] != count:
        value = value.T
    return value.astype(np.float32)


def comparison(left, right):
    if left.shape != right.shape:
        return {"shape_equal": False, "standalone_shape": list(left.shape), "dual_shape": list(right.shape), "max_abs_diff": None, "relative_energy_diff": None}
    denominator = max(float(np.sum(left * left)), 1e-12)
    return {"shape_equal": True, "standalone_shape": list(left.shape), "dual_shape": list(right.shape), "max_abs_diff": float(np.max(np.abs(left - right))), "relative_energy_diff": float(abs(np.sum(left * left) - np.sum(right * right)) / denominator)}


def main():
    source = POSES[0][0]
    layouts = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType
    standalone_binaural = make_sim(layouts.Binaural, 2)
    standalone_foa = make_sim(layouts.Ambisonics, 4)
    dual = make_sim(layouts.Binaural, 2)
    foa_spec = habitat_sim.AudioSensorSpec()
    foa_spec.uuid = "foa_sensor"
    foa_spec.enableMaterials = False
    foa_spec.channelLayout.type = layouts.Ambisonics
    foa_spec.channelLayout.channelCount = 4
    foa_spec.acousticsConfig.sampleRate = SAMPLE_RATE
    foa_spec.acousticsConfig.direct = True
    foa_spec.acousticsConfig.indirect = False
    foa_spec.acousticsConfig.diffraction = False
    foa_spec.acousticsConfig.transmission = False
    dual.add_sensor(foa_spec)
    try:
        set_pose(standalone_binaural, source)
        set_pose(standalone_foa, source)
        set_pose(dual, source)
        standalone_binaural_value = read(standalone_binaural, "audio_sensor", 2)
        standalone_foa_value = read(standalone_foa, "audio_sensor", 4)
        observations = dual.get_sensor_observations()
        dual_binaural_value = read(dual, "audio_sensor", 2)
        dual_foa_value = read(dual, "foa_sensor", 4)
    finally:
        standalone_binaural.close()
        standalone_foa.close()
        dual.close()
    result = {"binaural": comparison(standalone_binaural_value, dual_binaural_value), "foa": comparison(standalone_foa_value, dual_foa_value), "raw_dual_keys": sorted(observations.keys()), "paired_render_mode": "dual_sensor_single_sim_no_reset" if comparison(standalone_foa_value, dual_foa_value)["shape_equal"] else "sequential_sensor_lifecycle", "verdict": "PASS" if comparison(standalone_binaural_value, dual_binaural_value)["shape_equal"] and comparison(standalone_foa_value, dual_foa_value)["shape_equal"] else "FALLBACK"}
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "paired_render_equivalence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"binaural": result["binaural"], "foa": result["foa"], "paired_render_mode": result["paired_render_mode"], "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
