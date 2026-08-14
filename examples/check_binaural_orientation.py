import quaternion
import habitat_sim
import habitat_sim.sim
import numpy as np
from scipy.io import wavfile
from pathlib import Path


out_dir = Path("data/orientation_check")
out_dir.mkdir(parents=True, exist_ok=True)

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
backend_cfg.load_semantic_mesh = True
backend_cfg.enable_physics = False

agent_cfg = habitat_sim.agent.AgentConfiguration()

cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

audio_sensor_spec = habitat_sim.AudioSensorSpec()
audio_sensor_spec.uuid = "audio_sensor"

# 使用双声道 / 双耳布局
audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
audio_sensor_spec.channelLayout.channelCount = 2

audio_sensor_spec.position = [0.0, 1.5, 0.0]
audio_sensor_spec.acousticsConfig.sampleRate = 16000
audio_sensor_spec.acousticsConfig.indirect = True

sim.add_sensor(audio_sensor_spec)
audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]

# 固定声源位置
audio_sensor.setAudioSourceTransform(np.array([-8.56, 1.5, 0.50]))

# 如果使用 habitat-test-scenes，材料可不加载；如果你想加载材料，可取消下一行注释
# audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")

agent = sim.get_agent(0)

# 固定 agent 位置
base_state = agent.get_state()
base_state.position = np.array([-10.57, 0.0, -0.25])

# 四个朝向，单位：度
yaws = [0, 90, 180, 270]

for yaw in yaws:
    state = agent.get_state()
    state.position = base_state.position.copy()

    # Habitat 使用 quaternion 表示旋转；这里绕 Y 轴旋转
    state.rotation = quaternion.from_rotation_vector(
        np.array([0.0, np.deg2rad(yaw), 0.0])
    )

    agent.set_state(state, True)

    obs = np.array(sim.get_sensor_observations()["audio_sensor"])

    # 一些版本输出 shape 可能是 (2, N)，统一转成 (N, 2)
    if obs.ndim == 2 and obs.shape[0] == 2 and obs.shape[1] != 2:
        obs = obs.T

    out_path = out_dir / f"output_yaw{yaw}.wav"
    wavfile.write(str(out_path), 16000, obs)

    if obs.ndim == 2:
        left_energy = float(np.mean(obs[:, 0] ** 2))
        right_energy = float(np.mean(obs[:, 1] ** 2))
        ratio = left_energy / (right_energy + 1e-12)
        print(
            f"yaw={yaw:3d} | shape={obs.shape} | "
            f"L={left_energy:.8e} | R={right_energy:.8e} | L/R={ratio:.4f} | {out_path}"
        )
    else:
        print(f"yaw={yaw:3d} | shape={obs.shape} | mono output | {out_path}")

sim.close()
