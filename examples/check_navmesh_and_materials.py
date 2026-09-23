# M0 前置验证脚本：navmesh 可用性 + 离散动作 + 材料匹配率与 A/B 对比
# 用法：在 sound-spaces 根目录下运行
#   conda run -n ss python examples/check_navmesh_and_materials.py

import json
import sys

import quaternion  # 需先于 habitat_sim 导入，避免 invalid pointer（见 notes 4.6）
import habitat_sim
import habitat_sim.sim
import numpy as np

SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
MATERIALS_JSON = "data/mp3d_material_config.json"
# 与之前实验一致的固定位姿
SOURCE_POS = np.array([-8.56, 1.5, 0.50])
AGENT_POS = np.array([-10.57, 0.0, -0.25])


def make_sim(enable_materials):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.enableMaterials = enable_materials
    audio_sensor_spec.channelLayout.type = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    audio_sensor_spec.channelLayout.channelCount = 2
    audio_sensor_spec.position = [0.0, 1.5, 0.0]
    audio_sensor_spec.acousticsConfig.sampleRate = 16000
    audio_sensor_spec.acousticsConfig.indirect = True
    sim.add_sensor(audio_sensor_spec)

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(SOURCE_POS)
    if enable_materials:
        audio_sensor.setAudioMaterialsJSON(MATERIALS_JSON)

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = AGENT_POS
    state.sensor_states = {}
    agent.set_state(state, True)
    return sim


def render_ir(sim):
    obs = np.array(sim.get_sensor_observations()["audio_sensor"])
    # 统一成 (samples, channels)
    if obs.ndim == 2 and obs.shape[0] == 2 and obs.shape[1] != 2:
        obs = obs.T
    return obs


def ir_stats(ir):
    energy = float(np.mean(ir ** 2))
    # 粗略混响尾指标：后半段能量占总能量的比例，材料吸声越强该值越小
    half = ir.shape[0] // 2
    tail_ratio = float(np.sum(ir[half:] ** 2) / (np.sum(ir ** 2) + 1e-12))
    return energy, tail_ratio


def main():
    # ---------- 1. navmesh 与离散动作 ----------
    print("=== 1. Navmesh & discrete actions ===")
    sim = make_sim(enable_materials=False)

    pf = sim.pathfinder
    print("pathfinder.is_loaded:", pf.is_loaded)
    if pf.is_loaded:
        snapped = pf.snap_point(AGENT_POS)
        print("snap_point(AGENT_POS):", snapped)
        print("is navigable:", pf.is_navigable(snapped))

    agent = sim.get_agent(0)
    print("action space keys:", list(agent.agent_config.action_space.keys()))

    for i in range(5):
        acted = agent.act("move_forward")
        pos = agent.get_state().position
        print(f"step {i}: acted={acted} pos={np.round(pos, 3)}")

    # ---------- 2. 语义标签与材料库的匹配率 ----------
    print("\n=== 2. Semantic label match with material JSON ===")
    with open(MATERIALS_JSON) as f:
        mat_labels = set()
        for m in json.load(f)["materials"]:
            mat_labels.update(m.get("labels", []))

    semantic_scene = sim.semantic_scene
    cats = []
    for obj in semantic_scene.objects:
        if obj is not None and obj.category is not None:
            cats.append(obj.category.name().lower())
    print("场景语义对象数:", len(cats))
    uniq = sorted(set(cats))
    matched = [c for c in uniq if c in mat_labels]
    unmatched = [c for c in uniq if c not in mat_labels]
    print(f"唯一类别数: {len(uniq)} | 匹配: {len(matched)} | 未匹配: {len(unmatched)}")
    print("未匹配类别:", unmatched)

    # ---------- 3. 材料 A/B 对比 ----------
    # 注意：enableMaterials=True 在无语义网格的场景会断言崩溃（core dump），
    # 因此 materials ON 部分必须用 --materials-on 单独进程运行。
    print("\n=== 3. IR baseline: materials OFF ===")
    ir_off = render_ir(sim)
    e_off, t_off = ir_stats(ir_off)
    print(f"materials OFF: shape={ir_off.shape} energy={e_off:.6e} tail_ratio={t_off:.4f}")
    sim.close()

    if "--materials-on" in sys.argv:
        sim = make_sim(enable_materials=True)
        ir_on = render_ir(sim)
        e_on, t_on = ir_stats(ir_on)
        print(f"materials ON : shape={ir_on.shape} energy={e_on:.6e} tail_ratio={t_on:.4f}")
        sim.close()

        same_shape = ir_off.shape == ir_on.shape
        if same_shape:
            diff = float(np.mean(np.abs(ir_off - ir_on)))
            print(f"mean |IR_off - IR_on| = {diff:.6e}")
        print(f"energy 变化: {e_on / (e_off + 1e-12):.3f}x | tail_ratio 变化: {t_on / (t_off + 1e-12):.3f}x")


if __name__ == "__main__":
    main()
