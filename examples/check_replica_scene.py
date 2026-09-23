"""Replica 场景加载、材料映射、IR A/B 与导航验证。

用法（在 sound-spaces 根目录执行）：
    conda run -n ss --no-capture-output python examples/check_replica_scene.py
"""

import argparse
import copy
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import quaternion  # 必须先于 habitat_sim 导入
import habitat_sim
import numpy as np


DEFAULT_SCENE = (
    "data/scene_datasets/replica/office_0/habitat/mesh_semantic.ply"
)
DEFAULT_MP3D_MATERIALS = "data/mp3d_material_config.json"
DEFAULT_REPLICA_MATERIALS = "data/replica_material_config.json"


def normalize_label(value: Any) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[_-]+", " ", value)
    return re.sub(r"\s+", " ", value)


def add_label(labels: set, value: Any) -> None:
    if isinstance(value, str) and value.strip():
        labels.add(value.strip())


def collect_info_labels(info: Dict[str, Any]) -> List[str]:
    """兼容 Replica info_semantic.json 的常见 classes/objects 结构。"""
    labels = set()

    for key in ("classes", "categories", "semantic_classes", "labels"):
        collection = info.get(key, [])
        if isinstance(collection, dict):
            collection = list(collection.values())
        if not isinstance(collection, list):
            continue
        for item in collection:
            if isinstance(item, str):
                add_label(labels, item)
            elif isinstance(item, dict):
                for field in ("name", "label", "category", "class_name"):
                    add_label(labels, item.get(field))

    for item in info.get("objects", []):
        if not isinstance(item, dict):
            continue
        for field in ("class_name", "category", "label", "name"):
            add_label(labels, item.get(field))

    return sorted(labels, key=lambda value: (normalize_label(value), value))


def material_target(label: str) -> Tuple[str, bool]:
    """返回材料名与是否为明确物理映射。未覆盖项进入 Default。"""
    value = normalize_label(label)

    # 按 Replica 审查意见覆盖通用关键词规则，避免标签中的 wall/cabinet
    # 等词把物理属性判断带偏。
    manual_targets = {
        "wall cabinet": "Wood Floor",
        "wall plug": "Default",
        "nightstand": "wood, Thick",
        "stool": "wood, Thick",
        "chopping board": "wood, Thick",
        "shower stall": "Tile, Ceramic",
        "toilet": "Tile, Ceramic",
        "scarf": "Curtain",
        "beam": "Steel",
    }
    if value in manual_targets:
        target = manual_targets[value]
        return target, target != "Default"

    rules = [
        ("Glass", ("window", "mirror", "glass", "screen", "monitor", "blinds")),
        ("Gypsum Board", ("wall", "partition", "drywall", "plaster")),
        ("Acoustic Tile", ("ceiling",)),
        ("Carpet", ("carpet", "rug", "mat", "floor")),
        ("Tile, Ceramic", ("tile", "ceramic")),
        ("Brick", ("brick", "fireplace")),
        ("Concrete, Rough", ("concrete", "cement", "stone")),
        ("Curtain", (
            "curtain", "cloth", "fabric", "clothing", "blanket", "comforter",
            "cushion", "towel", "sofa", "bed", "backpack", "handbag",
        )),
        ("Foliage", ("plant", "foliage", "flower")),
        ("Steel", (
            "metal", "steel", "appliance", "microwave", "refrigerator", "sink",
            "railing", "handrail", "pipe", "bathtub",
        )),
        ("Wood Floor", ("cabinet", "stair", "wood floor")),
        ("wood, Thick", (
            "wood", "chair", "furniture", "door", "desk", "table", "shelf",
            "shelving", "counter", "drawer", "wardrobe", "seat", "bench",
        )),
    ]
    for target, keywords in rules:
        if any(keyword in value for keyword in keywords):
            return target, True

    if value in {"default", "void", "unknown", "unlabeled", "background", "none"}:
        return "Default", False
    return "Default", False


def build_material_config(
    labels: Iterable[str],
    source_path: str,
    output_path: str,
) -> Dict[str, Any]:
    with open(source_path, encoding="utf-8") as handle:
        config = json.load(handle)

    result = copy.deepcopy(config)
    by_material = {material["name"]: [] for material in result["materials"]}
    mapping = {}
    matched = {}
    for label in labels:
        target, is_matched = material_target(label)
        if target not in by_material:
            target = "Default"
            is_matched = False
        by_material[target].append(label)
        mapping[label] = target
        matched[label] = is_matched

    # Default 必须保留，且承接所有没有明确物理映射的 Replica 标签。
    by_material["Default"] = sorted(set(["default"] + by_material["Default"]))
    for material in result["materials"]:
        material["labels"] = sorted(set(by_material[material["name"]]))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return {
        "mapping": mapping,
        "matched": sorted(label for label, value in matched.items() if value),
        "unmatched": sorted(label for label, value in matched.items() if not value),
    }


def make_sim(
    scene: str,
    source_position: np.ndarray,
    agent_position: np.ndarray,
    enable_materials: bool,
    materials_json: str,
    scene_dataset_config: Optional[str] = None,
) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene
    if scene_dataset_config:
        backend_cfg.scene_dataset_config_file = scene_dataset_config
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    audio_spec = habitat_sim.AudioSensorSpec()
    audio_spec.uuid = "audio_sensor"
    audio_spec.enableMaterials = enable_materials
    audio_spec.channelLayout.type = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    audio_spec.channelLayout.channelCount = 2
    audio_spec.position = [0.0, 1.5, 0.0]
    audio_spec.acousticsConfig.sampleRate = 16000
    audio_spec.acousticsConfig.indirect = True
    sim.add_sensor(audio_spec)

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(source_position + np.array([0.0, 1.5, 0.0]))
    if enable_materials:
        audio_sensor.setAudioMaterialsJSON(materials_json)

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = agent_position
    state.sensor_states = {}
    agent.set_state(state, True)
    return sim


def render_ir(sim: habitat_sim.Simulator) -> np.ndarray:
    ir = np.asarray(sim.get_sensor_observations()["audio_sensor"])
    if ir.ndim == 2 and ir.shape[0] == 2 and ir.shape[1] != 2:
        ir = ir.T
    return ir


def ir_stats(ir: np.ndarray) -> Tuple[float, float]:
    energy = float(np.mean(ir ** 2))
    half = ir.shape[0] // 2
    tail_ratio = float(np.sum(ir[half:] ** 2) / (np.sum(ir ** 2) + 1e-12))
    return energy, tail_ratio


def vec(value: Any) -> List[float]:
    return [float(x) for x in np.asarray(value).reshape(-1)]


def semantic_categories(sim: habitat_sim.Simulator) -> List[str]:
    categories = []
    for obj in sim.semantic_scene.objects:
        if obj is not None and obj.category is not None:
            try:
                name = obj.category.name()
            except Exception:
                name = str(obj.category)
            if name:
                categories.append(str(name))
    return sorted(set(categories), key=lambda value: (normalize_label(value), value))


def island_radius(pathfinder: Any) -> Optional[float]:
    try:
        value = pathfinder.island_radius
        return float(value() if callable(value) else value)
    except Exception:
        return None


def sample_positions(pathfinder: Any) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    samples = []
    for _ in range(100):
        point = np.asarray(pathfinder.get_random_navigable_point(), dtype=float)
        if np.all(np.isfinite(point)):
            samples.append(point)
        if len(samples) >= 3:
            break
    if len(samples) < 3:
        raise RuntimeError("无法从 Replica navmesh 采样 3 个有效位姿")
    source = samples[0]
    agent = samples[1]
    if np.linalg.norm(source - agent) < 1.0:
        agent = samples[2]
    return source, agent, samples


def run_validation(
    scene: str,
    materials_json: str,
    enable_materials: bool,
    source: np.ndarray,
    agent_position: np.ndarray,
    scene_dataset_config: Optional[str] = None,
) -> Dict[str, Any]:
    sim = make_sim(
        scene,
        source,
        agent_position,
        enable_materials,
        materials_json,
        scene_dataset_config,
    )
    try:
        pathfinder = sim.pathfinder
        agent = sim.get_agent(0)
        before = vec(agent.get_state().position)
        moves = []
        for index in range(5):
            acted = bool(agent.act("move_forward"))
            moves.append({
                "step": index,
                "acted": acted,
                "position": vec(agent.get_state().position),
            })

        ir = render_ir(sim)
        energy, tail_ratio = ir_stats(ir)
        categories = semantic_categories(sim)
        return {
            "materials": enable_materials,
            "ir_shape": list(ir.shape),
            "energy": energy,
            "tail_ratio": tail_ratio,
            "pathfinder_is_loaded": bool(pathfinder.is_loaded),
            "island_radius": island_radius(pathfinder),
            "semantic_object_count": len(sim.semantic_scene.objects),
            "semantic_categories_from_sim": categories,
            "navigation": {
                "initial_position": before,
                "moves": moves,
                "position_changed": any(
                    np.linalg.norm(np.asarray(move["position"]) - before) > 1e-6
                    for move in moves
                ),
            },
        }
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument(
        "--scene-dataset-config",
        default=None,
        help="官方 SceneDatasetConfig；传入后 --scene 可使用 office_0 这类场景 ID",
    )
    parser.add_argument("--info-json", default=None)
    parser.add_argument("--mp3d-materials", default=DEFAULT_MP3D_MATERIALS)
    parser.add_argument("--replica-materials", default=DEFAULT_REPLICA_MATERIALS)
    parser.add_argument("--output-json", default="data/logs/replica_validation.json")
    parser.add_argument(
        "--materials-off-only",
        action="store_true",
        help="只验证场景、语义、navmesh 和材料 OFF，避免兼容性崩溃阻断结果落盘",
    )
    args = parser.parse_args()

    scene_dir = os.path.dirname(args.scene)
    info_json = args.info_json or os.path.join(scene_dir, "info_semantic.json")
    with open(info_json, encoding="utf-8") as handle:
        info = json.load(handle)
    info_labels = collect_info_labels(info)
    mapping_report = build_material_config(
        info_labels, args.mp3d_materials, args.replica_materials
    )

    # 先用一个无材料 sim 采样位姿；同一对位姿随后用于 OFF/ON A/B。
    probe = make_sim(
        args.scene,
        np.zeros(3),
        np.zeros(3),
        False,
        args.replica_materials,
        args.scene_dataset_config,
    )
    try:
        if not probe.pathfinder.is_loaded:
            raise RuntimeError("Replica navmesh 未加载")
        source, agent_position, samples = sample_positions(probe.pathfinder)
    finally:
        probe.close()

    off = run_validation(
        args.scene,
        args.replica_materials,
        False,
        source,
        agent_position,
        args.scene_dataset_config,
    )
    if args.materials_off_only:
        result = {
            "scene": args.scene,
            "info_json": info_json,
            "info_semantic_labels": info_labels,
            "mapping": mapping_report,
            "sampled_positions": [vec(point) for point in samples],
            "materials_off": off,
            "materials_on": None,
            "comparison": None,
            "materials_on_status": "not_run_by_request",
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    on = run_validation(
        args.scene,
        args.replica_materials,
        True,
        source,
        agent_position,
        args.scene_dataset_config,
    )
    result = {
        "scene": args.scene,
        "info_json": info_json,
        "info_semantic_labels": info_labels,
        "mapping": mapping_report,
        "sampled_positions": [vec(point) for point in samples],
        "materials_off": off,
        "materials_on": on,
        "comparison": {
            "energy_ratio_on_over_off": on["energy"] / (off["energy"] + 1e-12),
            "tail_ratio_on_over_off": on["tail_ratio"] / (off["tail_ratio"] + 1e-12),
        },
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
