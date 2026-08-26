"""Manifest-driven Golden top-down visualization for M3."""

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

from active_audition.config.loader import load_resolved_config
from active_audition.scene.simulator import create_scene_simulator


def _xz(point):
    return float(point[0]), float(point[2])


def render_golden_topdown(
    dataset_root: str,
    config_path: str,
    run_root: Path,
    episodes: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    viewpoints: Sequence[Mapping[str, object]],
):
    config = load_resolved_config(config_path)
    episode = sorted(episodes, key=lambda row: str(row["episode_id"]))[0]
    episode_candidates = [row for row in candidates if row["episode_id"] == episode["episode_id"]]
    initial = next(row for row in viewpoints if row["episode_id"] == episode["episode_id"] and row["viewpoint_id"] == "initial")
    points = [episode["source"]["position_world"], initial["base_position_world"], initial["sensor_position_world"]]
    for candidate in episode_candidates:
        for key in ("requested_base_position_world", "snapped_base_position_world"):
            if candidate.get(key) is not None:
                points.append(candidate[key])
        if candidate.get("path_points_world"):
            points.extend(candidate["path_points_world"])
    xz = np.asarray([_xz(point) for point in points], dtype=np.float64)
    min_x, max_x = float(np.min(xz[:, 0])), float(np.max(xz[:, 0]))
    min_z, max_z = float(np.min(xz[:, 1])), float(np.max(xz[:, 1]))
    margin = 2.0
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    with create_scene_simulator(config) as context:
        raw_pathfinder = context.pathfinder
        if hasattr(raw_pathfinder, "get_topdown_view"):
            try:
                occupancy = np.asarray(raw_pathfinder.get_topdown_view(0.05))
                if occupancy.ndim >= 2 and occupancy.size:
                    ax.imshow(occupancy, cmap="Greys", alpha=0.20, origin="lower", extent=(min_x - margin, max_x + margin, min_z - margin, max_z + margin))
            except Exception:
                pass
    source_x, source_z = _xz(episode["source"]["position_world"])
    initial_x, initial_z = _xz(initial["base_position_world"])
    ax.scatter([source_x], [source_z], c="red", marker="*", s=180, label="source GT", zorder=5)
    ax.scatter([initial_x], [initial_z], c="black", marker="o", s=70, label="initial base", zorder=5)
    ax.annotate("initial", (initial_x, initial_z), xytext=(6, 6), textcoords="offset points")
    yaw = math.radians(float(initial["yaw_deg"]))
    ax.arrow(initial_x, initial_z, -math.sin(yaw), -math.cos(yaw), color="black", width=0.008, head_width=0.12, length_includes_head=True)
    for candidate in sorted(episode_candidates, key=lambda row: str(row["candidate_id"])):
        candidate_id = str(candidate["candidate_id"])
        if candidate["action_type"] == "translation":
            requested = _xz(candidate["requested_base_position_world"])
            ax.scatter([requested[0]], [requested[1]], c="tab:orange", marker="x", s=60)
            if candidate.get("snapped_base_position_world") is not None:
                snapped = _xz(candidate["snapped_base_position_world"])
                ax.scatter([snapped[0]], [snapped[1]], c="tab:blue", marker="o", s=45)
                ax.plot([requested[0], snapped[0]], [requested[1], snapped[1]], "--", color="tab:orange", linewidth=0.8)
            if candidate.get("path_points_world"):
                path = np.asarray([_xz(point) for point in candidate["path_points_world"]])
                ax.plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=1.2, alpha=0.75)
            label_point = candidate.get("snapped_base_position_world") or candidate["requested_base_position_world"]
            lx, lz = _xz(label_point)
            ax.annotate(candidate_id, (lx, lz), xytext=(4, -12), textcoords="offset points", fontsize=7)
        else:
            angle = float(candidate["yaw_deg"] - initial["yaw_deg"])
            ax.add_patch(Arc((initial_x, initial_z), 0.8, 0.8, angle=0, theta1=0, theta2=angle, color="tab:green", linewidth=2))
            direction = 1.0 if angle >= 0 else -1.0
            ax.arrow(initial_x, initial_z, direction * 0.45, 0.0, color="tab:green", width=0.008, head_width=0.12, length_includes_head=True)
            ax.annotate(candidate_id, (initial_x, initial_z), xytext=(8, 18 if direction > 0 else -24), textcoords="offset points", fontsize=7, color="tab:green")
    ax.plot([initial["sensor_position_world"][0], source_x], [initial["sensor_position_world"][2], source_z], color="purple", linewidth=1.0, alpha=0.7, label="source-listener line")
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_z - margin, max_z + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x")
    ax.set_ylabel("world z")
    ax.set_title("Pipeline V0 M3 Golden top-down: {}".format(episode["episode_id"]))
    ax.legend(loc="best")
    output = run_root / "visualizations" / "golden_topdown.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output))
    plt.close(fig)
    return str(output)
