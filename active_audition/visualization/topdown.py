"""Manifest-driven top-down visualization for Pipeline V0 datasets."""

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

from active_audition.config.loader import load_resolved_config
from active_audition.scene.pose import yaw_to_forward_world
from active_audition.scene.simulator import create_scene_simulator


def _xz(point):
    return float(point[0]), float(point[2])


def occupancy_world_mapping(pathfinder, meters_per_pixel: float, height: float):
    """Return occupancy and its world-aligned x/z extent."""

    if not hasattr(pathfinder, "get_bounds"):
        raise ValueError("PathFinder does not expose world bounds")
    bounds_min, _ = pathfinder.get_bounds()
    occupancy = np.asarray(pathfinder.get_topdown_view(float(meters_per_pixel), float(height)))
    if occupancy.ndim != 2 or occupancy.size == 0:
        raise ValueError("PathFinder returned an invalid top-down occupancy grid")
    rows, columns = occupancy.shape
    min_x, min_z = float(bounds_min[0]), float(bounds_min[2])
    extent = (min_x, min_x + columns * meters_per_pixel, min_z, min_z + rows * meters_per_pixel)
    return occupancy, extent, (min_x, min_z)


def world_to_occupancy_pixel(point, bounds_min_xz, meters_per_pixel: float, shape):
    """Map world x/z to (row, column) using the PathFinder grid origin."""

    x, z = _xz(point)
    min_x, min_z = bounds_min_xz
    column = int(math.floor((x - float(min_x)) / float(meters_per_pixel)))
    row = int(math.floor((z - float(min_z)) / float(meters_per_pixel)))
    if row < 0 or column < 0 or row >= int(shape[0]) or column >= int(shape[1]):
        return None
    return row, column


def _forward_arrow(ax, position, yaw_deg, color, label=None, length=0.45):
    forward = yaw_to_forward_world(float(yaw_deg))
    ax.arrow(float(position[0]), float(position[1]), float(forward[0]) * length, float(forward[2]) * length, color=color, width=0.008, head_width=0.12, length_includes_head=True)
    if label:
        ax.annotate(label, (float(position[0]), float(position[1])), xytext=(6, 6), textcoords="offset points", color=color, fontsize=7)
    return forward


def render_golden_topdown(
    dataset_root: str,
    config_path: str,
    run_root: Path,
    episodes: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    viewpoints: Sequence[Mapping[str, object]],
    episode_id: str = None,
    output_filename: str = "golden_topdown.png",
    title_prefix: str = "Pipeline V0 Dataset",
):
    config = load_resolved_config(config_path)
    episode = next(
        (row for row in sorted(episodes, key=lambda row: str(row["episode_id"])) if episode_id is None or str(row["episode_id"]) == str(episode_id)),
        None,
    )
    if episode is None:
        raise ValueError("episode_id is not present in frozen episodes manifest: {}".format(episode_id))
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
    occupancy_extent = None
    with create_scene_simulator(config) as context:
        raw_pathfinder = context.pathfinder
        try:
            occupancy, occupancy_extent, bounds_min_xz = occupancy_world_mapping(raw_pathfinder, 0.05, float(initial["base_position_world"][1]))
            ax.imshow(occupancy, cmap="Greys", alpha=0.20, origin="lower", extent=occupancy_extent)
        except Exception:
            occupancy = None
    source_x, source_z = _xz(episode["source"]["position_world"])
    initial_x, initial_z = _xz(initial["base_position_world"])
    ax.scatter([source_x], [source_z], c="red", marker="*", s=180, label="source GT", zorder=5)
    ax.scatter([initial_x], [initial_z], c="black", marker="o", s=70, label="initial base", zorder=5)
    ax.annotate("initial", (initial_x, initial_z), xytext=(6, 6), textcoords="offset points")
    initial_forward = _forward_arrow(ax, (initial_x, initial_z), float(initial["yaw_deg"]), "black")
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
            start_angle = math.degrees(math.atan2(float(initial_forward[2]), float(initial_forward[0])))
            ax.add_patch(Arc((initial_x, initial_z), 0.8, 0.8, angle=0, theta1=start_angle, theta2=start_angle - angle, color="tab:green", linewidth=2))
            _forward_arrow(ax, (initial_x, initial_z), float(candidate["yaw_deg"]), "tab:green", candidate_id)
    ax.plot([initial["sensor_position_world"][0], source_x], [initial["sensor_position_world"][2], source_z], color="purple", linewidth=1.0, alpha=0.7, label="source-listener line")
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_z - margin, max_z + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x")
    ax.set_ylabel("world z")
    ax.set_title("{} top-down: {}".format(title_prefix, episode["episode_id"]))
    ax.legend(loc="best")
    output = run_root / "visualizations" / output_filename
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output))
    plt.close(fig)
    return str(output)
