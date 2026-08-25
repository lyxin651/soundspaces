"""Fixed and sampled episode generation for M1 planning."""

from typing import Any, Dict

import numpy as np

from active_audition.data.catalog import episode_seed
from active_audition.navigation.pathfinder import PathFinderAdapter
from active_audition.scene.pose import listener_sensor_position
from active_audition.types import EpisodeSpec, ListenerPose, SourceSpec


class EpisodeError(ValueError):
    """Raised when an episode violates the frozen V0 geometry contract."""


def _world_xyz(value: Any):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise EpisodeError("episode position must be finite xyz")
    return tuple(float(item) for item in array)


def _source_spec(config: Dict[str, Any], source_anchor_base: Any) -> SourceSpec:
    source_position = np.asarray(source_anchor_base, dtype=np.float64)
    source_position += np.asarray([0.0, float(config["source"]["height_m"]), 0.0])
    golden = config["golden"]
    return SourceSpec(
        position_world=_world_xyz(source_position),
        audio_id=str(golden["source_audio_id"]),
        segment_start_sec=float(golden["segment_start_sec"]),
        segment_duration_sec=float(config["dry_audio"]["segment_duration_sec"]),
        gain_db=float(golden["gain_db"]),
    )


def _episode(
    config: Dict[str, Any],
    episode_id: str,
    seed: int,
    listener_base: Any,
    source_anchor_base: Any,
    pathfinder: PathFinderAdapter,
) -> EpisodeSpec:
    listener_base = _world_xyz(listener_base)
    source_anchor_base = _world_xyz(source_anchor_base)
    if not pathfinder.is_navigable(listener_base):
        raise EpisodeError("listener base is not navigable")
    if not pathfinder.is_navigable(source_anchor_base):
        raise EpisodeError("source anchor is not navigable")
    if pathfinder.geodesic_distance(listener_base, source_anchor_base) is None:
        raise EpisodeError("listener and source are not in the same reachable region")
    listener = ListenerPose(
        base_position_world=listener_base,
        sensor_position_world=_world_xyz(listener_sensor_position(listener_base)),
        yaw_deg=float(config["golden"]["listener_yaw_deg"]),
    )
    return EpisodeSpec(
        schema_version=str(config["experiment"]["schema_version"]),
        episode_id=episode_id,
        scene_id=str(config["golden"]["scene_id"]),
        episode_seed=int(seed),
        source=_source_spec(config, source_anchor_base),
        listener_initial=listener,
    )


def fixed_golden_episode(
    config: Dict[str, Any], pathfinder: PathFinderAdapter, episode_id: str = "ep_000001"
) -> EpisodeSpec:
    """Build the frozen M0/M0.1 Golden Episode without resampling its pose."""

    golden = config["golden"]
    return _episode(
        config,
        episode_id,
        episode_seed(config["experiment"]["global_seed"], episode_id),
        golden["listener_base_position_world"],
        golden["source_anchor_base_position_world"],
        pathfinder,
    )


def sampled_episode(
    config: Dict[str, Any], pathfinder: PathFinderAdapter, episode_id: str
) -> EpisodeSpec:
    """Sample a listener/source pair with a per-episode deterministic RNG."""

    seed = episode_seed(config["experiment"]["global_seed"], episode_id)
    rng = np.random.default_rng(seed)
    for _ in range(128):
        listener = pathfinder.sample_navigable_point(rng)
        source = pathfinder.sample_navigable_point(rng)
        if pathfinder.geodesic_distance(listener, source) is not None:
            return _episode(config, episode_id, seed, listener, source, pathfinder)
    raise EpisodeError("could not sample a reachable listener/source pair")
