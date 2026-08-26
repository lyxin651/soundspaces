"""Fixed and sampled episode generation for M1 planning."""

from typing import Any, Dict

import numpy as np

from active_audition.data.catalog import episode_seed, load_dry_audio_registry
from active_audition.navigation.pathfinder import PathFinderAdapter
from active_audition.scene.pose import listener_sensor_position, normalize_yaw_deg
from active_audition.types import EpisodeSpec, ListenerPose, SourceSpec


class EpisodeError(ValueError):
    """Raised when an episode violates the frozen V0 geometry contract."""


def _world_xyz(value: Any):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise EpisodeError("episode position must be finite xyz")
    return tuple(float(item) for item in array)


def _source_spec(
    config: Dict[str, Any],
    source_anchor_base: Any,
    audio_id: str,
    segment_start_sec: float,
    gain_db: float,
) -> SourceSpec:
    source_position = np.asarray(source_anchor_base, dtype=np.float64)
    source_position += np.asarray([0.0, float(config["source"]["height_m"]), 0.0])
    return SourceSpec(
        position_world=_world_xyz(source_position),
        audio_id=str(audio_id),
        segment_start_sec=float(segment_start_sec),
        segment_duration_sec=float(config["dry_audio"]["segment_duration_sec"]),
        gain_db=float(gain_db),
    )


def _episode(
    config: Dict[str, Any],
    episode_id: str,
    seed: int,
    listener_base: Any,
    source_anchor_base: Any,
    pathfinder: PathFinderAdapter,
    listener_yaw_deg: float,
    scene_id: str,
    audio_id: str,
    segment_start_sec: float,
    gain_db: float,
) -> EpisodeSpec:
    listener_base = _world_xyz(listener_base)
    source_anchor_base = _world_xyz(source_anchor_base)
    if np.array_equal(listener_base, source_anchor_base):
        raise EpisodeError("listener and source anchors must not be numerically identical")
    if not pathfinder.is_navigable(listener_base):
        raise EpisodeError("listener base is not navigable")
    if not pathfinder.is_navigable(source_anchor_base):
        raise EpisodeError("source anchor is not navigable")
    if pathfinder.geodesic_distance(listener_base, source_anchor_base) is None:
        raise EpisodeError("listener and source are not in the same reachable region")
    listener = ListenerPose(
        base_position_world=listener_base,
        sensor_position_world=_world_xyz(listener_sensor_position(listener_base)),
        yaw_deg=normalize_yaw_deg(listener_yaw_deg),
    )
    source = _source_spec(config, source_anchor_base, audio_id, segment_start_sec, gain_db)
    source_listener_euclidean = float(
        np.linalg.norm(
            np.asarray(source.position_world, dtype=np.float64)
            - np.asarray(listener.sensor_position_world, dtype=np.float64)
        )
    )
    source_listener_geodesic = pathfinder.geodesic_distance(listener_base, source_anchor_base)
    if source_listener_geodesic is None:
        raise EpisodeError("listener and source are not in the same reachable region")
    return EpisodeSpec(
        schema_version=str(config["experiment"]["schema_version"]),
        episode_id=episode_id,
        scene_id=str(scene_id),
        episode_seed=int(seed),
        source=source,
        listener_initial=listener,
        source_listener_euclidean_m=source_listener_euclidean,
        source_listener_geodesic_m=float(source_listener_geodesic),
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
        float(golden["listener_yaw_deg"]),
        str(golden["scene_id"]),
        str(golden["source_audio_id"]),
        float(golden["segment_start_sec"]),
        float(golden["gain_db"]),
    )


def sampled_episode(
    config: Dict[str, Any],
    pathfinder: PathFinderAdapter,
    episode_id: str,
    dry_audio: Dict[str, Dict[str, Any]] = None,
    scene_id: str = None,
    sampling_diagnostics: Dict[str, int] = None,
) -> EpisodeSpec:
    """Sample a listener/source pair with a per-episode deterministic RNG."""

    def reject(name: str) -> None:
        if sampling_diagnostics is not None:
            sampling_diagnostics[name] = int(sampling_diagnostics.get(name, 0)) + 1

    if sampling_diagnostics is not None:
        sampling_diagnostics["total_sampling_attempts"] = int(sampling_diagnostics.get("total_sampling_attempts", 0))

    seed = episode_seed(config["experiment"]["global_seed"], episode_id)
    rng = np.random.default_rng(seed)
    if dry_audio is None:
        repo_root = config["_repo_root"]
        dry_audio = load_dry_audio_registry(config["registries"]["dry_audio_path"], repo_root)
    audio_ids = sorted(dry_audio)
    scene_ids = sorted(str(item) for item in config["scene"]["ids"])
    if scene_id is not None:
        scene_id = str(scene_id)
        if scene_id not in scene_ids:
            raise EpisodeError("sampled scene is not configured: {}".format(scene_id))
    if not audio_ids or not scene_ids:
        raise EpisodeError("sampled episode requires non-empty scene and dry-audio registries")
    scene_id = scene_id or scene_ids[int(rng.integers(0, len(scene_ids)))]
    audio_id = audio_ids[int(rng.integers(0, len(audio_ids)))]
    audio_duration = float(dry_audio[audio_id]["duration_sec"])
    segment_duration = float(config["dry_audio"]["segment_duration_sec"])
    if audio_duration < segment_duration:
        raise EpisodeError("sampled dry audio is shorter than the required segment")
    max_start = audio_duration - segment_duration
    segment_start_sec = float(rng.uniform(0.0, max_start)) if max_start else 0.0
    gain_db = float(config["source"]["gain_db"])
    listener_yaw_deg = float(rng.uniform(-180.0, 180.0))
    for _ in range(128):
        if sampling_diagnostics is not None:
            sampling_diagnostics["total_sampling_attempts"] += 1
        listener = pathfinder.sample_navigable_point(rng)
        source = pathfinder.sample_navigable_point(rng)
        if np.array_equal(listener, source):
            reject("identical_anchor_rejections")
            continue
        if not pathfinder.is_navigable(listener) or not pathfinder.is_navigable(source):
            reject("unreachable_rejections")
            continue
        path = pathfinder.shortest_path(listener, source)
        if path.found and path.geodesic_distance_m is not None and path.points_world:
            source_position = np.asarray(source, dtype=np.float64) + np.asarray([0.0, float(config["source"]["height_m"]), 0.0])
            listener_sensor = np.asarray(listener_sensor_position(listener), dtype=np.float64)
            source_listener_distance = float(np.linalg.norm(source_position - listener_sensor))
            minimum_distance = config["episode"].get("source_listener_min_distance_m")
            maximum_distance = config["episode"].get("source_listener_max_distance_m")
            if minimum_distance is not None and source_listener_distance < float(minimum_distance):
                reject("source_too_close_rejections")
                continue
            if maximum_distance is not None and source_listener_distance > float(maximum_distance):
                reject("source_too_far_rejections")
                continue
            if sampling_diagnostics is not None:
                sampling_diagnostics["accepted_episode_count"] = int(sampling_diagnostics.get("accepted_episode_count", 0)) + 1
            return _episode(
                config,
                episode_id,
                seed,
                listener,
                source,
                pathfinder,
                listener_yaw_deg,
                scene_id,
                audio_id,
                segment_start_sec,
                gain_db,
            )
        reject("unreachable_rejections")
    if sampling_diagnostics is not None:
        sampling_diagnostics["sampling_failure_count"] = int(sampling_diagnostics.get("sampling_failure_count", 0)) + 1
    raise EpisodeError("could not sample a reachable listener/source pair")


def generate_episodes(
    config: Dict[str, Any], pathfinder: PathFinderAdapter, dry_audio: Dict[str, Dict[str, Any]],
    sampling_diagnostics: Dict[str, int] = None,
):
    """Generate the configured deterministic Episode batch in one context."""

    mode = config["episode"]["mode"]
    count = int(config["episode"].get("count", 1))
    if mode in ("fixed", "fixed_or_sampled"):
        if count != 1:
            raise EpisodeError("fixed Golden planning expects episode.count=1")
        return [fixed_golden_episode(config, pathfinder)]
    if mode != "sampled":
        raise EpisodeError("unsupported episode mode: {}".format(mode))
    scene_ids = sorted(str(scene_id) for scene_id in config["scene"]["ids"])
    if len(scene_ids) != 1:
        raise EpisodeError("M3 sampled planning requires exactly one active scene")
    return [
        sampled_episode(
            config,
            pathfinder,
            "ep_{:06d}".format(index),
            dry_audio=dry_audio,
            scene_id=scene_ids[0],
            sampling_diagnostics=sampling_diagnostics,
        )
        for index in range(1, count + 1)
    ]
