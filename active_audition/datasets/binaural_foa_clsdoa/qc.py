"""Model-independent QC/report schema skeleton; no fake acoustic values."""

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .recipe import EpisodeRecipe
from .schema import RenderRecord


class QCError(ValueError):
    """Raised when a QC bin configuration is invalid."""


TINY_DEFAULT_AZIMUTH_BINS = (
    ("front", -22.5, 22.5),
    ("right", 22.5, 157.5),
    ("back", 157.5, 180.0),
    ("left", -180.0, -22.5),
)
TINY_DEFAULT_DISTANCE_BINS = (("near", 0.0, 1.0), ("mid", 1.0, 3.0), ("far", 3.0, math.inf))
TINY_DEFAULT_ELEVATION_BANDS = (("down", -90.0, -15.0), ("level", -15.0, 15.0), ("up", 15.0, 90.0))


def _normalize_bins(value: Any, default: Sequence[Sequence[Any]], name: str):
    if value is None:
        return tuple((str(label), float(low), float(high)) for label, low, high in default)
    if isinstance(value, Mapping):
        value = [(label, limits[0], limits[1]) for label, limits in value.items()]
    try:
        result = []
        for item in value:
            if isinstance(item, Mapping):
                label = item.get("label", item.get("name"))
                low = item.get("min", item.get("low"))
                high = item.get("max", item.get("high"))
            else:
                label, low, high = item
            low = float(low)
            high = float(high)
            if not isinstance(label, str) or not label or not math.isfinite(low) or math.isnan(high) or low >= high:
                raise QCError("invalid {} bin".format(name))
            result.append((label, low, high))
    except (TypeError, ValueError, IndexError) as exc:
        raise QCError("{} bins must contain label/min/max entries".format(name)) from exc
    if not result or len({item[0] for item in result}) != len(result):
        raise QCError("{} bins must be non-empty and uniquely named".format(name))
    return tuple(result)


def _configured_bins(qc_config: Optional[Mapping[str, Any]], name: str, default):
    config = qc_config or {}
    bins = config.get("bins", {}) if isinstance(config.get("bins", {}), Mapping) else {}
    return _normalize_bins(config.get(name, bins.get(name)), default, name)


def _histogram(values: Iterable[float], bins):
    result = {str(label): 0 for label, _, _ in bins}
    for value in values:
        if value is None or not math.isfinite(float(value)):
            continue
        for index, (label, low, high) in enumerate(bins):
            if low <= value < high or (index == len(bins) - 1 and value == high):
                result[str(label)] += 1
                break
    return result


def _cross_table(episodes: Sequence[EpisodeRecipe], value_fn, category_fn):
    table = defaultdict(Counter)
    for episode in episodes:
        table[str(value_fn(episode))][str(category_fn(episode))] += 1
    return {key: dict(sorted(values.items())) for key, values in sorted(table.items())}


def _reuse_statistics(counts: Iterable[int]) -> Mapping[str, Any]:
    values = np.asarray(sorted(int(value) for value in counts), dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(np.max(values)),
    }


def build_distribution_report(
    episodes: Sequence[EpisodeRecipe],
    renders: Sequence[RenderRecord],
    qc_config: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Aggregate configurable metadata distributions; acoustic metrics remain NOT_RUN."""

    class_counts = Counter(str(episode.class_id) for episode in episodes)
    split_counts = Counter(str(episode.split) for episode in episodes)
    scene_counts = Counter(str(episode.scene_family) for episode in episodes)
    source_dataset_counts = Counter(str(episode.source_dataset) for episode in episodes)
    azimuth_bins = _configured_bins(qc_config, "azimuth_bins", TINY_DEFAULT_AZIMUTH_BINS)
    distance_bins = _configured_bins(qc_config, "distance_bins", TINY_DEFAULT_DISTANCE_BINS)
    elevation_bands = _configured_bins(qc_config, "elevation_bands", TINY_DEFAULT_ELEVATION_BANDS)
    source_reuse = Counter(str(episode.base_clip_id) for episode in episodes)
    base_clip_ids_by_class = defaultdict(set)
    for episode in episodes:
        base_clip_ids_by_class[str(episode.class_id)].add(str(episode.base_clip_id))
    source_reuse_stats = _reuse_statistics(source_reuse.values())
    return {
        "episode_count": len(episodes),
        "render_counts": dict(sorted(Counter(record.render_status for record in renders).items())),
        "class_count": dict(sorted(class_counts.items())),
        "split_count": dict(sorted(split_counts.items())),
        "scene_family_count": dict(sorted(scene_counts.items())),
        "source_dataset_count": dict(sorted(source_dataset_counts.items())),
        "class_by_split": _cross_table(episodes, lambda episode: episode.class_id, lambda episode: episode.split),
        "class_by_scene_family": _cross_table(episodes, lambda episode: episode.class_id, lambda episode: episode.scene_family),
        "class_by_source_dataset": _cross_table(episodes, lambda episode: episode.class_id, lambda episode: episode.source_dataset),
        "class_by_azimuth_bin": _cross_table(episodes, lambda episode: episode.class_id, lambda episode: _bin_label(episode.azimuth_project_deg, azimuth_bins)),
        "class_by_distance_bin": _cross_table(episodes, lambda episode: episode.class_id, lambda episode: _bin_label(episode.distance_m, distance_bins)),
        "class_by_elevation_band": _cross_table(episodes, lambda episode: episode.class_id, lambda episode: _bin_label(episode.elevation_project_deg, elevation_bands)),
        "azimuth_bins": _histogram((episode.azimuth_project_deg for episode in episodes), azimuth_bins),
        "distance_bins": _histogram((episode.distance_m for episode in episodes), distance_bins),
        "elevation_bands": _histogram((episode.elevation_project_deg for episode in episodes), elevation_bands),
        "unique_base_clip_id_by_class": {key: len(values) for key, values in sorted(base_clip_ids_by_class.items())},
        "base_clip_ids_by_class": {key: sorted(values) for key, values in sorted(base_clip_ids_by_class.items())},
        "source_reuse": {
            "unique_source_identities": len(source_reuse),
            "identity_key": "base_clip_id",
            "statistics": source_reuse_stats,
            "mean": source_reuse_stats["mean"],
            "median": source_reuse_stats["median"],
            "p95": source_reuse_stats["p95"],
            "max": source_reuse_stats["max"],
            "counts": dict(sorted(source_reuse.items())),
        },
        "acoustic_qc": {"status": "NOT_RUN", "metrics": ["binaural_rms_l_r", "ild", "interaural_correlation", "foa_channel_energy"]},
    }


def _bin_label(value: float, bins) -> str:
    value = float(value)
    for index, (label, low, high) in enumerate(bins):
        if low <= value < high or (index == len(bins) - 1 and value == high):
            return label
    return "OUT_OF_CONFIG_BINS"
