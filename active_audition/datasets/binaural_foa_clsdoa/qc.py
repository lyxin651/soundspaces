"""Model-independent QC/report schema skeleton; no fake acoustic values."""

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .recipe import EpisodeRecipe
from .schema import RenderRecord


def _histogram(values, bins):
    result = {str(label): 0 for label, _, _ in bins}
    for value in values:
        for index, (label, low, high) in enumerate(bins):
            if low <= value < high or (index == len(bins) - 1 and value == high):
                result[str(label)] += 1
                break
    return result


def build_distribution_report(episodes: Sequence[EpisodeRecipe], renders: Sequence[RenderRecord]) -> Mapping[str, Any]:
    """Aggregate metadata distributions; acoustic metrics are intentionally absent."""

    class_counts = Counter(str(episode.class_id) for episode in episodes)
    split_counts = Counter(str(episode.split) for episode in episodes)
    scene_counts = Counter(str(episode.scene_family) for episode in episodes)
    source_dataset_counts = Counter(str(episode.source_dataset) for episode in episodes)
    source_reuse = Counter(str(episode.source_clip_id) for episode in episodes)
    return {
        "episode_count": len(episodes),
        "render_counts": Counter(record.render_status for record in renders),
        "class_count": dict(sorted(class_counts.items())),
        "class_by_split": {str(split): count for split, count in sorted(split_counts.items())},
        "scene_family_count": dict(sorted(scene_counts.items())),
        "source_dataset_count": dict(sorted(source_dataset_counts.items())),
        "azimuth_bins": _histogram((episode.azimuth_project_deg for episode in episodes), [("front", -22.5, 22.5), ("right", 22.5, 157.5), ("back", 157.5, 180.0), ("left", -180.0, -22.5)]),
        "distance_bins": _histogram((episode.distance_m for episode in episodes), [("near", 0.0, 1.0), ("mid", 1.0, 3.0), ("far", 3.0, float("inf"))]),
        "elevation_bands": _histogram((episode.elevation_project_deg for episode in episodes), [("down", -90.0, -15.0), ("level", -15.0, 15.0), ("up", 15.0, 90.0)]),
        "source_reuse": {
            "unique_source_identities": len(source_reuse),
            "mean": (sum(source_reuse.values()) / float(len(source_reuse))) if source_reuse else 0.0,
            "counts": dict(sorted(source_reuse.items())),
        },
        "acoustic_qc": {"status": "NOT_RUN", "metrics": ["binaural_rms_l_r", "ild", "interaural_correlation", "foa_channel_energy"]},
    }
