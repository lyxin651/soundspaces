"""Schema-only source registry validation for future Step 2A data."""

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from active_audition.data.audio import resample_waveform


SOURCE_FIELDS = (
    "source_clip_id", "base_clip_id", "canonical_class", "source_dataset", "source_label",
    "original_id", "original_path", "original_sample_rate", "original_channels", "duration_sec",
    "crop_start_sec", "crop_end_sec", "source_offset_policy", "license", "split", "qc_status",
    "qc_notes", "pretrain_seen_status", "sha256",
)


class SourceRegistryError(ValueError):
    """Raised when a source registry row violates the V1 schema."""


def validate_source_rows(rows: Iterable[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    normalized = [dict(row) for row in rows]
    seen = set()
    for row in normalized:
        missing = [key for key in SOURCE_FIELDS if key not in row]
        if missing:
            raise SourceRegistryError("source row missing fields: {}".format(", ".join(missing)))
        clip_id = str(row["source_clip_id"])
        if not clip_id or clip_id in seen:
            raise SourceRegistryError("duplicate or empty source_clip_id: {}".format(clip_id))
        seen.add(clip_id)
        duration = float(row["duration_sec"])
        if not math.isfinite(duration) or duration <= 0.0:
            raise SourceRegistryError("duration_sec must be finite and > 0")
        start = float(row["crop_start_sec"])
        end = float(row["crop_end_sec"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start or end > duration:
            raise SourceRegistryError("crop interval must lie inside the positive source duration")
        if row["split"] not in ("train", "val", "test", "UNASSIGNED"):
            raise SourceRegistryError("invalid source split: {}".format(row["split"]))
    return tuple(normalized)


def read_source_registry(path: str) -> Sequence[Mapping[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return validate_source_rows(csv.DictReader(handle))


def build_observation_timeline(
    waveform: Any,
    source_sample_rate_hz: int,
    source_offset_sec: float,
    target_sample_rate_hz: int = 24000,
    clip_duration_sec: float = 5.0,
) -> np.ndarray:
    """Place a short source on a deterministic zero-padded canonical timeline."""

    source = np.asarray(waveform)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise SourceRegistryError("short-source timeline requires finite non-empty mono audio")
    source_rate = int(source_sample_rate_hz)
    target_rate = int(target_sample_rate_hz)
    offset = float(source_offset_sec)
    if source_rate <= 0 or target_rate <= 0 or not math.isfinite(offset) or offset < 0.0:
        raise SourceRegistryError("invalid source timeline rate or offset")
    duration = float(source.size) / float(source_rate)
    if duration > float(clip_duration_sec) or offset > float(clip_duration_sec) - duration + 1.0e-9:
        raise SourceRegistryError("source offset must keep the complete source inside the 5 s timeline")
    canonical = resample_waveform(source, source_rate, target_rate)
    output_count = int(round(float(clip_duration_sec) * target_rate))
    start = int(round(offset * target_rate))
    if start + canonical.size > output_count:
        raise SourceRegistryError("resampled source does not fit the timeline")
    output = np.zeros(output_count, dtype=np.float32)
    output[start : start + canonical.size] = canonical
    return output
