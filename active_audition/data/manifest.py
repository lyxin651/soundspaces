"""Deterministic, atomic M1 Episode/Candidate manifest writing."""

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from active_audition.data.storage import DatasetStorage, StorageError
from active_audition.types import Candidate, EpisodeSpec


class ManifestError(ValueError):
    """Raised when manifest rows are invalid or duplicate."""


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _row(value: Any) -> Mapping[str, Any]:
    result = _json_safe(value)
    if not isinstance(result, dict):
        raise ManifestError("manifest row must be an object")
    return result


def _write_unique(storage: DatasetStorage, name: str, rows: Sequence[Mapping[str, Any]], key_name: str) -> Path:
    normalized = [_row(row) for row in rows]
    keys = [row.get(key_name) for row in normalized]
    if any(key is None for key in keys) or len(keys) != len(set(keys)):
        raise ManifestError("duplicate or missing {} manifest key".format(key_name))
    normalized.sort(key=lambda row: str(row[key_name]))
    return storage.atomic_write_jsonl(name, normalized)


def write_plan_manifests(
    output_root: str,
    episodes: Iterable[EpisodeSpec],
    candidates: Iterable[Candidate],
) -> Mapping[str, str]:
    """Atomically rewrite deterministic multi-Episode M1 manifests."""

    episode_list = list(episodes)
    candidate_list = list(candidates)
    storage = DatasetStorage(output_root)
    storage.ensure_writable()
    episodes_path = _write_unique(storage, "episodes.jsonl", episode_list, "episode_id")
    candidate_rows = [_row(candidate) for candidate in candidate_list]
    keys = [(row.get("episode_id"), row.get("candidate_id")) for row in candidate_rows]
    if any(None in key for key in keys) or len(keys) != len(set(keys)):
        raise ManifestError("duplicate or missing candidate manifest key")
    candidate_rows.sort(key=lambda row: (str(row["episode_id"]), str(row["candidate_id"])))
    candidates_path = storage.atomic_write_jsonl("candidates.jsonl", candidate_rows)
    return {"episodes": str(episodes_path), "candidates": str(candidates_path)}
