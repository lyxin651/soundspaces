"""Deterministic, atomic V1 EpisodeRecipe/RenderRecord manifests."""

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from .recipe import EpisodeRecipe
from .schema import RenderRecord
from .storage import V1DatasetStorage


class ManifestError(ValueError):
    """Raised for malformed or duplicate V1 manifest rows."""


def _json_line(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ManifestError("manifest contains non-JSON-safe value") from exc


def _write_rows(storage: V1DatasetStorage, name: str, rows: Sequence[Mapping[str, Any]], key_fn) -> Path:
    text = "".join(_json_line(row) + "\n" for row in sorted(rows, key=key_fn))
    return storage.write_bytes("manifests/{}".format(name), text.encode("utf-8"))


def write_manifests(storage: V1DatasetStorage, episodes: Iterable[EpisodeRecipe], renders: Iterable[RenderRecord]) -> Mapping[str, str]:
    episode_rows = [episode.to_dict() if isinstance(episode, EpisodeRecipe) else episode for episode in episodes]
    render_rows = [render.to_dict() if isinstance(render, RenderRecord) else render for render in renders]
    episode_ids = [row.get("episode_id") for row in episode_rows]
    if any(not isinstance(value, str) or not value for value in episode_ids) or len(set(episode_ids)) != len(episode_ids):
        raise ManifestError("duplicate or missing episode_id")
    render_keys = [(row.get("episode_id"), row.get("representation")) for row in render_rows]
    if any(None in key for key in render_keys) or len(set(render_keys)) != len(render_keys):
        raise ManifestError("duplicate or missing render key")
    # Serialize and validate all rows before either replacement, so bad input cannot rewrite half a batch.
    for row in episode_rows + render_rows:
        _json_line(row)
    storage.ensure_writable()
    episodes_path = _write_rows(storage, "episodes.jsonl", episode_rows, lambda row: str(row["episode_id"]))
    renders_path = _write_rows(storage, "renders.jsonl", render_rows, lambda row: (str(row["episode_id"]), str(row["representation"])))
    return {"episodes": str(episodes_path), "renders": str(renders_path)}


def read_jsonl(path: str) -> List[Mapping[str, Any]]:
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("invalid JSONL at {}:{}".format(path, line_number)) from exc
            if not isinstance(value, dict):
                raise ManifestError("manifest row must be an object")
            result.append(value)
    return result
