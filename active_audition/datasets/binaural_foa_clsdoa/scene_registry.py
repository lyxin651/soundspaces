"""Schema-only scene registry validation for future Step 2B admission."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCENE_FIELDS = ("scene_id", "scene_family", "scene_asset", "stage_config", "navmesh", "semantic_info", "materials_mode", "unit_scale", "resource_hash", "admitted", "exclude_reason", "split")


class SceneRegistryError(ValueError):
    """Raised when a scene registry row violates the V1 schema."""


def validate_scene_rows(rows: Mapping[str, Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    normalized = []
    seen = set()
    for scene_id, value in rows.items():
        row = dict(value)
        row.setdefault("scene_id", scene_id)
        missing = [key for key in SCENE_FIELDS if key not in row]
        if missing:
            raise SceneRegistryError("scene row missing fields: {}".format(", ".join(missing)))
        if row["scene_id"] in seen:
            raise SceneRegistryError("duplicate scene_id: {}".format(row["scene_id"]))
        seen.add(row["scene_id"])
        if row["admitted"] not in ("NOT_RUN", "UNASSIGNED", "PASS", "FAIL"):
            raise SceneRegistryError("invalid scene admission state: {}".format(row["admitted"]))
        if row["split"] not in ("train", "val", "test", "UNASSIGNED"):
            raise SceneRegistryError("invalid scene split: {}".format(row["split"]))
        if float(row["unit_scale"]) <= 0.0:
            raise SceneRegistryError("unit_scale must be positive")
        normalized.append(row)
    return tuple(sorted(normalized, key=lambda row: str(row["scene_id"])))


def read_scene_registry(path: str) -> Sequence[Mapping[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    scenes = raw.get("scenes", raw)
    if not isinstance(scenes, Mapping) or not scenes:
        raise SceneRegistryError("scene registry must be a non-empty mapping")
    return validate_scene_rows(scenes)
