"""Hard validation and future resume completeness hooks for V1."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .recipe import EpisodeRecipe, validate_recipe_geometry
from .schema import RenderRecord
from .storage import V1DatasetStorage, validate_relative_path
from .schema import CLIP_DURATION_SEC, NUM_SAMPLES, SAMPLE_RATE_HZ


class ValidationError(ValueError):
    """Raised on any hard V1 validation failure."""


def validate_normalization_config(config: Mapping[str, Any]) -> None:
    normalization = config.get("normalization", config)
    for key in ("per_render", "per_viewpoint", "separate_branch_normalization"):
        if bool(normalization.get(key, False)):
            raise ValidationError("{} must be false for V1".format(key))


def validate_audio_contract(config: Mapping[str, Any]) -> None:
    """Reject config drift from the frozen V1 24 kHz / 5 s contract."""

    audio = config.get("audio", {})
    if (int(audio.get("sample_rate_hz", -1)) != SAMPLE_RATE_HZ
            or float(audio.get("clip_duration_sec", -1.0)) != CLIP_DURATION_SEC
            or int(audio.get("num_samples", -1)) != NUM_SAMPLES
            or audio.get("dtype") != "float32"):
        raise ValidationError("audio contract must be 24000 Hz / 5 s / 120000 samples / float32")
    representations = config.get("representations", {})
    binaural = representations.get("binaural", {})
    foa = representations.get("foa", {})
    if binaural.get("channels") != 2 or binaural.get("channel_order") != ["LEFT", "RIGHT"]:
        raise ValidationError("binaural contract must be 2-channel LEFT/RIGHT")
    if foa.get("channels") != 4 or foa.get("canonical", {}).get("format") != "AmbiX ACN/SN3D":
        raise ValidationError("FOA contract must be 4-channel AmbiX ACN/SN3D")
    validate_normalization_config(config)


def validate_episode(recipe: EpisodeRecipe) -> None:
    try:
        validate_recipe_geometry(recipe)
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc
    if not -180.0 <= recipe.azimuth_project_deg < 180.0:
        raise ValidationError("azimuth must be in [-180, 180)")
    unit_norm = sum(value * value for value in recipe.doa_unit_project)
    if not math.isclose(unit_norm, 1.0, abs_tol=1.0e-6):
        raise ValidationError("DOA unit vector must have norm 1")


def validate_render(record: RenderRecord) -> None:
    if record.render_status != "complete":
        return
    for path in (record.audio_path, record.rir_path):
        try:
            validate_relative_path(path)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc


def validate_pairing(episodes: Sequence[EpisodeRecipe], renders: Sequence[RenderRecord]) -> None:
    episode_ids = [episode.episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValidationError("duplicate episode_id")
    keys = [(record.episode_id, record.representation) for record in renders]
    if len(set(keys)) != len(keys):
        raise ValidationError("duplicate render key")
    by_episode = defaultdict(set)
    for record in renders:
        validate_render(record)
        if record.episode_id not in episode_ids:
            raise ValidationError("render references missing episode: {}".format(record.episode_id))
        by_episode[record.episode_id].add(record.representation)
    for episode in episodes:
        validate_episode(episode)
        if by_episode[episode.episode_id] != {"binaural", "foa"}:
            raise ValidationError("episode does not have paired binaural/foa records: {}".format(episode.episode_id))


def validate_split_leakage(source_rows: Iterable[Mapping[str, Any]], scene_rows: Iterable[Mapping[str, Any]]) -> None:
    def check(rows, identity_key, label):
        memberships = defaultdict(set)
        for row in rows:
            split = row.get("split")
            if split in (None, "UNASSIGNED"):
                continue
            memberships[str(row[identity_key])].add(str(split))
        leaked = sorted(key for key, splits in memberships.items() if len(splits) > 1)
        if leaked:
            raise ValidationError("{} identity leakage across splits: {}".format(label, ", ".join(leaked)))

    check(source_rows, "base_clip_id", "source")
    check(scene_rows, "scene_id", "scene")


def payload_is_complete(storage: V1DatasetStorage, record: RenderRecord) -> bool:
    """Return whether a completed record can be safely skipped on resume."""

    if record.render_status != "complete" or not record.audio_path or not record.rir_path:
        return False
    try:
        audio = storage.payload_path(record.audio_path)
        rir = storage.payload_path(record.rir_path)
    except ValueError:
        return False
    return audio.is_file() and rir.is_file()


def validate_dataset_fixture(storage: V1DatasetStorage, episodes: Sequence[EpisodeRecipe], renders: Sequence[RenderRecord], config: Mapping[str, Any]) -> None:
    validate_audio_contract(config)
    validate_pairing(episodes, renders)
    for record in renders:
        if record.render_status == "complete" and not payload_is_complete(storage, record):
            raise ValidationError("complete render payload is missing: {} / {}".format(record.episode_id, record.representation))
