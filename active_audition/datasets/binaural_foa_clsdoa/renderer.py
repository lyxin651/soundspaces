"""Paired binaural/FOA render interface without a real SoundSpaces dependency."""

from abc import ABC, abstractmethod
from typing import Sequence

from .recipe import EpisodeRecipe
from .schema import RenderRecord, REPRESENTATIONS


class PairedRenderError(ValueError):
    """Raised when a paired renderer violates the shared recipe contract."""


class PairedRenderer(ABC):
    """Future backends implement one representation at a time from one recipe."""

    @abstractmethod
    def render_episode(self, recipe: EpisodeRecipe, representation: str) -> RenderRecord:
        raise NotImplementedError


def render_pair(renderer: PairedRenderer, recipe: EpisodeRecipe) -> Sequence[RenderRecord]:
    """Render both representations from the same immutable recipe object."""

    records = []
    for representation in REPRESENTATIONS:
        if representation not in recipe.representations:
            raise PairedRenderError("recipe does not require {}".format(representation))
        record = renderer.render_episode(recipe, representation)
        if not isinstance(record, RenderRecord) or record.episode_id != recipe.episode_id or record.representation != representation:
            raise PairedRenderError("renderer returned a record inconsistent with the shared recipe")
        records.append(record)
    return tuple(records)
