"""Immutable, model-independent ClassDOA V1 episode recipes."""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .geometry import project_geometry, validate_geometry_independently
from .schema import SCHEMA_VERSION, SchemaError, _finite_float, _required_string, _representations, _vector, validate_class_id


@dataclass(frozen=True)
class EpisodeRecipe:
    episode_id: str
    split: str
    scene_id: str
    scene_family: str
    source_clip_id: str
    base_clip_id: str
    source_dataset: str
    class_id: int
    source_position_world: Tuple[float, float, float]
    source_gain_db: float
    source_offset_sec: float
    listener_base_position_world: Tuple[float, float, float]
    listener_sensor_position_world: Tuple[float, float, float]
    listener_yaw_deg: float
    label_class_id: int
    azimuth_project_deg: float
    elevation_project_deg: float
    doa_unit_project: Tuple[float, float, float]
    distance_m: float
    representations: Tuple[str, ...] = ("binaural", "foa")
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_string(self.episode_id, "episode_id")
        _required_string(self.split, "split")
        _required_string(self.scene_id, "scene_id")
        _required_string(self.scene_family, "scene_family")
        _required_string(self.source_clip_id, "source_clip_id")
        _required_string(self.base_clip_id, "base_clip_id")
        _required_string(self.source_dataset, "source_dataset")
        validate_class_id(self.class_id)
        validate_class_id(self.label_class_id, "label_class_id")
        if self.class_id != self.label_class_id:
            raise SchemaError("label_class_id must equal class_id")
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported schema_version: {}".format(self.schema_version))
        _vector(self.source_position_world, "source_position_world")
        _vector(self.listener_base_position_world, "listener_base_position_world")
        _vector(self.listener_sensor_position_world, "listener_sensor_position_world")
        _vector(self.doa_unit_project, "doa_unit_project")
        _finite_float(self.source_gain_db, "source_gain_db")
        if _finite_float(self.source_offset_sec, "source_offset_sec") < 0.0:
            raise SchemaError("source_offset_sec must be non-negative")
        _finite_float(self.listener_yaw_deg, "listener_yaw_deg")
        distance = _finite_float(self.distance_m, "distance_m")
        if distance <= 0.0:
            raise SchemaError("distance_m must be positive")
        azimuth = _finite_float(self.azimuth_project_deg, "azimuth_project_deg")
        if not -180.0 <= azimuth < 180.0:
            raise SchemaError("azimuth_project_deg must be in [-180, 180)")
        _finite_float(self.elevation_project_deg, "elevation_project_deg")
        unit_norm = sum(float(value) * float(value) for value in self.doa_unit_project)
        if not abs(unit_norm - 1.0) <= 1.0e-6:
            raise SchemaError("doa_unit_project must have norm 1")
        _representations(self.representations)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "split": self.split,
            "scene": {"scene_id": self.scene_id, "scene_family": self.scene_family},
            "source": {
                "source_clip_id": self.source_clip_id,
                "base_clip_id": self.base_clip_id,
                "source_dataset": self.source_dataset,
                "class_id": self.class_id,
                "source_position_world": list(self.source_position_world),
                "source_gain_db": self.source_gain_db,
                "source_offset_sec": self.source_offset_sec,
            },
            "listener": {
                "base_position_world": list(self.listener_base_position_world),
                "sensor_position_world": list(self.listener_sensor_position_world),
                "yaw_deg": self.listener_yaw_deg,
            },
            "label": {
                "class_id": self.label_class_id,
                "azimuth_project_deg": self.azimuth_project_deg,
                "elevation_project_deg": self.elevation_project_deg,
                "doa_unit_project": list(self.doa_unit_project),
                "distance_m": self.distance_m,
            },
            "representations": {name: {"required": True} for name in self.representations},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeRecipe":
        required = ("episode_id", "split", "scene", "source", "listener", "label", "representations")
        missing = [key for key in required if key not in value]
        if missing:
            raise SchemaError("EpisodeRecipe missing fields: {}".format(", ".join(missing)))
        scene = value["scene"]
        source = value["source"]
        listener = value["listener"]
        label = value["label"]
        representations = value["representations"]
        if not all(isinstance(item, Mapping) for item in (scene, source, listener, label)):
            raise SchemaError("EpisodeRecipe nested sections must be objects")
        if isinstance(representations, Mapping):
            representations = tuple(key for key, item in representations.items() if isinstance(item, Mapping) and item.get("required") is True)
        return cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            episode_id=value["episode_id"],
            split=value["split"],
            scene_id=scene["scene_id"],
            scene_family=scene["scene_family"],
            source_clip_id=source["source_clip_id"],
            base_clip_id=source["base_clip_id"],
            source_dataset=source["source_dataset"],
            class_id=source["class_id"],
            source_position_world=tuple(source["source_position_world"]),
            source_gain_db=source["source_gain_db"],
            source_offset_sec=source["source_offset_sec"],
            listener_base_position_world=tuple(listener["base_position_world"]),
            listener_sensor_position_world=tuple(listener["sensor_position_world"]),
            listener_yaw_deg=listener["yaw_deg"],
            label_class_id=label["class_id"],
            azimuth_project_deg=label["azimuth_project_deg"],
            elevation_project_deg=label["elevation_project_deg"],
            doa_unit_project=tuple(label["doa_unit_project"]),
            distance_m=label["distance_m"],
            representations=tuple(representations),
        )


def make_episode_recipe(
    *, episode_id: str, split: str, scene_id: str, scene_family: str,
    source_clip_id: str, base_clip_id: str, source_dataset: str, class_id: int,
    source_position_world: Any, source_gain_db: float, source_offset_sec: float,
    listener_base_position_world: Any, listener_sensor_position_world: Any,
    listener_yaw_deg: float,
) -> EpisodeRecipe:
    label = project_geometry(source_position_world, listener_sensor_position_world, listener_yaw_deg)
    return EpisodeRecipe(
        episode_id=episode_id, split=split, scene_id=scene_id, scene_family=scene_family,
        source_clip_id=source_clip_id, base_clip_id=base_clip_id, source_dataset=source_dataset,
        class_id=class_id, source_position_world=tuple(source_position_world),
        source_gain_db=source_gain_db, source_offset_sec=source_offset_sec,
        listener_base_position_world=tuple(listener_base_position_world),
        listener_sensor_position_world=tuple(listener_sensor_position_world),
        listener_yaw_deg=listener_yaw_deg, label_class_id=class_id,
        azimuth_project_deg=label["azimuth_project_deg"],
        elevation_project_deg=label["elevation_project_deg"],
        doa_unit_project=tuple(label["doa_unit_project"]), distance_m=label["distance_m"],
    )


def validate_recipe_geometry(recipe: EpisodeRecipe) -> None:
    validate_geometry_independently(
        recipe.source_position_world,
        recipe.listener_sensor_position_world,
        recipe.listener_yaw_deg,
        {"distance_m": recipe.distance_m, "azimuth_project_deg": recipe.azimuth_project_deg,
         "elevation_project_deg": recipe.elevation_project_deg, "doa_unit_project": recipe.doa_unit_project},
    )
