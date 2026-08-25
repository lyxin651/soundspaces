"""Stable, Habitat-independent data contracts for Active Audition V0."""

from dataclasses import dataclass
from typing import Optional, Tuple


WorldXYZ = Tuple[float, float, float]


@dataclass(frozen=True)
class ListenerPose:
    base_position_world: WorldXYZ
    sensor_position_world: WorldXYZ
    yaw_deg: float


@dataclass(frozen=True)
class SourceSpec:
    position_world: WorldXYZ
    audio_id: str
    segment_start_sec: float
    segment_duration_sec: float
    gain_db: float


@dataclass(frozen=True)
class EpisodeSpec:
    schema_version: str
    episode_id: str
    scene_id: str
    episode_seed: int
    source: SourceSpec
    listener_initial: ListenerPose


@dataclass(frozen=True)
class Candidate:
    episode_id: str
    candidate_id: str

    action_type: str
    translation_direction: Optional[str]

    requested_translation_m: float
    rotation_deg: float

    requested_base_position_world: WorldXYZ
    snapped_base_position_world: Optional[WorldXYZ]
    sensor_position_world: Optional[WorldXYZ]
    yaw_deg: float

    snap_error_m: Optional[float]
    move_euclidean_m: float
    move_geodesic_m: Optional[float]

    is_navigable: bool
    has_path: bool
    path_points_world: Optional[Tuple[WorldXYZ, ...]]

    valid: bool
    invalid_reason: Optional[str]


@dataclass(frozen=True)
class Viewpoint:
    episode_id: str
    viewpoint_id: str
    candidate_id: Optional[str]

    action_type: str

    base_position_world: WorldXYZ
    sensor_position_world: WorldXYZ
    yaw_deg: float

    audio_path: str
    rir_id: Optional[str]
    rir_path: Optional[str]

    sample_rate_hz: int
    num_channels: int
    num_samples: int
    duration_sec: float
    dtype: str


@dataclass(frozen=True)
class AcousticMetrics:
    episode_id: str
    viewpoint_id: str

    rms_left_dbfs: float
    rms_right_dbfs: float
    rms_mean_dbfs: float
    ild_db: float

    peak_left: float
    peak_right: float

    clipping_fraction: float
    silence_fraction: float
    interaural_correlation: float
