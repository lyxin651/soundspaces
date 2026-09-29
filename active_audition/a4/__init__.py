"""A4 infrastructure contracts and immutable experiment-plan records.

The A4 namespace is intentionally separate from the legacy V0 records and
from the frozen A0--A3 runtime contracts.  This package contains only data
validation and identity construction; it does not render audio, run ASR, or
choose experimental geometries.
"""

from active_audition.a4.contract import (
    A4ContractError,
    A4_CONTRACT_VERSION,
    A4_GATE,
    canonical_contract_json,
    contract_sha256,
    load_contract,
    validate_contract,
)
from active_audition.a4.identity import (
    A4IdentityError,
    canonical_json,
    canonical_json_bytes,
    identity_sha256,
    sha256_bytes,
    stable_id,
)
from active_audition.a4.records import (
    BlockRecord,
    CalibrationArtifact,
    EpisodeRecord,
    GeometryRecord,
    PoseRecord,
    RecordError,
    validate_block_record,
    validate_calibration_artifact,
    validate_episode_record,
    validate_geometry_record,
    validate_pose_record,
)

__all__ = [
    "A4ContractError",
    "A4_CONTRACT_VERSION",
    "A4_GATE",
    "A4IdentityError",
    "BlockRecord",
    "CalibrationArtifact",
    "EpisodeRecord",
    "GeometryRecord",
    "PoseRecord",
    "RecordError",
    "canonical_contract_json",
    "canonical_json",
    "canonical_json_bytes",
    "contract_sha256",
    "identity_sha256",
    "load_contract",
    "sha256_bytes",
    "stable_id",
    "validate_block_record",
    "validate_calibration_artifact",
    "validate_contract",
    "validate_episode_record",
    "validate_geometry_record",
    "validate_pose_record",
]
