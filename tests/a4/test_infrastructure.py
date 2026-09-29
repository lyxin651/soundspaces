import copy
import unittest
from pathlib import Path

from active_audition.a4.contract import (
    A4ContractError,
    contract_sha256,
    load_contract,
    validate_contract,
)
from active_audition.a4.budget import (
    MOTION_COST_ALGORITHM_IDENTITY,
    MOTION_EXECUTION_IDENTITY,
    MotionParameters,
    compute_motion_cost,
)
from active_audition.a4.active_mask import ACTIVE_MASK_ALGORITHM_IDENTITY
from active_audition.a4.identity import (
    A4IdentityError,
    canonical_json_bytes,
    identity_sha256,
    stable_id,
)
from active_audition.a4.records import (
    BlockRecord,
    CalibrationArtifact,
    EpisodeRecord,
    GeometryRecord,
    PoseRecord,
    RecordError,
    pose_identity_payload,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs/active_audition/v1/a4_infrastructure_contract.yaml"


def _geometry_payload():
    return {
        "schema_version": "active-asr-a4-geometry-v1",
        "scene_id": "replica.engineering_0",
        "scene_resource_identities": {
            "mesh_sha256": "1" * 64,
            "navmesh_sha256": "2" * 64,
        },
        "initial_listener_requested_base_xyz": [0.0, 0.0, 0.0],
        "actual_listener_base_xyz": [0.0, 0.0, 0.0],
        "sensor_xyz": [0.0, 1.5, 0.0],
        "sensor_transform_identity": "sensor-transform-v1",
        "initial_yaw_deg": 0.0,
        "target_world_pose": {"position_xyz": [1.0, 1.5, 0.0], "yaw_deg": 0.0},
        "noise_world_pose": {"position_xyz": [-1.0, 1.5, 0.0], "yaw_deg": 180.0},
        "production_acoustic_policy_identity": "a3-native16-materials-off",
        "candidate_contract_identity": "candidate-contract-v1",
    }


def _geometry():
    payload = _geometry_payload()
    return GeometryRecord(geometry_id=stable_id("geometry", payload), **payload)


def _block(geometry):
    payload = {
        "geometry_id": geometry.geometry_id,
        "speaker_id": "speaker-1",
        "noise_parent_id": "musan-parent-1",
        "nominal_initial_snr_db": 0.0,
        "selection_utterance_ids": ["utt-s1", "utt-s2"],
        "evaluation_utterance_ids": ["utt-e1", "utt-e2"],
        "noise_segment_plan_identity": "noise-plan-v1",
        "global_gain_identity": "gain-v1",
    }
    return BlockRecord(schema_version="active-asr-a4-block-v1", block_id=stable_id("block", payload), **payload)


def _concrete_frozen_contract():
    contract = copy.deepcopy(load_contract(str(CONTRACT_PATH)))
    contract["contract"]["state"] = "FROZEN"
    contract["calibration_boundary"]["active_mask"] = ACTIVE_MASK_ALGORITHM_IDENTITY
    contract["calibration_boundary"]["algorithm"] = "selection-calibration-v1"
    contract["mixture_boundary"]["timeline"] = "source-time-timeline-v1"
    contract["mixture_boundary"]["reconstruction"] = "dual-source-reconstruction-v1"
    contract["mixture_boundary"]["algorithm"] = "synthetic-mixer-v1"
    contract["cache_resume"]["algorithm"] = "cache-resume-v1"
    return contract


class A4IdentityTests(unittest.TestCase):
    def test_canonical_json_and_sha_are_order_invariant(self):
        first = {"b": [1, 2.0], "a": {"x": "✓"}}
        second = {"a": {"x": "✓"}, "b": [1, 2.0]}
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(identity_sha256(first), identity_sha256(second))
        self.assertNotEqual(identity_sha256(first), identity_sha256({"a": {"x": "changed"}, "b": [1, 2.0]}))

    def test_nonfinite_and_unknown_semantic_result_fields_are_rejected(self):
        with self.assertRaises(A4IdentityError):
            identity_sha256({"value": float("nan")})
        with self.assertRaises(A4IdentityError):
            identity_sha256({"value": float("inf")})
        geometry = _geometry_payload()
        geometry["target_world_pose"]["wer"] = 0.2
        with self.assertRaises((A4IdentityError, RecordError)):
            GeometryRecord(geometry_id=stable_id("geometry", geometry), **geometry)


class A4ContractTests(unittest.TestCase):
    def test_draft_contract_validates_and_hash_is_stable(self):
        contract = load_contract(str(CONTRACT_PATH))
        self.assertEqual(contract["contract"]["state"], "DRAFT")
        self.assertEqual(contract["contract"]["gate"], "A4")
        self.assertEqual(contract_sha256(contract), contract_sha256(copy.deepcopy(contract)))
        reordered = {key: contract[key] for key in reversed(list(contract))}
        self.assertEqual(contract_sha256(contract), contract_sha256(reordered))

    def test_require_frozen_rejects_draft_and_unresolved_semantics(self):
        contract = copy.deepcopy(load_contract(str(CONTRACT_PATH)))
        with self.assertRaises(A4ContractError):
            validate_contract(contract, require_frozen=True)
        contract["contract"]["state"] = "FROZEN"
        with self.assertRaises(A4ContractError):
            validate_contract(contract, require_frozen=True)

    def test_concrete_frozen_fixture_validates_without_smoke_manifest(self):
        contract = _concrete_frozen_contract()
        validate_contract(contract, require_frozen=True)
        self.assertNotIn("engineering_smoke_manifest", contract)

    def test_all_deferred_later_slice_draft_validates(self):
        contract = copy.deepcopy(load_contract(str(CONTRACT_PATH)))
        contract["calibration_boundary"]["active_mask"] = "DEFERRED_TO_A4_2"
        validate_contract(contract)

    def test_partially_concrete_draft_validates(self):
        contract = copy.deepcopy(load_contract(str(CONTRACT_PATH)))
        self.assertEqual(contract["calibration_boundary"]["active_mask"], ACTIVE_MASK_ALGORITHM_IDENTITY)
        self.assertEqual(contract["calibration_boundary"]["algorithm"], "DEFERRED_TO_A4_2")
        validate_contract(contract)

    def test_fully_concrete_draft_validates(self):
        contract = _concrete_frozen_contract()
        contract["contract"]["state"] = "DRAFT"
        validate_contract(contract)

    def test_frozen_with_any_deferred_field_rejects(self):
        contract = copy.deepcopy(load_contract(str(CONTRACT_PATH)))
        contract["contract"]["state"] = "FROZEN"
        with self.assertRaises(A4ContractError):
            validate_contract(contract)

    def test_smoke_manifest_is_not_an_infrastructure_contract_field(self):
        contract = copy.deepcopy(load_contract(str(CONTRACT_PATH)))
        contract["engineering_smoke_manifest"] = {}
        with self.assertRaises(A4ContractError):
            validate_contract(contract)

    def test_unknown_and_missing_contract_fields_rejected(self):
        contract = copy.deepcopy(load_contract(str(CONTRACT_PATH)))
        contract["unknown"] = True
        with self.assertRaises(A4ContractError):
            validate_contract(contract)
        contract = copy.deepcopy(load_contract(str(CONTRACT_PATH)))
        del contract["typed_records"]
        with self.assertRaises(A4ContractError):
            validate_contract(contract)


class A4RecordTests(unittest.TestCase):
    def test_geometry_identity_changes_with_target_or_noise_pose(self):
        first = _geometry()
        changed = _geometry_payload()
        changed["target_world_pose"]["position_xyz"][0] = 1.25
        second = GeometryRecord(geometry_id=stable_id("geometry", changed), **changed)
        self.assertNotEqual(first.geometry_id, second.geometry_id)
        changed = _geometry_payload()
        changed["noise_world_pose"]["position_xyz"][0] = -1.25
        third = GeometryRecord(geometry_id=stable_id("geometry", changed), **changed)
        self.assertNotEqual(first.geometry_id, third.geometry_id)

    def test_pose_record_has_stable_identity_and_rejects_illegal_shape(self):
        geometry = _geometry()
        parameters = MotionParameters(
            schema_version="active-asr-a4-motion-parameters-v1",
            algorithm_identity=MOTION_COST_ALGORITHM_IDENTITY,
            execution_identity=MOTION_EXECUTION_IDENTITY,
            translation_speed_mps=0.25,
            rotation_speed_dps=90.0,
            settling_sec=0.5,
            budget_sec=10.0,
        )
        cost = compute_motion_cost(
            (0.0, 0.0, 0.0), 0.0, ((0.0, 0.0, 0.0),), 0.0, 0.0, parameters
        )
        payload = {
            "schema_version": "active-asr-a4-pose-v2",
            "geometry_id": geometry.geometry_id,
            "position_id": "position-0",
            "yaw_id": "yaw-0",
            "requested_base_xyz": [0.0, 0.0, 0.0],
            "actual_snapped_base_xyz": [0.0, 0.0, 0.0],
            "sensor_transform_identity": "sensor-transform-v1",
            "sensor_xyz": [0.0, 1.5, 0.0],
            "yaw_deg": 0.0,
            "snap_error_m": 0.0,
            "path_polyline": [[0.0, 0.0, 0.0]],
            "geodesic_path_length_m": 0.0,
            "initial_to_path_turn_deg": 0.0,
            "internal_path_turn_deg": 0.0,
            "final_turn_deg": 0.0,
            "settling_sec": 0.0,
            "total_cost_sec": 0.0,
            "motion_contract_identity": cost.motion_contract_identity,
            "path_polyline_length_m": 0.0,
            "translation_speed_mps": cost.translation_speed_mps,
            "rotation_speed_dps": cost.rotation_speed_dps,
            "translation_sec": 0.0,
            "rotation_sec": 0.0,
            "budget_sec": 10.0,
            "budget_feasible": True,
            "geometry_legality": "LEGAL",
            "invalid_reason": None,
        }
        pose = PoseRecord(pose_id=stable_id("pose", pose_identity_payload(payload)), **payload)
        self.assertEqual(pose.to_payload()["pose_id"], pose.pose_id)
        with self.assertRaises((A4IdentityError, RecordError)):
            PoseRecord(pose_id="pose-" + "0" * 64, **payload)

    def test_block_identity_is_plan_only_and_calibration_is_separate(self):
        block = _block(_geometry())
        self.assertNotIn("alpha", block.identity_payload())
        self.assertNotIn("measured_snr_db", block.identity_payload())
        self.assertNotIn("calibration_state", block.to_payload())
        self.assertNotIn("calibration_artifact_id", block.to_payload())
        self.assertNotIn("selection_episode_ids", block.to_payload())
        self.assertNotIn("engineering_only", block.to_payload())
        with self.assertRaises((RecordError, TypeError)):
            block.selection_utterance_ids[0] = "mutate"

    def test_same_block_id_cannot_back_two_serialized_plan_payloads(self):
        block = _block(_geometry())
        changed = dict(block.to_payload(), speaker_id="speaker-2")
        with self.assertRaises(RecordError):
            BlockRecord.from_payload(changed)
        with self.assertRaises(RecordError):
            BlockRecord.from_payload(
                dict(block.to_payload(), selection_episode_ids=["episode-selection-1", "episode-selection-2"])
            )

    def test_every_block_plan_field_changes_block_id(self):
        block = _block(_geometry())
        changed_geometry_payload = _geometry_payload()
        changed_geometry_payload["initial_yaw_deg"] = 15.0
        changed_geometry = GeometryRecord(
            geometry_id=stable_id("geometry", changed_geometry_payload), **changed_geometry_payload
        )
        mutations = {
            "geometry_id": changed_geometry.geometry_id,
            "speaker_id": "speaker-2",
            "noise_parent_id": "musan-parent-2",
            "nominal_initial_snr_db": 3.0,
            "global_gain_identity": "gain-v2",
            "selection_utterance_ids": ["utt-s1", "utt-s3"],
            "evaluation_utterance_ids": ["utt-e1", "utt-e3"],
            "noise_segment_plan_identity": "noise-plan-v2",
        }
        for field, value in mutations.items():
            payload = dict(block.to_payload())
            payload[field] = value
            payload.pop("block_id")
            identity_payload = {key: payload[key] for key in block.identity_payload()}
            candidate = BlockRecord(block_id=stable_id("block", identity_payload), **payload)
            self.assertNotEqual(block.block_id, candidate.block_id, field)

    def test_episode_identity_changes_with_utterance_or_noise_segment(self):
        block = _block(_geometry())
        payload = {
            "schema_version": "active-asr-a4-episode-v1",
            "block_id": block.block_id,
            "role": "selection",
            "utterance_identity": {"utterance_id": "utt-s1", "decoded_waveform_sha256": "5" * 64},
            "reference_identity": {"reference_sha256": "6" * 64, "normalization_version": "v1"},
            "fixed_dry_noise_segment_identity": {"parent_id": "musan-parent-1", "sha256": "7" * 64},
            "target_source_duration_sec": 3.5,
            "noise_source_time_start_sec": -2.0,
            "noise_source_time_end_sec": 4.0,
            "noise_segment_duration_sec": 6.0,
        }
        episode = EpisodeRecord(episode_id=stable_id("episode", payload), **payload)
        self.assertEqual(episode.noise_source_time_start_sec, -2.0)
        utterance_changed = copy.deepcopy(payload)
        utterance_changed["utterance_identity"]["utterance_id"] = "utt-s2"
        changed_utterance = EpisodeRecord(episode_id=stable_id("episode", utterance_changed), **utterance_changed)
        self.assertNotEqual(episode.episode_id, changed_utterance.episode_id)
        segment_changed = copy.deepcopy(payload)
        segment_changed["fixed_dry_noise_segment_identity"]["sha256"] = "8" * 64
        changed_segment = EpisodeRecord(episode_id=stable_id("episode", segment_changed), **segment_changed)
        self.assertNotEqual(episode.episode_id, changed_segment.episode_id)

    def test_episode_zero_start_is_valid_and_invalid_timeline_fields_rejected(self):
        block = _block(_geometry())
        payload = {
            "schema_version": "active-asr-a4-episode-v1",
            "block_id": block.block_id,
            "role": "evaluation",
            "utterance_identity": {"utterance_id": "utt-e1", "decoded_waveform_sha256": "5" * 64},
            "reference_identity": {"reference_sha256": "6" * 64, "normalization_version": "v1"},
            "fixed_dry_noise_segment_identity": {"parent_id": "musan-parent-1", "sha256": "7" * 64},
            "target_source_duration_sec": 4.0,
            "noise_source_time_start_sec": 0.0,
            "noise_source_time_end_sec": 4.0,
            "noise_segment_duration_sec": 4.0,
        }
        valid = EpisodeRecord(episode_id=stable_id("episode", payload), **payload)
        self.assertEqual(valid.noise_source_time_start_sec, 0.0)
        for field, value in (
            ("noise_source_time_end_sec", -1.0),
            ("noise_segment_duration_sec", 3.0),
            ("target_source_duration_sec", 0.0),
        ):
            invalid = dict(payload)
            invalid[field] = value
            with self.assertRaises(RecordError):
                EpisodeRecord(episode_id=stable_id("episode", invalid), **invalid)

    def test_calibration_artifact_has_independent_identity_and_block_reference(self):
        block = _block(_geometry())
        payload = {
            "schema_version": "active-asr-a4-calibration-v1",
            "block_id": block.block_id,
            "calibration_contract_identity": "calibration-contract-v1",
            "selection_episode_ids": ["episode-selection-1", "episode-selection-2"],
            "input_component_identities": {"target_sha256": "9" * 64, "noise_sha256": "a" * 64},
            "ps": 1.0,
            "pn": 0.5,
            "active_sample_count": 100,
            "alpha": 2.0,
            "nominal_snr_db": 3.0,
            "measured_snr_db": 3.01,
            "status": "CALIBRATED",
            "provenance": {"algorithm_version": "future-a4-2"},
        }
        artifact = CalibrationArtifact(calibration_artifact_id=stable_id("calibration", payload), **payload)
        self.assertEqual(artifact.block_id, block.block_id)
        self.assertNotIn("calibration_artifact_id", block.to_payload())
        self.assertNotIn("alpha", block.identity_payload())
        self.assertNotIn("measured_snr_db", block.identity_payload())

    def test_result_fields_are_not_accepted_as_identity_inputs(self):
        geometry = _geometry_payload()
        geometry["noise_world_pose"]["hypothesis"] = "RESULT"
        with self.assertRaises((A4IdentityError, RecordError)):
            GeometryRecord(geometry_id=stable_id("geometry", geometry), **geometry)


if __name__ == "__main__":
    unittest.main()
