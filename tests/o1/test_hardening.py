import copy
import json
import unittest
from pathlib import Path

import numpy as np

from active_audition.a4.identity import stable_id
from active_audition.o1.component_snr import build_component_snr_record, ComponentSnrError
from active_audition.o1.landscape import (
    O1LandscapeSummary,
    O1PoseScore,
    _o1_result_id,
    _paired_improvement,
    _selection,
    _transfer_selection,
)
from active_audition.o1.manifest import O1ExploratoryManifest, O1_MANIFEST_PRE_ASR_STATUS
from active_audition.o1.noise_audit import (
    O1_NOISE_AUDIT_POLICY_IDENTITY,
    O1_NOISE_AUDIT_SCHEMA_VERSION,
    O1NoiseAuditError,
    O1NoiseParentAuditRecord,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "runs/active_asr_v1/o1_replica_apartment_2_c8c821eae922/o1_exploratory_manifest.json"


def _score(pose_id, error_count=0, motion=1.0, episode="episode", snr=None):
    return O1PoseScore(
        schema_version="active-asr-o1-landscape-v1", score_id="unused", block_id="block-1",
        episode_id=episode, role="selection" if episode.startswith("selection") else "evaluation",
        utterance_id=episode, frontend="mean_lr", pose_id=pose_id,
        position_id="position-1" if pose_id in ("pose-a", "pose-c") else "position-2",
        yaw_id="yaw-" + pose_id, geometry_id="geometry-1", geometry_legality="LEGAL",
        actual_snapped_base_xyz=(0.0, 0.0, 0.0), sensor_xyz=(0.0, 1.5, 0.0), yaw_deg=0.0,
        motion_cost_sec=motion, budget_feasible_at_manifest_budget=True,
        S=error_count, D=0, I=0, N=10, WER=error_count / 10.0,
        reference="REFERENCE", hypothesis="HYPOTHESIS", component_snr_db=snr,
    )


class O1HardeningTests(unittest.TestCase):
    def test_pre_audit_manifest_is_explicitly_not_scientific(self):
        manifest = O1ExploratoryManifest.from_payload(json.loads(MANIFEST.read_text(encoding="utf-8")))
        self.assertEqual(manifest.scientific_use_status, O1_MANIFEST_PRE_ASR_STATUS)
        with self.assertRaises(ValueError):
            manifest.require_scientific_noise_audit()

    def test_noise_audit_round_trip_and_pending_gate(self):
        preview = {
            "relative_preview_path": "previews/clip_01.wav", "clip_index": 1, "center_fraction": 0.08,
            "start_sample": 0, "end_sample": 240000, "start_sec": 0.0, "end_sec": 15.0,
            "clip_sample_count": 240000, "expected_source_slice_float32_sha256": "1" * 64,
            "written_clip_decoded_float32_sha256": "1" * 64, "verification_status": "SAMPLE_EXACT",
        }
        previews = tuple(dict(preview, clip_index=index, relative_preview_path="previews/clip_{:02d}.wav".format(index)) for index in range(1, 7))
        record = O1NoiseParentAuditRecord(
            schema_version=O1_NOISE_AUDIT_SCHEMA_VERSION,
            audit_policy_identity=O1_NOISE_AUDIT_POLICY_IDENTITY,
            decision_source="user_manual_listening", parent_recording_id="noise/free-sound/noise-free-sound-0015",
            relative_source_path="noise/free-sound/noise-free-sound-0015.wav", source_file_sha256="2" * 64,
            decoded_waveform_sha256="3" * 64, sample_rate_hz=16000, sample_count=240000,
            reviewed_preview_identities=previews, review_scope={"dimensions": ["speech_leakage", "strong_reverberation", "indoor_localized_source_compatibility"]},
            manual_decision_provenance={"decision_source": "user_manual_listening"},
            speech_leakage="PENDING_USER", strong_reverberation="PENDING_USER",
            indoor_localized_source_compatibility="PENDING_USER", selected_for_o1_scientific_use=False,
            exclusion_reason="awaiting user audit", engineering_only=True,
        )
        self.assertEqual(O1NoiseParentAuditRecord.from_payload(record.to_payload()).to_payload(), record.to_payload())
        tampered = copy.deepcopy(record.to_payload())
        tampered["audit_record_sha256"] = "0" * 64
        with self.assertRaises(O1NoiseAuditError):
            O1NoiseParentAuditRecord.from_payload(tampered)

    def test_component_snr_uses_two_ear_power_and_strict_tamper(self):
        target = np.asarray([[2.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        noise = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        record = build_component_snr_record(
            block_id="block-1", episode_id="episode-1", role="selection", utterance_id="utt-1", pose_id="pose-1",
            timeline_identity=stable_id("receiver-timeline", {"fixture": 1}),
            target_component_identity=stable_id("target-component", {"fixture": 1}),
            noise_component_identity=stable_id("noise-component", {"fixture": 1}),
            receiver_mask_identity=stable_id("receiver-mask", {"fixture": 1}),
            calibration_artifact_identity=stable_id("calibration", {"fixture": 1}), alpha=2.0,
            target_binaural=target, noise_binaural=noise, receiver_mask=(True, True),
        )
        self.assertAlmostEqual(record.target_power, 2.0)
        self.assertAlmostEqual(record.noise_power, 1.0)
        self.assertAlmostEqual(record.component_snr_db, 10.0 * np.log10(0.5))
        tampered = dict(record.to_payload(), component_snr_db=0.0)
        with self.assertRaises(ComponentSnrError):
            type(record).from_payload(tampered)

    def test_max_snr_tie_break_is_not_wer(self):
        rows = [_score("pose-z", error_count=0, motion=2.0, snr=3.0), _score("pose-a", error_count=9, motion=1.0, snr=3.0)]
        record = _selection("block-1", "mean_lr", 2.0, "Max-SNR", "episode", rows, rows, episode_id="episode-1")
        self.assertEqual(record.selected_pose_id, "pose-a")

    def test_paired_improvement_is_episode_matched(self):
        records = []
        for episode, stay_error, rotate_error in (("evaluation-0", 5, 4), ("evaluation-1", 1, 3)):
            records.append(_selection("block-1", "mean_lr", 10.0, "Stay", "episode", [_score("stay", stay_error, episode=episode)], [_score("stay", stay_error, episode=episode)], episode_id=episode))
            records.append(_selection("block-1", "mean_lr", 10.0, "Rotate-best", "episode", [_score("rotate", rotate_error, episode=episode)], [_score("rotate", rotate_error, episode=episode)], episode_id=episode))
        paired = _paired_improvement(records, "Rotate-best", "mean_lr", 10.0)
        self.assertEqual(paired["paired_episode_count"], 2)
        self.assertEqual(paired["improved_count"], 1)
        self.assertEqual(paired["worsened_count"], 1)

    def test_transfer_candidate_ids_are_the_actual_restricted_sets(self):
        selection = [_score("pose-a", 0, 1.0, "selection-0"), _score("pose-b", 1, 1.5, "selection-0")]
        selection += [_score("pose-a", 0, 1.0, "selection-1"), _score("pose-b", 1, 1.5, "selection-1")]
        evaluation = [_score("pose-a", 2, 1.0, "evaluation-0"), _score("pose-b", 0, 1.5, "evaluation-0")]
        block = {"block_record": {"block_id": "block-1"}, "episodes": [{"episode_id": "selection-0", "role": "selection"}, {"episode_id": "selection-1", "role": "selection"}, {"episode_id": "evaluation-0", "role": "evaluation"}], "geometry_record": {"target_world_pose": {"position_xyz": [10.0, 1.5, 10.0]}}}
        record, _ = _transfer_selection(block, "mean_lr", 2.0, selection, evaluation, "O_transfer")
        self.assertEqual(set(record.candidate_pose_ids), {"pose-a", "pose-b"})
        rotate, _ = _transfer_selection(block, "mean_lr", 2.0, selection, evaluation, "Rotate-transfer")
        self.assertEqual(set(rotate.candidate_pose_ids), {"pose-a"})

    def test_result_summary_identity_allows_descriptive_wer_fields(self):
        payload = {
            "schema_version": "active-asr-o1-landscape-v1",
            "analysis_kind": "FAMILIAR_ENGINEERING_DRY_RUN",
            "algorithm_identity": "active-asr-o1-descriptive-landscape-v1",
            "frontend_policy_identity": "active-asr-o1-mean-lr-primary-fixed-channel-sensitivity-v1",
            "infrastructure_contract_sha256": "a" * 64,
            "source_manifest_identity": "manifest-1",
            "blocks": 1,
            "episodes": 1,
            "poses": 1,
            "attempted": 1,
            "valid": 1,
            "incomplete": 0,
            "summary": {"cost_vs_wer": [{"pose_id": "pose-1", "WER": 0.25}]},
        }
        payload["summary_id"] = _o1_result_id("o1-summary", payload)
        value = O1LandscapeSummary.from_payload(payload)
        self.assertEqual(value.to_payload(), payload)


if __name__ == "__main__":
    unittest.main()
