import copy
import json
import unittest
from pathlib import Path

import numpy as np

from active_audition.a4.identity import stable_id
from active_audition.o1.component_snr import build_component_snr_record, ComponentSnrError, O1ComponentSnrRecord, merge_reused_component_snr_records
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
from active_audition.o1.replacement import (
    O1_REPLACEMENT_BATCH_SCHEMA_VERSION,
    O1_REPLACEMENT_POLICY_IDENTITY,
    O1ReplacementCandidateBatch,
    O1ReplacementError,
)
from active_audition.o1.replacement_audit import O1FinalizedReplacementAuditBatch
from scripts.run_o1_production import _require_scientific_manifest


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

    def test_finalized_audit_decisions_gate_scientific_use(self):
        preview = {
            "relative_preview_path": "previews/clip_01.wav", "clip_index": 1, "center_fraction": 0.08,
            "start_sample": 0, "end_sample": 240000, "start_sec": 0.0, "end_sec": 15.0,
            "clip_sample_count": 240000, "expected_source_slice_float32_sha256": "1" * 64,
            "written_clip_decoded_float32_sha256": "1" * 64, "verification_status": "SAMPLE_EXACT",
        }
        previews = tuple(dict(preview, clip_index=index, relative_preview_path="previews/clip_{:02d}.wav".format(index)) for index in range(1, 7))
        base = dict(
            schema_version=O1_NOISE_AUDIT_SCHEMA_VERSION,
            audit_policy_identity=O1_NOISE_AUDIT_POLICY_IDENTITY,
            decision_source="user_manual_listening", parent_recording_id="noise/free-sound/noise-free-sound-0032",
            relative_source_path="noise/free-sound/noise-free-sound-0032.wav", source_file_sha256="2" * 64,
            decoded_waveform_sha256="3" * 64, sample_rate_hz=16000, sample_count=240000,
            reviewed_preview_identities=previews, review_scope={"dimensions": list(("speech_leakage", "strong_reverberation", "indoor_localized_source_compatibility"))},
            manual_decision_provenance={"decision_source": "user_manual_listening", "decision_status": "FINALIZED_USER_DECISION"},
            speech_leakage="PASS", strong_reverberation="PASS", indoor_localized_source_compatibility="PASS",
            selected_for_o1_scientific_use=True, exclusion_reason="", engineering_only=True,
        )
        self.assertTrue(O1NoiseParentAuditRecord.from_payload(O1NoiseParentAuditRecord(**base).to_payload()).selected_for_o1_scientific_use)
        excluded = dict(base, speech_leakage="NOT_EVALUATED", strong_reverberation="NOT_EVALUATED", indoor_localized_source_compatibility="FLAGGED", selected_for_o1_scientific_use=False, exclusion_reason="MOVING_CAR_DRIVING_SOURCE")
        self.assertFalse(O1NoiseParentAuditRecord(**excluded).selected_for_o1_scientific_use)

    def test_replacement_batch_is_lexical_and_exclusion_strict(self):
        def candidate(index, parent):
            return {
                "candidate_index": index, "parent_recording_id": parent, "relative_source_path": parent + ".wav",
                "source_file_sha256": "1" * 64, "decoded_waveform_sha256": "2" * 64, "sample_rate_hz": 16000, "sample_count": 900000,
                "technical_audit": {
                    "corpus": "MUSAN", "subset": "noise", "excluded": False, "readable": True, "too_short": False, "extreme_silence": False,
                    "sample_rate_hz": 16000, "source_file_exists": True, "source_file_sha256_matches": True,
                    "decoded_waveform_sha256_matches": True, "decoded_mono_finite_float32": True, "sample_count_sufficient": True,
                },
            }
        payload = dict(
            schema_version=O1_REPLACEMENT_BATCH_SCHEMA_VERSION,
            selection_policy_identity=O1_REPLACEMENT_POLICY_IDENTITY,
            registry_relative_path="registries/active_asr_a3/musan_noise.jsonl", registry_sha256="3" * 64,
            required_parent_sample_count=882720, required_parent_duration_sec=55.17,
            excluded_parent_recording_ids=("noise/free-sound/noise-free-sound-0015",),
            candidate_records=tuple(candidate(i, "noise/free-sound/noise-free-sound-00{:02d}".format(i + 41)) for i in range(1, 5)),
            engineering_only=True, batch_id="", batch_sha256="",
        )
        batch = O1ReplacementCandidateBatch(**payload)
        self.assertEqual(O1ReplacementCandidateBatch.from_payload(batch.to_payload()).to_payload(), batch.to_payload())
        tampered = dict(batch.to_payload(), candidate_records=tuple(reversed(batch.candidate_records)))
        with self.assertRaises(O1ReplacementError):
            O1ReplacementCandidateBatch.from_payload(tampered)
        tampered = dict(batch.to_payload(), excluded_parent_recording_ids=tuple(sorted(batch.excluded_parent_recording_ids + (batch.candidate_records[0]["parent_recording_id"],))))
        with self.assertRaises(O1ReplacementError):
            O1ReplacementCandidateBatch.from_payload(tampered)

    def test_finalized_replacement_batch_binds_slots_and_round_trips(self):
        path = ROOT / "data/logs/o1_noise_replacement_audit_batch01/o1_noise_audit_records_finalized.json"
        if not path.is_file():
            self.skipTest("manual replacement package is local-only")
        batch = O1FinalizedReplacementAuditBatch.from_payload(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(
            batch.selected_replacement_slots["slot1"]["parent_recording_id"],
            "noise/free-sound/noise-free-sound-0041",
        )
        self.assertEqual(
            batch.selected_replacement_slots["slot2"]["parent_recording_id"],
            "noise/free-sound/noise-free-sound-0073",
        )
        self.assertEqual(O1FinalizedReplacementAuditBatch.from_payload(batch.to_payload()).to_payload(), batch.to_payload())
        tampered = dict(batch.to_payload())
        tampered["selected_replacement_slots"] = dict(batch.selected_replacement_slots, slot1=dict(batch.selected_replacement_slots["slot1"], parent_recording_id="noise/free-sound/noise-free-sound-0073"))
        with self.assertRaises(ValueError):
            O1FinalizedReplacementAuditBatch.from_payload(tampered)

    def test_scientific_runner_is_fail_closed_for_pre_audit_manifest(self):
        old = O1ExploratoryManifest.from_payload(json.loads(MANIFEST.read_text(encoding="utf-8")))
        _require_scientific_manifest(old, "rir")
        for stage in ("mixture", "component-snr", "asr", "diagnostics"):
            with self.assertRaises(RuntimeError):
                _require_scientific_manifest(old, stage)

    def test_scientific_runner_accepts_finalized_v2_manifest(self):
        path = ROOT / "runs/active_asr_v1/o1_replica_apartment_2_3155ddba31ea/o1_scientific_manifest.json"
        if not path.is_file():
            self.skipTest("scientific O1 manifest is local-only")
        manifest = O1ExploratoryManifest.from_payload(json.loads(path.read_text(encoding="utf-8")))
        for stage in ("rir", "mixture", "component-snr", "asr", "diagnostics"):
            _require_scientific_manifest(manifest, stage)

    def test_retained_component_snr_reuse_is_content_bound(self):
        current_path = ROOT / "runs/active_asr_v1/o1_replica_apartment_2_3155ddba31ea/o1_component_snr.jsonl"
        reference_path = ROOT / "runs/active_asr_v1/o1_replica_apartment_2_c8c821eae922/o1_component_snr.jsonl"
        if not current_path.is_file() or not reference_path.is_file():
            self.skipTest("O1 component-SNR artifacts are local-only")
        current = [O1ComponentSnrRecord.from_payload(json.loads(line)) for line in current_path.read_text().splitlines() if line]
        reference = [O1ComponentSnrRecord.from_payload(json.loads(line)) for line in reference_path.read_text().splitlines() if line]
        retained = {"block-863f1f715ef9efbfb58782daf4c165224c23d8a02ce736ac76a8658dfae3a081", "block-19e0884003cd820714023f79ca48339d1ef27998b1eb368234b44674d644d560"}
        merged = merge_reused_component_snr_records(current, reference, retained)
        by_key = {(item.block_id, item.episode_id, item.pose_id): item for item in merged}
        ref_by_key = {(item.block_id, item.episode_id, item.pose_id): item for item in reference}
        self.assertEqual(sum(by_key[key].to_payload() == ref_by_key[key].to_payload() for key in ref_by_key if key[0] in retained), 384)


if __name__ == "__main__":
    unittest.main()
