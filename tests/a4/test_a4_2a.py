import copy
import math
import sys
import unittest

from active_audition.a4.active_mask import (
    ACTIVE_MASK_ALGORITHM_IDENTITY,
    ActiveMaskContract,
    ActiveMaskError,
    ActiveMaskRecord,
    build_active_mask,
)
from active_audition.a4.identity import canonical_json_bytes, stable_id
from active_audition.a4.noise_segments import (
    A3_NOISE_PARENT_ADAPTER_IDENTITY,
    EPISODE_SLOT_ORDER,
    EpisodeNoiseRequest,
    NoiseParentMetadata,
    NoiseSegmentError,
    NoiseSegmentPlannerParameters,
    noise_parent_from_a3_provenance,
    plan_noise_segments,
)
from active_audition.a4.records import BlockRecord, EpisodeRecord
from active_audition.a4.mixer import GlobalGainSpec


def _parent(sample_count=1000000, sample_rate_hz=16000):
    parent_payload = {"fixture": "synthetic-noise-parent", "sample_rate_hz": sample_rate_hz}
    return NoiseParentMetadata(
        noise_parent_id=stable_id("noise-parent", parent_payload),
        decoded_resampled_payload_sha256="a" * 64,
        sample_rate_hz=sample_rate_hz,
        sample_count=sample_count,
    )


def _planner(pre=2.0, post=2.0, guard=0.25, sample_rate_hz=16000):
    return NoiseSegmentPlannerParameters(
        sample_rate_hz=sample_rate_hz,
        pre_roll_sec=pre,
        post_roll_sec=post,
        guard_interval_sec=guard,
    )


def _episodes():
    # Deliberately caller-shuffled; the planner must use fixed slot order.
    return [
        EpisodeNoiseRequest("evaluation", 1, "utterance-e2", 12003),
        EpisodeNoiseRequest("selection", 0, "utterance-s1", 16000),
        EpisodeNoiseRequest("evaluation", 0, "utterance-e1", 20001),
        EpisodeNoiseRequest("selection", 1, "utterance-s2", 8001),
    ]


def _plan_block_episode_chain():
    """Construct the complete DAG without any placeholder episode IDs."""

    parent = _parent()
    plan = plan_noise_segments(parent, _episodes(), _planner())
    block_payload = {
        "geometry_id": stable_id("geometry", {"fixture": "dag-geometry"}),
        "speaker_id": "speaker-dag",
        "noise_parent_id": parent.noise_parent_id,
        "nominal_initial_snr_db": 0.0,
        "selection_utterance_ids": ["utterance-s1", "utterance-s2"],
        "evaluation_utterance_ids": ["utterance-e1", "utterance-e2"],
        "noise_segment_plan_identity": plan.plan_id,
        "global_gain_identity": GlobalGainSpec(1.0).identity,
    }
    block = BlockRecord(
        schema_version="active-asr-a4-block-v1",
        block_id=stable_id("block", block_payload),
        **block_payload,
    )
    episodes = []
    for segment in plan.segments:
        episode_payload = {
            "schema_version": "active-asr-a4-episode-v1",
            "block_id": block.block_id,
            "role": segment.role,
            "utterance_identity": {
                "utterance_id": segment.utterance_identity,
                "decoded_waveform_sha256": "b" * 64,
            },
            "reference_identity": {"reference_sha256": "c" * 64, "normalization_version": "v1"},
            "fixed_dry_noise_segment_identity": segment.to_payload(),
            "target_source_duration_sec": segment.target_duration_sec,
            "noise_source_time_start_sec": segment.source_time_start_sec,
            "noise_source_time_end_sec": segment.source_time_end_sec,
            "noise_segment_duration_sec": segment.source_time_end_sec - segment.source_time_start_sec,
        }
        episodes.append(EpisodeRecord(episode_id=stable_id("episode", episode_payload), **episode_payload))
    return parent, plan, block, tuple(episodes)


class NoiseSegmentPlannerTests(unittest.TestCase):
    def test_fixed_slots_and_dag_plan_does_not_contain_final_episode_ids(self):
        parent, plan, block, episodes = _plan_block_episode_chain()
        self.assertEqual([(item.role, item.role_index) for item in plan.segments], list(EPISODE_SLOT_ORDER))
        self.assertEqual(len(episodes), 4)
        self.assertEqual(block.noise_segment_plan_identity, plan.plan_id)
        serialized = plan.to_payload()
        self.assertNotIn("episode_id", serialized)
        self.assertTrue(all("episode_id" not in segment for segment in serialized["segments"]))
        self.assertEqual(parent.noise_parent_id, block.noise_parent_id)
        self.assertEqual([episode.block_id for episode in episodes], [block.block_id] * 4)

    def test_stable_order_and_four_nonoverlapping_segments(self):
        plan = plan_noise_segments(_parent(), _episodes(), _planner())
        self.assertEqual([(item.role, item.role_index) for item in plan.segments], list(EPISODE_SLOT_ORDER))
        for first, second in zip(plan.segments, plan.segments[1:]):
            self.assertEqual(second.start_sample, first.end_sample + first.guard_after_samples)
            self.assertLessEqual(first.end_sample, second.start_sample)
        self.assertEqual(plan.segments[0].source_time_start_sec, -2.0)
        self.assertEqual(plan.segments[0].source_time_end_sec, 3.0)
        self.assertEqual(plan.segments[-1].guard_after_samples, 0)
        self.assertEqual(plan.required_parent_sample_count, plan.segments[-1].end_sample)

    def test_same_input_is_byte_identical_and_round_trips(self):
        first = plan_noise_segments(_parent(), _episodes(), _planner())
        second = plan_noise_segments(_parent(), list(reversed(_episodes())), _planner())
        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(canonical_json_bytes(first.to_payload()), canonical_json_bytes(second.to_payload()))
        restored = first.from_payload(first.to_payload())
        self.assertEqual(restored.plan_id, first.plan_id)
        self.assertEqual(canonical_json_bytes(restored.to_payload()), canonical_json_bytes(first.to_payload()))

    def test_target_sample_count_is_authority_for_fractional_duration(self):
        params = _planner(pre=0.25, post=0.25, guard=0.05, sample_rate_hz=10)
        episodes = [
            EpisodeNoiseRequest(role, index, "utt-{}-{}".format(role, index), count)
            for role, index, count in (
                ("selection", 0, 1), ("selection", 1, 2), ("evaluation", 0, 3), ("evaluation", 1, 4)
            )
        ]
        plan = plan_noise_segments(_parent(sample_count=1000, sample_rate_hz=10), episodes, params)
        first = plan.segments[0]
        # 0.25 s rounds half-up to 3 samples; target count remains exactly 1.
        self.assertEqual(first.start_sample, 0)
        self.assertEqual(first.end_sample - first.start_sample, 3 + 1 + 3)
        self.assertEqual(first.target_sample_count, 1)
        self.assertEqual(first.target_duration_sec, 0.1)
        self.assertEqual(first.source_time_start_sec, -0.3)
        self.assertEqual(first.source_time_end_sec, 0.4)
        self.assertEqual(plan.required_parent_sample_count, plan.segments[-1].end_sample)

    def test_insufficient_parent_fails_without_looping(self):
        large = plan_noise_segments(_parent(), _episodes(), _planner())
        short_parent = _parent(sample_count=large.required_parent_sample_count - 1)
        with self.assertRaisesRegex(NoiseSegmentError, "INSUFFICIENT_NOISE_PARENT_LENGTH"):
            plan_noise_segments(short_parent, _episodes(), _planner())

    def test_serialized_plan_consistency_rejects_tampering(self):
        plan = plan_noise_segments(_parent(), _episodes(), _planner())
        payload = plan.to_payload()
        mutations = []
        changed = copy.deepcopy(payload)
        changed["segments"][1]["role_index"] = 0
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["segments"][1]["guard_after_samples"] += 1
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["segments"][2]["start_sample"] += 1
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["segments"][0]["start_sample"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["segments"][3]["guard_after_samples"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["required_parent_sample_count"] -= 1
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(NoiseSegmentError):
                plan.from_payload(mutation)

    def test_parent_payload_episode_and_contract_identity_bindings(self):
        first = plan_noise_segments(_parent(), _episodes(), _planner())
        changed_parent = NoiseParentMetadata(
            noise_parent_id=stable_id("noise-parent", {"fixture": "other-parent"}),
            decoded_resampled_payload_sha256="b" * 64,
            sample_rate_hz=16000,
            sample_count=1000000,
        )
        changed = plan_noise_segments(changed_parent, _episodes(), _planner())
        self.assertNotEqual(first.plan_id, changed.plan_id)
        self.assertNotEqual(first.segments[0].segment_id, changed.segments[0].segment_id)
        changed_slots = list(_episodes())
        changed_slots[1] = EpisodeNoiseRequest("selection", 0, "utterance-s1-changed", 16000)
        changed_plan = plan_noise_segments(_parent(), changed_slots, _planner())
        self.assertNotEqual(first.segments[0].segment_id, changed_plan.segments[0].segment_id)

    def test_a3_path_like_parent_requires_explicit_provenance_adapter(self):
        with self.assertRaises(NoiseSegmentError):
            NoiseParentMetadata("parent_recording_id/path.wav", "a" * 64, 16000, 1000)
        parent = noise_parent_from_a3_provenance(
            "parent_recording_id/path.wav", "1" * 64, "2" * 64, "3" * 64, 16000, 1000
        )
        self.assertTrue(parent.noise_parent_id.startswith("noise-parent-"))
        self.assertEqual(A3_NOISE_PARENT_ADAPTER_IDENTITY, "active-asr-a4-a3-parent-provenance-adapter-v1")

    def test_api_has_no_rir_snr_or_asr_inputs(self):
        with self.assertRaises(TypeError):
            plan_noise_segments(_parent(), _episodes(), _planner(), rir="forbidden")

    def test_parameter_rebinding_changes_fixture_identity_not_algorithm_identity(self):
        base = _planner()
        changed = _planner(pre=3.0, post=1.0, guard=0.5)
        self.assertNotEqual(base.identity, changed.identity)
        self.assertEqual(base.algorithm_identity, changed.algorithm_identity)


class ActiveMaskTests(unittest.TestCase):
    def test_exact_frame_hop_conversion_and_sample_expansion(self):
        speech = tuple([1.0] * 400 + [0.0] * 1200)
        result = build_active_mask(speech, 16000, ActiveMaskContract())
        self.assertEqual(result.frame_samples, 400)
        self.assertEqual(result.hop_samples, 160)
        self.assertEqual(result.frame_count, 8)
        self.assertEqual(result.active_frame_indices, (0, 1, 2))
        self.assertEqual(result.active_sample_count, 720)
        self.assertEqual(sum(result.mask[:720]), 720)
        self.assertEqual(sum(result.mask[720:]), 0)

    def test_reference_percentile_and_threshold_are_deterministic(self):
        speech = tuple([1.0] * 1600)
        contract = ActiveMaskContract(reference_percentile=95.0, threshold_db=-40.0)
        first = build_active_mask(speech, 16000, contract)
        second = build_active_mask(speech, 16000, contract)
        self.assertEqual(first.mask_id, second.mask_id)
        self.assertEqual(first.reference_rms, 1.0)
        self.assertTrue(math.isclose(first.threshold_rms, 0.01, rel_tol=1.0e-12))
        self.assertEqual(first.active_sample_count, 1520)

    def test_mask_record_round_trip_and_tamper_rejection(self):
        result = build_active_mask(tuple([1.0] * 1600), 16000, ActiveMaskContract())
        restored = ActiveMaskRecord.from_payload(result.to_payload())
        self.assertEqual(restored.mask_id, result.mask_id)
        mutations = []
        changed = dict(result.to_payload())
        changed["frame_count"] += 1
        mutations.append(changed)
        changed = dict(result.to_payload())
        changed["active_frame_indices"] = [0, 1]
        mutations.append(changed)
        changed = dict(result.to_payload())
        changed["mask"] = list(result.mask)
        changed["mask"][1599] = True
        mutations.append(changed)
        changed = dict(result.to_payload())
        changed["mask"] = list(result.mask)
        changed["mask"][0] = False
        mutations.append(changed)
        changed = dict(result.to_payload())
        changed["active_sample_count"] += 1
        mutations.append(changed)
        changed = dict(result.to_payload())
        changed["mask_id"] = "active-mask-" + "0" * 64
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(ActiveMaskError):
                ActiveMaskRecord.from_payload(mutation)

    def test_finite_and_silence_validation(self):
        with self.assertRaisesRegex(ActiveMaskError, "finite"):
            build_active_mask(tuple([1.0] * 399 + [float("nan")] + [1.0] * 1200), 16000, ActiveMaskContract())
        with self.assertRaisesRegex(ActiveMaskError, "SILENT_OR_UNUSABLE_SPEECH"):
            build_active_mask(tuple([0.0] * 1600), 16000, ActiveMaskContract())
        with self.assertRaises(ActiveMaskError):
            build_active_mask(tuple([1.0] * 1600), 16000, ActiveMaskContract(frame_duration_ms=0.03))

    def test_amplitude_scaling_changes_metrics_not_waveform_or_activity(self):
        speech = tuple([0.5] * 1600)
        original = copy.deepcopy(speech)
        scaled = build_active_mask(tuple(value * 3.0 for value in speech), 16000, ActiveMaskContract())
        baseline = build_active_mask(speech, 16000, ActiveMaskContract())
        self.assertEqual(speech, original)
        self.assertEqual(scaled.mask, baseline.mask)
        self.assertTrue(math.isclose(scaled.reference_rms, baseline.reference_rms * 3.0, rel_tol=1.0e-12))
        self.assertTrue(math.isclose(scaled.threshold_rms, baseline.threshold_rms * 3.0, rel_tol=1.0e-12))

    def test_contract_round_trip_and_no_rir_or_asr_api(self):
        contract = ActiveMaskContract()
        self.assertEqual(ActiveMaskContract.from_payload(contract.to_payload()).identity, contract.identity)
        changed = ActiveMaskContract(threshold_db=-35.0)
        self.assertNotEqual(changed.identity, contract.identity)
        self.assertEqual(changed.algorithm_identity, contract.algorithm_identity)
        with self.assertRaises(TypeError):
            build_active_mask(tuple([1.0] * 1600), 16000, contract, rir="forbidden")
        self.assertEqual(ACTIVE_MASK_ALGORITHM_IDENTITY, "active-asr-a4-dry-speech-active-mask-v1")

    def test_pure_a4_2a_modules_do_not_import_habitat_or_speechbrain(self):
        self.assertNotIn("habitat_sim", sys.modules)
        self.assertNotIn("speechbrain", sys.modules)


if __name__ == "__main__":
    unittest.main()
