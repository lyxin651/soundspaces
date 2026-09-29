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
    EpisodeNoiseRequest,
    NoiseParentMetadata,
    NoiseSegmentError,
    NoiseSegmentPlannerParameters,
    plan_noise_segments,
)


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
    return [
        EpisodeNoiseRequest("episode-e2", "utterance-e2", "evaluation", "e2", 0.75),
        EpisodeNoiseRequest("episode-s1", "utterance-s1", "selection", "s1", 1.0),
        EpisodeNoiseRequest("episode-e1", "utterance-e1", "evaluation", "e1", 1.25),
        EpisodeNoiseRequest("episode-s2", "utterance-s2", "selection", "s2", 0.5),
    ]


class NoiseSegmentPlannerTests(unittest.TestCase):
    def test_stable_order_and_four_nonoverlapping_segments(self):
        plan = plan_noise_segments(_parent(), _episodes(), _planner())
        self.assertEqual([item.role for item in plan.segments], ["selection", "selection", "evaluation", "evaluation"])
        self.assertEqual([item.episode_id for item in plan.segments], ["episode-s1", "episode-s2", "episode-e1", "episode-e2"])
        for first, second in zip(plan.segments, plan.segments[1:]):
            self.assertLessEqual(first.end_sample, second.start_sample)
        self.assertEqual(plan.segments[0].source_time_start_sec, -2.0)
        self.assertEqual(plan.segments[0].source_time_end_sec, 3.0)
        self.assertEqual(plan.segments[0].end_sample - plan.segments[0].start_sample, 5 * 16000)
        self.assertEqual(plan.segments[0].guard_after_samples, int(0.25 * 16000))

    def test_selection_and_evaluation_ranges_are_nonoverlapping(self):
        plan = plan_noise_segments(_parent(), _episodes(), _planner())
        selection = plan.segments[:2]
        evaluation = plan.segments[2:]
        self.assertLessEqual(selection[-1].end_sample, evaluation[0].start_sample)
        self.assertEqual(plan.required_parent_sample_count, plan.segments[-1].end_sample)

    def test_same_input_is_byte_identical_and_round_trips(self):
        first = plan_noise_segments(_parent(), _episodes(), _planner())
        second = plan_noise_segments(_parent(), list(reversed(_episodes())), _planner())
        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(canonical_json_bytes(first.to_payload()), canonical_json_bytes(second.to_payload()))
        restored = first.from_payload(first.to_payload())
        self.assertEqual(restored.plan_id, first.plan_id)
        self.assertEqual(canonical_json_bytes(restored.to_payload()), canonical_json_bytes(first.to_payload()))

    def test_segment_identity_binds_parent_payload_episode_and_contract(self):
        first = plan_noise_segments(_parent(), _episodes(), _planner())
        changed_parent = _parent()
        changed_parent = NoiseParentMetadata(
            noise_parent_id=stable_id("noise-parent", {"fixture": "other-parent"}),
            decoded_resampled_payload_sha256="b" * 64,
            sample_rate_hz=16000,
            sample_count=1000000,
        )
        changed = plan_noise_segments(changed_parent, _episodes(), _planner())
        self.assertNotEqual(first.plan_id, changed.plan_id)
        self.assertNotEqual(first.segments[0].segment_id, changed.segments[0].segment_id)
        changed_episodes = list(_episodes())
        changed_episodes[1] = EpisodeNoiseRequest("episode-s1", "utterance-s1-changed", "selection", "s1", 1.0)
        changed_episode_plan = plan_noise_segments(_parent(), changed_episodes, _planner())
        self.assertNotEqual(first.segments[0].segment_id, changed_episode_plan.segments[0].segment_id)

    def test_insufficient_parent_fails_without_looping(self):
        large = plan_noise_segments(_parent(), _episodes(), _planner())
        short_parent = _parent(sample_count=large.required_parent_sample_count - 1)
        with self.assertRaisesRegex(NoiseSegmentError, "INSUFFICIENT_NOISE_PARENT_LENGTH"):
            plan_noise_segments(short_parent, _episodes(), _planner())

    def test_sample_boundary_and_guard_accounting_are_half_open(self):
        params = _planner(pre=0.2, post=0.3, guard=0.1, sample_rate_hz=10)
        episodes = [
            EpisodeNoiseRequest("episode-{}".format(index), "utterance-{}".format(index), role, order, 0.5)
            for index, (role, order) in enumerate((
                ("selection", "s1"), ("selection", "s2"), ("evaluation", "e1"), ("evaluation", "e2")
            ))
        ]
        plan = plan_noise_segments(_parent(sample_count=1000, sample_rate_hz=10), episodes, params)
        self.assertEqual(plan.segments[0].start_sample, 0)
        self.assertEqual(plan.segments[0].end_sample, 10)
        self.assertEqual(plan.segments[1].start_sample, 11)
        self.assertEqual(plan.segments[-1].end_sample, 43)
        self.assertEqual(plan.required_parent_sample_count, 43)

    def test_int_and_float_metadata_normalize_to_same_identity(self):
        integer = _planner(pre=2, post=2, guard=0)
        floating = _planner(pre=2.0, post=2.0, guard=0.0)
        source_episodes = [
            EpisodeNoiseRequest("episode-{}".format(index), "utterance-{}".format(index), role, order, duration)
            for index, (role, order, duration) in enumerate((
                ("selection", "s1", 1), ("selection", "s2", 2), ("evaluation", "e1", 3), ("evaluation", "e2", 4)
            ))
        ]
        integer_episodes = [
            EpisodeNoiseRequest(item.episode_id, item.utterance_identity, item.role, item.stable_order_key, int(item.target_duration_sec))
            for item in source_episodes
        ]
        float_episodes = [
            EpisodeNoiseRequest(item.episode_id, item.utterance_identity, item.role, item.stable_order_key, float(item.target_duration_sec))
            for item in integer_episodes
        ]
        first = plan_noise_segments(_parent(), integer_episodes, integer)
        second = plan_noise_segments(_parent(), float_episodes, floating)
        self.assertEqual(first.plan_id, second.plan_id)

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

    def test_mask_record_round_trip_and_unknown_fields_reject(self):
        result = build_active_mask(tuple([1.0] * 1600), 16000, ActiveMaskContract())
        restored = ActiveMaskRecord.from_payload(result.to_payload())
        self.assertEqual(restored.mask_id, result.mask_id)
        payload = dict(result.to_payload(), unknown=True)
        with self.assertRaises(ActiveMaskError):
            ActiveMaskRecord.from_payload(payload)

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
