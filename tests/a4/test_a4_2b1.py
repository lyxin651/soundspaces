import copy
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import yaml

from active_audition.a4.active_mask import ActiveMaskContract, ActiveMaskError, build_active_mask
from active_audition.a4.calibration import (
    CalibrationContract,
    CalibrationError,
    SelectionComponentAtInitial,
    calibrate_block_noise_gain,
)
from active_audition.a4.identity import canonical_json_bytes, stable_id
from active_audition.a4.mixer import GlobalGainSpec
from active_audition.a4.records import BlockRecord
from active_audition.a4.timeline import (
    TimelineContract,
    TimelineError,
    a2_direct_onset_samples,
    build_common_receiver_timeline,
    map_dry_mask_to_receiver_time,
)


class A42B1FixtureMixin:
    def timeline(self, target_delay=(2, 4), noise_delay=(7, 9)):
        target_source = np.ones(1600, dtype=np.float32)
        noise_source = np.ones(1800, dtype=np.float32)
        target_rir = np.zeros((16, 2), dtype=np.float32)
        noise_rir = np.zeros((16, 2), dtype=np.float32)
        target_rir[target_delay[0], 0] = 1.0
        target_rir[target_delay[1], 1] = 1.0
        noise_rir[noise_delay[0], 0] = 1.0
        noise_rir[noise_delay[1], 1] = 1.0
        contract = TimelineContract()
        timeline = build_common_receiver_timeline(
            target_source,
            noise_source,
            target_rir,
            noise_rir,
            stable_id("noise-segment", {"fixture": "target"}),
            stable_id("noise-segment", {"fixture": "noise"}),
            -100,
            contract,
        )
        mask = build_active_mask(target_source, 16000, ActiveMaskContract())
        receiver_mask = map_dry_mask_to_receiver_time(mask, timeline)
        return timeline, mask, receiver_mask

    def block(self):
        payload = {
            "geometry_id": stable_id("geometry", {"fixture": "a4-2b1"}),
            "speaker_id": "speaker-a4-2b1",
            "noise_parent_id": stable_id("noise-parent", {"fixture": "a4-2b1"}),
            "nominal_initial_snr_db": 0.0,
            "selection_utterance_ids": ["selection-0", "selection-1"],
            "evaluation_utterance_ids": ["evaluation-0", "evaluation-1"],
            "noise_segment_plan_identity": stable_id("noise-segment-plan", {"fixture": "a4-2b1"}),
            "global_gain_identity": GlobalGainSpec(1.0).identity,
        }
        return BlockRecord(
            schema_version="active-asr-a4-block-v1",
            block_id=stable_id("block", payload),
            **payload,
        )

    def component(self, block, timeline, mask, receiver_mask, index, target_level=2.0, noise_level=1.0):
        active = np.asarray(receiver_mask.mask, dtype=bool)
        target = np.zeros(timeline.target_binaural.shape, dtype=np.float32)
        noise = np.zeros(timeline.noise_binaural.shape, dtype=np.float32)
        target[active, :] = target_level
        noise[active, :] = noise_level
        return SelectionComponentAtInitial(
            episode_id=stable_id("episode", {"fixture": "a4-2b1", "index": index}),
            block_id=block.block_id,
            role="selection",
            utterance_id="selection-{}".format(index),
            pose_scope_identity="initial_pose_only_v1",
            timeline_identity=timeline.timeline_id,
            dry_mask_identity=mask.mask_id,
            receiver_mask_identity=receiver_mask.mask_id,
            target_component_identity=stable_id("target-component", {"fixture": "a4-2b1", "index": index}),
            noise_component_identity=stable_id("noise-component", {"fixture": "a4-2b1", "index": index}),
            target_direct_onset_samples=dict(timeline.target_direct_onset_samples),
            target_binaural=target,
            noise_binaural=noise,
            receiver_time_mask=receiver_mask,
        )


class TimelineTests(unittest.TestCase, A42B1FixtureMixin):
    def test_common_timeline_preserves_relative_target_noise_and_channel_delays(self):
        timeline, _, _ = self.timeline()
        self.assertEqual(timeline.receiver_end_sample_exclusive - timeline.receiver_start_sample, timeline.target_binaural.shape[0])
        target_l = np.flatnonzero(np.abs(timeline.target_binaural[:, 0]) > 1.0e-6)
        target_r = np.flatnonzero(np.abs(timeline.target_binaural[:, 1]) > 1.0e-6)
        noise_l = np.flatnonzero(np.abs(timeline.noise_binaural[:, 0]) > 1.0e-6)
        self.assertEqual(int(target_l[0] - timeline.target_receiver_offset_samples), 2)
        self.assertEqual(int(target_r[0] - timeline.target_receiver_offset_samples), 4)
        self.assertEqual(int(noise_l[0] - timeline.noise_receiver_offset_samples), 7)
        self.assertLess(int(noise_l[0]), int(target_l[0]))
        self.assertNotEqual(int(target_l[0]), int(target_r[0]))

    def test_no_direct_peak_or_left_right_alignment_and_common_shape(self):
        timeline, _, _ = self.timeline(target_delay=(3, 8), noise_delay=(11, 14))
        self.assertEqual(timeline.target_direct_onset_samples, {"L": 3, "R": 8})
        self.assertEqual(timeline.noise_direct_onset_samples, {"L": 11, "R": 14})
        self.assertEqual(timeline.target_binaural.shape, timeline.noise_binaural.shape)
        self.assertEqual(timeline.target_receiver_offset_samples, 100)
        self.assertEqual(timeline.noise_receiver_offset_samples, 0)

    def test_a2_onset_helper_equivalence_fixture(self):
        rir = np.zeros((32, 2), dtype=np.float32)
        rir[4, 0] = 0.5
        rir[9, 1] = 1.0
        from active_audition.receiver.qualification import _direct_window

        contract_path = Path(__file__).resolve().parents[2] / "configs/active_audition/v1/metric_contract_oracle_v3.yaml"
        with contract_path.open(encoding="utf-8") as handle:
            a2_contract = yaml.safe_load(handle)
        frozen = _direct_window(rir, 16000, a2_contract["production_native16"])
        self.assertEqual(a2_direct_onset_samples(rir), {"L": 4, "R": 9})
        self.assertEqual(frozen["channel_onset_samples"], {"L": 4, "R": 9})

    def test_timeline_metadata_is_repeated_byte_identical(self):
        first, _, _ = self.timeline()
        second, _, _ = self.timeline()
        self.assertEqual(first.timeline_id, second.timeline_id)
        self.assertEqual(canonical_json_bytes(first.to_payload()), canonical_json_bytes(second.to_payload()))

    def test_rir_delay_changes_receiver_placement_not_dry_segment_identity(self):
        first, _, _ = self.timeline(target_delay=(2, 4))
        changed, _, _ = self.timeline(target_delay=(5, 7))
        self.assertEqual(first.target_source_segment_identity, changed.target_source_segment_identity)
        self.assertEqual(first.noise_source_segment_identity, changed.noise_source_segment_identity)
        self.assertNotEqual(first.timeline_id, changed.timeline_id)
        self.assertNotEqual(first.target_direct_onset_samples, changed.target_direct_onset_samples)

    def test_mask_mapping_uses_target_delay_and_rejects_bad_support(self):
        timeline, dry_mask, receiver_mask = self.timeline()
        self.assertEqual(receiver_mask.dry_mask_identity, dry_mask.mask_id)
        self.assertEqual(receiver_mask.active_sample_count, dry_mask.active_sample_count)
        self.assertEqual(receiver_mask.active_start_sample, timeline.target_source_start_sample + 2)
        bad = SimpleNamespace(
            sample_rate_hz=16000,
            sample_count=dry_mask.sample_count,
            mask=tuple(dry_mask.mask[:-1]),
            mask_id=dry_mask.mask_id,
        )
        with self.assertRaises(TimelineError):
            map_dry_mask_to_receiver_time(bad, timeline)


class CalibrationTests(unittest.TestCase, A42B1FixtureMixin):
    def components(self):
        timeline, mask, receiver_mask = self.timeline()
        block = self.block()
        return (
            block,
            timeline,
            mask,
            receiver_mask,
            self.component(block, timeline, mask, receiver_mask, 0, 2.0, 1.0),
            self.component(block, timeline, mask, receiver_mask, 1, 4.0, 2.0),
        )

    def test_accumulates_two_selection_episodes_with_two_ear_power(self):
        block, _, _, _, first, second = self.components()
        artifact = calibrate_block_noise_gain(block, [second, first], CalibrationContract(nominal_snr_db=0.0))
        self.assertAlmostEqual(artifact.ps, 10.0)
        self.assertAlmostEqual(artifact.pn, 2.5)
        self.assertAlmostEqual(artifact.alpha, 2.0)
        self.assertAlmostEqual(artifact.measured_snr_db, 0.0, places=12)
        self.assertEqual(artifact.active_sample_count, first.active_sample_count + second.active_sample_count)

    def test_two_ear_mean_square_is_not_mean_lr_power(self):
        block, timeline, mask, receiver_mask, _, _ = self.components()
        component = self.component(block, timeline, mask, receiver_mask, 0, 1.0, 1.0)
        target = component.target_binaural.copy()
        target[np.asarray(receiver_mask.mask, dtype=bool), 1] = 3.0
        component = SelectionComponentAtInitial(
            episode_id=component.episode_id,
            block_id=component.block_id,
            role=component.role,
            utterance_id=component.utterance_id,
            pose_scope_identity=component.pose_scope_identity,
            timeline_identity=component.timeline_identity,
            dry_mask_identity=component.dry_mask_identity,
            receiver_mask_identity=component.receiver_mask_identity,
            target_component_identity=component.target_component_identity,
            noise_component_identity=component.noise_component_identity,
            target_direct_onset_samples=component.target_direct_onset_samples,
            target_binaural=target,
            noise_binaural=component.noise_binaural,
            receiver_time_mask=component.receiver_time_mask,
        )
        other = self.component(block, timeline, mask, receiver_mask, 1, 1.0, 1.0)
        artifact = calibrate_block_noise_gain(block, [component, other], CalibrationContract(nominal_snr_db=0.0))
        self.assertAlmostEqual(artifact.ps, 3.0)
        self.assertAlmostEqual(artifact.alpha, 3.0 ** 0.5)

    def test_selection_only_initial_pose_and_exact_two_boundary(self):
        block, timeline, mask, receiver_mask, first, second = self.components()
        with self.assertRaises(CalibrationError):
            calibrate_block_noise_gain(block, [first, second], CalibrationContract(nominal_snr_db=1.0))
        with self.assertRaises(CalibrationError):
            calibrate_block_noise_gain(block, [first], CalibrationContract(nominal_snr_db=0.0))
        with self.assertRaises(CalibrationError):
            calibrate_block_noise_gain(block, [first, first], CalibrationContract(nominal_snr_db=0.0))
        with self.assertRaises(CalibrationError):
            SelectionComponentAtInitial(
                episode_id=first.episode_id,
                block_id=first.block_id,
                role="evaluation",
                utterance_id=first.utterance_id,
                pose_scope_identity=first.pose_scope_identity,
                timeline_identity=timeline.timeline_id,
                dry_mask_identity=mask.mask_id,
                receiver_mask_identity=receiver_mask.mask_id,
                target_component_identity=first.target_component_identity,
                noise_component_identity=first.noise_component_identity,
                target_direct_onset_samples=first.target_direct_onset_samples,
                target_binaural=first.target_binaural,
                noise_binaural=first.noise_binaural,
                receiver_time_mask=receiver_mask,
            )

    def test_artifact_round_trip_and_tamper_rejection(self):
        block, _, _, _, first, second = self.components()
        artifact = calibrate_block_noise_gain(block, [first, second], CalibrationContract(nominal_snr_db=0.0))
        rebuilt = type(artifact).from_payload(artifact.to_payload())
        self.assertEqual(rebuilt.calibration_artifact_id, artifact.calibration_artifact_id)
        self.assertEqual(rebuilt.identity_payload(), artifact.identity_payload())
        tampered = copy.deepcopy(artifact.to_payload())
        tampered["alpha"] += 0.1
        with self.assertRaises(Exception):
            type(artifact).from_payload(tampered)

    def test_power_and_finite_failures_are_explicit(self):
        block, timeline, mask, receiver_mask, first, second = self.components()
        zero_noise = np.zeros_like(first.noise_binaural)
        zero_noise_component = SelectionComponentAtInitial(
            episode_id=first.episode_id,
            block_id=first.block_id,
            role=first.role,
            utterance_id=first.utterance_id,
            pose_scope_identity=first.pose_scope_identity,
            timeline_identity=first.timeline_identity,
            dry_mask_identity=first.dry_mask_identity,
            receiver_mask_identity=first.receiver_mask_identity,
            target_component_identity=first.target_component_identity,
            noise_component_identity=first.noise_component_identity,
            target_direct_onset_samples=first.target_direct_onset_samples,
            target_binaural=first.target_binaural,
            noise_binaural=zero_noise,
            receiver_time_mask=receiver_mask,
        )
        with self.assertRaises(CalibrationError):
            zero_noise_second = self.component(block, timeline, mask, receiver_mask, 1, 4.0, 0.0)
            calibrate_block_noise_gain(block, [zero_noise_component, zero_noise_second], CalibrationContract(nominal_snr_db=0.0))
        with self.assertRaises(CalibrationError):
            SelectionComponentAtInitial(
                episode_id=first.episode_id,
                block_id=first.block_id,
                role=first.role,
                utterance_id=first.utterance_id,
                pose_scope_identity=first.pose_scope_identity,
                timeline_identity=timeline.timeline_id,
                dry_mask_identity=mask.mask_id,
                receiver_mask_identity=receiver_mask.mask_id,
                target_component_identity=first.target_component_identity,
                noise_component_identity=first.noise_component_identity,
                target_direct_onset_samples=first.target_direct_onset_samples,
                target_binaural=np.full_like(first.target_binaural, np.nan),
                noise_binaural=first.noise_binaural,
                receiver_time_mask=receiver_mask,
            )

    def test_calibration_api_has_no_rir_or_asr_arguments(self):
        block, _, _, _, first, second = self.components()
        with self.assertRaises(TypeError):
            calibrate_block_noise_gain(
                block,
                [first, second],
                CalibrationContract(nominal_snr_db=0.0),
                rir="forbidden",
            )


class ActiveMaskRegressionTests(unittest.TestCase):
    def test_exact_native16_frame_hop_and_silence_failure(self):
        record = build_active_mask(np.ones(1600, dtype=np.float32), 16000, ActiveMaskContract())
        self.assertEqual(record.frame_samples, 400)
        self.assertEqual(record.hop_samples, 160)
        self.assertEqual(record.frame_count, 8)
        self.assertEqual(record.active_sample_count, sum(record.mask))
        with self.assertRaises(ActiveMaskError):
            build_active_mask(np.zeros(1600, dtype=np.float32), 16000, ActiveMaskContract())


if __name__ == "__main__":
    unittest.main()
