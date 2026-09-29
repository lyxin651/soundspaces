import copy
import inspect
import sys
import unittest

import numpy as np

from active_audition.a4.active_mask import ActiveMaskContract, build_active_mask
from active_audition.a4.calibration import (
    CalibrationContract,
    SelectionComponentAtInitial,
    calibrate_block_noise_gain,
)
from active_audition.a4.identity import canonical_json_bytes, stable_id
from active_audition.a4.mixer import (
    ARITHMETIC_IDENTITY,
    GlobalGainSpec,
    MixerError,
    MixtureArtifact,
    MixtureContract,
    WaveformComponent,
    build_mixture,
    validate_mixture_reconstruction,
)
from active_audition.a4.noise_segments import (
    EpisodeNoiseRequest,
    NoiseParentMetadata,
    NoiseSegmentPlannerParameters,
    plan_noise_segments,
)
from active_audition.a4.records import BlockRecord, EpisodeRecord, CalibrationArtifact
from active_audition.a4.timeline import (
    TimelineContract,
    build_common_receiver_timeline,
    map_dry_mask_to_receiver_time,
)


def _fixture(gain_value=1.0):
    parent_payload = {"fixture": "a4-2b2-mixer-parent"}
    parent = NoiseParentMetadata(
        noise_parent_id=stable_id("noise-parent", parent_payload),
        decoded_resampled_payload_sha256="a" * 64,
        sample_rate_hz=16000,
        sample_count=100000,
    )
    requests = [
        EpisodeNoiseRequest("evaluation", 1, "evaluation-1", 900),
        EpisodeNoiseRequest("selection", 0, "selection-0", 800),
        EpisodeNoiseRequest("evaluation", 0, "evaluation-0", 850),
        EpisodeNoiseRequest("selection", 1, "selection-1", 820),
    ]
    planner = NoiseSegmentPlannerParameters(16000, 0.1, 0.1, 0.01)
    plan = plan_noise_segments(parent, requests, planner)
    gain = GlobalGainSpec(gain_value)
    block_payload = {
        "geometry_id": stable_id("geometry", {"fixture": "a4-2b2-mixer-geometry"}),
        "speaker_id": "speaker-a4-2b2-mixer",
        "noise_parent_id": parent.noise_parent_id,
        "nominal_initial_snr_db": 0.0,
        "selection_utterance_ids": ["selection-0", "selection-1"],
        "evaluation_utterance_ids": ["evaluation-0", "evaluation-1"],
        "noise_segment_plan_identity": plan.plan_id,
        "global_gain_identity": gain.identity,
    }
    block = BlockRecord(
        schema_version="active-asr-a4-block-v1",
        block_id=stable_id("block", block_payload),
        **block_payload
    )
    episodes = []
    for segment in plan.segments:
        payload = {
            "schema_version": "active-asr-a4-episode-v1",
            "block_id": block.block_id,
            "role": segment.role,
            "utterance_identity": {
                "utterance_id": segment.utterance_identity,
                "decoded_waveform_sha256": "b" * 64,
            },
            "reference_identity": {"reference_sha256": "c" * 64},
            "fixed_dry_noise_segment_identity": segment.to_payload(),
            "target_source_duration_sec": segment.target_duration_sec,
            "noise_source_time_start_sec": segment.source_time_start_sec,
            "noise_source_time_end_sec": segment.source_time_end_sec,
            "noise_segment_duration_sec": segment.source_time_end_sec - segment.source_time_start_sec,
        }
        episodes.append(EpisodeRecord(episode_id=stable_id("episode", payload), **payload))
    target_source = np.ones(800, dtype=np.float32)
    noise_source = np.ones(900, dtype=np.float32)
    target_rir = np.zeros((8, 2), dtype=np.float32)
    noise_rir = np.zeros((8, 2), dtype=np.float32)
    target_rir[1, 0] = 1.0
    target_rir[3, 1] = 1.0
    noise_rir[5, 0] = 1.0
    noise_rir[2, 1] = 1.0
    timeline = build_common_receiver_timeline(
        target_source,
        noise_source,
        target_rir,
        noise_rir,
        stable_id("noise-segment", {"fixture": "target-component"}),
        plan.segments[0].segment_id,
        -10,
        TimelineContract(),
    )
    dry_mask = build_active_mask(target_source, 16000, ActiveMaskContract())
    receiver_mask = map_dry_mask_to_receiver_time(dry_mask, timeline)

    def selection_component(index, target_level, noise_level):
        target = np.zeros(timeline.target_binaural.shape, dtype=np.float32)
        noise = np.zeros(timeline.noise_binaural.shape, dtype=np.float32)
        active = np.asarray(receiver_mask.mask, dtype=bool)
        target[active, :] = target_level
        noise[active, :] = noise_level
        episode = episodes[index]
        return SelectionComponentAtInitial(
            episode_id=episode.episode_id,
            block_id=block.block_id,
            role="selection",
            utterance_id=episode.utterance_identity["utterance_id"],
            pose_scope_identity="initial_pose_only_v1",
            timeline_identity=timeline.timeline_id,
            dry_mask_identity=dry_mask.mask_id,
            receiver_mask_identity=receiver_mask.mask_id,
            target_component_identity=stable_id("target-component", {"calibration": index}),
            noise_component_identity=stable_id("noise-component", {"calibration": index}),
            target_direct_onset_samples=dict(timeline.target_direct_onset_samples),
            target_binaural=target,
            noise_binaural=noise,
            receiver_time_mask=receiver_mask,
        )

    first = selection_component(0, 2.0, 1.0)
    second = selection_component(1, 4.0, 2.0)
    calibration = calibrate_block_noise_gain(
        block,
        [first, second],
        CalibrationContract(nominal_snr_db=0.0),
    )
    target_component = WaveformComponent.from_array("target", timeline.timeline_id, timeline.target_binaural)
    noise_component = WaveformComponent.from_array("noise", timeline.timeline_id, timeline.noise_binaural)
    pose_id = stable_id("pose", {"fixture": "a4-2b2", "index": 0})
    return {
        "parent": parent,
        "plan": plan,
        "block": block,
        "episodes": tuple(episodes),
        "timeline": timeline,
        "calibration": calibration,
        "gain": gain,
        "target_component": target_component,
        "noise_component": noise_component,
        "pose_id": pose_id,
    }


class MixerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _fixture()
        self.contract = MixtureContract()

    def build(self, **updates):
        values = {
            "block": self.fixture["block"],
            "episode": self.fixture["episodes"][0],
            "pose_id": self.fixture["pose_id"],
            "timeline": self.fixture["timeline"],
            "target_component": self.fixture["target_component"],
            "noise_component": self.fixture["noise_component"],
            "calibration_artifact": self.fixture["calibration"],
            "global_gain": self.fixture["gain"],
            "contract": self.contract,
        }
        values.update(updates)
        return build_mixture(**values)

    def test_known_formula_and_channel_order_are_exact(self):
        artifact = self.build()
        expected = self.fixture["target_component"].payload.astype(np.float64) + 2.0 * self.fixture["noise_component"].payload.astype(np.float64)
        expected = expected.astype(np.float32)
        np.testing.assert_array_equal(artifact.mixture_binaural, expected)
        self.assertEqual(artifact.channel_order, ("L", "R"))
        self.assertEqual(artifact.dtype, "float32")
        self.assertEqual(artifact.alpha, self.fixture["calibration"].alpha)
        self.assertEqual(artifact.global_gain, 1.0)

    def test_residual_uses_actual_mixture_peak_denominator(self):
        artifact = self.build()
        payload = copy.deepcopy(artifact.to_payload())
        expected = artifact.mixture_binaural.copy()
        peak_index = np.unravel_index(np.argmax(np.abs(expected)), expected.shape)
        payload["mixture_samples"][peak_index[0]][peak_index[1]] += 1.5e-6
        actual = np.asarray(payload["mixture_samples"], dtype=np.float32)
        residual = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        actual_relative = residual / max(1.0, float(np.max(np.abs(actual.astype(np.float64)))))
        expected_relative = residual / max(1.0, float(np.max(np.abs(expected.astype(np.float64)))))
        self.assertNotEqual(actual_relative, expected_relative)
        payload["mixture_payload_sha256"] = __import__("hashlib").sha256(actual.tobytes()).hexdigest()
        payload["max_abs_residual"] = residual
        payload["relative_residual"] = actual_relative
        payload["reconstruction_status"] = "PASS"
        identity_payload = {
            key: value for key, value in payload.items()
            if key not in ("mixture_artifact_id", "mixture_samples")
        }
        payload["mixture_artifact_id"] = stable_id("mixture", identity_payload)
        tampered = MixtureArtifact.from_payload(payload)
        validate_mixture_reconstruction(
            tampered, self.fixture["block"], self.fixture["episodes"][0], self.fixture["pose_id"],
            self.fixture["timeline"], self.fixture["target_component"], self.fixture["noise_component"],
            self.fixture["calibration"], self.fixture["gain"], self.contract,
        )
        self.assertAlmostEqual(tampered.relative_residual, actual_relative, places=15)
        self.assertGreater(abs(tampered.relative_residual - expected_relative), 0.0)

    def test_alpha_only_comes_from_calibration_artifact(self):
        parameters = inspect.signature(build_mixture).parameters
        self.assertNotIn("alpha", parameters)
        with self.assertRaises(TypeError):
            self.build(alpha=3.0)
        self.assertEqual(self.build().alpha, self.fixture["calibration"].alpha)

    def test_pure_mixer_import_has_no_habitat_or_speechbrain_dependency(self):
        self.assertNotIn("habitat_sim", sys.modules)
        self.assertNotIn("quaternion", sys.modules)
        self.assertNotIn("speechbrain", sys.modules)

    def test_wrong_block_and_non_calibrated_artifact_are_rejected(self):
        payload = dict(self.fixture["block"].to_payload())
        payload["speaker_id"] = "different-speaker"
        payload.pop("block_id")
        bad_block = BlockRecord(block_id=stable_id("block", {key: payload[key] for key in self.fixture["block"].identity_payload()}), **payload)
        with self.assertRaises(MixerError):
            self.build(block=bad_block)

        failed = copy.deepcopy(self.fixture["calibration"].to_payload())
        failed["status"] = "FAILED"
        failed.pop("calibration_artifact_id")
        failed["calibration_artifact_id"] = stable_id("calibration", failed)
        failed_artifact = CalibrationArtifact.from_payload(failed)
        with self.assertRaises(MixerError):
            self.build(calibration_artifact=failed_artifact)

    def test_global_gain_identity_binds_value_and_is_reusable_per_block(self):
        first = self.build()
        second = self.build(pose_id=stable_id("pose", {"fixture": "a4-2b2", "index": 1}))
        self.assertEqual(first.global_gain_identity, second.global_gain_identity)
        self.assertNotEqual(first.mixture_artifact_id, second.mixture_artifact_id)
        self.assertNotEqual(GlobalGainSpec(1.0).identity, GlobalGainSpec(2.0).identity)
        with self.assertRaises(MixerError):
            self.build(global_gain=GlobalGainSpec(2.0))

    def test_non_unit_gain_changes_waveform_and_block_mixture_identity(self):
        non_unit = _fixture(2.0)
        artifact = build_mixture(
            block=non_unit["block"],
            episode=non_unit["episodes"][0],
            pose_id=non_unit["pose_id"],
            timeline=non_unit["timeline"],
            target_component=non_unit["target_component"],
            noise_component=non_unit["noise_component"],
            calibration_artifact=non_unit["calibration"],
            global_gain=non_unit["gain"],
            contract=self.contract,
        )
        expected = 2.0 * (
            non_unit["target_component"].payload.astype(np.float64)
            + non_unit["calibration"].alpha * non_unit["noise_component"].payload.astype(np.float64)
        ).astype(np.float32)
        np.testing.assert_array_equal(artifact.mixture_binaural, expected)
        self.assertEqual(non_unit["block"].global_gain_identity, non_unit["gain"].identity)
        self.assertNotEqual(non_unit["gain"].identity, self.fixture["gain"].identity)
        self.assertNotEqual(non_unit["block"].block_id, self.fixture["block"].block_id)
        self.assertNotEqual(artifact.mixture_artifact_id, self.build().mixture_artifact_id)
        self.assertEqual(artifact.mixer_algorithm_identity, self.build().mixer_algorithm_identity)
        self.assertEqual(artifact.reconstruction_identity, self.build().reconstruction_identity)

    def test_shape_dtype_finite_and_independent_component_requirements(self):
        parameters = inspect.signature(build_mixture).parameters
        self.assertNotIn("rir", parameters)
        bad_shape = WaveformComponent.from_array(
            "noise", self.fixture["timeline"].timeline_id,
            np.zeros((self.fixture["timeline"].noise_binaural.shape[0] - 1, 2), dtype=np.float32),
        )
        with self.assertRaises(MixerError):
            self.build(noise_component=bad_shape)
        with self.assertRaises(MixerError):
            WaveformComponent.from_array(
                "target", self.fixture["timeline"].timeline_id,
                self.fixture["timeline"].target_binaural.astype(np.float64),
            )
        nan_payload = self.fixture["timeline"].noise_binaural.copy()
        nan_payload[0, 0] = np.nan
        with self.assertRaises(MixerError):
            WaveformComponent.from_array("noise", self.fixture["timeline"].timeline_id, nan_payload)

    def test_repeat_hash_and_serialized_round_trip_are_deterministic(self):
        first = self.build()
        second = self.build()
        self.assertEqual(first.mixture_artifact_id, second.mixture_artifact_id)
        self.assertEqual(first.mixture_payload_sha256, second.mixture_payload_sha256)
        self.assertEqual(canonical_json_bytes(first.to_payload()), canonical_json_bytes(second.to_payload()))
        restored = MixtureArtifact.from_payload(first.to_payload())
        self.assertEqual(restored.mixture_artifact_id, first.mixture_artifact_id)
        self.assertEqual(restored.identity_payload(), first.identity_payload())

    def test_component_and_artifact_metadata_tampering_is_rejected(self):
        artifact = self.build()
        tampered = copy.deepcopy(artifact.to_payload())
        tampered["target_payload_sha256"] = "f" * 64
        with self.assertRaises(MixerError):
            MixtureArtifact.from_payload(tampered)
        tampered = copy.deepcopy(artifact.to_payload())
        tampered["alpha"] += 0.25
        with self.assertRaises(MixerError):
            MixtureArtifact.from_payload(tampered)
        tampered = copy.deepcopy(artifact.to_payload())
        tampered["mixture_payload_sha256"] = "e" * 64
        with self.assertRaises(MixerError):
            MixtureArtifact.from_payload(tampered)
        tampered = copy.deepcopy(artifact.to_payload())
        tampered["max_abs_residual"] = 1.0
        with self.assertRaises(MixerError):
            MixtureArtifact.from_payload(tampered)

    def test_component_shape_dtype_metadata_round_trip_and_tamper_rejection(self):
        artifact = self.build()
        payload = artifact.to_payload()
        self.assertEqual(payload["target_shape"], payload["noise_shape"])
        self.assertEqual(payload["noise_shape"], payload["mixture_shape"])
        self.assertEqual(payload["target_dtype"], "float32")
        self.assertEqual(payload["noise_dtype"], "float32")
        self.assertEqual(payload["mixture_dtype"], "float32")
        restored = MixtureArtifact.from_payload(payload)
        self.assertEqual(restored.target_shape, tuple(self.fixture["target_component"].payload.shape))
        for field, value in (("target_shape", [1, 2]), ("noise_shape", [1, 2]), ("mixture_shape", [1, 2])):
            tampered = copy.deepcopy(payload)
            tampered[field] = value
            with self.assertRaises(MixerError):
                MixtureArtifact.from_payload(tampered)
        for field in ("target_dtype", "noise_dtype", "mixture_dtype"):
            tampered = copy.deepcopy(payload)
            tampered[field] = "float64"
            with self.assertRaises(MixerError):
                MixtureArtifact.from_payload(tampered)

    def test_reconstruction_gate_and_tampered_waveform_fail_without_normalization(self):
        artifact = self.build()
        self.assertEqual(artifact.max_abs_residual, 0.0)
        self.assertEqual(artifact.relative_residual, 0.0)
        self.assertEqual(artifact.reconstruction_status, "PASS")
        self.assertEqual(artifact.residual_gate_threshold, 1.0e-6)
        tampered = copy.deepcopy(artifact.to_payload())
        tampered["mixture_samples"][0][0] += 1.0
        tampered["mixture_payload_sha256"] = __import__("hashlib").sha256(
            np.asarray(tampered["mixture_samples"], dtype=np.float32).tobytes()
        ).hexdigest()
        tampered["max_abs_residual"] = 1.0
        tampered["relative_residual"] = 1.0
        tampered["reconstruction_status"] = "FAIL"
        identity_payload = {key: value for key, value in tampered.items() if key not in ("mixture_artifact_id", "mixture_samples")}
        tampered["mixture_artifact_id"] = stable_id("mixture", identity_payload)
        invalid = MixtureArtifact.from_payload(tampered)
        with self.assertRaises(MixerError):
            validate_mixture_reconstruction(
                invalid, self.fixture["block"], self.fixture["episodes"][0], self.fixture["pose_id"],
                self.fixture["timeline"], self.fixture["target_component"], self.fixture["noise_component"],
                self.fixture["calibration"], self.fixture["gain"], self.contract,
            )

    def test_peak_over_unit_is_diagnostic_only_and_mean_lr_is_not_calibration(self):
        artifact = self.build()
        self.assertTrue(artifact.diagnostics["peak_over_unit"])
        self.assertGreater(artifact.diagnostics["peak_over_unit_max_abs"], 1.0)
        self.assertEqual(artifact.mixture_binaural.max(), np.float32(3.0))
        self.assertEqual(artifact.alpha, self.fixture["calibration"].alpha)
        self.assertIn("diagnostic_snr_db_two_ear", artifact.diagnostics)
        self.assertIn("diagnostic_snr_db_mean_lr", artifact.diagnostics)
        self.assertEqual(artifact.arithmetic_identity, ARITHMETIC_IDENTITY)

    def test_episode_segment_and_calibration_block_bindings_are_enforced(self):
        wrong_timeline = build_common_receiver_timeline(
            np.ones(800, dtype=np.float32),
            np.ones(900, dtype=np.float32),
            np.pad(np.eye(2, dtype=np.float32), ((0, 6), (0, 0))),
            np.pad(np.eye(2, dtype=np.float32), ((0, 6), (0, 0))),
            self.fixture["timeline"].target_source_segment_identity,
            stable_id("noise-segment", {"wrong": True}),
            -10,
            TimelineContract(),
        )
        with self.assertRaises(MixerError):
            self.build(
                timeline=wrong_timeline,
                target_component=WaveformComponent.from_array("target", wrong_timeline.timeline_id, wrong_timeline.target_binaural),
                noise_component=WaveformComponent.from_array("noise", wrong_timeline.timeline_id, wrong_timeline.noise_binaural),
            )


if __name__ == "__main__":
    unittest.main()
