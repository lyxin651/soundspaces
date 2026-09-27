import unittest

import numpy as np

from active_audition.asr.qualification import (
    A3_MANIFEST_SCHEMA_VERSIONS,
    add_noise_at_snr,
    canonical_json_bytes,
    deterministic_broadband_noise,
    manifest_sha256,
    waveform_identity,
)


class A3QualificationPrimitiveTest(unittest.TestCase):
    def test_noise_is_deterministic_and_same_realization_scales_only(self):
        first = deterministic_broadband_noise(16000)
        second = deterministic_broadband_noise(16000)
        np.testing.assert_array_equal(first, second)
        target = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
        mix_high, high = add_noise_at_snr(target, first, 10.0)
        mix_low, low = add_noise_at_snr(target, first, -10.0)
        self.assertAlmostEqual(high["achieved_snr_db"], 10.0, places=5)
        self.assertAlmostEqual(low["achieved_snr_db"], -10.0, places=5)
        np.testing.assert_allclose(
            (mix_high - target) / np.float32(high["noise_scale"]),
            (mix_low - target) / np.float32(low["noise_scale"]),
            rtol=2e-5,
            atol=2e-6,
        )
        self.assertEqual(high["postmix_normalization"], "none")

    def test_manifest_hash_binds_records_and_q4_paths_are_distinct(self):
        speech = {
            "utterance_id": "u", "split": "dev-clean", "speaker_id": "1", "chapter_id": "2",
            "relative_source_path": "dev-clean/u.flac", "source_file_sha256": "1" * 64,
            "decoded_waveform_sha256": "2" * 64, "samples": 80000, "duration_sec": 5.0,
            "normalized_transcript": "ONE TWO", "reference_word_count": 2, "complete_utterance": True,
        }
        rir_case = {
            "case_id": "front", "geometry_id": "shoebox", "geometry_registry_sha256": "3" * 64,
            "mesh_sha256": "4" * 64, "receiver_sensor_position_world": [0, 6, 0],
            "listener_yaw_deg": 0.0, "source_position_world": [0, 6, -1],
            "relative_azimuth_deg": 0.0, "distance_m": 1.0, "line_of_sight": "LOS",
            "runtime_config_path": "config.yaml", "runtime_config_sha256": "5" * 64,
            "a2_runtime_lock_path": "runtime.json", "a2_runtime_lock_sha256": "6" * 64,
            "expected_effective_acoustics": {},
        }
        record = {
            "record_id": "u__front", "speech": speech, "rir_case": rir_case,
            "frontends": ["mean_lr", "fixed_L", "fixed_R"],
            "path_a": {"renderer": "native_SS2", "sample_rate_hz": 16000},
            "path_b": {"renderer": "native_SS2", "sample_rate_hz": 24000, "resample_to_hz": 16000, "resampler": "resample_poly"},
            "separate_normalization": False, "a2_q2_q3_provenance": {},
        }
        manifest = {
            "schema_version": A3_MANIFEST_SCHEMA_VERSIONS["q4"],
            "kind": "q4",
            "selection_policy": {"before_wer": True},
            "records": [record],
        }
        first = manifest_sha256(manifest, "q4")
        changed = {**manifest, "records": [{**record, "path_b": {"renderer": "native_SS2", "sample_rate_hz": 16000, "resample_to_hz": 16000, "resampler": "none"}}]}
        self.assertNotEqual(first, manifest_sha256(changed, "q4"))

    def test_waveform_hash_is_dtype_canonical(self):
        value = np.asarray([0.25, -0.25], dtype=np.float32)
        self.assertEqual(waveform_identity(value), waveform_identity(value.astype(np.float64)))

    def test_artifact_serialization_is_mapping_order_invariant(self):
        self.assertEqual(canonical_json_bytes({"b": 2, "a": 1}), canonical_json_bytes({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
