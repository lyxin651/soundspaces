import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from active_audition.data.audio import (
    generate_golden_probe,
    prepare_dry_segment,
    resample_waveform,
)
from active_audition.data.catalog import load_dry_audio_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


class DryAudioContractTests(unittest.TestCase):
    def test_golden_probe_registry_matches_wav_metadata_and_sha(self):
        registry = load_dry_audio_registry(
            str(REPO_ROOT / "registries/dry_audio.csv"), str(REPO_ROOT)
        )
        entry = registry["golden_probe_v0"]
        sample_rate, signal = wavfile.read(entry["path"])
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(signal.shape, (80000,))
        self.assertEqual(signal.dtype, np.float32)
        self.assertEqual(entry["channels"], "1")
        self.assertEqual(entry["source_dataset"], "synthetic_pipeline_v0_golden_probe")
        self.assertEqual(
            entry["sha256"],
            "df63d31dcbe1a7b0868163b81206f97f5d9928685a70e50e568d43bd90b8c54d",
        )

    def test_golden_probe_is_deterministic_and_unormalized(self):
        first = generate_golden_probe()
        second = generate_golden_probe()
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first.shape, (80000,))
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(np.max(np.abs(first))), 0.22226334, places=7)
        self.assertAlmostEqual(float(np.sqrt(np.mean(first * first))), 0.05012242, places=7)

    def test_resampling_common_rates_has_exact_canonical_count(self):
        for source_rate in (24000, 44100, 48000):
            source_count = int(5.0 * source_rate)
            source = np.zeros(source_count, dtype=np.float32)
            source[0] = 1.0
            result = resample_waveform(source, source_rate, 16000, 80000)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, (80000,))
            self.assertTrue(np.isfinite(result).all())
            self.assertLessEqual(float(np.max(np.abs(result))), 1.0)

    def test_stereo_to_mono_is_arithmetic_mean(self):
        stereo = np.asarray([[1.0, 3.0], [-2.0, 4.0]], dtype=np.float32)
        result = prepare_dry_segment(stereo, 16000, 0.0, 2.0 / 16000.0)
        np.testing.assert_array_equal(result, np.asarray([2.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
