import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.clsdoa_v1.model_c1.foa_fixture import (
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE_HZ,
    load_wav_dtype_aware,
    make_canonical_foa_fixture,
    scaling_comparison,
    write_float32_wav,
)


class ModelC1FOAFixtureTests(unittest.TestCase):
    def test_fixture_contract_and_nonzero_directional_channels(self):
        fixture = make_canonical_foa_fixture("directional")
        self.assertEqual(fixture.shape, (NUM_SAMPLES, NUM_CHANNELS))
        self.assertEqual(fixture.dtype, np.float32)
        self.assertTrue(np.isfinite(fixture).all())
        self.assertTrue(np.all(np.sqrt(np.mean(fixture * fixture, axis=0)) > 0.0))

    def test_float32_wav_is_not_divided_by_32768_in_dtype_aware_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "fixture.wav"
            original = make_canonical_foa_fixture("directional")
            write_float32_wav(wav_path, original)
            loaded, sample_rate, raw_dtype = load_wav_dtype_aware(wav_path)
            self.assertEqual(sample_rate, SAMPLE_RATE_HZ)
            self.assertEqual(raw_dtype, "float32")
            np.testing.assert_allclose(loaded, original, atol=1e-7)

    def test_int16_scale_is_supported_without_peak_or_rms_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "fixture_int16.wav"
            original = make_canonical_foa_fixture("generic")
            sf.write(str(wav_path), original, SAMPLE_RATE_HZ, subtype="PCM_16")
            loaded, sample_rate, raw_dtype = load_wav_dtype_aware(wav_path)
            self.assertEqual(sample_rate, SAMPLE_RATE_HZ)
            self.assertEqual(raw_dtype, "int16")
            expected = np.round(np.clip(original, -1.0, 1.0) * 32767.0).astype(np.int16).astype(np.float32) / 32768.0
            np.testing.assert_allclose(loaded, expected, atol=1.0 / 32768.0)
            self.assertNotAlmostEqual(float(np.max(np.abs(loaded))), 1.0, places=2)

    def test_shape_sample_rate_sample_count_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "bad_sr.wav"
            sf.write(str(wav_path), make_canonical_foa_fixture("generic"), 16000, subtype="FLOAT")
            with self.assertRaises(ValueError):
                load_wav_dtype_aware(wav_path)

            bad_channels = Path(tmp) / "bad_channels.wav"
            sf.write(str(bad_channels), np.zeros((NUM_SAMPLES, 2), dtype=np.float32), SAMPLE_RATE_HZ, subtype="FLOAT")
            with self.assertRaises(ValueError):
                load_wav_dtype_aware(bad_channels)

            bad_length = Path(tmp) / "bad_length.wav"
            sf.write(str(bad_length), np.zeros((100, NUM_CHANNELS), dtype=np.float32), SAMPLE_RATE_HZ, subtype="FLOAT")
            with self.assertRaises(ValueError):
                load_wav_dtype_aware(bad_length)

    def test_nonfinite_fixture_is_rejected_on_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "bad.wav"
            fixture = make_canonical_foa_fixture("generic")
            fixture[0, 0] = np.nan
            with self.assertRaises(ValueError):
                write_float32_wav(wav_path, fixture)

    def test_simulated_int16_assumption_loader_attenuates_float32(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "fixture.wav"
            write_float32_wav(wav_path, make_canonical_foa_fixture("directional"))
            comparison = scaling_comparison(wav_path)
            self.assertAlmostEqual(comparison["stock_loader"]["attenuation_ratio"], 1.0 / 32768.0, places=8)
            self.assertLess(comparison["stock_loader"]["attenuation_db"], -90.0)
            self.assertTrue(comparison["dtype_aware_adapter"]["amplitude_preserved"])


if __name__ == "__main__":
    unittest.main()
