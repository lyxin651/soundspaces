import unittest

import numpy as np

from active_audition.asr.g6_failure_attribution import (
    _edc_fit,
    _gain_summary,
    _make_rir_variant,
    _safe_gain,
    _stats,
)


class G6FailureAttributionUnitTests(unittest.TestCase):
    def test_waveform_stats_are_raw_and_deterministic(self):
        value = np.asarray([0.5, -0.5, 0.0, 0.25], dtype=np.float32)
        stats = _stats(value, 16000)
        self.assertEqual(stats["samples"], 4)
        self.assertAlmostEqual(stats["sum_square_energy"], 0.5625)
        self.assertAlmostEqual(stats["rms"], np.sqrt(0.5625 / 4.0))
        self.assertFalse(stats["clipping"])
        self.assertEqual(stats["sha256"], _stats(value, 16000)["sha256"])

    def test_direct_variant_preserves_time_axis_and_does_not_normalize(self):
        rir = np.arange(20, dtype=np.float32).reshape(10, 2)
        variant = _make_rir_variant(rir, 2, 5)
        expected = np.zeros_like(rir)
        expected[2:5] = rir[2:5]
        np.testing.assert_array_equal(variant, expected)
        self.assertEqual(variant.shape, rir.shape)

    def test_gain_is_acoustic_and_clipping_is_explicit(self):
        waves = [np.asarray([0.2, -0.3], dtype=np.float32), np.asarray([0.1, 0.25], dtype=np.float32)]
        self.assertTrue(_safe_gain(waves, 2.0))
        self.assertFalse(_safe_gain(waves, 4.0))
        summary = _gain_summary(waves, 2.0, "acoustic_energy_only")
        self.assertEqual(summary["source"], "acoustic_energy_only")
        self.assertFalse(summary["unsafe_clipping"])
        self.assertIn("sum_square", summary["formula"])

    def test_decay_fit_reports_applicability_without_claiming_rt60(self):
        # A synthetic 20 dB/sec raw EDC, sampled at 1 kHz.
        time = np.arange(0, 3.0, 0.001)
        edc = np.power(10.0, -2.0 * time).astype(np.float64)
        fit = _edc_fit(edc, 0, 1000, 0.0, -10.0)
        self.assertEqual(fit["status"], "APPLICABLE")
        self.assertNotIn("rt60", fit)


if __name__ == "__main__":
    unittest.main()
