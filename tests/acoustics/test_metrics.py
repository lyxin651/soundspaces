import unittest

import numpy as np

from active_audition.acoustics.metrics import (
    ANALYSIS_SAMPLES,
    MetricsError,
    compute_rir_metrics,
    compute_waveform_metrics,
    interaural_correlation_and_lag,
    waveform_qc_window,
)


class MetricsTests(unittest.TestCase):
    def test_fixed_window_and_too_short_reject(self):
        waveform = np.zeros((ANALYSIS_SAMPLES + 20, 2), dtype=np.float32)
        self.assertEqual(waveform_qc_window(waveform).shape, (ANALYSIS_SAMPLES, 2))
        with self.assertRaises(MetricsError):
            waveform_qc_window(waveform[: ANALYSIS_SAMPLES - 1])

    def test_rms_ild_peak_silence_and_over_unit(self):
        waveform = np.zeros((ANALYSIS_SAMPLES, 2), dtype=np.float32)
        waveform[:100, 0] = 0.5
        waveform[:100, 1] = 0.25
        waveform[100, 0] = 1.0
        original = waveform.copy()
        metrics = compute_waveform_metrics(waveform)
        self.assertAlmostEqual(metrics["rms_left_dbfs"], 20.0 * np.log10(np.sqrt(0.26 / 800.0)), places=5)
        self.assertGreater(metrics["ild_db"], 0.0)
        self.assertEqual(metrics["peak_left"], 1.0)
        self.assertEqual(metrics["peak_right"], 0.25)
        self.assertAlmostEqual(metrics["silence_fraction"], (ANALYSIS_SAMPLES - 101) / float(ANALYSIS_SAMPLES))
        self.assertEqual(metrics["clipping_fraction"], 1.0 / float(ANALYSIS_SAMPLES * 2))
        np.testing.assert_array_equal(waveform, original)

    def test_power_ild_sign(self):
        left_stronger = np.ones((ANALYSIS_SAMPLES, 2), dtype=np.float32)
        left_stronger[:, 0] *= 2.0
        left_stronger[:, 1] *= 1.0
        self.assertGreater(compute_waveform_metrics(left_stronger)["ild_db"], 0.0)
        left_stronger[:, [0, 1]] = left_stronger[:, [1, 0]]
        self.assertLess(compute_waveform_metrics(left_stronger)["ild_db"], 0.0)

    def test_correlation_lag_zero_positive_negative_and_tie_break(self):
        rng = np.random.default_rng(7)
        left = rng.normal(size=ANALYSIS_SAMPLES).astype(np.float64)
        right = left.copy()
        correlation, lag = interaural_correlation_and_lag(left, right)
        self.assertAlmostEqual(correlation, 1.0, places=10)
        self.assertEqual(lag, 0)
        right = np.concatenate([np.zeros(5), left[:-5]])
        self.assertEqual(interaural_correlation_and_lag(left, right)[1], 5)
        left_delayed = np.concatenate([np.zeros(7), right[:-7]])
        self.assertEqual(interaural_correlation_and_lag(left_delayed, right)[1], -7)
        zeros = np.zeros(ANALYSIS_SAMPLES)
        self.assertEqual(interaural_correlation_and_lag(zeros, zeros)[1], 0)

    def test_full_energy_and_rir_tail_variable_length(self):
        waveform = np.ones((ANALYSIS_SAMPLES + 3, 2), dtype=np.float32)
        metrics = compute_waveform_metrics(waveform)
        self.assertEqual(metrics["full_num_samples"], ANALYSIS_SAMPLES + 3)
        self.assertEqual(metrics["full_energy_total"], float((ANALYSIS_SAMPLES + 3) * 2))
        rir = np.zeros((2000, 2), dtype=np.float32)
        rir[-1] = [1.0, 1.0]
        result = compute_rir_metrics(rir)
        self.assertEqual(result["rir_num_samples"], 2000)
        self.assertAlmostEqual(result["rir_tail_energy_ratio_100ms"], 1.0)
        with self.assertRaises(MetricsError):
            compute_waveform_metrics(np.full((ANALYSIS_SAMPLES, 2), np.nan, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
