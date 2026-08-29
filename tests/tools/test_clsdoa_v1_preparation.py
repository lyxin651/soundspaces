import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.clsdoa_v1.prepare_source_pool import _normalize, _safe_name


class SourcePreparationTests(unittest.TestCase):
    def test_normalization_is_one_scalar_gain_with_peak_guard(self):
        waveform = np.ones(24000, dtype=np.float32) * 0.01
        output, stats = _normalize(waveform)
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(float(np.max(np.abs(output))), 0.50)
        self.assertEqual(stats["peak_guard_limited"], "false")
        self.assertAlmostEqual(float(np.sqrt(np.mean(output ** 2))), 10 ** (-24 / 20), places=5)

    def test_safe_name_is_stable(self):
        self.assertEqual(_safe_name("desed:train/a"), "desed__train__a")


if __name__ == "__main__":
    unittest.main()
