import unittest

import numpy as np
from scipy.signal import fftconvolve

from active_audition.acoustics.renderer import convolve_binaural


class RendererTests(unittest.TestCase):
    def test_full_convolution_reference_and_channel_mapping(self):
        dry = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        rir = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
        result = convolve_binaural(dry, rir)
        expected = np.column_stack((fftconvolve(dry, rir[:, 0], mode="full"), fftconvolve(dry, rir[:, 1], mode="full"))).astype(np.float32)
        self.assertEqual(result.shape, (4, 2))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_no_normalization_and_dry_input_is_not_mutated(self):
        dry = np.asarray([2.0, 2.0], dtype=np.float32)
        original = dry.copy()
        rir_a = np.asarray([[1.0, 1.0]], dtype=np.float32)
        rir_b = np.asarray([[2.0, 2.0]], dtype=np.float32)
        first = convolve_binaural(dry, rir_a)
        second = convolve_binaural(dry, rir_b)
        np.testing.assert_array_equal(dry, original)
        self.assertGreater(float(np.max(np.abs(second))), 1.0)
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
