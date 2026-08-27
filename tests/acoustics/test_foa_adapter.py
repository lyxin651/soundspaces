import unittest

import numpy as np

from examples.foa_adapter import (
    find_shared_direct_sample,
    measure_shared_direct_coefficients,
    native_foa_to_ambix,
    native_foa_to_canonical,
    project_to_dcase_azimuth,
)


class FOAAdapterTests(unittest.TestCase):
    def test_shared_direct_sample_ignores_later_weaker_total_reflection(self):
        values = np.zeros((4, 12), dtype=np.float32)
        values[:, 3] = [1.0, 0.5, 0.2, 0.1]
        values[1, 8] = 1.1
        measured = measure_shared_direct_coefficients(values, window_radius=1)
        self.assertEqual(find_shared_direct_sample(values), 3)
        self.assertEqual(measured["direct_sample"], 3)
        np.testing.assert_array_equal(measured["signed_coefficients"], values[:, 3])

    def test_n3d_to_sn3d_scale_is_explicit_and_does_not_mutate(self):
        values = np.ones((4, 4), dtype=np.float32)
        original = values.copy()
        result = native_foa_to_ambix(values)
        np.testing.assert_array_equal(values, original)
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1:], 1.0 / np.sqrt(3.0))

    def test_canonical_converter_rotates_world_frame_and_flips_dcase_x(self):
        values = np.zeros((4, 1), dtype=np.float32)
        values[0, 0] = 1.0
        values[3, 0] = np.sqrt(3.0)
        result = native_foa_to_canonical(values, listener_yaw_deg=90.0)
        self.assertAlmostEqual(float(result[3, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(result[2, 0]), 1.0, places=6)

    def test_project_dcase_azimuth_mapping(self):
        self.assertEqual(project_to_dcase_azimuth(90.0), -90.0)
        self.assertEqual(project_to_dcase_azimuth(-90.0), 90.0)

    def test_invalid_shape_is_rejected_and_no_clipping_is_applied(self):
        with self.assertRaises(ValueError):
            native_foa_to_canonical(np.zeros((3, 4), dtype=np.float32))
        values = np.full((4, 1), 4.0, dtype=np.float32)
        result = native_foa_to_ambix(values)
        self.assertGreater(float(result.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
