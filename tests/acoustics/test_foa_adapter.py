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

    def test_canonical_converter_rotates_world_frame_into_dcase_axes(self):
        values = np.zeros((4, 1), dtype=np.float32)
        values[0, 0] = 1.0
        values[3, 0] = np.sqrt(3.0)
        result = native_foa_to_canonical(values, listener_yaw_deg=90.0)
        self.assertAlmostEqual(float(result[1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(result[2, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(result[3, 0]), 1.0, places=6)

    def test_yaw_transform_matches_independent_geometry_for_all_required_views(self):
        directions = {
            "front": (0.0, 0.0, -1.0), "back": (0.0, 0.0, 1.0),
            "left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0),
            "front-left": (-2.0 ** -0.5, 0.0, -2.0 ** -0.5),
            "front-right": (2.0 ** -0.5, 0.0, -2.0 ** -0.5),
            "elevated": (0.0, 1.0, 0.0), "depressed": (0.0, -1.0, 0.0),
        }
        for yaw in (0.0, 45.0, 90.0, -45.0):
            angle = np.deg2rad(yaw)
            for name, (x_world, y_world, z_world) in directions.items():
                native = np.asarray([[1.0], [np.sqrt(3.0) * y_world],
                                     [np.sqrt(3.0) * z_world], [np.sqrt(3.0) * x_world]], dtype=np.float32)
                converted = native_foa_to_canonical(native, yaw)
                right_local = np.cos(angle) * x_world + np.sin(angle) * z_world
                up_local = y_world
                back_local = -np.sin(angle) * x_world + np.cos(angle) * z_world
                expected = np.asarray([-back_local, -right_local, up_local])
                actual = np.asarray([converted[3, 0], converted[1, 0], converted[2, 0]])
                np.testing.assert_allclose(actual, expected, atol=2e-6, err_msg="{} yaw {}".format(name, yaw))
                self.assertEqual(converted.dtype, np.float32)
                self.assertTrue(np.isfinite(converted).all())
                self.assertAlmostEqual(float(converted[0, 0]), 1.0, places=6)

    def test_corrected_yaw_regression_catches_legacy_90_and_180_degree_errors(self):
        def legacy_dcase(world_x, world_z, yaw_deg):
            angle = np.deg2rad(yaw_deg)
            old_x = np.cos(angle) * world_x - np.sin(angle) * world_z
            old_z = np.sin(angle) * world_x + np.cos(angle) * world_z
            return np.asarray([-old_z, -old_x])

        for yaw, expected_error in ((45.0, 90.0), (90.0, 180.0)):
            native = np.asarray([[1.0], [0.0], [0.0], [np.sqrt(3.0)]], dtype=np.float32)
            actual = native_foa_to_canonical(native, yaw)
            actual_direction = np.asarray([actual[3, 0], actual[1, 0]])
            angle = np.deg2rad(yaw)
            expected_direction = np.asarray([-(-np.sin(angle)), -np.cos(angle)])
            corrected_error = np.rad2deg(np.arccos(np.clip(np.dot(actual_direction, expected_direction), -1.0, 1.0)))
            legacy_direction = legacy_dcase(1.0, 0.0, yaw)
            legacy_error = np.rad2deg(np.arccos(np.clip(np.dot(legacy_direction, expected_direction), -1.0, 1.0)))
            self.assertLess(corrected_error, 1.0)
            self.assertAlmostEqual(float(legacy_error), expected_error, places=5)

    def test_canonical_axis_mapping_matches_front_left_up(self):
        # Native RLR basis is [right, up, back]; canonical is [front, left, up].
        cases = {
            "front": (0.0, 0.0, -1.0),
            "right": (-1.0, 0.0, 0.0),
            "left": (1.0, 0.0, 0.0),
            "back": (0.0, 0.0, 1.0),
            "elevated": (0.0, 1.0, 0.0),
        }
        for name, (front, left, up) in cases.items():
            native = np.asarray([[1.0], [np.sqrt(3.0) * up], [np.sqrt(3.0) * (-front)], [np.sqrt(3.0) * (-left)]], dtype=np.float32)
            canonical = native_foa_to_canonical(native)
            actual = np.asarray([canonical[3, 0], canonical[1, 0], canonical[2, 0]])
            expected = np.asarray([front, left, up])
            np.testing.assert_allclose(actual, expected, atol=1e-6, err_msg=name)

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
