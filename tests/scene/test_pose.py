import math
import unittest

import numpy as np

from active_audition.scene.pose import (
    agent_local_direction_to_world,
    listener_sensor_position,
    normalize_yaw_deg,
    relative_azimuth_deg,
    rotate_left_yaw,
    rotate_right_yaw,
    yaw_to_forward_world,
    yaw_to_right_world,
)
from active_audition.types import SourceSpec


class PoseTests(unittest.TestCase):
    def test_forward_vectors_match_habitat_convention(self):
        expected = {
            0.0: (0.0, 0.0, -1.0),
            45.0: (-math.sqrt(0.5), 0.0, -math.sqrt(0.5)),
            -45.0: (math.sqrt(0.5), 0.0, -math.sqrt(0.5)),
            90.0: (-1.0, 0.0, 0.0),
        }
        for yaw, vector in expected.items():
            np.testing.assert_allclose(yaw_to_forward_world(yaw), vector, atol=1e-6)

    def test_right_vector_and_local_translation_follow_yaw(self):
        np.testing.assert_allclose(yaw_to_right_world(0.0), (1.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(
            agent_local_direction_to_world("forward", 90.0), (-1.0, 0.0, 0.0), atol=1e-6
        )
        np.testing.assert_allclose(
            agent_local_direction_to_world("right", 90.0), (0.0, 0.0, -1.0), atol=1e-6
        )

    def test_relative_azimuth_is_front_right_left_back(self):
        listener = (0.0, 1.5, 0.0)
        self.assertAlmostEqual(relative_azimuth_deg((0.0, 1.5, -1.0), listener, 0.0), 0.0)
        self.assertAlmostEqual(relative_azimuth_deg((1.0, 1.5, 0.0), listener, 0.0), 90.0)
        self.assertAlmostEqual(relative_azimuth_deg((-1.0, 1.5, 0.0), listener, 0.0), -90.0)
        self.assertAlmostEqual(abs(relative_azimuth_deg((0.0, 1.5, 1.0), listener, 0.0)), 180.0)

    def test_rotation_actions_are_geometrically_named(self):
        self.assertAlmostEqual(rotate_left_yaw(0.0), 45.0)
        self.assertAlmostEqual(rotate_right_yaw(0.0), -45.0)
        self.assertLess(yaw_to_forward_world(rotate_left_yaw(0.0))[0], 0.0)
        self.assertGreater(yaw_to_forward_world(rotate_right_yaw(0.0))[0], 0.0)

    def test_sensor_height_is_added_once_and_source_is_already_world_position(self):
        base = (1.0, -0.96887, 2.0)
        np.testing.assert_allclose(listener_sensor_position(base), (1.0, 0.53113, 2.0))
        source = SourceSpec((1.0, 0.53113, 2.0), "audio", 0.0, 5.0, 0.0)
        self.assertEqual(source.position_world, (1.0, 0.53113, 2.0))

    def test_yaw_normalization_is_half_open(self):
        self.assertAlmostEqual(normalize_yaw_deg(180.0), -180.0)
        self.assertAlmostEqual(normalize_yaw_deg(-180.0), -180.0)
        self.assertAlmostEqual(normalize_yaw_deg(540.0), -180.0)


if __name__ == "__main__":
    unittest.main()
