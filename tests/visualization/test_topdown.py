import unittest

import numpy as np

from active_audition.scene.pose import yaw_to_forward_world
from active_audition.visualization.topdown import occupancy_world_mapping, world_to_occupancy_pixel


class TopdownGeometryTests(unittest.TestCase):
    def test_occupancy_extent_comes_from_pathfinder_bounds(self):
        class FakePathFinder:
            def get_bounds(self):
                return (np.asarray((-1.0, -0.5, -2.0)), np.asarray((1.0, 1.0, 2.0)))

            def get_topdown_view(self, meters_per_pixel, height):
                self.args = (meters_per_pixel, height)
                return np.ones((40, 30), dtype=bool)

        pathfinder = FakePathFinder()
        occupancy, extent, origin = occupancy_world_mapping(pathfinder, 0.2, -0.5)
        self.assertEqual(occupancy.shape, (40, 30))
        self.assertEqual(extent, (-1.0, 5.0, -2.0, 6.0))
        self.assertEqual(origin, (-1.0, -2.0))
        self.assertEqual(pathfinder.args, (0.2, -0.5))

    def test_world_to_occupancy_pixel_uses_world_bounds_and_resolution(self):
        self.assertEqual(world_to_occupancy_pixel((1.24, 0.0, 2.36), (-1.0, -2.0), 0.2, (30, 20)), (21, 11))
        self.assertIsNone(world_to_occupancy_pixel((-1.1, 0.0, 2.36), (-1.0, -2.0), 0.2, (30, 20)))

    def test_frozen_yaw_forward_vectors_for_rotation_arrows(self):
        np.testing.assert_allclose(yaw_to_forward_world(0.0), (0.0, 0.0, -1.0), atol=1e-7)
        np.testing.assert_allclose(yaw_to_forward_world(45.0), (-2 ** -0.5, 0.0, -2 ** -0.5), atol=1e-7)
        np.testing.assert_allclose(yaw_to_forward_world(-45.0), (2 ** -0.5, 0.0, -2 ** -0.5), atol=1e-7)

    def test_rotation_endpoint_is_initial_base(self):
        initial = np.asarray((0.5, 0.0, 1.5))
        for yaw in (45.0, -45.0):
            endpoint = initial.copy()
            np.testing.assert_array_equal(endpoint, initial)


if __name__ == "__main__":
    unittest.main()
