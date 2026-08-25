import unittest
from unittest.mock import Mock
from unittest.mock import patch

from active_audition.navigation.candidates import generate_candidates
from active_audition.navigation.pathfinder import PathFinderAdapter, PathResult
from active_audition.types import ListenerPose


LISTENER = ListenerPose((0.0, 0.0, 0.0), (0.0, 1.5, 0.0), 0.0)
CONFIG = {
    "navigation": {"thresholds_enabled": False},
    "candidate": {
        "translation": {"distance_m": 1.0, "directions": ["forward"]},
        "rotation": {"angle_deg": 45.0, "directions": []},
    },
}


def candidate_for(fake_pathfinder):
    return generate_candidates("ep_000001", LISTENER, fake_pathfinder, CONFIG)[0]


class CandidateReasonTests(unittest.TestCase):
    def test_frozen_translation_invalid_reasons(self):
        snap_failure = Mock()
        snap_failure.snap_point.return_value = None
        self.assertEqual(candidate_for(snap_failure).invalid_reason, "snap_failed")

        outside_navmesh = Mock()
        outside_navmesh.snap_point.return_value = (0.0, 0.0, -1.0)
        outside_navmesh.is_navigable.return_value = False
        self.assertEqual(candidate_for(outside_navmesh).invalid_reason, "outside_navmesh")

        too_small = Mock()
        too_small.snap_point.return_value = (0.0, 0.0, 0.0)
        too_small.is_navigable.return_value = True
        self.assertEqual(candidate_for(too_small).invalid_reason, "actual_move_too_small")

        no_path = Mock()
        no_path.snap_point.return_value = (0.0, 0.0, -1.0)
        no_path.is_navigable.return_value = True
        no_path.shortest_path.return_value = PathResult(
            LISTENER.base_position_world,
            (0.0, 0.0, -1.0),
            None,
            None,
            False,
        )
        self.assertEqual(candidate_for(no_path).invalid_reason, "no_path")

    def test_rotation_does_not_call_pathfinder(self):
        pathfinder = Mock()
        config = {
            "navigation": {"thresholds_enabled": False},
            "candidate": {
                "translation": {"distance_m": 1.0, "directions": []},
                "rotation": {"angle_deg": 45.0, "directions": ["left", "right"]},
            },
        }
        candidates = generate_candidates("ep_000001", LISTENER, pathfinder, config)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate.has_path for candidate in candidates))
        self.assertTrue(all(candidate.path_points_world is None for candidate in candidates))
        pathfinder.assert_not_called()


class PathFinderReachabilityTests(unittest.TestCase):
    def setUp(self):
        self.adapter = PathFinderAdapter(Mock())

    def test_geodesic_returns_finite_distance_for_found_path(self):
        result = PathResult((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.25, ((0.0, 0.0, 0.0),), True)
        with patch.object(self.adapter, "shortest_path", return_value=result):
            self.assertEqual(self.adapter.geodesic_distance((0, 0, 0), (1, 0, 0)), 1.25)

    def test_geodesic_returns_none_for_unreachable_path(self):
        result = PathResult((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), None, None, False)
        with patch.object(self.adapter, "shortest_path", return_value=result):
            self.assertIsNone(self.adapter.geodesic_distance((0, 0, 0), (1, 0, 0)))

    def test_geodesic_returns_none_for_not_found_finite_distance(self):
        result = PathResult((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.25, None, False)
        with patch.object(self.adapter, "shortest_path", return_value=result):
            self.assertIsNone(self.adapter.geodesic_distance((0, 0, 0), (1, 0, 0)))


if __name__ == "__main__":
    unittest.main()
