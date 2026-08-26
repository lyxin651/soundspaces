import unittest
from unittest.mock import Mock

from active_audition.config.loader import load_resolved_config
from active_audition.navigation.candidates import generate_candidates
from active_audition.navigation.pathfinder import PathResult
from active_audition.scene.episode import sampled_episode
from active_audition.scene.pose import listener_sensor_position
from active_audition.types import ListenerPose


LISTENER = ListenerPose((0.0, 0.0, 0.0), listener_sensor_position((0.0, 0.0, 0.0)), 0.0)


def config_for(**overrides):
    navigation = {
        "thresholds_enabled": True,
        "max_snap_error_m": 0.5,
        "min_actual_translation_m": 0.5,
        "max_actual_translation_m": 1.5,
        "max_geodesic_detour_ratio": 2.0,
        "duplicate_position_tolerance_m": 0.25,
    }
    navigation.update(overrides)
    return {
        "navigation": navigation,
        "candidate": {
            "translation": {"distance_m": 1.0, "directions": ["forward"]},
            "rotation": {"angle_deg": 45.0, "directions": []},
        },
    }


def pathfinder_for(snapped, geodesic=None):
    pathfinder = Mock()
    pathfinder.snap_point.return_value = snapped
    pathfinder.is_navigable.return_value = True
    pathfinder.shortest_path.return_value = PathResult(
        LISTENER.base_position_world,
        snapped,
        float(geodesic if geodesic is not None else 1.0),
        (LISTENER.base_position_world, snapped),
        True,
    )
    return pathfinder


class M4ThresholdTests(unittest.TestCase):
    def test_source_distance_boundaries_use_acoustic_positions_and_are_inclusive(self):
        config = load_resolved_config("configs/active_audition/v0_replica_m4_pilot.yaml")
        dry_audio = {"golden_probe_v0": {"duration_sec": 5.0}}

        class SamplingPathFinder:
            def __init__(self, distances):
                self.points = iter(point for distance in distances for point in ((0.0, 0.0, 0.0), (distance, 0.0, 0.0)))

            def sample_navigable_point(self, rng):
                return next(self.points)

            def is_navigable(self, point):
                return True

            def shortest_path(self, start, end):
                return PathResult(start, end, float(abs(end[0] - start[0])), (start, end), True)

            def geodesic_distance(self, start, end):
                return float(abs(end[0] - start[0]))

        def sample(distances):
            diagnostics = {}
            episode = sampled_episode(config, SamplingPathFinder(distances), "ep_boundary", dry_audio=dry_audio, scene_id="replica.office_0", sampling_diagnostics=diagnostics)
            return episode, diagnostics

        close_boundary, close_boundary_stats = sample([0.50])
        self.assertAlmostEqual(close_boundary.source_listener_euclidean_m, 0.50, places=7)
        self.assertEqual(close_boundary_stats.get("source_too_close_rejections", 0), 0)
        far_boundary, far_boundary_stats = sample([3.00])
        self.assertAlmostEqual(far_boundary.source_listener_euclidean_m, 3.00, places=7)
        self.assertEqual(far_boundary_stats.get("source_too_far_rejections", 0), 0)
        _, below_min_stats = sample([0.49, 0.50])
        self.assertEqual(below_min_stats["source_too_close_rejections"], 1)
        self.assertEqual(below_min_stats["total_sampling_attempts"], 2)
        _, above_max_stats = sample([3.01, 3.00])
        self.assertEqual(above_max_stats["source_too_far_rejections"], 1)
        self.assertEqual(above_max_stats["total_sampling_attempts"], 2)

    def test_source_distance_gate_resamples_deterministically(self):
        config = load_resolved_config("configs/active_audition/v0_replica_m4_pilot.yaml")
        dry_audio = {"golden_probe_v0": {"duration_sec": 5.0}}

        class SamplingPathFinder:
            def __init__(self, pairs):
                self.points = iter(point for pair in pairs for point in pair)

            def sample_navigable_point(self, rng):
                return next(self.points)

            def is_navigable(self, point):
                return True

            def shortest_path(self, start, end):
                return PathResult(start, end, 1.0, (start, end), True)

            def geodesic_distance(self, start, end):
                return 1.0

        pairs = [((0.0, 0.0, 0.0), (0.4, 0.0, 0.0)), ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0))]
        first_diagnostics = {}
        first = sampled_episode(config, SamplingPathFinder(pairs), "ep_000001", dry_audio=dry_audio, scene_id="replica.office_0", sampling_diagnostics=first_diagnostics)
        second_diagnostics = {}
        second = sampled_episode(config, SamplingPathFinder(pairs), "ep_000001", dry_audio=dry_audio, scene_id="replica.office_0", sampling_diagnostics=second_diagnostics)
        self.assertEqual(first, second)
        self.assertEqual(first_diagnostics["total_sampling_attempts"], 2)
        self.assertEqual(first_diagnostics["source_too_close_rejections"], 1)
        self.assertEqual(first_diagnostics["accepted_episode_count"], 1)

    def test_snap_boundary_is_inclusive(self):
        self.assertTrue(generate_candidates("ep", LISTENER, pathfinder_for((0.5, 0.0, -1.0)), config_for())[0].valid)
        candidate = generate_candidates("ep", LISTENER, pathfinder_for((0.51, 0.0, -1.0)), config_for())[0]
        self.assertEqual(candidate.invalid_reason, "snap_too_far")

    def test_actual_translation_boundaries_are_inclusive(self):
        config = config_for(max_snap_error_m=2.0)
        self.assertTrue(generate_candidates("ep", LISTENER, pathfinder_for((0.5, 0.0, 0.0)), config)[0].valid)
        small = generate_candidates("ep", LISTENER, pathfinder_for((0.49, 0.0, 0.0)), config)[0]
        self.assertEqual(small.invalid_reason, "actual_move_too_small")
        self.assertTrue(generate_candidates("ep", LISTENER, pathfinder_for((1.5, 0.0, 0.0)), config)[0].valid)
        large = generate_candidates("ep", LISTENER, pathfinder_for((1.51, 0.0, 0.0)), config)[0]
        self.assertEqual(large.invalid_reason, "actual_move_too_large")

    def test_detour_boundary_is_inclusive(self):
        self.assertTrue(generate_candidates("ep", LISTENER, pathfinder_for((0.0, 0.0, -1.0), 2.0), config_for())[0].valid)
        candidate = generate_candidates("ep", LISTENER, pathfinder_for((0.0, 0.0, -1.0), 2.01), config_for())[0]
        self.assertEqual(candidate.invalid_reason, "geodesic_detour_too_large")

    def test_duplicate_strict_boundary_and_lower_snap_error_priority(self):
        config = config_for(max_snap_error_m=2.0)
        config["candidate"]["translation"]["directions"] = ["forward", "backward"]
        pathfinder = Mock()
        pathfinder.snap_point.side_effect = [(0.0, 0.0, -1.0), (0.0, 0.0, -0.75)]
        pathfinder.is_navigable.return_value = True
        pathfinder.shortest_path.side_effect = [
            PathResult(LISTENER.base_position_world, (0.0, 0.0, -1.0), 1.0, ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0)), True),
            PathResult(LISTENER.base_position_world, (0.0, 0.0, -0.75), 0.75, ((0.0, 0.0, 0.0), (0.0, 0.0, -0.75)), True),
        ]
        candidates = generate_candidates("ep", LISTENER, pathfinder, config)
        self.assertTrue(candidates[0].valid)
        self.assertTrue(candidates[1].valid)  # exactly 0.25 m is not duplicate

        pathfinder.snap_point.side_effect = [(0.0, 0.0, -1.0), (0.0, 0.0, -0.9)]
        pathfinder.shortest_path.side_effect = [
            PathResult(LISTENER.base_position_world, (0.0, 0.0, -1.0), 1.0, (), True),
            PathResult(LISTENER.base_position_world, (0.0, 0.0, -0.9), 0.9, (), True),
        ]
        candidates = generate_candidates("ep", LISTENER, pathfinder, config)
        self.assertTrue(candidates[0].valid)
        self.assertEqual(candidates[1].invalid_reason, "duplicate_candidate")

    def test_rotation_ignores_translation_thresholds(self):
        pathfinder = Mock()
        config = config_for()
        config["candidate"]["translation"]["directions"] = []
        config["candidate"]["rotation"]["directions"] = ["left", "right"]
        candidates = generate_candidates("ep", LISTENER, pathfinder, config)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate.valid for candidate in candidates))
        self.assertTrue(all(candidate.move_euclidean_m == 0.0 for candidate in candidates))
        self.assertTrue(all(candidate.move_geodesic_m == 0.0 for candidate in candidates))
        pathfinder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
