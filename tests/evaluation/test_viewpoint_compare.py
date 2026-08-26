import unittest

from active_audition.evaluation.viewpoint_compare import build_viewpoint_comparisons


class ViewpointCompareTests(unittest.TestCase):
    def test_delta_direction_rotation_and_source_distance(self):
        episodes = [{"episode_id": "ep_000001", "source": {"position_world": [0.0, 1.5, -2.0]}}]
        candidates = [{"episode_id": "ep_000001", "candidate_id": "rot_left_45", "action_type": "rotation", "valid": True, "move_euclidean_m": 0.0, "move_geodesic_m": 0.0}]
        viewpoints = [
            {"episode_id": "ep_000001", "viewpoint_id": "initial", "candidate_id": None, "sensor_position_world": [0.0, 1.5, 0.0], "yaw_deg": 0.0},
            {"episode_id": "ep_000001", "viewpoint_id": "rot_left_45", "candidate_id": "rot_left_45", "sensor_position_world": [0.0, 1.5, 0.0], "yaw_deg": 45.0},
        ]
        metrics = {
            ("ep_000001", "initial"): {"rms_mean_dbfs": -10.0, "ild_db": 1.0, "peak_max": 0.2, "interaural_correlation": 0.8, "interaural_lag_samples": 0},
            ("ep_000001", "rot_left_45"): {"rms_mean_dbfs": -8.0, "ild_db": -1.0, "peak_max": 0.3, "interaural_correlation": 0.7, "interaural_lag_samples": 2},
        }
        rows = build_viewpoint_comparisons(episodes, candidates, viewpoints, metrics)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delta_rms_mean_dbfs"], 2.0)
        self.assertEqual(rows[0]["delta_ild_db"], -2.0)
        self.assertEqual(rows[0]["movement_euclidean_m"], 0.0)
        self.assertAlmostEqual(rows[0]["delta_source_distance_m"], 0.0)

    def test_invalid_candidate_without_viewpoint_is_excluded(self):
        episodes = [{"episode_id": "ep_000001", "source": {"position_world": [0.0, 1.5, -2.0]}}]
        candidates = [{"episode_id": "ep_000001", "candidate_id": "trans_forward_r100", "action_type": "translation", "valid": False, "move_euclidean_m": 0.0, "move_geodesic_m": None}]
        viewpoints = [{"episode_id": "ep_000001", "viewpoint_id": "initial", "candidate_id": None, "sensor_position_world": [0.0, 1.5, 0.0], "yaw_deg": 0.0}]
        metrics = {("ep_000001", "initial"): {"rms_mean_dbfs": -10.0, "ild_db": 1.0, "peak_max": 0.2, "interaural_correlation": 0.8, "interaural_lag_samples": 0}}
        self.assertEqual(build_viewpoint_comparisons(episodes, candidates, viewpoints, metrics), [])


if __name__ == "__main__":
    unittest.main()
