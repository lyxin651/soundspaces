import unittest
from pathlib import Path

from active_audition.config.loader import load_resolved_config
from active_audition.navigation.candidates import generate_candidates
from active_audition.navigation.pathfinder import PathFinderAdapter
from active_audition.scene.episode import fixed_golden_episode
from active_audition.scene.simulator import create_scene_simulator


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/active_audition/v0_replica_debug.yaml"


class M1GoldenPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_resolved_config(str(CONFIG))
        cls.context = create_scene_simulator(cls.config)
        cls.pathfinder = PathFinderAdapter(cls.context.pathfinder)

    @classmethod
    def tearDownClass(cls):
        cls.context.close()

    def test_golden_has_six_structurally_valid_candidates(self):
        episode = fixed_golden_episode(self.config, self.pathfinder)
        candidates = generate_candidates(
            episode.episode_id, episode.listener_initial, self.pathfinder, self.config
        )
        self.assertEqual(len(candidates), 6)
        self.assertTrue(all(candidate.valid for candidate in candidates))
        self.assertEqual(
            [candidate.candidate_id for candidate in candidates],
            [
                "trans_forward_r100",
                "trans_backward_r100",
                "trans_left_r100",
                "trans_right_r100",
                "rot_left_45",
                "rot_right_45",
            ],
        )

        translations = candidates[:4]
        self.assertTrue(all(candidate.yaw_deg == episode.listener_initial.yaw_deg for candidate in translations))
        self.assertTrue(all(candidate.path_points_world for candidate in translations))
        self.assertTrue(all(candidate.move_geodesic_m is not None for candidate in translations))
        self.assertTrue(all(candidate.requested_translation_m == 1.0 for candidate in translations))

        rotations = candidates[4:]
        for candidate in rotations:
            self.assertEqual(candidate.requested_base_position_world, episode.listener_initial.base_position_world)
            self.assertEqual(candidate.snapped_base_position_world, episode.listener_initial.base_position_world)
            self.assertEqual(candidate.sensor_position_world, episode.listener_initial.sensor_position_world)
            self.assertIsNone(candidate.path_points_world)
            self.assertIsNone(candidate.move_geodesic_m)

    def test_candidate_generation_does_not_depend_on_source(self):
        episode = fixed_golden_episode(self.config, self.pathfinder)
        first = generate_candidates(
            episode.episode_id, episode.listener_initial, self.pathfinder, self.config
        )
        altered_source = episode.source.position_world
        self.assertNotEqual(altered_source, episode.listener_initial.base_position_world)
        second = generate_candidates(
            episode.episode_id, episode.listener_initial, self.pathfinder, self.config
        )
        self.assertEqual(first, second)

    def test_rotation_world_geometry(self):
        episode = fixed_golden_episode(self.config, self.pathfinder)
        candidates = generate_candidates(
            episode.episode_id, episode.listener_initial, self.pathfinder, self.config
        )
        self.assertEqual(candidates[4].yaw_deg, 45.0)
        self.assertEqual(candidates[5].yaw_deg, -45.0)


if __name__ == "__main__":
    unittest.main()
