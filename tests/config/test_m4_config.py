import unittest
from pathlib import Path

from active_audition.config.loader import load_resolved_config
from active_audition.pipeline.v0 import SUCCESS_TEXT


REPO_ROOT = Path(__file__).resolve().parents[2]


class M4ConfigTests(unittest.TestCase):
    def test_m4_config_enables_only_frozen_pilot_thresholds(self):
        config = load_resolved_config(str(REPO_ROOT / "configs/active_audition/v0_replica_m4_pilot.yaml"))
        self.assertEqual(config["episode"]["count"], 100)
        self.assertEqual(config["episode"]["source_listener_min_distance_m"], 0.5)
        self.assertEqual(config["episode"]["source_listener_max_distance_m"], 3.0)
        self.assertTrue(config["navigation"]["thresholds_enabled"])
        self.assertEqual(config["navigation"]["duplicate_position_tolerance_m"], 0.25)

    def test_existing_m3_config_keeps_thresholds_disabled(self):
        config = load_resolved_config(str(REPO_ROOT / "configs/active_audition/v0_replica_m3_diag_002.yaml"))
        self.assertFalse(config["navigation"]["thresholds_enabled"])
        self.assertIsNone(config["navigation"]["max_snap_error_m"])
        self.assertIsNone(config["episode"]["source_listener_min_distance_m"])

    def test_success_marker_is_milestone_agnostic(self):
        self.assertEqual(SUCCESS_TEXT, "Pipeline V0 dataset finalized\n")
        self.assertNotRegex(SUCCESS_TEXT, r"M[234]")


if __name__ == "__main__":
    unittest.main()
