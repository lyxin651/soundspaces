import unittest

from tools.clsdoa_v1.admit_scenes import (
    inclusive_distance,
    paired_geometry_equal,
    stable_seed,
    stable_split,
    validate_rir,
)
from tools.clsdoa_v1.split_admitted_scenes import split_results


class SceneAdmissionLogicTests(unittest.TestCase):
    def test_seed_and_split_are_stable(self):
        self.assertEqual(stable_seed("v1", "MP3D", "mp3d.a"), stable_seed("v1", "MP3D", "mp3d.a"))
        self.assertEqual(stable_split("Replica", "replica.office_0"), stable_split("Replica", "replica.office_0"))

    def test_distance_boundaries_are_inclusive(self):
        self.assertTrue(inclusive_distance(1.0, 1.0, 4.0))
        self.assertTrue(inclusive_distance(4.0, 1.0, 4.0))
        self.assertFalse(inclusive_distance(0.999, 1.0, 4.0))
        self.assertFalse(inclusive_distance(4.001, 1.0, 4.0))

    def test_rir_hard_validation(self):
        self.assertTrue(validate_rir([[1.0, 0.0], [0.0, 1.0]], 2)["nonzero"])
        self.assertFalse(validate_rir([[0.0], [0.0]], 2)["nonzero"])
        self.assertFalse(validate_rir([[float("nan")]], 2)["finite"])

    def test_paired_geometry_requires_all_pose_fields(self):
        row = {"listener_base_position_world": [0, 0, 0], "listener_sensor_position_world": [0, 1.5, 0], "listener_yaw_deg": 10, "source_position_world": [1, 1, 0]}
        self.assertTrue(paired_geometry_equal(row, dict(row)))
        altered = dict(row)
        altered["listener_yaw_deg"] = 11
        self.assertFalse(paired_geometry_equal(row, altered))

    def test_split_excludes_failures_and_keeps_unassigned(self):
        rows = [
            {"scene_id": "replica.a", "scene_family": "Replica", "admitted_status": "PASS"},
            {"scene_id": "mp3d.b", "scene_family": "MP3D", "admitted_status": "FAIL", "exclude_reason": "FAIL_FOA_RENDER"},
        ]
        result = split_results(rows, "v1")
        self.assertEqual(len(result["scenes"]), 1)
        self.assertEqual(result["excluded"][0].get("split"), None)


if __name__ == "__main__":
    unittest.main()
