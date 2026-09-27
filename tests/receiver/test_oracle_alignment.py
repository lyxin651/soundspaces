import copy
import unittest
from pathlib import Path

import numpy as np

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.receiver.oracle_alignment import (
    OracleAlignmentError,
    canonical_oracle_json,
    enumerate_fixed_world_yaw_cases,
    load_oracle_contract,
    oracle_contract_sha256,
    validate_oracle_contract,
)
from active_audition.receiver.qualification import load_metric_contract, metric_contract_sha256
from active_audition.scene.pose import relative_azimuth_deg as scene_relative_azimuth


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_PATH = REPO_ROOT / "configs/active_audition/v1/metric_contract.yaml"
V3_PATH = REPO_ROOT / "configs/active_audition/v1/metric_contract_oracle_v3.yaml"


class ActiveASRA2OracleAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_oracle_contract(str(V3_PATH))

    def test_v3_is_strict_canonical_and_keeps_v2_parent_immutable(self):
        self.assertEqual(oracle_contract_sha256(self.contract), "d423cdd97225d5f98261c1804f771164afa55fdd4085f8a3d6eeac5dd014aa65")
        reordered = {key: self.contract[key] for key in reversed(list(self.contract))}
        self.assertEqual(canonical_oracle_json(self.contract), canonical_oracle_json(reordered))
        self.assertEqual(oracle_contract_sha256(self.contract), oracle_contract_sha256(reordered))
        self.assertEqual(metric_contract_sha256(load_metric_contract(str(V2_PATH))), "f1185d2c5fe81091199d5525f224822a3227c700c60bba9ab168e6d9ddc38f83")

        unknown = copy.deepcopy(self.contract)
        unknown["hard_gates"]["P8_native16_pose_gain_preservation"]["future_gate"] = True
        with self.assertRaisesRegex(OracleAlignmentError, "unknown oracle contract field"):
            validate_oracle_contract(unknown)

    def test_fixed_world_source_yaw_cases_are_relative_and_deterministic(self):
        receiver = (0.0, 6.0, 0.0)
        source = (0.0, 6.0, -4.0)
        yaws = [-90, -60, -30, 0, 30, 60, 90, 180]
        first = enumerate_fixed_world_yaw_cases(receiver, source, yaws)
        second = enumerate_fixed_world_yaw_cases(receiver, source, yaws)
        self.assertEqual(first, second)
        self.assertTrue(all(case["source_position_world"] == list(source) for case in first))
        for case in first:
            expected = scene_relative_azimuth(source, receiver, case["listener_yaw_deg"])
            self.assertAlmostEqual(case["geometry_relative_azimuth_deg"], expected, places=9)
        by_yaw = {case["listener_yaw_deg"]: case["geometry_relative_azimuth_deg"] for case in first}
        self.assertAlmostEqual(by_yaw[0], 0.0)
        self.assertAlmostEqual(abs(by_yaw[180]), 180.0)
        self.assertAlmostEqual(by_yaw[90], -90.0)
        self.assertAlmostEqual(by_yaw[-90], 90.0)

    def test_mean_lr_formula_and_global_gain_control_are_linear_without_normalization(self):
        dry = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
        rir = np.asarray([[1.0, 0.5], [0.0, 1.0]], dtype=np.float32)
        low = convolve_binaural(dry * 0.25, rir)
        high = convolve_binaural(dry * 0.50, rir)
        np.testing.assert_array_equal(high, low * 2.0)
        mean_lr = (low[:, 0] + low[:, 1]) * 0.5
        swapped_mean = (low[:, 1] + low[:, 0]) * 0.5
        np.testing.assert_array_equal(mean_lr, swapped_mean)
        self.assertEqual(low.shape, (dry.shape[0] + rir.shape[0] - 1, 2))
        self.assertTrue(np.isfinite(low).all())


if __name__ == "__main__":
    unittest.main()
