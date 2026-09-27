import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.receiver.oracle_alignment import (
    OracleAlignmentError,
    _historical_gate_reference,
    _pose_waveform_record,
    evaluate_pose_gain_invariants,
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
        self.assertEqual(oracle_contract_sha256(self.contract), "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c")
        self.assertEqual(self.contract["contract"]["state"], "FROZEN")
        reordered = {key: self.contract[key] for key in reversed(list(self.contract))}
        self.assertEqual(canonical_oracle_json(self.contract), canonical_oracle_json(reordered))
        self.assertEqual(oracle_contract_sha256(self.contract), oracle_contract_sha256(reordered))
        self.assertEqual(metric_contract_sha256(load_metric_contract(str(V2_PATH))), "f1185d2c5fe81091199d5525f224822a3227c700c60bba9ab168e6d9ddc38f83")

        unknown = copy.deepcopy(self.contract)
        unknown["hard_gates"]["P8_native16_pose_gain_preservation"]["future_gate"] = True
        with self.assertRaisesRegex(OracleAlignmentError, "unknown oracle contract field"):
            validate_oracle_contract(unknown)

    def test_historical_frozen_evidence_hashes_and_missing_artifacts(self):
        historical = _historical_gate_reference(REPO_ROOT, "runs/active_asr_v1/a2_failure_attribution_run3", self.contract)
        self.assertEqual(historical["status"], "PASS")
        self.assertEqual(historical["P1_native_channel_mapping"]["status"], "PASS")
        self.assertEqual(historical["P1_native_channel_mapping"]["canonical_downstream_order"], ["L", "R"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = REPO_ROOT / "runs/active_asr_v1/a2_failure_attribution_run3"
            for name in ("direction_channel_calibration.json", "a2_summary.json", "qualification_cases.jsonl"):
                shutil.copyfile(source / name, root / name)
            contract = copy.deepcopy(self.contract)
            contract["historical_v2_provenance"]["historical_artifact_root"] = str(root)
            mismatch = json.loads((root / "a2_summary.json").read_text(encoding="utf-8"))
            mismatch["status"] = "PASS"
            (root / "a2_summary.json").write_text(json.dumps(mismatch), encoding="utf-8")
            failed = _historical_gate_reference(REPO_ROOT, str(root), contract)
            self.assertEqual(failed["status"], "BLOCKED")
            self.assertEqual(failed["reason"], "HISTORICAL_EVIDENCE_INTEGRITY_FAILURE")

            (root / "a2_summary.json").unlink()
            missing = _historical_gate_reference(REPO_ROOT, str(root), contract)
            self.assertEqual(missing["status"], "BLOCKED")
            self.assertEqual(missing["reason"], "HISTORICAL_EVIDENCE_MISSING")

    def test_p1_top_level_pass_with_invalid_h2_evidence_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = REPO_ROOT / "runs/active_asr_v1/a2_failure_attribution_run3"
            for name in ("direction_channel_calibration.json", "a2_summary.json", "qualification_cases.jsonl"):
                shutil.copyfile(source / name, root / name)
            direction = json.loads((root / "direction_channel_calibration.json").read_text(encoding="utf-8"))
            direction["hypothesis_summary"]["H2_native_ch0_R_ch1_L"]["pass"] = 3
            direction["hypothesis_summary"]["H2_native_ch0_R_ch1_L"]["sign_accuracy"] = 0.75
            direction["hypothesis_summary"]["supported_hypothesis"] = "H1_native_ch0_L_ch1_R"
            (root / "direction_channel_calibration.json").write_text(json.dumps(direction), encoding="utf-8")
            contract = copy.deepcopy(self.contract)
            contract["historical_v2_provenance"]["historical_artifact_root"] = str(root)
            contract["historical_v2_provenance"]["historical_artifact_sha256"]["direction_channel_calibration.json"] = __import__("hashlib").sha256((root / "direction_channel_calibration.json").read_bytes()).hexdigest()
            failed = _historical_gate_reference(REPO_ROOT, str(root), contract)
            self.assertEqual(failed["status"], "BLOCKED")
            self.assertEqual(failed["reason"], "P1_HISTORICAL_EVIDENCE_INVALID")
            self.assertEqual(failed["P1_native_channel_mapping"]["status"], "BLOCKED")

    def test_p8_equalized_pose_fixture_fails_even_with_perfect_gain_linearity(self):
        rows = []
        for pose_id in ("center", "closer", "farther"):
            rows.append({
                "pose_id": pose_id,
                "gain_linearity_control": {"max_abs_waveform_error": 0.0},
                "waveforms": {
                    "0.25": {"output": {"mean_lr": {"sum_square": 1.0}}},
                    "0.5": {"output": {"mean_lr": {"sum_square": 4.0}}},
                },
            })
        result = evaluate_pose_gain_invariants(rows, self.contract)
        self.assertTrue(result["global_gain_linearity_max_abs_ratio_error"] == 0.0)
        self.assertFalse(result["pose_dependent_energy_variation_observed"])
        self.assertFalse(result["passed"])

    def test_p8_real_pose_variation_and_linearity_pass(self):
        rows = []
        for pose_id, energy in (("center", 1.0), ("closer", 2.0), ("farther", 0.5)):
            rows.append({
                "pose_id": pose_id,
                "gain_linearity_control": {"max_abs_waveform_error": 0.0},
                "waveforms": {
                    "0.25": {"output": {"mean_lr": {"sum_square": energy}}},
                    "0.5": {"output": {"mean_lr": {"sum_square": energy * 4.0}}},
                },
            })
        result = evaluate_pose_gain_invariants(rows, self.contract)
        self.assertTrue(result["pose_dependent_energy_variation_observed"])
        self.assertTrue(result["passed"])

    def test_p9_clipping_threshold_is_read_from_contract(self):
        dry = np.asarray([1.0], dtype=np.float32)
        rir = np.asarray([[0.75, 0.75]], dtype=np.float32)
        contract = copy.deepcopy(self.contract)
        contract["hard_gates"]["P9_native16_waveform_validity"]["max_abs_waveform_before_clipping"] = 0.5
        record = _pose_waveform_record(dry, rir, 1.0, contract, "runtime", {})
        self.assertEqual(record["hard_clipping_smoke"]["threshold_abs"], 0.5)
        self.assertFalse(record["hard_clipping_smoke"]["passed"])

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
