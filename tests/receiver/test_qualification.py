import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from active_audition.receiver.qualification import (
    BLOCKED,
    NA,
    PASS,
    QUALIFICATION_CASE_SCHEMA_VERSION,
    QualificationError,
    canonical_json,
    compute_direct_metrics,
    load_metric_contract,
    make_probe,
    metric_contract_sha256,
    run_a2_qualification,
    run_resampler_band_calibration,
    run_synthetic_metric_fixtures,
    validate_a2_runtime_lock,
    validate_metric_contract,
    validate_qualification_case,
    validate_sample_rate_ab,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_CONTRACT_PATH = REPO_ROOT / "configs/active_audition/v1/metric_contract.yaml"
RUNTIME_CONFIG_PATH = REPO_ROOT / "configs/active_audition/v0_replica_debug.yaml"


class ActiveASRA2MetricTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_metric_contract(str(METRIC_CONTRACT_PATH))

    def test_metric_contract_is_strict_and_hash_is_order_invariant(self):
        reordered = copy.deepcopy(self.contract)
        reordered["metrics"] = {key: reordered["metrics"][key] for key in reversed(list(reordered["metrics"]))}
        reordered["contract"] = {key: reordered["contract"][key] for key in reversed(list(reordered["contract"]))}
        self.assertEqual(canonical_json(self.contract), canonical_json(reordered))
        self.assertEqual(metric_contract_sha256(self.contract), metric_contract_sha256(reordered))

        invalid = copy.deepcopy(self.contract)
        invalid["future_gate"] = True
        with self.assertRaisesRegex(QualificationError, "unknown metric contract field"):
            validate_metric_contract(invalid)

        invalid = copy.deepcopy(self.contract)
        invalid["metrics"]["itd"]["definition"] = "tL_minus_tR"
        with self.assertRaisesRegex(QualificationError, "metrics.itd.definition"):
            validate_metric_contract(invalid)

    def test_known_delay_gain_sign_units_and_direct_window(self):
        positive = self._fixture(delay=6, left_gain=1.0, right_gain=0.5)
        result = compute_direct_metrics(positive, 16000, self.contract)
        self.assertEqual(result["applicability"], "APPLICABLE")
        self.assertEqual(result["itd_samples"], 6)
        self.assertAlmostEqual(result["itd_seconds"], 6.0 / 16000.0)
        self.assertAlmostEqual(result["ild_db"], 10.0 * __import__("math").log10(4.0), places=6)
        self.assertEqual(result["direct_window"]["length_samples"], 166)
        self.assertIn("post_direct_tail_energy_db", result)
        self.assertNotIn("interaural_lag_samples", result)
        self.assertNotIn("rir_tail_energy_ratio_100ms", result)

        negative = compute_direct_metrics(self._fixture(delay=-6, left_gain=0.5, right_gain=1.0), 16000, self.contract)
        self.assertEqual(negative["itd_samples"], -6)
        self.assertAlmostEqual(negative["ild_db"], -10.0 * __import__("math").log10(4.0), places=6)

    def test_low_energy_is_na_and_probe_support_is_deterministic(self):
        silent = compute_direct_metrics([[0.0, 0.0]] * 256, 16000, self.contract)
        self.assertEqual(silent["applicability"], NA)
        self.assertEqual(silent["na_reason"], "LOW_ENERGY")
        for kind in ("impulse", "broadband_noise", "chirp", "voiced_tone"):
            first = make_probe(kind, 16000)
            second = make_probe(kind, 16000)
            self.assertEqual(first.dtype.name, "float32")
            self.assertEqual(first.tolist(), second.tolist())

    def test_synthetic_fixtures_pass_before_runtime(self):
        result = run_synthetic_metric_fixtures(self.contract)
        self.assertEqual(result["status"], PASS)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["resampling_convention"]["separate_normalization"], False)

    def test_geometry_registry_and_case_enumeration_are_strict_and_deterministic(self):
        from active_audition.receiver.geometry import (
            enumerate_controlled_cases,
            generate_geometry_asset,
            geometry_sanity,
            load_geometry_registry,
        )

        registry = load_geometry_registry(str(REPO_ROOT / "registries/active_asr_a2/qualification_geometry.yaml"), str(REPO_ROOT))
        self.assertEqual(len(registry["geometries"]), 2)
        self.assertTrue(all(geometry_sanity(entry)["all_sources_inside"] for entry in registry["geometries"]))
        self.assertTrue(all(geometry_sanity(entry)["all_direct_windows_clear"] for entry in registry["geometries"]))
        cases = enumerate_controlled_cases(registry)
        self.assertEqual(sum(case["line_of_sight"] == "LOS" for case in cases), 160)
        self.assertEqual(sum(case["line_of_sight"] == "NLOS" for case in cases), 10)
        self.assertEqual(sorted(set(case["relative_angle_deg"] for case in cases if case["line_of_sight"] == "LOS")), [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 180.0])
        with tempfile.TemporaryDirectory() as directory:
            first_mesh = Path(directory) / "first.ply"
            second_mesh = Path(directory) / "second.ply"
            generate_geometry_asset(first_mesh, "a2_symmetric_shoebox_v1")
            generate_geometry_asset(second_mesh, "a2_symmetric_shoebox_v1")
            self.assertEqual(first_mesh.read_bytes(), second_mesh.read_bytes())
            bad = Path(directory) / "bad.yaml"
            bad.write_text((REPO_ROOT / "registries/active_asr_a2/qualification_geometry.yaml").read_text(encoding="utf-8").replace("schema_version:", "future_field: true\nschema_version:"), encoding="utf-8")
            with self.assertRaises(Exception):
                load_geometry_registry(str(bad), str(REPO_ROOT))
            bad_hash = Path(directory) / "bad_hash.yaml"
            bad_hash.write_text((REPO_ROOT / "registries/active_asr_a2/qualification_geometry.yaml").read_text(encoding="utf-8").replace("c9bbf650ef35e1ac1d0c337f9b7550afeaab31ec77f9c9273f3a378a832e7e4e", "0" * 64), encoding="utf-8")
            with self.assertRaises(Exception):
                load_geometry_registry(str(bad_hash), str(REPO_ROOT))

    def test_geometry_reflection_timing_and_resampler_calibration(self):
        from active_audition.receiver.geometry import first_reflection_sanity, source_position_world

        source = source_position_world([0.0, 6.0, 0.0], 0.0, 90.0, 4.0)
        reflection = first_reflection_sanity({"x": 24.0, "y": 12.0, "z": 24.0}, [0.0, 6.0, 0.0], source)
        self.assertGreater(reflection["nearest_first_reflection_extra_time_ms"], 10.0)
        calibration = run_resampler_band_calibration(self.contract)
        self.assertIn(calibration["status"], (PASS, "FAIL"))
        self.assertEqual(calibration["separate_normalization"], False)

    def test_case_schema_keeps_na_separate_from_blocked(self):
        row = {
            "schema_version": QUALIFICATION_CASE_SCHEMA_VERSION,
            "case_id": "missing-geometry",
            "case_type": "controlled_geometry_requirement",
            "geometry_id": None,
            "geometry": {},
            "source_transform": {},
            "receiver_transform": {},
            "line_of_sight": "UNKNOWN",
            "relative_angle_deg": None,
            "distance_m": None,
            "probe": "impulse",
            "sample_rate_hz": 16000,
            "ray_preset": {},
            "ir_length_sec": None,
            "repeat_id": 0,
            "applicability": NA,
            "raw_metrics": None,
            "status": NA,
            "reason": "AUTHORITATIVE_CONTROLLED_QUALIFICATION_GEOMETRY_MISSING",
            "runtime_sha256": "0" * 64,
            "contract_sha256": metric_contract_sha256(self.contract),
            "resource_hashes": {},
        }
        self.assertIs(validate_qualification_case(row), row)
        blocked = dict(row, status=BLOCKED)
        self.assertIs(validate_qualification_case(blocked), blocked)

    def test_runtime_and_sample_rate_artifact_schemas_are_strict(self):
        lock = {
            "schema_version": "active-asr-a2-runtime-lock-v1",
            "gate": "A2",
            "status": BLOCKED,
            "contract_sha256": "0" * 64,
            "runtime_fingerprint": {},
            "receiver_effective": None,
            "resource_hashes": {},
            "geometry_registry_reason": "missing",
        }
        self.assertIs(validate_a2_runtime_lock(lock), lock)
        with self.assertRaises(QualificationError):
            validate_a2_runtime_lock(dict(lock, future_field=True))
        sample = {
            "schema_version": "active-asr-a2-sample-rate-ab-v1",
            "status": BLOCKED,
            "applicability": NA,
            "reason": "missing",
            "normalization": "none",
        }
        self.assertIs(validate_sample_rate_ab(sample), sample)
        with self.assertRaises(QualificationError):
            validate_sample_rate_ab(dict(sample, normalization="per_render"))

    def test_runtime_output_is_reproducible_with_valid_geometry_fixture(self):
        formal = {
            "status": BLOCKED,
            "blocker": "FIXTURE_GATE_BLOCKED",
            "cases": [],
            "errors": [],
            "geometry_sanity": {},
            "effective_baseline_ray_preset": {},
            "effective_baseline_ir_length_sec": None,
            "gates": {name: {"status": BLOCKED, "attempted": 0, "applicable": 0, "pass": 0, "fail": 0, "na": 0, "reason": "fixture"} for name in ("direction", "repeatability", "sample_rate_ab", "ray_tail_convergence", "mirror_yaw", "near_far_los_direct_energy")},
            "runtime_lock": {
                "schema_version": "active-asr-a2-runtime-lock-v1",
                "gate": "A2",
                "status": BLOCKED,
                "contract_sha256": metric_contract_sha256(self.contract),
                "runtime_fingerprint": {"git_commit": "fixture"},
                "receiver_effective": {},
                "resource_hashes": {},
                "geometry_registry_reason": "VALIDATED",
            },
        }
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            with patch("active_audition.receiver.qualification._run_controlled_qualification", return_value=formal):
                first = run_a2_qualification(str(METRIC_CONTRACT_PATH), first_dir, str(RUNTIME_CONFIG_PATH))
            with patch("active_audition.receiver.qualification._run_controlled_qualification", return_value=formal):
                second = run_a2_qualification(str(METRIC_CONTRACT_PATH), second_dir, str(RUNTIME_CONFIG_PATH))
            self.assertEqual(first["status"], BLOCKED)
            self.assertEqual(first["contract_sha256"], second["contract_sha256"])
            for name in ("runtime.lock.json", "qualification_cases.jsonl", "sample_rate_ab.json", "a2_summary.json", "a2_report.md"):
                self.assertEqual((Path(first_dir) / name).read_bytes(), (Path(second_dir) / name).read_bytes())
            summary = json.loads((Path(first_dir) / "a2_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], BLOCKED)
            self.assertEqual(summary["gates"]["direction"]["na"], 0)

    @staticmethod
    def _fixture(delay, left_gain, right_gain):
        import numpy as np

        result = np.zeros((2048, 2), dtype=np.float32)
        left = 400
        result[left, 0] = left_gain
        result[left + delay, 1] = right_gain
        result[max(left, left + delay) + 200, :] = [0.02, 0.01]
        return result


if __name__ == "__main__":
    unittest.main()
