import json
import tempfile
import unittest
from pathlib import Path

from active_audition.data.storage import DatasetStorage
from active_audition.evaluation.qc import QCError, _load_execution_evidence, _summary_markdown, _threshold_proposals
from active_audition.pipeline.v0 import _append_generation_event


class QCHardeningTests(unittest.TestCase):
    @staticmethod
    def _evidence(spot_count=3, expected=None):
        determinism = {
            "episodes_manifest_sha256": "episodes",
            "candidates_manifest_sha256": "candidates",
            "repeated_match": True,
            "acoustic_spot_checks": [
                {"episode_id": "ep_{:06d}".format(index), "shape": [4, 2], "max_abs_diff": 0.0, "exact_equal": True}
                for index in range(1, spot_count + 1)
            ],
        }
        if expected is not None:
            determinism["expected_spot_check_count"] = expected
        return {
            "determinism": determinism,
            "resume": {
                "rendered": 0,
                "skipped": 1,
                "viewpoint_count": 1,
                "manifest_unique": True,
                "wav_payload_hashes_unchanged": True,
                "rir_payload_hashes_unchanged": True,
                "viewpoints_manifest_hash_unchanged": True,
            },
            "runtime_provenance": {},
        }

    def test_threshold_report_keeps_percentile_as_evidence_not_proposal(self):
        distribution = {"count": 3, "min": 0.1, "p05": 0.2, "p25": 0.3, "median": 0.4, "p75": 0.5, "p95": 0.6, "max": 0.7}
        geometry = {
            "episode_source_listener_euclidean_m": distribution,
            "translation_snap_error_m": distribution,
            "translation_actual_euclidean_m": distribution,
            "translation_geodesic_detour_ratio": distribution,
            "translation_pairwise_distance_m": distribution,
        }
        acoustic = {"rir_tail_energy_ratio_100ms": distribution}
        result = _threshold_proposals(geometry, acoustic)
        for item in result.values():
            self.assertIsNone(item["proposed_value"])
            self.assertIn("observed_distribution", item)
            self.assertIn("risk_if_too_loose", item)
            self.assertEqual(item["status"], "PROVISIONAL / REQUIRES HUMAN REVIEW")

    def test_generation_events_are_append_only_and_distinguish_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(temp_dir)
            _append_generation_event(storage, {"mode": "render", "rendered": 7})
            _append_generation_event(storage, {"mode": "resume", "rendered": 0, "skipped": 7})
            rows = [json.loads(line) for line in storage.path("logs", "generation_events.jsonl").read_text().splitlines()]
            self.assertEqual([row["mode"] for row in rows], ["render", "resume"])
            self.assertEqual(rows[0]["rendered"], 7)
            self.assertEqual(rows[1]["skipped"], 7)

    def test_missing_evidence_is_unknown_and_incomplete_evidence_rejected(self):
        self.assertEqual(_load_execution_evidence(None)["status"], "UNKNOWN")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence.json"
            path.write_text(json.dumps({"determinism": {}}), encoding="utf-8")
            with self.assertRaises(QCError):
                _load_execution_evidence(str(path))

    def test_evidence_accepts_one_three_and_five_spot_checks(self):
        for count in (1, 3, 5):
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "evidence.json"
                path.write_text(json.dumps(self._evidence(count)), encoding="utf-8")
                self.assertEqual(_load_execution_evidence(str(path))["status"], "RECORDED_FROM_EXECUTION_EVIDENCE")

    def test_evidence_expected_count_and_values_are_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence.json"
            path.write_text(json.dumps(self._evidence(3, expected=5)), encoding="utf-8")
            with self.assertRaises(QCError):
                _load_execution_evidence(str(path))
            for field in ("episode_id", "shape", "max_abs_diff", "exact_equal"):
                evidence = self._evidence(1)
                del evidence["determinism"]["acoustic_spot_checks"][0][field]
                path.write_text(json.dumps(evidence), encoding="utf-8")
                with self.assertRaises(QCError):
                    _load_execution_evidence(str(path))
            for value in (float("nan"), -1.0):
                evidence = self._evidence(1)
                evidence["determinism"]["acoustic_spot_checks"][0]["max_abs_diff"] = value
                path.write_text(json.dumps(evidence), encoding="utf-8")
                with self.assertRaises(QCError):
                    _load_execution_evidence(str(path))

    def test_evidence_empty_list_and_bad_tolerance_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence.json"
            evidence = self._evidence(1)
            evidence["determinism"]["acoustic_spot_checks"] = []
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaises(QCError):
                _load_execution_evidence(str(path))
            for tolerance in (0.0, -1.0, float("nan")):
                evidence = self._evidence(1)
                evidence["determinism"]["acceptance_tolerance"] = tolerance
                path.write_text(json.dumps(evidence), encoding="utf-8")
                with self.assertRaises(QCError):
                    _load_execution_evidence(str(path))

    def test_threshold_summary_is_dynamic_and_distinguishes_pilot_validity(self):
        distribution = {"count": 1, "min": 0.1, "p05": 0.1, "p25": 0.1, "median": 0.1, "p75": 0.1, "p95": 0.1, "max": 0.1}
        geometry = {
            "translation_snap_error_m": distribution,
            "translation_actual_euclidean_m": distribution,
            "translation_geodesic_detour_ratio": distribution,
            "translation_pairwise_distance_m": distribution,
            "candidate_counts": {"valid": 1, "invalid": 0},
        }
        acoustic = {"rir_tail_energy_db_100ms": distribution}
        config = {"experiment": {"name": "v0_test"}, "storage": {"dataset_id": "test"}, "scene": {"ids": ["replica.office_test"]}, "navigation": {"thresholds_enabled": True}}
        pilot = {
            "candidate_counts": {"translation_valid": 1, "translation_attempted": 1, "translation_valid_rate": 1.0, "rotation_valid": 0, "rotation_attempted": 0, "rotation_valid_rate": 0.0, "invalid_reason_counts": {}},
            "episode_source_distance_gate": {"min_m": 0.5, "max_m": 3.0, "accepted_episode_count": 100, "sampling": {}},
            "per_direction": {key: {"valid_rate": 1.0} for key in ("forward", "backward", "left", "right")},
            "viewpoint_counts": {"expected": 1, "actual": 1, "render_failure_count": 0},
        }
        text = _summary_markdown([{"episode_id": "ep_000001"}] * 100, [], [{"viewpoint_id": "initial"}], geometry, acoustic, {"status": "UNKNOWN"}, 0, {}, pilot, config)
        self.assertIn("# Pipeline V0 Dataset QC", text)
        self.assertNotIn("M3", text)
        self.assertNotIn("20-Episode", text)
        self.assertIn("valid after structural and configured Pilot quality gates", text)
        self.assertIn("triangle inequality", text)


if __name__ == "__main__":
    unittest.main()
