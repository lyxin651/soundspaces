import json
import tempfile
import unittest
from pathlib import Path

from active_audition.data.storage import DatasetStorage
from active_audition.evaluation.qc import QCError, _load_execution_evidence, _threshold_proposals
from active_audition.pipeline.v0 import _append_generation_event


class QCHardeningTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
