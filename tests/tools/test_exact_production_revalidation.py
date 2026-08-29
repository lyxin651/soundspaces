import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.clsdoa_v1.validate_exact_production_revalidation import EvidenceValidationError, validate_rows

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/audits/clsdoa_v1/scene_admission/exact_production_acoustic_revalidation.json"
REGISTRY = ROOT / "registries/clsdoa_v1_scenes.yaml"


class ExactProductionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads(EVIDENCE.read_text())
        cls.registry = yaml.safe_load(REGISTRY.read_text())

    def assert_invalid(self, mutate, expected_message):
        evidence = copy.deepcopy(self.evidence)
        mutate(evidence)
        with self.assertRaisesRegex(EvidenceValidationError, expected_message):
            validate_rows(evidence["rows"], self.registry, evidence["historical_acoustic_evidence"])

    def test_authoritative_evidence_passes(self):
        self.assertEqual(validate_rows(self.evidence["rows"], self.registry, self.evidence["historical_acoustic_evidence"])["status"], "PASS")

    def test_contract_failure_cases(self):
        cases = [
            (lambda e: e["rows"].append(copy.deepcopy(e["rows"][0])), "row/probe count mismatch"),
            (lambda e: e["rows"][0].__setitem__("sample_rate_hz", 16000), "wrong sample rate"),
            (lambda e: e["rows"][0].__setitem__("materials_enabled", True), "Materials enabled"),
            (lambda e: e["rows"][0].__setitem__("indirect_ray_count", 1), "ray config mismatch"),
            (lambda e: e["rows"][0].__setitem__("normalization_applied", True), "normalization enabled"),
            (lambda e: e["rows"][0].__setitem__("receiver_error_m", 1e-3), "receiver error above tolerance"),
            (lambda e: e["rows"][0].__setitem__("foa_receiver_error_m", 1e-3), "FOA receiver error above tolerance"),
            (lambda e: e["rows"][0]["binaural"].__setitem__("channel_count", 1), "wrong Binaural channels"),
            (lambda e: e["rows"][0]["binaural"].__setitem__("finite", False), "Binaural finite/nonzero"),
            (lambda e: e["rows"][0]["foa"].__setitem__("channel_count", 2), "wrong FOA channels"),
            (lambda e: e["rows"][0]["foa"].__setitem__("nonzero", False), "FOA finite/nonzero"),
            (lambda e: e.__setitem__("historical_acoustic_evidence", "WRONG"), "historical superseded marker"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                self.assert_invalid(mutate, message)

    def test_duplicate_key_and_missing_probe_are_rejected(self):
        self.assert_invalid(lambda e: e["rows"][1].__setitem__("probe_id", e["rows"][0]["probe_id"]), "duplicate scene\\+probe key")
        self.assert_invalid(lambda e: e["rows"][1].__setitem__("probe_id", "probe_2"), "missing probe_0/probe_1 coverage")

    def test_python_optimized_mode_rejects_bad_evidence(self):
        bad = copy.deepcopy(self.evidence)
        bad["rows"][0]["sample_rate_hz"] = 16000
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "evidence.json"
            evidence.write_text(json.dumps(bad))
            result = subprocess.run([sys.executable, "-O", str(ROOT / "tools/clsdoa_v1/validate_exact_production_revalidation.py"), "--evidence", str(evidence), "--registry", str(REGISTRY)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wrong sample rate", result.stderr)


if __name__ == "__main__":
    unittest.main()
