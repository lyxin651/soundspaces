import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "docs/audits/clsdoa_v1/integration"


class IntegrationEvidenceHygieneTests(unittest.TestCase):
    def test_leakage_report_uses_detected_semantics(self):
        report = json.loads((INTEGRATION / "split_leakage_report.json").read_text())
        self.assertFalse(report["source_identity_leakage_detected"])
        self.assertFalse(report["scene_identity_leakage_detected"])
        self.assertTrue(all(value == 0 for value in report["source_split_overlap"].values()))
        self.assertEqual(report["scene_split_overlap"], 0)
        self.assertEqual(report["status"], "PASS")

    def test_authority_pointer_separates_final_and_superseded(self):
        pointer = json.loads((INTEGRATION / "step2b_evidence_authority.json").read_text())
        self.assertEqual(pointer["authoritative"]["final_commit"], "86d8b6f8c65220fa5801d81437009ed4382ab34d")
        authoritative = set(pointer["authoritative"]["acoustic_evidence"])
        superseded = pointer["superseded_or_historical"]["evidence"]
        self.assertIn("intermediate correct-receiver revalidation at 954deee8d3ce5cab070c436deea91118a416977d", superseded)
        self.assertTrue(all(item not in authoritative for item in superseded))

    def test_legacy_admission_warning_is_present_without_receiver_rewrite(self):
        source = (ROOT / "tools/clsdoa_v1/admit_scenes.py").read_text()
        self.assertIn("LEGACY / HISTORICAL STEP 2B ADMISSION TOOL", source)
        self.assertIn("not authorized for classdoa v1", source.lower())
        self.assertIn("spec.position = [0.0, 0.0, 0.0]", source)


if __name__ == "__main__":
    unittest.main()
