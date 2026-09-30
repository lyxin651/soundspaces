import json
import tempfile
import unittest
from pathlib import Path

from active_audition.asr.historical_materials_audit import _audit_material_config


class HistoricalMaterialsAuditTests(unittest.TestCase):
    def test_label_mapping_is_bound_to_semantic_classes_without_inventing_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            config = root_path / "replica_material_config.json"
            semantic = root_path / "info_semantic.json"
            config.write_text(json.dumps({
                "materials": [
                    {
                        "name": "Default",
                        "absorption": [],
                        "scattering": [],
                        "transmission": [],
                        "labels": ["wall"],
                        "damping": [],
                    }
                ]
            }), encoding="utf-8")
            semantic.write_text(json.dumps({"classes": [{"name": "wall"}]}), encoding="utf-8")
            result = _audit_material_config(root_path, config, [semantic])
            self.assertEqual(result["json_parse"], "PASS")
            self.assertEqual(result["schema"]["unknown_top_level_keys"], [])
            self.assertEqual(result["mapping"]["unmatched_semantic_labels"][0]["labels"], [])
            self.assertFalse(result["mapping"]["explicit_fallback_field"])

    def test_missing_semantic_label_is_not_silently_defaulted(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            config = root_path / "replica_material_config.json"
            semantic = root_path / "info_semantic.json"
            config.write_text(json.dumps({
                "materials": [{
                    "name": "Default",
                    "absorption": [],
                    "scattering": [],
                    "transmission": [],
                    "labels": ["wall"],
                    "damping": [],
                }]
            }), encoding="utf-8")
            semantic.write_text(json.dumps({"classes": [{"name": "chair"}]}), encoding="utf-8")
            result = _audit_material_config(root_path, config, [semantic])
            self.assertEqual(result["mapping"]["unmatched_semantic_labels"][0]["labels"], ["chair"])


if __name__ == "__main__":
    unittest.main()
