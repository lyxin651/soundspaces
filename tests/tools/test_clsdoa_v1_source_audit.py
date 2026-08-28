import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.clsdoa_v1.audit_source_mappings import DATASETS, MAPPING_TYPES, audit_sources, write_outputs


REPO_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY = REPO_ROOT / "registries/ontology.yaml"


class ClassDOAV1SourceAuditTests(unittest.TestCase):
    def test_missing_resources_still_emit_exactly_twelve_by_three_rows(self):
        rows = audit_sources(ONTOLOGY)
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["source_dataset"] for row in rows}, set(DATASETS))
        self.assertEqual({row["canonical_class_id"] for row in rows}, set(range(12)))
        for class_id in range(12):
            self.assertEqual(sum(row["canonical_class_id"] == class_id for row in rows), 3)

    def test_mapping_enum_and_resource_blocker_invariants(self):
        rows = audit_sources(ONTOLOGY)
        self.assertTrue({row["mapping_type"] for row in rows}.issubset(set(MAPPING_TYPES)))
        for row in rows:
            if row["mapping_type"] != "EXACT":
                self.assertTrue(row["manual_review_required"])
            if row["mapping_type"] in ("NONE", "BLOCKED_RESOURCE"):
                self.assertFalse(row["exact_match"])
                self.assertFalse(row["semantic_match"])

    def test_output_is_deterministic_and_does_not_write_audio(self):
        evidence = {dataset: {"evidence": "fixture"} for dataset in DATASETS}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "audit"
            rows = audit_sources(ONTOLOGY, evidence=evidence)
            write_outputs(output, ONTOLOGY, rows, evidence)
            first = {path.name: path.read_bytes() for path in output.iterdir()}
            write_outputs(output, ONTOLOGY, rows, evidence)
            second = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(first, second)
            self.assertEqual(list(output.rglob("*.wav")), [])
            with (output / "source_class_mapping_audit.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 36)
            inventory = json.loads((output / "source_dataset_inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["canonical_class_count"], 12)


if __name__ == "__main__":
    unittest.main()
