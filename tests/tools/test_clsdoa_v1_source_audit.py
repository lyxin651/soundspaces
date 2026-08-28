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
            self.assertEqual(row["mapping_status"], row["mapping_type"])
            if row["mapping_type"] != "EXACT":
                self.assertTrue(row["manual_review_required"])
            if row["mapping_type"] in ("NONE", "BLOCKED_RESOURCE"):
                self.assertFalse(row["exact_match"])
                self.assertFalse(row["semantic_match"])

    def test_desed_mapping_status_is_separate_from_missing_audio_resource(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = Path(temp_dir) / "event_occurrences.json"
            metadata.write_text(json.dumps([
                "Alarm_bell_ringing", "Dishes", "Frying", "Running_water", "Speech", "Vacuum_cleaner",
            ]), encoding="utf-8")
            rows = audit_sources(ONTOLOGY, desed_metadata=metadata)
        by_class = {
            row["canonical_class"]: row for row in rows
            if row["source_dataset"] == "DESED isolated foreground"
        }
        for name in ("speech", "running_water", "dishes", "frying", "vacuum_cleaner"):
            self.assertEqual(by_class[name]["mapping_status"], "EXACT")
            self.assertEqual(by_class[name]["resource_status"], "MISSING")
            self.assertIsNone(by_class[name]["candidate_count"])
        self.assertEqual(by_class["clock_alarm"]["mapping_status"], "SEMANTIC_STRONG")
        self.assertTrue(by_class["clock_alarm"]["manual_review_required"])
        self.assertEqual(by_class["clock_alarm"]["resource_status"], "MISSING")
        self.assertNotIn("BLOCKED_RESOURCE", {row["mapping_status"] for row in by_class.values()})

    def test_pseld_crosswalk_uses_official_label_mid_and_unique_clip_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "cls_indices_train.tsv"
            test = root / "cls_indices_test.tsv"
            train.write_text("24\t/m/01m4t\tPrinter\t194\t1259.5\n", encoding="utf-8")
            test.write_text("24\t/m/01m4t\tPrinter\t22\t174.0\n", encoding="utf-8")
            source = root / "_m_01m4t.tsv"
            source.write_text(
                "123\t1.0\t/m/01m4t,/t/dd00077\ttrain\n"
                "123\t1.0\t/m/01m4t,/t/dd00077\ttrain\n"
                "456\t2.0\t/m/01m4t,/t/dd00077\ttest\n",
                encoding="utf-8",
            )
            rows = audit_sources(
                ONTOLOGY,
                pseld_train_index=train,
                pseld_test_index=test,
                pseld_source_tsv_dir=root,
            )
        printer = next(row for row in rows if row["canonical_class"] == "printer" and row["source_dataset"] == "PSELD-selected FSD50K")
        vacuum = next(row for row in rows if row["canonical_class"] == "vacuum_cleaner" and row["source_dataset"] == "PSELD-selected FSD50K")
        self.assertEqual(printer["mapping_status"], "EXACT")
        self.assertEqual(printer["source_label_id"], "/m/01m4t")
        self.assertEqual(printer["candidate_count"], 3)
        self.assertEqual(printer["independent_identity_count"], 2)
        self.assertEqual(printer["resource_status"], "PRESENT_METADATA_ONLY")
        self.assertEqual(vacuum["mapping_status"], "NONE")
        self.assertEqual(vacuum["candidate_count"], 0)

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
