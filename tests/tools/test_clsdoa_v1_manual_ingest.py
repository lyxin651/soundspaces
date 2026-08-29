import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.clsdoa_v1.ingest_manual_qc import ingest
from tools.clsdoa_v1.ingest_incremental_manual_qc import ingest_incremental


class ManualQCIngestTests(unittest.TestCase):
    def test_fills_only_blank_target_and_emits_minimal_incremental_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue.csv"
            fields = [
                "review_order", "canonical_class", "source_dataset", "source_label",
                "original_id", "base_clip_id", "identity_key", "audio_path",
                "duration_sec", "active_duration_estimate_sec", "auto_qc_status",
                "auto_qc_reasons", "mapping_type", "license_status",
                "pool_candidate_role", "preview_start_sec", "preview_end_sec",
                "preview_audio_path", "manual_decision", "manual_reason",
                "manual_notes", "manual_crop_override_candidate",
            ]
            rows = []
            for order, role, decision, original in (
                ("1", "TARGET", "", "target_blank"),
                ("2", "TARGET", "REJECT", "target_reject"),
                ("3", "RESERVE", "", "reserve_blank_a"),
                ("4", "RESERVE", "", "reserve_blank_b"),
                ("5", "RESERVE", "REJECT", "reserve_reject"),
            ):
                rows.append({
                    "review_order": order, "canonical_class": "dishes",
                    "source_dataset": "DESED isolated foreground",
                    "source_label": "Dishes", "original_id": original,
                    "base_clip_id": "desed:train:" + original,
                    "identity_key": "DESED:" + original,
                    "audio_path": str(root / (original + ".wav")),
                    "duration_sec": "1", "active_duration_estimate_sec": "1",
                    "auto_qc_status": "AUTO_PASS", "auto_qc_reasons": "",
                    "mapping_type": "EXACT", "license_status": "PER_RECORDING_METADATA",
                    "pool_candidate_role": role, "preview_start_sec": "0",
                    "preview_end_sec": "1", "preview_audio_path": str(root / (original + ".wav")),
                    "manual_decision": decision, "manual_reason": "",
                    "manual_notes": "", "manual_crop_override_candidate": "false",
                })
            with queue.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

            qc = root / "qc.csv"
            qc_fields = [
                "source_dataset", "canonical_class", "source_label", "original_id",
                "base_clip_id", "source_clip_id", "raw_relpath", "resource_status",
                "mapping_type", "auto_qc_status", "auto_qc_reason", "license_status",
                "license_raw", "provenance_source", "pretrain_seen_status", "raw_sha256",
                "active_duration_estimate_sec",
            ]
            with qc.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=qc_fields, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "source_dataset": row["source_dataset"],
                        "canonical_class": row["canonical_class"],
                        "source_label": row["source_label"],
                        "original_id": row["original_id"],
                        "base_clip_id": row["base_clip_id"],
                        "source_clip_id": row["original_id"],
                        "raw_relpath": Path(row["audio_path"]).name,
                        "resource_status": "PRESENT", "mapping_type": "EXACT",
                        "auto_qc_status": "AUTO_PASS", "auto_qc_reason": "",
                        "license_status": "PER_RECORDING_METADATA", "license_raw": "CC0",
                        "provenance_source": "test.tsv", "pretrain_seen_status": "unknown",
                        "raw_sha256": row["original_id"],
                        "active_duration_estimate_sec": "1",
                    })
            snapshot = root / "snapshot.csv"
            stats_path = root / "stats.json"
            membership = root / "membership.csv"
            incremental = root / "incremental.csv"
            stats = ingest(
                queue, [qc], root, snapshot, stats_path, membership, incremental,
                target_minimum=2,
            )
            with queue.open(newline="", encoding="utf-8") as handle:
                after = list(csv.DictReader(handle))
            self.assertEqual(after[0]["manual_decision"], "ACCEPT")
            self.assertEqual(after[1]["manual_decision"], "REJECT")
            self.assertEqual(after[2]["manual_decision"], "")
            self.assertEqual(after[4]["manual_decision"], "REJECT")
            self.assertEqual(stats["target_accept"], 1)
            self.assertEqual(stats["target_reject"], 1)
            self.assertEqual(stats["incremental_manual_review"]["rows"], 1)
            self.assertEqual(json.loads(stats_path.read_text())["status"],
                             "STEP 2A INCREMENTAL MANUAL QC REQUIRED")
            with incremental.open(newline="", encoding="utf-8") as handle:
                incremental_rows = list(csv.DictReader(handle))
            self.assertEqual(incremental_rows[0]["manual_decision"], "")
            self.assertEqual(incremental_rows[0]["incremental_review_required"], "true")

    def test_ingests_authorized_incremental_accept_without_touching_other_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue.csv"
            fields = [
                "review_order", "canonical_class", "source_dataset", "source_label",
                "original_id", "base_clip_id", "identity_key", "audio_path",
                "duration_sec", "active_duration_estimate_sec", "auto_qc_status",
                "auto_qc_reasons", "mapping_type", "license_status",
                "pool_candidate_role", "preview_start_sec", "preview_end_sec",
                "preview_audio_path", "manual_decision", "manual_reason",
                "manual_notes", "manual_crop_override_candidate",
            ]
            def queue_row(order, original, role, decision=""):
                return {
                    "review_order": str(order), "canonical_class": "dishes",
                    "source_dataset": "DESED isolated foreground", "source_label": "Dishes",
                    "original_id": original, "base_clip_id": "desed:train:" + original,
                    "identity_key": "DESED:" + original, "audio_path": str(root / (original + ".wav")),
                    "duration_sec": "1", "active_duration_estimate_sec": "1",
                    "auto_qc_status": "AUTO_PASS", "auto_qc_reasons": "",
                    "mapping_type": "EXACT", "license_status": "PER_RECORDING_METADATA",
                    "pool_candidate_role": role, "preview_start_sec": "0", "preview_end_sec": "1",
                    "preview_audio_path": str(root / (original + ".wav")),
                    "manual_decision": decision, "manual_reason": "", "manual_notes": "",
                    "manual_crop_override_candidate": "false",
                }
            rows = [queue_row(1, "target", "TARGET", "ACCEPT"),
                    queue_row(2, "incremental", "RESERVE"),
                    queue_row(3, "untouched", "RESERVE")]
            with queue.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            incremental = root / "incremental.csv"
            inc_fields = fields + [
                "incremental_review_order", "incremental_review_required", "incremental_reason",
            ]
            inc_row = queue_row(2, "incremental", "RESERVE")
            inc_row.update({
                "incremental_review_order": "1",
                "incremental_review_required": "true",
                "incremental_reason": "class_below_30_accept_identity_minimum",
            })
            with incremental.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=inc_fields, lineterminator="\n")
                writer.writeheader()
                writer.writerow(inc_row)
            qc = root / "qc.csv"
            qc_fields = [
                "source_dataset", "canonical_class", "source_label", "original_id",
                "base_clip_id", "source_clip_id", "raw_relpath", "resource_status",
                "mapping_type", "auto_qc_status", "auto_qc_reason", "license_status",
                "license_raw", "provenance_source", "pretrain_seen_status", "raw_sha256",
            ]
            with qc.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=qc_fields, lineterminator="\n")
                writer.writeheader()
                for original in ("target", "incremental"):
                    writer.writerow({
                        "source_dataset": "DESED isolated foreground", "canonical_class": "dishes",
                        "source_label": "Dishes", "original_id": original,
                        "base_clip_id": "desed:train:" + original, "source_clip_id": original,
                        "raw_relpath": original + ".wav", "resource_status": "PRESENT",
                        "mapping_type": "EXACT", "auto_qc_status": "AUTO_PASS",
                        "auto_qc_reason": "", "license_status": "PER_RECORDING_METADATA",
                        "license_raw": "CC0", "provenance_source": "test.tsv",
                        "pretrain_seen_status": "unknown", "raw_sha256": original,
                    })
            stats = ingest_incremental(
                queue, incremental, [qc], root, root / "snapshot.csv",
                root / "membership.csv", root / "stats.json", root / "split.json",
                "clsdoa_v1_source_split_20260828",
            )
            with queue.open(newline="", encoding="utf-8") as handle:
                after = list(csv.DictReader(handle))
            self.assertEqual(after[1]["manual_decision"], "ACCEPT")
            self.assertEqual(after[1]["manual_notes"],
                             "human-authorized batch acceptance: dishes incremental review")
            self.assertEqual(after[2]["manual_decision"], "")
            self.assertEqual(stats["accepted_total"], 2)
            self.assertEqual(stats["manual_ingest"]["other_reserve_or_formal_auto_accepted"], False)


if __name__ == "__main__":
    unittest.main()
