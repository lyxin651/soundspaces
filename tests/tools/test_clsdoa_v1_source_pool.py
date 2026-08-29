import csv
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from tools.clsdoa_v1.build_source_candidates import build_candidates
from tools.clsdoa_v1.build_fs50k_candidates import crosswalk_entries
from tools.clsdoa_v1.build_desed_candidates import build_candidates as build_desed_candidates
from tools.clsdoa_v1.build_manual_review_queue import build_queue
from tools.clsdoa_v1.compress_manual_review_queue import build_compressed_queue
from tools.clsdoa_v1.qc_source_audio import run_qc


class SourcePoolToolTests(unittest.TestCase):
    def test_fs50k_builder_reads_step1a_crosswalk_without_string_guessing(self):
        evidence = Path(__file__).parents[2] / "docs/audits/clsdoa_v1/source_mapping/source_mapping_evidence.json"
        entries = crosswalk_entries(evidence)
        self.assertEqual(len(entries), 17)
        self.assertIn("mechanical_fan", {entry["canonical_class"] for entry in entries})
        self.assertIn("microwave_oven", {entry["canonical_class"] for entry in entries})
        self.assertIn("printer", {entry["canonical_class"] for entry in entries})
        self.assertTrue(all(entry["tsv_path"].endswith(".tsv") for entry in entries))

    def test_candidate_inventory_uses_exact_mapping_and_preserves_base_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "assets"
            esc = asset / "raw/esc50/ESC-50-master"
            (esc / "audio").mkdir(parents=True)
            mapping = root / "mapping.csv"
            mapping.write_text(
                "canonical_class_id,canonical_class,source_dataset,source_label,mapping_status,primary_candidate,mapping_type\n"
                "3,vacuum_cleaner,ESC-50,vacuum_cleaner,EXACT,true,EXACT\n"
                "3,vacuum_cleaner,DESED isolated foreground,Vacuum_cleaner,EXACT,false,EXACT\n",
                encoding="utf-8",
            )
            metadata = esc / "meta.csv"
            metadata.write_text(
                "filename,fold,target,category,src_file,take\n"
                "1-100210-A-36.wav,1,36,vacuum_cleaner,100210,A\n"
                "1-100210-B-36.wav,1,36,vacuum_cleaner,100210,B\n",
                encoding="utf-8",
            )
            for name in ("1-100210-A-36.wav", "1-100210-B-36.wav"):
                with wave.open(str(esc / "audio" / name), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(8000)
                    handle.writeframes((np.ones(8000) * 1000).astype("<i2").tobytes())
            out = root / "candidate.csv"
            rows = build_candidates(mapping, metadata, esc, asset, out)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["base_clip_id"] for row in rows}, {"esc50:100210"})
            self.assertEqual(rows[0]["mapping_type"], "EXACT")

    def test_qc_marks_nonfinite_or_zero_as_reject_and_valid_audio_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "ok.wav"
            with wave.open(str(wav), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes((np.ones(8000) * 1000).astype("<i2").tobytes())
            candidate = root / "candidate.csv"
            with candidate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["raw_relpath"])
                writer.writeheader()
                writer.writerow({"raw_relpath": "ok.wav"})
            result = run_qc(candidate, root, root / "qc.csv")
            self.assertEqual(result[0]["auto_qc_status"], "AUTO_PASS")
            self.assertEqual(result[0]["channels"], "1")

    def test_manual_queue_keeps_esc_all_and_limits_fsd_exact_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "a.wav"
            with wave.open(str(wav), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes((np.ones(8000) * 1000).astype("<i2").tobytes())
            qc = root / "qc.csv"
            fields = ["source_dataset", "canonical_class", "source_label",
                      "original_id", "raw_relpath", "resource_status",
                      "mapping_type", "auto_qc_status", "auto_qc_reason",
                      "duration_sec", "active_duration_estimate_sec"]
            with qc.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for dataset, count in (("ESC-50", 2), ("FSD", 60)):
                    for index in range(count):
                        writer.writerow({
                            "source_dataset": dataset,
                            "canonical_class": "speech",
                            "source_label": "Speech",
                            "original_id": "{}-{}".format(dataset, index),
                            "raw_relpath": "a.wav",
                            "resource_status": "PRESENT",
                            "mapping_type": "EXACT",
                            "auto_qc_status": "AUTO_PASS",
                            "auto_qc_reason": "",
                            "duration_sec": "1",
                            "active_duration_estimate_sec": "1",
                        })
            output = root / "queue.csv"
            rows = build_queue([qc], root, output, target_per_class=40,
                               reserve_per_class=12)
            self.assertEqual(sum(row["source_dataset"] == "ESC-50" for row in rows), 2)
            self.assertEqual(sum(row["source_dataset"] == "FSD" and
                                 row["pool_candidate_role"] == "TARGET" for row in rows), 40)
            self.assertEqual(sum(row["source_dataset"] == "FSD" and
                                 row["pool_candidate_role"] == "RESERVE" for row in rows), 12)

    def test_desed_builder_excludes_mixtures_and_keeps_alarm_semantic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extracted = root / "extracted"
            foreground = extracted / "audio/train/soundbank/foreground"
            for label in ("Speech", "Alarm_bell_ringing", "Cat"):
                (foreground / label).mkdir(parents=True)
            (extracted / "audio/eval/soundbank/foreground_on_off/Speech").mkdir(parents=True)
            for label in ("Speech", "Alarm_bell_ringing", "Cat"):
                wav = foreground / label / "123_0.wav"
                with wave.open(str(wav), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(8000)
                    handle.writeframes((np.ones(8000) * 1000).astype("<i2").tobytes())
            (extracted / "license_training.tsv").write_text(
                "filename\tid\tdataset\tdownload\tlicense\tname\tusername\n"
                "/x/training/soundbank/foreground/Speech/123_0.wav\t123\tfreesound\t\tCC0\t123.wav\tu\n"
                "/x/training/soundbank/foreground/Alarm_bell_ringing/123_0.wav\t123\tfreesound\t\tCC0\t123.wav\tu\n"
                "/x/training/soundbank/foreground/Cat/123_0.wav\t123\tfreesound\t\tCC0\t123.wav\tu\n",
                encoding="utf-8",
            )
            (extracted / "license_eval.tsv").write_text(
                "dataset\tdownload\tfilename\tid\tlicense\tname\tusername\n",
                encoding="utf-8",
            )
            rows, excluded = build_desed_candidates(extracted, root, root / "out.csv")
            self.assertEqual({row["canonical_class"] for row in rows}, {"speech", "clock_alarm"})
            self.assertEqual(excluded, 0)
            self.assertEqual(
                next(row["mapping_type"] for row in rows if row["canonical_class"] == "clock_alarm"),
                "SEMANTIC_STRONG",
            )

    def test_compressed_queue_deduplicates_globally_and_sets_preview_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "long.wav"
            with wave.open(str(wav), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes((np.ones(80000) * 1000).astype("<i2").tobytes())
            qc = root / "qc.csv"
            fields = ["source_dataset", "canonical_class", "source_label",
                      "original_id", "base_clip_id", "source_clip_id", "raw_relpath",
                      "resource_status", "mapping_type", "auto_qc_status",
                      "auto_qc_reason", "license_status", "provenance_source",
                      "raw_sha256", "duration_sec", "active_duration_estimate_sec"]
            with qc.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                rows = [
                    {"source_dataset": "DESED isolated foreground", "canonical_class": "speech",
                     "source_label": "Speech", "original_id": "same", "base_clip_id": "desed:train:same",
                     "source_clip_id": "train/same", "raw_relpath": "long.wav", "resource_status": "PRESENT",
                     "mapping_type": "EXACT", "auto_qc_status": "AUTO_FLAG", "auto_qc_reason": "low_activity",
                     "license_status": "PER_RECORDING_METADATA", "provenance_source": "license.tsv",
                     "raw_sha256": "a", "duration_sec": "10", "active_duration_estimate_sec": "1"},
                    {"source_dataset": "DESED isolated foreground", "canonical_class": "speech",
                     "source_label": "Speech", "original_id": "same", "base_clip_id": "desed:eval:same",
                     "source_clip_id": "eval/same", "raw_relpath": "long.wav", "resource_status": "PRESENT",
                     "mapping_type": "EXACT", "auto_qc_status": "AUTO_PASS", "auto_qc_reason": "",
                     "license_status": "PER_RECORDING_METADATA", "provenance_source": "license.tsv",
                     "raw_sha256": "b", "duration_sec": "10", "active_duration_estimate_sec": "2"},
                ]
                writer.writerows(rows)
            queue_path = root / "queue.csv"
            summary_path = root / "summary.json"
            output, summary = build_compressed_queue(
                [qc], root, queue_path, summary_json=summary_path,
                target_per_class=1, reserve_per_class=0,
            )
            self.assertEqual(summary["candidate_rows_read"], 2)
            self.assertEqual(summary["identity_count_before_dedup"], 2)
            self.assertEqual(summary["identity_count_after_dedup"], 1)
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0]["auto_qc_status"], "AUTO_PASS")
            self.assertEqual(output[0]["preview_start_sec"], "2.500000")
            self.assertEqual(output[0]["preview_end_sec"], "7.500000")
            self.assertEqual(output[0]["manual_decision"], "")
            self.assertEqual(output[0]["manual_crop_override_candidate"], "false")

    def test_compressed_queue_keeps_class_source_role_before_qc_tie_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "a.wav"
            with wave.open(str(wav), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes((np.ones(8000) * 1000).astype("<i2").tobytes())
            qc = root / "qc.csv"
            fields = ["source_dataset", "canonical_class", "source_label",
                      "original_id", "base_clip_id", "source_clip_id", "raw_relpath",
                      "resource_status", "mapping_type", "auto_qc_status",
                      "auto_qc_reason", "license_status", "provenance_source",
                      "raw_sha256", "duration_sec", "active_duration_estimate_sec"]
            with qc.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for dataset, status in (("ESC-50", "AUTO_FLAG"),
                                        ("DESED isolated foreground", "AUTO_PASS")):
                    writer.writerow({
                        "source_dataset": dataset, "canonical_class": "vacuum_cleaner",
                        "source_label": "vacuum_cleaner", "original_id": dataset,
                        "base_clip_id": dataset, "source_clip_id": dataset,
                        "raw_relpath": "a.wav", "resource_status": "PRESENT",
                        "mapping_type": "EXACT", "auto_qc_status": status,
                        "auto_qc_reason": "flag" if status == "AUTO_FLAG" else "",
                        "license_status": "DATASET_LEVEL_VERIFIED",
                        "provenance_source": "test", "raw_sha256": "hash",
                        "duration_sec": "1", "active_duration_estimate_sec": "1",
                    })
            output, _ = build_compressed_queue(
                [qc], root, root / "queue.csv", target_per_class=1,
                reserve_per_class=0,
            )
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0]["source_dataset"], "ESC-50")
            self.assertEqual(output[0]["auto_qc_status"], "AUTO_FLAG")


if __name__ == "__main__":
    unittest.main()
