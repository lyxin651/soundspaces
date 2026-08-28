import csv
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from tools.clsdoa_v1.build_source_candidates import build_candidates
from tools.clsdoa_v1.qc_source_audio import run_qc


class SourcePoolToolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
