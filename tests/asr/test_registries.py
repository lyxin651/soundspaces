import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from active_audition.data.noise_registry import build_noise_registry, parent_split_audit
from active_audition.data.speech_registry import (
    SPEECH_REGISTRY_SCHEMA_VERSION,
    build_speech_registry,
    decoded_waveform_sha256,
    select_clean_qualification,
    sha256_file,
    speaker_split_audit,
)


def speech_row(split, speaker, index, eligible=True):
    utterance_id = "{}-1-{:04d}".format(speaker, index)
    transcript = "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN"
    return {
        "schema_version": SPEECH_REGISTRY_SCHEMA_VERSION,
        "corpus": "LibriSpeech",
        "split": split,
        "speaker_id": str(speaker),
        "chapter_id": "1",
        "utterance_id": utterance_id,
        "relative_source_path": split + "/" + utterance_id + ".flac",
        "source_file_sha256": "1" * 64,
        "decoded_waveform_sha256": "2" * 64,
        "source_sample_rate_hz": 16000,
        "decode_sample_rate_hz": 16000,
        "dtype": "float32",
        "samples": 80000,
        "duration_sec": 5.0,
        "raw_transcript": transcript,
        "normalized_transcript": transcript,
        "reference_word_count": 10,
        "complete_utterance": True,
        "eligible": eligible,
        "exclusion_reason": "" if eligible else "TEST_EXCLUSION",
    }


class RegistryTest(unittest.TestCase):
    def test_real_decode_binds_source_and_decoded_hash_and_keeps_full_utterance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for speaker, split in enumerate(("dev-clean", "dev-other", "test-clean", "test-other"), 1):
                chapter = root / split / "1" / "1"
                chapter.mkdir(parents=True)
                utterance_id = "{}-1-0001".format(speaker)
                waveform = np.linspace(-0.1, 0.1, 16000 * 5, dtype=np.float32)
                audio = chapter / (utterance_id + ".flac")
                sf.write(str(audio), waveform, 16000)
                (chapter / "{}-1.trans.txt".format(speaker)).write_text(utterance_id + " ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN\n", encoding="utf-8")
            rows = build_speech_registry(str(root))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["source_file_sha256"], sha256_file(root / rows[0]["relative_source_path"]))
            decoded, _ = sf.read(str(root / rows[0]["relative_source_path"]), dtype="float32")
            self.assertEqual(rows[0]["decoded_waveform_sha256"], decoded_waveform_sha256(decoded))
            self.assertEqual(rows[0]["samples"], 16000 * 5)
            self.assertTrue(rows[0]["complete_utterance"])

    def test_metadata_only_selection_is_deterministic_and_speaker_broad(self):
        rows = []
        for split, base in (("dev-clean", 100), ("dev-other", 200)):
            for speaker in range(base, base + 12):
                rows.append(speech_row(split, speaker, 1))
        first = select_clean_qualification(rows)
        second = select_clean_qualification(list(reversed(rows)))
        self.assertEqual([row["utterance_id"] for row in first], [row["utterance_id"] for row in second])
        self.assertEqual(len(first), 24)
        self.assertEqual(len({row["speaker_id"] for row in first}), 24)

    def test_speaker_split_audit_detects_overlap(self):
        rows = [speech_row("dev-clean", 1, 1), speech_row("test-clean", 1, 2)]
        self.assertEqual(speaker_split_audit(rows)["status"], "FAIL")

    def test_noise_technical_rejection_and_parent_split_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "musan"
            noise = root / "noise" / "free-sound"
            noise.mkdir(parents=True)
            sf.write(str(noise / "valid.wav"), np.ones(16000 * 5, dtype=np.float32) * 0.01, 16000)
            sf.write(str(noise / "short.wav"), np.ones(16000, dtype=np.float32) * 0.01, 16000)
            sf.write(str(noise / "silent.wav"), np.zeros(16000 * 5, dtype=np.float32), 16000)
            rows = build_noise_registry(str(root))
            by_id = {Path(row["parent_recording_id"]).name: row for row in rows}
            self.assertFalse(by_id["valid"]["excluded"])
            self.assertEqual(by_id["valid"]["speech_leakage_audit"]["status"], "NOT_AUDITED")
            self.assertTrue(by_id["short"]["technical_audit"]["too_short"])
            self.assertTrue(by_id["silent"]["technical_audit"]["extreme_silence"])
        audit = parent_split_audit([
            {"parent_recording_id": "recording", "split": "o1"},
            {"parent_recording_id": "recording", "split": "o2"},
        ])
        self.assertEqual(audit["status"], "FAIL")

    def test_corrupt_noise_is_rejected_not_registered_as_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            noise = Path(temporary) / "musan" / "noise" / "source"
            noise.mkdir(parents=True)
            (noise / "corrupt.wav").write_bytes(b"not a wave file")
            with self.assertRaises(Exception):
                build_noise_registry(str(Path(temporary) / "musan"))

    def test_selection_api_has_no_asr_output_or_wer_input(self):
        import inspect

        parameters = inspect.signature(select_clean_qualification).parameters
        self.assertEqual(list(parameters), ["rows", "per_split"])
        self.assertNotIn("wer", parameters)
        self.assertNotIn("hypothesis", parameters)


if __name__ == "__main__":
    unittest.main()
