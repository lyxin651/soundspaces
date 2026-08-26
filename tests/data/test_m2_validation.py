import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from active_audition.config.loader import load_resolved_config
from active_audition.data.storage import DatasetStorage, StorageError
from active_audition.data.validation import payload_is_complete
from active_audition.pipeline.v0 import _storage_summary, final_filesystem_bytes


class M2StorageAndResumeTests(unittest.TestCase):
    def test_resume_detects_missing_and_corrupt_payloads_without_manifest_duplication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(temp_dir)
            storage.ensure_writable()
            audio = storage.viewpoint_audio_path("replica.office_0", "ep_000001", "initial")
            rir = storage.rir_cache_path("ep_000001__initial")
            waveform = np.zeros((4, 2), dtype=np.float32)
            storage.atomic_write_bytes(audio, _wav_bytes(waveform))
            storage.atomic_write_npz(rir, {"rir": np.zeros((2, 2), dtype=np.float32), "sample_rate_hz": np.int64(16000), "num_samples": np.int64(2)})
            row = {
                "audio_path": str(audio.relative_to(storage.root)), "rir_id": "ep_000001__initial",
                "rir_path": str(rir.relative_to(storage.root)), "sample_rate_hz": 16000,
                "num_channels": 2, "dtype": "float32",
                "num_samples": 4, "duration_sec": 4.0 / 16000.0,
                "ground_truth": {"segment_duration_sec": 3.0 / 16000.0},
            }
            self.assertTrue(payload_is_complete(temp_dir, row))
            audio.write_bytes(b"corrupt")
            self.assertFalse(payload_is_complete(temp_dir, row))
            self.assertEqual(len({("ep_000001", "initial")}), 1)

    def test_success_makes_dataset_immutable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(temp_dir)
            storage.ensure_writable()
            storage.atomic_write_text(storage.success_path, "done\n")
            with self.assertRaises(StorageError):
                storage.ensure_writable()

    def test_storage_summary_pre_final_measurement_and_final_filesystem_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(temp_dir)
            storage.ensure_writable()
            storage.atomic_write_text(storage.manifest_path("episodes.jsonl"), "{}\n")
            wav = storage.root / "episodes/scene/ep/audio/initial.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"wav")
            rir = storage.rir_cache_path("ep__initial")
            storage.atomic_write_bytes(rir, b"rir")
            summary = _storage_summary(storage)
            self.assertEqual(summary["wav_count"], 1)
            self.assertEqual(summary["wav_bytes"], 3)
            self.assertEqual(summary["rir_count"], 1)
            self.assertEqual(summary["rir_bytes"], 3)
            self.assertEqual(summary["manifest_bytes"], 3)
            self.assertEqual(summary["measured_bytes_before_storage_summary_finalize"], 9)
            storage.atomic_write_json(storage.path("reports", "storage_summary.json"), summary)
            storage.atomic_write_text(storage.success_path, "done\n")
            self.assertEqual(final_filesystem_bytes(storage), sum(path.stat().st_size for path in storage.root.rglob("*") if path.is_file()))

    def test_payload_manifest_channel_and_dtype_mismatch_cannot_resume_skip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(temp_dir)
            storage.ensure_writable()
            audio = storage.viewpoint_audio_path("replica.office_0", "ep_000001", "initial")
            rir = storage.rir_cache_path("ep_000001__initial")
            waveform = np.zeros((4, 2), dtype=np.float32)
            storage.atomic_write_bytes(audio, _wav_bytes(waveform))
            storage.atomic_write_npz(rir, {"rir": np.zeros((2, 2), dtype=np.float32), "sample_rate_hz": np.int64(16000), "num_samples": np.int64(2)})
            row = {
                "audio_path": str(audio.relative_to(storage.root)), "rir_id": "ep_000001__initial",
                "rir_path": str(rir.relative_to(storage.root)), "sample_rate_hz": 16000,
                "num_channels": 2, "dtype": "float32", "num_samples": 4,
                "duration_sec": 4.0 / 16000.0,
                "ground_truth": {"segment_duration_sec": 3.0 / 16000.0},
            }
            self.assertTrue(payload_is_complete(temp_dir, row))
            row["num_channels"] = 1
            self.assertFalse(payload_is_complete(temp_dir, row))
            row["num_channels"] = 2
            row["dtype"] = "int16"
            self.assertFalse(payload_is_complete(temp_dir, row))


def _wav_bytes(waveform):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        wavfile.write(handle.name, 16000, waveform)
        handle.flush()
        return Path(handle.name).read_bytes()


if __name__ == "__main__":
    unittest.main()
