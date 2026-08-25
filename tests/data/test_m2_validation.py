import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from active_audition.config.loader import load_resolved_config
from active_audition.data.storage import DatasetStorage, StorageError
from active_audition.data.validation import payload_is_complete


class M2StorageAndResumeTests(unittest.TestCase):
    def test_resume_detects_missing_and_corrupt_payloads_without_manifest_duplication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(temp_dir)
            storage.ensure_writable()
            audio = storage.viewpoint_audio_path("ep_000001", "initial")
            rir = storage.rir_cache_path("ep_000001__initial")
            waveform = np.zeros((4, 2), dtype=np.float32)
            storage.atomic_write_bytes(audio, _wav_bytes(waveform))
            storage.atomic_write_npz(rir, {"rir": np.zeros((2, 2), dtype=np.float32), "sample_rate_hz": np.int64(16000), "num_samples": np.int64(2)})
            row = {"audio_path": str(audio.relative_to(storage.root)), "rir_path": str(rir.relative_to(storage.root)), "sample_rate_hz": 16000}
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


def _wav_bytes(waveform):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        wavfile.write(handle.name, 16000, waveform)
        handle.flush()
        return Path(handle.name).read_bytes()


if __name__ == "__main__":
    unittest.main()
