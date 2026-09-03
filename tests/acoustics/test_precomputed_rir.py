import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from active_audition.acoustics.precomputed_rir import PrecomputedRIRBackend, PrecomputedRIRError
from active_audition.acoustics.resampling import canonicalize_rir, resample_array


class PrecomputedRIRTests(unittest.TestCase):
    def test_header_contract_index_and_sha_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "sceneA" / "receiver-r1" / "source-s1" / "heading_0.wav"
            asset.parent.mkdir(parents=True)
            wavfile.write(str(asset), 16000, np.column_stack((np.arange(8), np.arange(8) + 1)).astype(np.float32))
            backend = PrecomputedRIRBackend(str(root))
            row = backend.locate_rir("sceneA", "r1", "s1", 0)
            self.assertEqual(row.original_sample_rate_hz, 16000)
            self.assertEqual(row.original_num_channels, 2)
            self.assertEqual(row.original_dtype, "float32")
            self.assertEqual(row.sha256, backend.locate_rir("sceneA", "r1", "s1", 0).sha256)

    def test_wrong_channels_and_missing_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "sceneA" / "receiver-r1" / "source-s1" / "heading_0.wav"
            asset.parent.mkdir(parents=True)
            wavfile.write(str(asset), 16000, np.zeros((8, 1), dtype=np.float32))
            backend = PrecomputedRIRBackend(str(root))
            with self.assertRaises(PrecomputedRIRError): backend.locate_rir("sceneA", "r1", "s1", 0)
            with self.assertRaises(PrecomputedRIRError): backend.locate_rir("sceneA", "r1", "missing", 0)

    def test_two_stage_resampling_is_deterministic_and_not_normalized(self):
        rir = np.column_stack((np.linspace(0, 2, 441), np.linspace(1, 3, 441))).astype(np.float32)
        first = canonicalize_rir(rir, 44100)
        second = canonicalize_rir(rir, 44100)
        self.assertEqual(first.shape[1], 2)
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first.shape[0], 240)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(first[:, 1].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
