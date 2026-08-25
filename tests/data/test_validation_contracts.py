import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from active_audition.config.loader import load_resolved_config
from active_audition.data.storage import DatasetStorage
from active_audition.data.validation import validate_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/active_audition/v0_replica_debug.yaml"


class ValidationContractTests(unittest.TestCase):
    def test_missing_initial_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage, episodes, candidates, viewpoints = _fixture(temp_dir)
            storage.atomic_write_jsonl("viewpoints.jsonl", [viewpoints[1]])
            result = validate_dataset(temp_dir, load_resolved_config(str(CONFIG)))
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("initial" in failure for failure in result["failures"]))

    def test_duplicate_candidate_key_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage, episodes, candidates, viewpoints = _fixture(temp_dir)
            storage.atomic_write_jsonl("candidates.jsonl", candidates + [candidates[0]])
            result = validate_dataset(temp_dir, load_resolved_config(str(CONFIG)))
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("candidate key" in failure for failure in result["failures"]))

    def test_candidate_viewpoint_relation_and_manifest_metadata_are_hard_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage, episodes, candidates, viewpoints = _fixture(temp_dir)
            viewpoints[1]["action_type"] = "translation"
            viewpoints[1]["num_channels"] = 1
            storage.atomic_write_jsonl("viewpoints.jsonl", viewpoints)
            result = validate_dataset(temp_dir, load_resolved_config(str(CONFIG)))
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("candidate/viewpoint" in failure for failure in result["failures"]))
            self.assertTrue(any("metadata" in failure for failure in result["failures"]))

    def test_same_dry_hash_and_duration_mismatch_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage, episodes, candidates, viewpoints = _fixture(temp_dir)
            viewpoints[1]["ground_truth"]["gain_db"] = 3.0
            storage.atomic_write_jsonl("viewpoints.jsonl", viewpoints)
            result = validate_dataset(temp_dir, load_resolved_config(str(CONFIG)))
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("ground_truth" in failure or "same Episode dry" in failure for failure in result["failures"]))

    def test_candidate_referencing_missing_episode_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage, episodes, candidates, viewpoints = _fixture(temp_dir)
            candidates.append({"episode_id": "ep_missing", "candidate_id": "rot_right_45", "action_type": "rotation", "valid": False})
            storage.atomic_write_jsonl("candidates.jsonl", candidates)
            result = validate_dataset(temp_dir, load_resolved_config(str(CONFIG)))
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("candidate references missing episode" in failure for failure in result["failures"]))


def _fixture(temp_dir):
    storage = DatasetStorage(temp_dir)
    storage.ensure_writable()
    episode = {
        "schema_version": "v0.1", "episode_id": "ep_000001", "scene_id": "replica.office_0", "episode_seed": 1,
        "source": {"position_world": [1.0, 0.5, 1.0], "audio_id": "golden_probe_v0", "segment_start_sec": 0.0, "segment_duration_sec": 5.0, "gain_db": 0.0},
        "listener_initial": {"base_position_world": [0.0, 0.0, 0.0], "sensor_position_world": [0.0, 1.5, 0.0], "yaw_deg": 0.0},
    }
    candidate = {"episode_id": "ep_000001", "candidate_id": "rot_left_45", "action_type": "rotation", "valid": True}
    dry_hash = "dry-hash"
    gt = {"scene_id": "replica.office_0", "source_position_world": [1.0, 0.5, 1.0], "dry_audio_id": "golden_probe_v0", "segment_start_sec": 0.0, "segment_duration_sec": 5.0, "gain_db": 0.0, "dry_hash": dry_hash}
    viewpoints = [
        _viewpoint(storage, "initial", None, "initial", [0.0, 0.0, 0.0], [0.0, 1.5, 0.0], 0.0, gt),
        _viewpoint(storage, "rot_left_45", "rot_left_45", "rotation", [0.0, 0.0, 0.0], [0.0, 1.5, 0.0], 45.0, gt),
    ]
    storage.atomic_write_jsonl("episodes.jsonl", [episode])
    storage.atomic_write_jsonl("candidates.jsonl", [candidate])
    storage.atomic_write_jsonl("viewpoints.jsonl", viewpoints)
    return storage, [episode], [candidate], viewpoints


def _viewpoint(storage, viewpoint_id, candidate_id, action_type, base, sensor, yaw, gt):
    audio = storage.viewpoint_audio_path("replica.office_0", "ep_000001", viewpoint_id)
    rir = storage.rir_cache_path("ep_000001__" + viewpoint_id)
    waveform = np.zeros((80001, 2), dtype=np.float32)
    audio.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(audio), 16000, waveform)
    storage.atomic_write_npz(rir, {"rir": np.zeros((2, 2), dtype=np.float32), "sample_rate_hz": np.int64(16000), "num_samples": np.int64(2)})
    return {"episode_id": "ep_000001", "viewpoint_id": viewpoint_id, "candidate_id": candidate_id, "action_type": action_type, "base_position_world": base, "sensor_position_world": sensor, "yaw_deg": yaw, "audio_path": str(audio.relative_to(storage.root)), "rir_id": "ep_000001__" + viewpoint_id, "rir_path": str(rir.relative_to(storage.root)), "sample_rate_hz": 16000, "num_channels": 2, "num_samples": 80001, "duration_sec": 80001 / 16000.0, "dtype": "float32", "ground_truth": dict(gt)}


if __name__ == "__main__":
    unittest.main()
