import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_audition.config.loader import load_resolved_config
from active_audition.data.storage import DatasetStorage, StorageError
from active_audition.pipeline.v0 import render_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/active_audition/v0_replica_debug.yaml"


class FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class RecoveryTests(unittest.TestCase):
    def test_real_payload_recovery_and_orphan_rejection(self):
        config = load_resolved_config(str(CONFIG))
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(temp_dir)
            storage.ensure_writable()
            storage.atomic_write_jsonl("episodes.jsonl", [_episode()])
            storage.atomic_write_jsonl("candidates.jsonl", [_candidate()])
            native_rir = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            patches = [
                patch("active_audition.pipeline.v0.DatasetStorage.from_config", return_value=storage),
                patch("active_audition.pipeline.v0.create_scene_simulator", return_value=FakeContext()),
                patch("active_audition.pipeline.v0.load_dry_audio_registry", return_value={"golden_probe_v0": {"path": str(REPO_ROOT / "res/active_audition/golden_probe_v0.wav")}}),
                patch("active_audition.pipeline.v0.render_native_rir", return_value=native_rir),
            ]
            for item in patches:
                item.start()
            try:
                first = render_dataset(str(CONFIG))
                with self.assertRaisesRegex(StorageError, "incomplete dataset"):
                    render_dataset(str(CONFIG))
                forced = render_dataset(str(CONFIG), overwrite=True)
                self.assertEqual(forced["rendered"], 2)
                rows = _read_rows(storage)
                complete_hash = _sha(storage.root / rows["rot_left_45"]["audio_path"])
                initial_audio = storage.root / rows["initial"]["audio_path"]
                initial_audio.write_bytes(b"corrupt")
                recovered = render_dataset(str(CONFIG), resume=True)
                self.assertEqual(recovered["recovery"], ["initial"])
                self.assertEqual(_sha(storage.root / rows["rot_left_45"]["audio_path"]), complete_hash)
                rows = _read_rows(storage)
                (storage.root / rows["initial"]["rir_path"]).unlink()
                recovered_rir = render_dataset(str(CONFIG), resume=True)
                self.assertEqual(recovered_rir["recovery"], ["initial"])
                self.assertEqual(len(_read_rows(storage)), 2)
                orphan = storage.root / "episodes/replica_office_0/ep_000001/audio/orphan.wav"
                orphan.parent.mkdir(parents=True, exist_ok=True)
                orphan.write_bytes(b"orphan")
                with self.assertRaises(StorageError):
                    render_dataset(str(CONFIG), resume=True)
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_orphan_is_rejected_without_viewpoints_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(temp_dir)
            storage.ensure_writable()
            storage.atomic_write_jsonl("episodes.jsonl", [_episode()])
            storage.atomic_write_jsonl("candidates.jsonl", [_candidate()])
            audio = storage.root / "episodes/replica_office_0/ep_000001/audio/orphan.wav"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"orphan")
            storage.atomic_write_npz(storage.rir_cache_path("orphan"), {"rir": np.zeros((1, 2), dtype=np.float32)})
            with patch("active_audition.pipeline.v0.DatasetStorage.from_config", return_value=storage):
                with self.assertRaisesRegex(StorageError, "orphan"):
                    render_dataset(str(CONFIG), resume=True)


def _episode():
    return {"schema_version": "v0.1", "episode_id": "ep_000001", "scene_id": "replica.office_0", "episode_seed": 1, "source": {"position_world": [1.0, 0.5, 1.0], "audio_id": "golden_probe_v0", "segment_start_sec": 0.0, "segment_duration_sec": 5.0, "gain_db": 0.0}, "listener_initial": {"base_position_world": [0.0, 0.0, 0.0], "sensor_position_world": [0.0, 1.5, 0.0], "yaw_deg": 0.0}}


def _candidate():
    return {"episode_id": "ep_000001", "candidate_id": "rot_left_45", "action_type": "rotation", "valid": True, "snapped_base_position_world": [0.0, 0.0, 0.0], "sensor_position_world": [0.0, 1.5, 0.0], "yaw_deg": 45.0}


def _read_rows(storage):
    import json
    rows = {}
    with storage.manifest_path("viewpoints.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["viewpoint_id"]] = row
    return rows


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
