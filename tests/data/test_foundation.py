import json
import tempfile
import unittest
from pathlib import Path

from active_audition.config.loader import load_resolved_config
from active_audition.data.catalog import episode_seed, load_dry_audio_registry, load_scene_registry
from active_audition.data.storage import DatasetStorage


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/active_audition/v0_replica_debug.yaml"


class FoundationTests(unittest.TestCase):
    def test_config_and_registries_load(self):
        config = load_resolved_config(str(CONFIG))
        self.assertIs(config["acoustics"]["materials_enabled"], False)
        self.assertEqual(config["acoustics"]["sample_rate_hz"], 16000)
        scenes = load_scene_registry(config["registries"]["scenes_path"], config["_repo_root"])
        dry = load_dry_audio_registry(config["registries"]["dry_audio_path"], config["_repo_root"])
        self.assertEqual(list(scenes), ["replica.office_0"])
        self.assertIn(config["golden"]["source_audio_id"], dry)

    def test_episode_seed_is_stable_and_distinct(self):
        first = episode_seed(20260824, "ep_000001")
        second = episode_seed(20260824, "ep_000001")
        self.assertEqual(first, second)
        self.assertNotEqual(first, episode_seed(20260824, "ep_000002"))

    def test_storage_jsonl_is_atomic_and_sorted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = DatasetStorage(str(Path(temp_dir) / "dataset"))
            storage.ensure_writable()
            path = storage.atomic_write_jsonl(
                "episodes.jsonl", [{"episode_id": "ep_000001", "x": 1}]
            )
            expected = json.dumps(
                {"episode_id": "ep_000001", "x": 1},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ) + "\n"
            self.assertEqual(path.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
