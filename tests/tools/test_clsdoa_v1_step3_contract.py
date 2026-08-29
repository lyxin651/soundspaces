import unittest
from pathlib import Path

import yaml

from active_audition.datasets.binaural_foa_clsdoa.scene_registry import read_scene_registry, resolve_generation_scene_resources
from active_audition.datasets.binaural_foa_clsdoa.source_registry import read_source_registry
from tools.clsdoa_v1.pilot_dataset import stable_int


ROOT = Path(__file__).resolve().parents[2]


class Step3ContractTests(unittest.TestCase):
    def test_finalized_source_reader_consumes_all_rows(self):
        rows = read_source_registry(str(ROOT / "registries/source_audio.csv"))
        self.assertEqual(len(rows), 422)
        self.assertEqual(len({row["base_clip_id"] for row in rows}), 422)

    def test_finalized_scene_reader_keeps_failures_and_resolves_direct_core(self):
        rows = read_scene_registry(str(ROOT / "registries/clsdoa_v1_scenes.yaml"))
        self.assertEqual(len(rows), 108)
        self.assertEqual(sum(row["admitted"] == "PASS" for row in rows), 103)
        self.assertEqual(sum(row["admitted"] == "FAIL" for row in rows), 5)
        replica = next(row for row in rows if row["scene_family"] == "Replica")
        resolved = resolve_generation_scene_resources(dict(replica, scene_asset=str(ROOT / replica["scene_asset"]), navmesh=str(ROOT / replica["navmesh"]), semantic_info=str(ROOT / replica["semantic_info"])))
        self.assertTrue(resolved["scene_asset"].endswith("mesh_semantic.ply"))
        self.assertTrue(resolved["navmesh"].endswith("mesh_semantic.navmesh"))
        self.assertTrue(resolved["semantic_info"].endswith("info_semantic.json"))

    def test_stable_seed_is_not_python_hash(self):
        self.assertEqual(stable_int("x", 1), stable_int("x", 1))
        self.assertNotEqual(stable_int("x", 1), stable_int("x", 2))

    def test_pilot_policy_is_frozen(self):
        config = yaml.safe_load((ROOT / "configs/active_audition/clsdoa_v1_pilot_001.yaml").read_text())
        self.assertEqual(config["dataset_id"], "clsdoa_v1_pilot_001")
        self.assertTrue(config["storage"]["save_rir"])
        self.assertTrue(config["storage"]["require_rir"])
        self.assertEqual(config["global_seed"], 20260829)


if __name__ == "__main__":
    unittest.main()
