import json
import tempfile
import unittest
from pathlib import Path

from tools.clsdoa_v1.audit_scene_readiness import audit_replica, candidate


class SceneReadinessTests(unittest.TestCase):
    def make_scene(self, stage_text=None, nav=True, semantic=True):
        temp = Path(tempfile.mkdtemp())
        habitat = temp / "scene" / "habitat"
        habitat.mkdir(parents=True)
        (habitat / "mesh_semantic.ply").write_bytes(b"ply\n")
        if nav:
            (habitat / "mesh_semantic.navmesh").write_bytes(b"nav\n")
        if semantic:
            (habitat / "info_semantic.json").write_text("{}\n", encoding="utf-8")
        if stage_text is None:
            stage_text = json.dumps({"render_asset": "mesh_semantic.ply", "semantic_asset": "mesh_semantic.ply", "nav_asset": "mesh_semantic.navmesh"})
        (habitat / "scene.stage_config.json").write_text(stage_text, encoding="utf-8")
        return temp

    def test_complete_fixture(self):
        entries = audit_replica(self.make_scene())
        self.assertEqual(entries[0]["readiness_status"], "PRESENT_COMPLETE")

    def test_missing_navmesh_is_partial(self):
        entries = audit_replica(self.make_scene(nav=False))
        self.assertEqual(entries[0]["readiness_status"], "PRESENT_PARTIAL")

    def test_missing_semantic_metadata_is_partial(self):
        entries = audit_replica(self.make_scene(semantic=False))
        self.assertEqual(entries[0]["readiness_status"], "PRESENT_PARTIAL")

    def test_broken_stage_reference(self):
        stage = json.dumps({"render_asset": "missing.ply", "semantic_asset": "mesh_semantic.ply", "nav_asset": "mesh_semantic.navmesh"})
        entries = audit_replica(self.make_scene(stage_text=stage))
        self.assertEqual(entries[0]["readiness_status"], "BROKEN_PATH")

    def test_candidate_is_unassigned_and_not_admitted(self):
        item = candidate(audit_replica(self.make_scene())[0])
        self.assertEqual(item["split"], "UNASSIGNED")
        self.assertEqual(item["admitted"], "NOT_RUN")
        self.assertEqual(item["materials_mode"], "off")


if __name__ == "__main__":
    unittest.main()
