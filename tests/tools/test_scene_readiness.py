import json
import tempfile
import unittest
from pathlib import Path

from tools.clsdoa_v1.audit_scene_readiness import audit_mp3d, audit_replica, candidate


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

    def make_mp3d_scene(self, missing=()):
        temp = Path(tempfile.mkdtemp())
        scan = temp / "ABCD1234"
        scan.mkdir()
        files = {
            "ABCD1234.glb": b"glb\n",
            "ABCD1234_semantic.ply": b"ply\n",
            "ABCD1234.navmesh": b"nav\n",
            "ABCD1234.house": b"house\n",
        }
        for name, content in files.items():
            if name not in missing:
                (scan / name).write_bytes(content)
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

    def test_mp3d_official_layout_is_complete(self):
        result = audit_mp3d(self.make_mp3d_scene())
        self.assertEqual(result["scan_count"], 1)
        self.assertEqual(result["scans"][0]["readiness_status"], "PRESENT_COMPLETE")
        item = candidate(result["scans"][0])
        self.assertEqual(item["admitted"], "NOT_RUN")
        self.assertEqual(item["split"], "UNASSIGNED")

    def test_mp3d_missing_core_resource_is_partial(self):
        result = audit_mp3d(self.make_mp3d_scene(missing={"ABCD1234.navmesh"}))
        self.assertEqual(result["scans"][0]["readiness_status"], "PRESENT_PARTIAL")
        self.assertEqual(result["scans"][0]["missing_fields"], ["navmesh"])


if __name__ == "__main__":
    unittest.main()
