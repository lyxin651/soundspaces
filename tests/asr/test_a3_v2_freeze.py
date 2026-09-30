import copy
import json
import unittest
from pathlib import Path

from active_audition.asr.a3_v2_freeze import (
    INVENTORY_SCHEMA,
    SCENE_MANIFEST_SCHEMA,
    SPEECH_MANIFEST_SCHEMA,
    validate_scene_inventory,
    validate_scene_manifest,
    validate_speech_manifest,
)
from active_audition.asr.static_mesh_los import STATIC_MESH_LOS_METHOD, synthetic_sanity_results
from active_audition.asr.contract_v2 import (
    A3_V2_VERSION,
    A3V2ContractError,
    a3_v2_contract_sha256,
    load_a3_v2_contract,
    validate_a3_v2_contract,
)


ROOT = Path(__file__).resolve().parents[2]
V2_PATH = ROOT / "configs/active_audition/v1/asr_contract_v2.yaml"


class A3V2FreezeTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_a3_v2_contract(str(V2_PATH), require_frozen=True, repo_root=str(ROOT))

    def test_contract_is_frozen_strict_and_canonical(self):
        self.assertEqual(self.contract["contract"]["version"], A3_V2_VERSION)
        self.assertEqual(a3_v2_contract_sha256(self.contract), "70864c814a55db5d184ef8a6835b65cb564c4c90fc86112f8705b7ecf6df1ffe")
        reordered = dict(reversed(list(self.contract.items())))
        self.assertEqual(a3_v2_contract_sha256(reordered), a3_v2_contract_sha256(self.contract))
        invalid = copy.deepcopy(self.contract)
        invalid["realistic_domain"]["unknown"] = True
        with self.assertRaises(A3V2ContractError):
            validate_a3_v2_contract(invalid, require_frozen=True, repo_root=str(ROOT))

    def test_parent_and_v1_identity_are_bound(self):
        self.assertEqual(self.contract["parents"]["a3_v1_contract_sha256"], "b972d3ca2354ead8a10d2954a60602896ebbc5204adb7e8692c22f2b340497e2")
        self.assertEqual(self.contract["historical_v1"]["g6_status"], "FAIL")
        self.assertEqual(self.contract["historical_v1"]["g6_threshold_fraction"], 0.50)
        self.assertEqual(self.contract["office0"]["forbidden_for_v2_acceptance"], True)
        self.assertEqual(self.contract["production_acoustics"]["materials"], "OFF")

    def test_manifests_are_strict_and_pre_wer(self):
        paths = [
            (ROOT / "registries/active_asr_a3_v2/replica_scene_inventory.json", validate_scene_inventory, INVENTORY_SCHEMA),
            (ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_scene_manifest.json", validate_scene_manifest, SCENE_MANIFEST_SCHEMA),
            (ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_speech_manifest.json", validate_speech_manifest, SPEECH_MANIFEST_SCHEMA),
        ]
        for path, validator, schema in paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            validator(value)
            self.assertEqual(value["schema_version"], schema)
            if schema == INVENTORY_SCHEMA:
                self.assertFalse(value["asr_run_before_inventory"])
                self.assertFalse(value["wer_run_before_inventory"])
            else:
                self.assertFalse(value["asr_run_before_freeze"])
                self.assertFalse(value["wer_run_before_freeze"])
            invalid = copy.deepcopy(value)
            invalid["unexpected_future_field"] = True
            with self.assertRaises(Exception):
                validator(invalid)

    def test_scene_and_speech_selection_are_held_out(self):
        scene = json.loads((ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_scene_manifest.json").read_text())
        self.assertEqual(scene["selected_scene_ids"], sorted(scene["selected_scene_ids"]))
        self.assertNotIn("replica.office_0", scene["selected_scene_ids"])
        self.assertEqual(sum(len(item["cases"]) for item in scene["scenes"]), 8)
        self.assertEqual({item["category"] for scene_row in scene["scenes"] for item in scene_row["cases"]}, {"near_front_like", "moderate_front_like", "far_front_like", "far_off_axis"})
        self.assertEqual(scene["selection_policy"]["visibility_method"], STATIC_MESH_LOS_METHOD)
        self.assertTrue(scene["visibility_sanity"]["synthetic"]["passed"])
        self.assertTrue(all(item["passed"] for item in scene["visibility_sanity"]["floor_checks"]))
        self.assertTrue(scene["visibility_sanity"]["determinism"]["passed"])
        for scene_row in scene["scenes"]:
            for case in scene_row["cases"]:
                self.assertEqual(case["los_status"], "VERIFIED_GEOMETRIC_LOS_STATIC_MESH")
                self.assertTrue(case["visibility_evidence"]["clear_line_of_sight"])
                self.assertEqual(case["visibility_evidence"]["intersection_count"], 0)
                self.assertEqual(case["visibility_evidence"]["mesh_sha256"], case["scene_asset_sha256"])
        speech = json.loads((ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_speech_manifest.json").read_text())
        self.assertEqual(len(speech["records"]), 24)
        self.assertEqual(speech["speaker_audit"]["selected_historical_speaker_overlap"], [])
        self.assertTrue(speech["speaker_audit"]["speaker_disjoint_from_history"])

    def test_static_mesh_synthetic_sanity(self):
        details = synthetic_sanity_results()
        self.assertTrue(all(item["passed"] for item in details.values() if isinstance(item, dict) and "passed" in item))

    def test_all_nonhistorical_scenes_pass_renderer_construction(self):
        inventory = json.loads((ROOT / "registries/active_asr_a3_v2/replica_scene_inventory.json").read_text())
        self.assertEqual(len(inventory["scenes"]), 18)
        self.assertEqual(sum(row["technical_eligibility"] for row in inventory["scenes"]), 17)
        for row in inventory["scenes"]:
            if row["scene_id"] == "replica.office_0":
                continue
            smoke = row["runtime_scene_load_smoke"]
            self.assertTrue(row["technical_eligibility"])
            self.assertTrue(smoke["audio_sensor_created"])
            self.assertEqual(smoke["effective_sample_rate_hz"], 16000)
            self.assertEqual(smoke["effective_channel_layout"], "binaural")
            self.assertEqual(smoke["effective_channel_count"], 2)
            self.assertFalse(smoke["materials_effective"])
            self.assertEqual(smoke["renderer_construction_status"], "PASS")

    def test_speech_manifest_byte_identity(self):
        path = ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_speech_manifest.json"
        import hashlib
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "9f8fc5e4a62a9b2aee9af0b250517cfb85250fdb24b599d8c0dd818794755c99",
        )


if __name__ == "__main__":
    unittest.main()
