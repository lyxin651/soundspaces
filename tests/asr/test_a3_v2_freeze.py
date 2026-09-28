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
        self.assertEqual(a3_v2_contract_sha256(self.contract), "afc4528f815bf2cc4a8b705eacedc807b5f8ae541b6a17892a687769f2acbfd2")
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
        self.assertEqual(scene["selected_scene_ids"], ["replica.apartment_0", "replica.apartment_1"])
        self.assertEqual(sum(len(item["cases"]) for item in scene["scenes"]), 8)
        self.assertEqual({item["category"] for scene_row in scene["scenes"] for item in scene_row["cases"]}, {"near_front_like", "moderate_front_like", "far_front_like", "far_off_axis"})
        speech = json.loads((ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_speech_manifest.json").read_text())
        self.assertEqual(len(speech["records"]), 24)
        self.assertEqual(speech["speaker_audit"]["selected_historical_speaker_overlap"], [])
        self.assertTrue(speech["speaker_audit"]["speaker_disjoint_from_history"])


if __name__ == "__main__":
    unittest.main()
