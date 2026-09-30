import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from active_audition.config.loader import load_resolved_config as load_legacy_config
from active_audition.config.v1 import (
    ContractError,
    canonical_json,
    contract_sha256,
    load_resolved_config,
    validate_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_CONFIG = REPO_ROOT / "configs/active_audition/v1/experiment_contract.yaml"
V0_CONFIG = REPO_ROOT / "configs/active_audition/v0_replica_debug.yaml"


class ActiveASRV1ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_resolved_config(str(V1_CONFIG))

    def test_a0_freezes_experiment_boundary(self):
        self.assertEqual(self.contract["contract"], {"namespace": "active-asr", "version": "active-asr-v1.1", "gate": "A0", "state": "FROZEN"})
        self.assertEqual(self.contract["audio"]["render_sample_rate_hz"], 16000)
        self.assertEqual(self.contract["audio"]["asr_sample_rate_hz"], 16000)
        self.assertEqual(
            self.contract["audio"]["sample_rate_qualification_reference"]["render_sample_rate_hz"],
            24000,
        )
        self.assertEqual(
            self.contract["audio"]["sample_rate_qualification_reference"]["scope"],
            "A2_only",
        )
        self.assertEqual(self.contract["audio"]["channel_order"], ["L", "R"])
        self.assertEqual(self.contract["actions"]["kind"], "safe_waypoint_macro_action")
        self.assertEqual(self.contract["actions"]["execution"], ["move", "stop", "listen"])
        self.assertFalse(self.contract["actions"]["primitive_controls_in_action_space"])
        self.assertEqual(
            self.contract["sources"]["target"]["split_policy"],
            {
                "o1": ["dev-clean", "dev-other"],
                "o2": ["test-clean", "test-other"],
                "speaker_disjoint": True,
                "source_registry_freeze": "A3",
            },
        )
        self.assertEqual(self.contract["sources"]["noise"]["dataset"], "MUSAN")
        self.assertEqual(self.contract["asr"]["family"], "LibriSpeech Transformer family")
        self.assertEqual(
            self.contract["asr"]["default_checkpoint"],
            "speechbrain/asr-transformer-transformerlm-librispeech",
        )
        self.assertEqual(self.contract["asr"]["checkpoint_status"], "reference_only_not_A3_frozen")
        self.assertEqual(self.contract["asr"]["revision_decoder_freeze"], "A3")
        self.assertFalse(self.contract["mix"]["normalization"]["per_pose"])
        self.assertEqual(self.contract["mix"]["noise_gain"]["calibration_pose"], "initial")
        self.assertNotIn("attempt", self.contract["seed_policy"]["component_order"])

    def test_hash_is_independent_of_mapping_order(self):
        reordered = json.loads(json.dumps(self.contract))
        reordered["audio"] = {key: reordered["audio"][key] for key in reversed(list(reordered["audio"]))}
        reordered["contract"] = {key: reordered["contract"][key] for key in reversed(list(reordered["contract"]))}
        self.assertEqual(canonical_json(self.contract), canonical_json(reordered))
        self.assertEqual(contract_sha256(self.contract), contract_sha256(reordered))

    def test_changed_frozen_value_is_rejected(self):
        changed = copy.deepcopy(self.contract)
        changed["audio"]["render_sample_rate_hz"] = 24000
        with self.assertRaises(ContractError):
            validate_contract(changed)

    def test_unknown_and_invalid_fields_are_rejected(self):
        unknown = copy.deepcopy(self.contract)
        unknown["future_gate"] = {"enabled": True}
        with self.assertRaisesRegex(ContractError, "unknown config field"):
            validate_contract(unknown)

        invalid = copy.deepcopy(self.contract)
        invalid["sources"]["target"]["split_policy"]["o1"] = ["test-clean"]
        with self.assertRaisesRegex(ContractError, "dev-clean"):
            validate_contract(invalid)

        invalid = copy.deepcopy(self.contract)
        invalid["mix"]["normalization"]["per_pose"] = True
        with self.assertRaisesRegex(ContractError, "per_pose"):
            validate_contract(invalid)

    def test_yaml_serialization_has_stable_identity_without_path_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.yaml"
            second = Path(temp_dir) / "second.yaml"
            first.write_text(yaml.safe_dump(self.contract, sort_keys=True), encoding="utf-8")
            second.write_text(yaml.safe_dump(self.contract, sort_keys=False), encoding="utf-8")
            first_contract = load_resolved_config(str(first))
            second_contract = load_resolved_config(str(second))
        self.assertEqual(contract_sha256(first_contract), contract_sha256(second_contract))

    def test_v1_loader_rejects_legacy_config_and_legacy_loader_still_works(self):
        with self.assertRaises(ContractError):
            load_resolved_config(str(V0_CONFIG))
        legacy = load_legacy_config(str(V0_CONFIG))
        self.assertEqual(legacy["experiment"]["schema_version"], "v0.1")


if __name__ == "__main__":
    unittest.main()
