import ast
import copy
import json
import unittest
from pathlib import Path

from active_audition.a4.cache_resume import CacheExpectedManifest
from active_audition.a4.contract import contract_sha256, load_contract, validate_contract
from active_audition.a4.smoke_manifest import EngineeringSmokeManifest, EngineeringSmokeManifestError


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/active_audition/v1/a4_infrastructure_contract.yaml"
CONTRACT_SHA = ROOT / "configs/active_audition/v1/a4_infrastructure_contract.sha256"
SMOKE = ROOT / "configs/active_audition/v1/a4_engineering_smoke_manifest.json"
CACHE_PLAN = ROOT / "configs/active_audition/v1/a4_engineering_smoke_cache_expected_manifest.json"


class A44PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(SMOKE.read_text(encoding="utf-8"))
        cls.manifest = EngineeringSmokeManifest.from_payload(cls.payload)

    def test_contract_is_frozen_and_sha_matches_sidecar(self):
        contract = load_contract(str(CONTRACT), require_frozen=True)
        self.assertEqual(contract["contract"]["state"], "FROZEN")
        self.assertEqual(contract_sha256(contract), CONTRACT_SHA.read_text(encoding="utf-8").strip())
        validate_contract(contract, require_frozen=True)
        self.assertEqual(self.manifest.infrastructure_contract_sha256, contract_sha256(contract))

    def test_exact_two_blocks_four_episodes_and_selection_boundary(self):
        self.assertEqual(len(self.manifest.blocks), 2)
        for block in self.manifest.blocks:
            self.assertTrue(block["geometry_record"]["scene_id"].startswith("replica."))
            self.assertEqual(len(block["episodes"]), 4)
            self.assertEqual([item["role"] for item in block["episodes"]], ["selection", "selection", "evaluation", "evaluation"])
            self.assertEqual(len(block["speech_sources"]), 4)
            self.assertEqual(len(block["poses"]), 48)
            self.assertEqual(len(block["sampler_output"]["probe_attempts"]), 64)
            self.assertEqual(len(block["sampler_output"]["yaw_plans"]), 48)
            self.assertTrue(block["noise_audit"]["engineering_only"])

    def test_round_trip_and_identity_are_byte_stable(self):
        restored = EngineeringSmokeManifest.from_payload(self.manifest.to_payload())
        self.assertEqual(restored.to_payload(), self.manifest.to_payload())
        self.assertEqual(
            SMOKE.read_bytes(),
            json.dumps(self.manifest.to_payload(), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n",
        )

    def test_tamper_and_unknown_fields_fail_closed(self):
        raw = copy.deepcopy(self.payload)
        raw["unexpected"] = True
        with self.assertRaises(EngineeringSmokeManifestError):
            EngineeringSmokeManifest.from_payload(raw)
        raw = copy.deepcopy(self.payload)
        raw["blocks"][0]["block_record"]["block_id"] = raw["blocks"][1]["block_record"]["block_id"]
        with self.assertRaises(Exception):
            EngineeringSmokeManifest.from_payload(raw)
        raw = copy.deepcopy(self.payload)
        raw["manifest_sha256"] = "0" * 64
        with self.assertRaises(EngineeringSmokeManifestError):
            EngineeringSmokeManifest.from_payload(raw)

    def test_engineering_only_and_o2_exclusion_are_explicit(self):
        self.assertTrue(self.manifest.engineering_only)
        self.assertTrue(self.manifest.o2_exclusion["excluded"])
        self.assertEqual(tuple(self.manifest.expected_frontends), ("mean_lr", "fixed_L", "fixed_R"))
        self.assertIn("WER", self.manifest.forbidden_result_dependent_selection)
        self.assertIn("Oracle", self.manifest.forbidden_result_dependent_selection)

    def test_future_cache_plan_is_strict_and_has_deterministic_counts(self):
        cache = CacheExpectedManifest.from_payload(json.loads(CACHE_PLAN.read_text(encoding="utf-8")))
        self.assertEqual(dict(cache.expected_counts), {"rir": 192, "mixture": 384, "asr": 1152})
        self.assertEqual(cache.manifest_id, json.loads(CACHE_PLAN.read_text(encoding="utf-8"))["manifest_id"])

    def test_smoke_manifest_module_is_pure(self):
        for name in ("smoke_manifest.py", "smoke_preparation.py"):
            source = (ROOT / "active_audition/a4" / name).read_text(encoding="utf-8")
            imported = []
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
            self.assertFalse(any(name.startswith("habitat") for name in imported))
            self.assertFalse(any(name.startswith("speechbrain") for name in imported))


if __name__ == "__main__":
    unittest.main()
