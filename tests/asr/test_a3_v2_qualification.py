import json
import unittest
from pathlib import Path

import numpy as np

from active_audition.asr.a3_v2_qualification import (
    V2_CONTRACT_SHA,
    _load_v2_inputs,
    _onset,
    _rir_diagnostics,
    _scene_case_lookup,
)
from active_audition.asr.contract_v2 import a3_v2_contract_sha256, load_a3_v2_contract


ROOT = Path(__file__).resolve().parents[2]


class A3V2QualificationTests(unittest.TestCase):
    def test_frozen_inputs_are_bound_before_phase_b(self):
        contract, inventory, scene_manifest, speech_manifest, digest = _load_v2_inputs(
            ROOT, "configs/active_audition/v1/asr_contract_v2.yaml"
        )
        self.assertEqual(digest, V2_CONTRACT_SHA)
        self.assertEqual(len(inventory["scenes"]), 18)
        self.assertEqual(len(_scene_case_lookup(scene_manifest)), 8)
        self.assertEqual(len(speech_manifest["records"]), 24)
        self.assertEqual(a3_v2_contract_sha256(contract), V2_CONTRACT_SHA)

    def test_onset_and_rir_diagnostic_are_deterministic_and_raw(self):
        value = np.zeros((32, 2), dtype=np.float32)
        value[5, 0] = 1.0
        value[7, 1] = 0.5
        self.assertEqual(_onset(value[:, 0])["sample"], 5)
        self.assertEqual(_onset(value[:, 1])["sample"], 7)
        first = _rir_diagnostics(value)
        second = _rir_diagnostics(value.copy())
        self.assertEqual(first, second)
        self.assertEqual(first["channel_order"], ["L", "R"])
        self.assertTrue(first["finite"])
        self.assertTrue(first["no_normalization"])
        self.assertEqual(first["channels"]["L"]["sum_square_energy"], 1.0)
        self.assertEqual(first["channels"]["R"]["sum_square_energy"], 0.25)

    def test_phase_a_manifests_are_not_wer_selected(self):
        path = ROOT / "registries/active_asr_a3_v2/a3_v2_realistic_domain_speech_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        forbidden = set(value["selection_policy"]["forbidden_inputs"])
        self.assertTrue({"ASR", "WER", "RIR", "energy", "Oracle"}.issubset(forbidden))
        self.assertFalse(value["asr_run_before_freeze"])


if __name__ == "__main__":
    unittest.main()
