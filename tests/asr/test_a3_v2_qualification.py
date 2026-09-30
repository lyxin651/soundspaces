import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_audition.asr.a3_v2_qualification import (
    V2_CONTRACT_SHA,
    V1_Q4_MANIFEST_SHA,
    _g8,
    _load_v2_inputs,
    _onset,
    _rir_diagnostics,
    _repeatability_summary,
    _scene_case_lookup,
)
from active_audition.asr.contract_v2 import a3_v2_contract_sha256, load_a3_v2_contract
from active_audition.asr.run_qualification import _q4_bridge, _q4_reference_plan


ROOT = Path(__file__).resolve().parents[2]


class _FakeASROutput:
    def __init__(self, hypothesis):
        self.hypothesis = hypothesis

    def to_dict(self):
        return {"hypothesis": self.hypothesis}


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

    def test_q4_reference_plan_is_record_major_by_frontend(self):
        manifest = json.loads((ROOT / "registries/active_asr_a3/q4_manifest.json").read_text(encoding="utf-8"))
        references, metadata = _q4_reference_plan(manifest)
        records = manifest["records"]
        expected_utterances = [
            record["speech"]["utterance_id"]
            for record in records
            for _frontend in ("mean_lr", "fixed_L", "fixed_R")
        ]
        expected_frontends = [
            frontend
            for _record in records
            for frontend in ("mean_lr", "fixed_L", "fixed_R")
        ]
        self.assertEqual(len(references), len(records) * 3)
        self.assertEqual([item["utterance_id"] for item in references], expected_utterances)
        self.assertEqual([item["utterance_id"] for item in metadata], expected_utterances)
        self.assertEqual([item["frontend"] for item in metadata], expected_frontends)

    def test_q4_v2_matches_v1_logic_and_restores_pose_evidence(self):
        manifest = json.loads((ROOT / "registries/active_asr_a3/q4_manifest.json").read_text(encoding="utf-8"))
        utterances = []
        for record in manifest["records"]:
            utterance_id = record["speech"]["utterance_id"]
            if utterance_id not in utterances:
                utterances.append(utterance_id)
        references_by_index = {index + 1: "HYPOTHESIS_{}".format(index) for index, _ in enumerate(utterances)}
        references_by_utterance = {
            record["speech"]["utterance_id"]: record["speech"]["normalized_transcript"]
            for record in manifest["records"]
        }
        waveform_index = {utterance_id: index + 1 for index, utterance_id in enumerate(utterances)}

        def fake_decode(_repo, _source_root, speech):
            return np.asarray([waveform_index[speech["utterance_id"]]], dtype=np.float32), None

        def fake_transcribe(_adapter, waveforms, _frontend):
            return [_FakeASROutput(references_by_index[int(value[0])]) for value in waveforms]

        def fake_rir(_waveform, _rir):
            return _waveform

        def fake_resample(waveform, _source_rate, _target_rate):
            return waveform

        def fake_frontend(waveform, _frontend):
            return waveform

        rir = {
            (str(record["rir_case"]["case_id"]), rate): np.ones(1, dtype=np.float32)
            for record in manifest["records"]
            for rate in (16000, 24000)
        }
        v1_contract = {"qualification": {"manifests": {"q4": {"path": "registries/active_asr_a3/q4_manifest.json"}}}}
        with patch("active_audition.asr.run_qualification._decode_speech", side_effect=fake_decode), patch(
            "active_audition.asr.run_qualification._transcribe_many", side_effect=fake_transcribe
        ), patch("active_audition.asr.run_qualification.convolve_binaural", side_effect=fake_rir), patch(
            "active_audition.asr.run_qualification.resample_array", side_effect=fake_resample
        ), patch("active_audition.asr.run_qualification.apply_frontend", side_effect=fake_frontend):
            v1 = _q4_bridge(ROOT, ROOT, manifest, rir, None)
            v2 = _g8(ROOT, ROOT, None, v1_contract, rir)

        self.assertEqual(v2["path_a"], v1["path_a"])
        self.assertEqual(v2["path_b"], v1["path_b"])
        self.assertEqual(v2["paired"], v1["paired"])
        self.assertEqual(v2["pose_ordering"], v1["pose_ordering"])
        for row in v2["paired"]:
            expected_reference = references_by_utterance[row["utterance_id"]]
            self.assertEqual(row["path_a"]["utterance_id"], row["utterance_id"])
            self.assertEqual(row["path_b"]["utterance_id"], row["utterance_id"])
            self.assertEqual(row["path_a"]["reference"], expected_reference)
            self.assertEqual(row["path_b"]["reference"], expected_reference)
        expected_groups = {
            (utterance_id, frontend)
            for utterance_id in utterances
            for frontend in ("mean_lr", "fixed_L", "fixed_R")
        }
        actual_groups = {(item["utterance_id"], item["frontend"]) for item in v2["pose_ordering"]}
        self.assertEqual(actual_groups, expected_groups)
        self.assertEqual(len(v2["pose_ordering"]), 12)

        self.assertEqual(v2["provenance"]["manifest_sha256"], V1_Q4_MANIFEST_SHA)

    def test_repeatability_rejects_numeric_or_shape_mismatch(self):
        first = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.assertEqual(_repeatability_summary(first, first.copy())["status"], "PASS")
        changed = first.copy()
        changed[0, 0] += 1.0e-4
        mismatch = _repeatability_summary(first, changed)
        self.assertEqual(mismatch["status"], "FAIL")
        self.assertFalse(mismatch["array_sha_equal"])
        self.assertGreater(mismatch["max_abs_difference"], 0.0)
        shape_mismatch = _repeatability_summary(first, first[:1])
        self.assertEqual(shape_mismatch["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
