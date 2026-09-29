import copy
import json
import unittest
from pathlib import Path

from active_audition.a4.noise_audit import NoiseParentAuditRecord, NoiseAuditError
from active_audition.a4.smoke_manifest import EngineeringSmokeManifest, EngineeringSmokeManifestError


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "registries/active_asr_a4/a4_engineering_noise_parent_audit_v1.json"
SMOKE = ROOT / "configs/active_audition/v1/a4_engineering_smoke_manifest.json"


class A44ManualAuditTests(unittest.TestCase):
    def test_audit_records_round_trip_and_user_decisions(self):
        payload = json.loads(AUDIT.read_text(encoding="utf-8"))
        records = [NoiseParentAuditRecord.from_payload(item) for item in payload["records"]]
        self.assertEqual(
            {record.parent_recording_id for record in records},
            {
                "noise/free-sound/noise-free-sound-0048",
                "noise/free-sound/noise-free-sound-0270",
                "noise/free-sound/noise-free-sound-0002",
                "noise/free-sound/noise-free-sound-0020",
            },
        )
        selected = [record for record in records if record.selected_for_a4_smoke]
        self.assertEqual(
            {record.parent_recording_id for record in selected},
            {"noise/free-sound/noise-free-sound-0002", "noise/free-sound/noise-free-sound-0020"},
        )
        for record in records:
            self.assertEqual(NoiseParentAuditRecord.from_payload(record.to_payload()).to_payload(), record.to_payload())

    def test_audit_tamper_fails_closed(self):
        payload = json.loads(AUDIT.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(payload["records"][2])
        tampered["speech_leakage"] = "FLAGGED"
        with self.assertRaises(NoiseAuditError):
            NoiseParentAuditRecord.from_payload(tampered)
        tampered = copy.deepcopy(payload["records"][2])
        tampered["source_file_sha256"] = "0" * 64
        with self.assertRaises(NoiseAuditError):
            NoiseParentAuditRecord.from_payload(tampered)

    def test_manifest_v2_requires_audited_selected_noise(self):
        payload = json.loads(SMOKE.read_text(encoding="utf-8"))
        manifest = EngineeringSmokeManifest.from_payload(payload)
        self.assertEqual(manifest.schema_version, "active-asr-a4-engineering-smoke-manifest-v2")
        for block in manifest.blocks:
            self.assertTrue(block["noise_audit_record"]["selected_for_a4_smoke"])
            self.assertEqual(
                [block["noise_audit_record"][field] for field in (
                    "speech_leakage",
                    "strong_reverberation",
                    "indoor_localized_source_compatibility",
                )],
                ["PASS", "PASS", "PASS"],
            )
        tampered = copy.deepcopy(payload)
        tampered["blocks"][0]["noise_audit_record"]["speech_leakage"] = "NOT_EVALUATED"
        with self.assertRaises(EngineeringSmokeManifestError):
            EngineeringSmokeManifest.from_payload(tampered)


if __name__ == "__main__":
    unittest.main()
