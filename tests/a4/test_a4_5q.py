import copy
import unittest

from active_audition.a4.qualification import (
    A4QualificationArtifact,
    QualificationError,
)


def _artifact():
    gates = {"G{}".format(index): {"status": "PASS", "evidence": {"count": index}} for index in range(1, 10)}
    return A4QualificationArtifact(
        infrastructure_contract_sha256="1" * 64,
        smoke_manifest_sha256="2" * 64,
        code_head="deadbeef",
        gate_records=gates,
        expected_counts={"rir": 192, "mixture": 384, "asr": 1152},
        valid_counts={"rir": 192, "mixture": 384, "asr": 1152},
        calibration_evidence={"blocks": [{"block_index": 1, "alpha": 0.1}]},
        cache_reconciliation_identity={"record_id": "record"},
        completion_marker_identity={"marker_id": "marker"},
        resource_profile_references={"sha256": "3" * 64},
        asr_diagnostic_references={"summary_sha256": "4" * 64, "record_count": 1152},
        qualification_state="RESOURCE_PROFILE_PENDING_GPU_REPLAY",
    )


class QualificationArtifactTests(unittest.TestCase):
    def test_round_trip_and_identity(self):
        original = _artifact()
        rebuilt = A4QualificationArtifact.from_payload(original.to_payload())
        self.assertEqual(rebuilt.to_payload(), original.to_payload())
        self.assertEqual(rebuilt.artifact_id, original.artifact_id)
        self.assertEqual(rebuilt.artifact_sha256, original.artifact_sha256)

    def test_code_head_is_provenance_not_semantic_identity(self):
        original = _artifact()
        payload = original.to_payload()
        payload["code_head"] = "another-head"
        rebuilt = A4QualificationArtifact.from_payload(payload)
        self.assertEqual(rebuilt.artifact_id, original.artifact_id)

    def test_tampered_gate_status_rejected(self):
        payload = _artifact().to_payload()
        payload["gate_records"]["G7"]["status"] = "FAIL"
        with self.assertRaises(QualificationError):
            A4QualificationArtifact.from_payload(payload)

    def test_tampered_artifact_id_rejected(self):
        payload = _artifact().to_payload()
        payload["artifact_id"] = "a4-qualification-" + "0" * 64
        with self.assertRaises(QualificationError):
            A4QualificationArtifact.from_payload(payload)

    def test_result_dependent_selection_fields_rejected(self):
        payload = _artifact().to_payload()
        payload["gate_records"]["G9"]["evidence"]["best_pose"] = "forbidden"
        with self.assertRaises(QualificationError):
            A4QualificationArtifact.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
