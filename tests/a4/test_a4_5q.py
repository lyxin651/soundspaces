from types import SimpleNamespace
import unittest

from active_audition.a4.qualification import (
    A4QualificationArtifact,
    PENDING_RESOURCE_PROFILE,
    QualificationError,
    RESOURCE_PROFILE_ALGORITHM_IDENTITY,
    RESOURCE_PROFILE_SCHEMA_VERSION,
)
from active_audition.a4.identity import identity_sha256
from scripts.qualify_a4 import _validate_replay_profile


def _semantic_payload(rtf=0.02, allocated=100, asr_bytes=300):
    return {
        "schema_version": RESOURCE_PROFILE_SCHEMA_VERSION,
        "algorithm_identity": RESOURCE_PROFILE_ALGORITHM_IDENTITY,
        "rir": {"count": 192, "total_render_seconds": 1.0, "mean_render_seconds": 0.1, "median_render_seconds": 0.1, "p95_render_seconds": 0.2, "min_render_seconds": 0.01, "max_render_seconds": 0.3},
        "production_asr": {"expected": 1152, "new_decodes": 1152, "total_audio_seconds": 10.0, "decode_wall_seconds": 0.2, "rtf": rtf},
        "gpu_replay": {"expected": 1152, "decoded": 1152, "batch_size": 4, "frontends": ["mean_lr", "fixed_L", "fixed_R"], "total_audio_seconds": 10.0, "wall_seconds": 0.2, "rtf": 0.02, "hypothesis_mismatch_count_against_cached_diagnostic": 0, "cache_write_count": 0, "peak_memory_allocated_bytes": allocated, "peak_memory_reserved_bytes": 200, "device": "cuda:0", "device_name": "NVIDIA GeForce RTX 4090", "a3_v2_contract_sha256": "7" * 64, "final_manifest_id": "cache-manifest-1", "final_manifest_sha256": "8" * 64},
        "effective_cache_footprint": {"effective_rir_cache_bytes": 100, "effective_mixture_v3_cache_bytes": 200, "effective_asr_cache_bytes": asr_bytes, "effective_cache_total_bytes": 300 + asr_bytes},
    }


def _artifact(state="RESOURCE_PROFILE_PENDING_GPU_REPLAY", semantic=None, references=None):
    semantic = semantic or _semantic_payload()
    references = references or {"path": "runs/resource.json", "sha256": "3" * 64, "run_evidence_bytes_snapshot": 1000}
    gates = {
        "G{}".format(index): {
            "status": PENDING_RESOURCE_PROFILE if state == "RESOURCE_PROFILE_PENDING_GPU_REPLAY" and index == 9 else "PASS",
            "evidence": {"count": index},
        }
        for index in range(1, 10)
    }
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
        resource_profile_identity={"schema_version": RESOURCE_PROFILE_SCHEMA_VERSION, "semantic_sha256": identity_sha256(semantic)},
        resource_profile_semantic_payload=semantic,
        resource_profile_references=references,
        asr_diagnostic_references={"summary_sha256": "4" * 64, "record_count": 1152},
        qualification_state=state,
    )


def _replay_profile(**overrides):
    profile = {
        "schema_version": "active-asr-a4-asr-resource-profile-replay-v1",
        "profile_status": "DIAGNOSTIC_RESOURCE_REPLAY",
        "semantic_cache_authority": "NOT_SEMANTIC_CACHE_AUTHORITY",
        "expected": 1152,
        "decoded": 1152,
        "batch_size": 4,
        "frontends": ["mean_lr", "fixed_L", "fixed_R"],
        "total_audio_seconds": 17194.305749999967,
        "wall_seconds": 380.0,
        "rtf": 0.0221,
        "hypothesis_mismatch_count_against_cached_diagnostic": 0,
        "peak_memory_allocated_bytes": 100,
        "peak_memory_reserved_bytes": 200,
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 4090",
        "cuda_visible_devices": "5",
        "torch": "2.5.1+cu121",
        "torch_cuda": "12.1",
        "python": "3.11.16",
        "a3_v2_contract_sha256": "70864c814a55db5d184ef8a6835b65cb564c4c90fc86112f8705b7ecf6df1ffe",
        "final_manifest_id": "cache-manifest-1",
        "final_manifest_sha256": "8" * 64,
        "cache_write_count": 0,
    }
    profile.update(overrides)
    return profile


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

    def test_resource_reporting_provenance_is_not_semantic_identity(self):
        original = _artifact("SERVER_RUN_COMPLETE_PENDING_REVIEW")
        payload = original.to_payload()
        payload["resource_profile_references"] = {"path": "runs/another-report.json", "sha256": "4" * 64, "run_evidence_bytes_snapshot": 999999}
        rebuilt = A4QualificationArtifact.from_payload(payload)
        self.assertEqual(rebuilt.artifact_id, original.artifact_id)
        self.assertEqual(rebuilt.artifact_sha256, original.artifact_sha256)

    def test_resource_semantic_change_changes_qualification_identity(self):
        original = _artifact("SERVER_RUN_COMPLETE_PENDING_REVIEW")
        semantic = _semantic_payload(rtf=0.03)
        payload = original.to_payload()
        payload["resource_profile_semantic_payload"] = semantic
        payload["resource_profile_identity"] = {"schema_version": RESOURCE_PROFILE_SCHEMA_VERSION, "semantic_sha256": identity_sha256(semantic)}
        payload["artifact_id"] = ""
        payload["artifact_sha256"] = ""
        rebuilt = A4QualificationArtifact.from_payload(payload)
        self.assertNotEqual(rebuilt.artifact_id, original.artifact_id)
        self.assertNotEqual(rebuilt.artifact_sha256, original.artifact_sha256)

    def test_each_resource_semantic_authority_changes_identity(self):
        original = _artifact("SERVER_RUN_COMPLETE_PENDING_REVIEW")
        variants = (
            _semantic_payload(rtf=0.03),
            _semantic_payload(allocated=101),
            _semantic_payload(asr_bytes=301),
        )
        variants[-1]["gpu_replay"]["final_manifest_id"] = "cache-manifest-2"
        for semantic in variants:
            payload = original.to_payload()
            payload["resource_profile_semantic_payload"] = semantic
            payload["resource_profile_identity"] = {"schema_version": RESOURCE_PROFILE_SCHEMA_VERSION, "semantic_sha256": identity_sha256(semantic)}
            payload["artifact_id"] = ""
            payload["artifact_sha256"] = ""
            rebuilt = A4QualificationArtifact.from_payload(payload)
            self.assertNotEqual(rebuilt.artifact_sha256, original.artifact_sha256)

    def test_resource_semantic_payload_hash_tamper_rejected(self):
        payload = _artifact("SERVER_RUN_COMPLETE_PENDING_REVIEW").to_payload()
        payload["resource_profile_semantic_payload"]["production_asr"]["rtf"] = 0.03
        with self.assertRaises(QualificationError):
            A4QualificationArtifact.from_payload(payload)

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

    def test_pending_state_requires_g9_pending(self):
        artifact = _artifact()
        self.assertEqual(artifact.gate_records["G9"]["status"], PENDING_RESOURCE_PROFILE)
        self.assertEqual(artifact.qualification_state, "RESOURCE_PROFILE_PENDING_GPU_REPLAY")

    def test_pending_state_with_g9_pass_rejected(self):
        payload = _artifact().to_payload()
        payload["gate_records"]["G9"]["status"] = "PASS"
        with self.assertRaises(QualificationError):
            A4QualificationArtifact.from_payload(payload)

    def test_final_state_requires_all_gates_pass(self):
        artifact = _artifact("SERVER_RUN_COMPLETE_PENDING_REVIEW")
        self.assertTrue(all(record["status"] == "PASS" for record in artifact.gate_records.values()))

    def test_final_state_with_pending_g9_rejected(self):
        payload = _artifact("SERVER_RUN_COMPLETE_PENDING_REVIEW").to_payload()
        payload["gate_records"]["G9"]["status"] = PENDING_RESOURCE_PROFILE
        with self.assertRaises(QualificationError):
            A4QualificationArtifact.from_payload(payload)

    def test_valid_replay_profile(self):
        final = SimpleNamespace(manifest_id="cache-manifest-1", manifest_sha256="8" * 64)
        replay = _replay_profile()
        validated = _validate_replay_profile(replay, final, {"total_audio_seconds": 17194.305749999967})
        self.assertEqual(validated["peak_memory_allocated_bytes"], 100)

    def test_replay_missing_peak_memory_rejected(self):
        final = SimpleNamespace(manifest_id="cache-manifest-1", manifest_sha256="8" * 64)
        replay = _replay_profile()
        del replay["peak_memory_allocated_bytes"]
        with self.assertRaises(RuntimeError):
            _validate_replay_profile(replay, final, {"total_audio_seconds": 17194.305749999967})

    def test_replay_reserved_below_allocated_rejected(self):
        final = SimpleNamespace(manifest_id="cache-manifest-1", manifest_sha256="8" * 64)
        with self.assertRaises(RuntimeError):
            _validate_replay_profile(_replay_profile(peak_memory_reserved_bytes=99), final, {"total_audio_seconds": 17194.305749999967})

    def test_replay_mismatch_rejected(self):
        final = SimpleNamespace(manifest_id="cache-manifest-1", manifest_sha256="8" * 64)
        with self.assertRaises(RuntimeError):
            _validate_replay_profile(_replay_profile(hypothesis_mismatch_count_against_cached_diagnostic=1), final, {"total_audio_seconds": 17194.305749999967})

    def test_replay_cache_write_rejected(self):
        final = SimpleNamespace(manifest_id="cache-manifest-1", manifest_sha256="8" * 64)
        with self.assertRaises(RuntimeError):
            _validate_replay_profile(_replay_profile(cache_write_count=1), final, {"total_audio_seconds": 17194.305749999967})

    def test_replay_a3_identity_rejected(self):
        final = SimpleNamespace(manifest_id="cache-manifest-1", manifest_sha256="8" * 64)
        with self.assertRaises(RuntimeError):
            _validate_replay_profile(_replay_profile(a3_v2_contract_sha256="9" * 64), final, {"total_audio_seconds": 17194.305749999967})

    def test_replay_identity_rejected(self):
        final = SimpleNamespace(manifest_id="cache-manifest-other", manifest_sha256="8" * 64)
        with self.assertRaises(RuntimeError):
            _validate_replay_profile(_replay_profile(), final, {"total_audio_seconds": 17194.305749999967})


if __name__ == "__main__":
    unittest.main()
