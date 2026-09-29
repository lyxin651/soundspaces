import copy
import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from active_audition.a4.cache import (
    A2_ACOUSTIC_CONTRACT_SHA256,
    A3_ASR_CONTRACT_SHA256,
    ASR_RESULT_SCHEMA_VERSION,
    AsrCacheKey,
    CacheStore,
    INVALID_CORRUPT,
    MISSING_METADATA,
    MISSING_PAYLOAD,
    MixtureCacheKey,
    NOISE_SOURCE_TIME_IDENTITY,
    PAYLOAD_HASH_MISMATCH,
    RirCacheKey,
)
from active_audition.a4.cache_resume import (
    CACHE_COMPLETION_MARKER_SCHEMA_VERSION,
    CACHE_EXPECTED_MANIFEST_SCHEMA_VERSION,
    COMPLETION_MARKER_VALID,
    INVALID_COMPLETION_MARKER,
    REBUILD_CORRUPT,
    REBUILD_MISSING,
    REUSE_VALID,
    STALE_COMPLETION_MARKER,
    CacheExpectedManifest,
    CacheResumeError,
    completion_marker_path,
    read_completion_marker,
    reconcile_cache_manifest,
    write_completion_marker,
)
from active_audition.a4.identity import stable_id


def _rir_key(source_x=1.0):
    return RirCacheKey(
        scene_resource_identities={
            "mesh_sha256": "a" * 64,
            "navmesh_sha256": "b" * 64,
            "stage_sha256": "c" * 64,
        },
        acoustic_contract_identity=A2_ACOUSTIC_CONTRACT_SHA256,
        renderer_algorithm_identity="SoundSpaces2_HabitatSim0.2.2_RLRAudioPropagation-v1",
        materials_policy="OFF",
        source_world_transform={"position_xyz": [source_x, 1.5, -2.0], "yaw_deg": 0.0},
        receiver_sensor_transform={"position_xyz": [0.0, 1.5, 0.0], "yaw_deg": 45.0},
        receiver_yaw_deg=45.0,
        replicate_identity="deterministic-render-replicate-0",
    )


def _mixture_key(rir_key):
    return MixtureCacheKey(
        target_rir_cache_key=rir_key.cache_key,
        noise_rir_cache_key=_rir_key(4.0).cache_key,
        target_dry_waveform_sha256="1" * 64,
        noise_segment_payload_sha256="2" * 64,
        noise_segment_identity=stable_id("noise-segment", {"fixture": "resume-noise"}),
        noise_source_time_identity=NOISE_SOURCE_TIME_IDENTITY,
        calibration_artifact_identity=stable_id("calibration", {"fixture": "resume-calibration"}),
        global_gain_identity=stable_id("global-gain", {"fixture": "resume-gain"}),
        timeline_identity=stable_id("receiver-timeline", {"fixture": "resume-timeline"}),
        mixer_contract_identity=stable_id("mixture-contract", {"fixture": "resume-mixer"}),
    )


def _asr_key(frontend="mean_lr"):
    return AsrCacheKey(
        mono_payload_sha256="3" * 64,
        frontend=frontend,
        model_identity={"repo_id": "speechbrain/model", "revision": "r1", "weights_sha256": "4" * 64},
        language_model_identity={"name": "lm", "sha256": "5" * 64},
        tokenizer_identity={"name": "spm", "sha256": "6" * 64},
        decoder_identity={"beam_size": 66, "temperature": 1.15},
        precision_runtime_identity={"precision": "float32", "torch": "2.5.1", "device": "cuda:0"},
        asr_contract_identity=A3_ASR_CONTRACT_SHA256,
    )


def _asr_payload(key):
    return {
        "schema_version": ASR_RESULT_SCHEMA_VERSION,
        "hypothesis": "resume fixture",
        "score": -1.0,
        "score_semantics": "raw_decoder_sequence_score_not_confidence_or_probability",
        "input_waveform_sha256": key.mono_payload_sha256,
        "frontend": key.frontend,
        "model_identity": key.model_identity,
        "decoder_identity": key.decoder_identity,
        "asr_contract_identity": key.asr_contract_identity,
        "raw_decoder_metadata": {"token_count": 2},
    }


class CacheResumeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = CacheStore(self.tempdir.name)
        self.rir = _rir_key()
        self.rir_extra = _rir_key(2.0)
        self.mixture = _mixture_key(self.rir)
        self.asr = _asr_key()
        self.array = np.asarray([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32)
        self.manifest = CacheExpectedManifest(
            infrastructure_contract_sha256="e" * 64,
            expected_rir_keys=(self.rir,),
            expected_mixture_keys=(self.mixture,),
            expected_asr_keys=(self.asr,),
            logical_labels={self.rir.cache_key: "target-rir"},
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _populate_all(self, include_extra=False):
        self.store.write_rir(self.rir, self.array)
        self.store.write_mixture(self.mixture, self.array)
        self.store.write_asr(self.asr, _asr_payload(self.asr), (2,))
        if include_extra:
            self.store.write_rir(self.rir_extra, self.array)

    def test_manifest_is_canonical_immutable_and_strict(self):
        reordered = CacheExpectedManifest(
            infrastructure_contract_sha256="e" * 64,
            expected_rir_keys=(self.rir,),
            expected_mixture_keys=(self.mixture,),
            expected_asr_keys=(self.asr,),
            logical_labels={self.rir.cache_key: "target-rir"},
        )
        self.assertEqual(self.manifest.manifest_id, reordered.manifest_id)
        self.assertEqual(self.manifest.to_payload(), reordered.to_payload())
        self.assertEqual(
            CacheExpectedManifest.from_payload(self.manifest.to_payload()).to_payload(),
            self.manifest.to_payload(),
        )
        raw = self.manifest.to_payload()
        raw["logical_labels"][self.rir.cache_key] = "changed-label"
        self.assertEqual(CacheExpectedManifest.from_payload(raw).manifest_id, self.manifest.manifest_id)
        with self.assertRaises(CacheResumeError):
            CacheExpectedManifest(
                infrastructure_contract_sha256="e" * 64,
                expected_rir_keys=(self.rir, self.rir),
            )
        raw = self.manifest.to_payload()
        raw["expected_counts"]["rir"] = 2
        with self.assertRaises(CacheResumeError):
            CacheExpectedManifest.from_payload(raw)
        raw = self.manifest.to_payload()
        raw["expected_rir_keys"] = [self.mixture.to_payload()]
        with self.assertRaises(CacheResumeError):
            CacheExpectedManifest.from_payload(raw)
        raw = self.manifest.to_payload()
        raw["unexpected"] = True
        with self.assertRaises(CacheResumeError):
            CacheExpectedManifest.from_payload(raw)

    def test_empty_cache_rebuilds_all_and_does_not_write_marker(self):
        record = reconcile_cache_manifest(self.manifest, self.store)
        self.assertFalse(record.complete)
        self.assertEqual(record.rebuild_keys, tuple(entry.cache_key for entry in record.entries))
        self.assertEqual(dict(record.missing_counts), {"rir": 1, "mixture": 1, "asr": 1})
        self.assertEqual(dict(record.corrupt_counts), {"rir": 0, "mixture": 0, "asr": 0})
        self.assertTrue(all(entry.status == REBUILD_MISSING for entry in record.entries))
        self.assertTrue(all(entry.reason == MISSING_METADATA for entry in record.entries))
        with self.assertRaises(CacheResumeError):
            write_completion_marker(self.store, self.manifest)

    def test_clean_cache_reuses_all_and_marker_round_trips(self):
        self._populate_all()
        first = reconcile_cache_manifest(self.manifest, self.store)
        self.assertTrue(first.complete)
        self.assertTrue(all(entry.status == REUSE_VALID for entry in first.entries))
        marker = write_completion_marker(self.store, self.manifest)
        self.assertEqual(marker.schema_version, CACHE_COMPLETION_MARKER_SCHEMA_VERSION)
        self.assertEqual(read_completion_marker(self.store, self.manifest).to_payload(), marker.to_payload())
        second = reconcile_cache_manifest(self.manifest, self.store)
        self.assertTrue(second.complete)
        self.assertEqual(second.completion_marker_status, COMPLETION_MARKER_VALID)
        self.assertEqual(first.record_id, second.record_id)
        self.assertEqual(first.record_sha256, second.record_sha256)
        self.assertEqual(first.identity_payload(), second.identity_payload())

    def test_interrupted_run_reuses_valid_and_orders_only_rebuilds(self):
        self.store.write_rir(self.rir, self.array)
        record = reconcile_cache_manifest(self.manifest, self.store)
        self.assertEqual([entry.status for entry in record.entries], [REUSE_VALID, REBUILD_MISSING, REBUILD_MISSING])
        self.assertEqual([entry.layer for entry in record.entries], ["rir", "mixture", "asr"])
        self.assertEqual(record.rebuild_keys, (self.mixture.cache_key, self.asr.cache_key))
        self.assertEqual(record.reasons_by_cache_key[self.mixture.cache_key], MISSING_METADATA)

    def test_missing_and_corrupt_entries_preserve_exact_reasons(self):
        self._populate_all()
        write_completion_marker(self.store, self.manifest)
        (self.store.entry_dir(self.rir) / "payload.npy").unlink()
        result = reconcile_cache_manifest(self.manifest, self.store)
        rir_entry = next(entry for entry in result.entries if entry.layer == "rir")
        self.assertEqual((rir_entry.status, rir_entry.reason), (REBUILD_CORRUPT, MISSING_PAYLOAD))
        self.assertFalse(result.complete)
        self.assertEqual(result.completion_marker_status, STALE_COMPLETION_MARKER)

        self.store.write_rir(self.rir, self.array)
        self.store.write_mixture(self.mixture, self.array)
        self.store.write_asr(self.asr, _asr_payload(self.asr), (2,))
        path = self.store.entry_dir(self.mixture) / "payload.npy"
        path.write_bytes(path.read_bytes() + b"corrupt")
        result = reconcile_cache_manifest(self.manifest, self.store)
        mixture_entry = next(entry for entry in result.entries if entry.layer == "mixture")
        self.assertEqual((mixture_entry.status, mixture_entry.reason), (REBUILD_CORRUPT, PAYLOAD_HASH_MISMATCH))

        self.store.write_mixture(self.mixture, self.array)
        path = self.store.entry_dir(self.asr) / "payload.json"
        path.write_bytes(path.read_bytes() + b"corrupt")
        result = reconcile_cache_manifest(self.manifest, self.store)
        asr_entry = next(entry for entry in result.entries if entry.layer == "asr")
        self.assertEqual((asr_entry.status, asr_entry.reason), (REBUILD_CORRUPT, PAYLOAD_HASH_MISMATCH))

    def test_marker_never_bypasses_revalidation(self):
        self._populate_all()
        first = reconcile_cache_manifest(self.manifest, self.store)
        write_completion_marker(self.store, self.manifest)
        (self.store.entry_dir(self.asr) / "payload.json").unlink()
        result = reconcile_cache_manifest(self.manifest, self.store)
        self.assertFalse(result.complete)
        self.assertEqual(result.completion_marker_status, STALE_COMPLETION_MARKER)
        self.assertEqual(
            next(entry for entry in result.entries if entry.layer == "asr").reason,
            MISSING_PAYLOAD,
        )

    def test_manifest_or_contract_mutation_invalidates_old_marker(self):
        self._populate_all(include_extra=True)
        first = reconcile_cache_manifest(self.manifest, self.store)
        write_completion_marker(self.store, self.manifest)

        changed_manifest = CacheExpectedManifest(
            infrastructure_contract_sha256=self.manifest.infrastructure_contract_sha256,
            expected_rir_keys=(self.rir, self.rir_extra),
            expected_mixture_keys=(self.mixture,),
            expected_asr_keys=(self.asr,),
        )
        changed = reconcile_cache_manifest(changed_manifest, self.store)
        self.assertTrue(changed.complete)
        self.assertEqual(changed.completion_marker_status, "ABSENT")
        self.assertEqual(dict(changed.reuse_counts), {"rir": 2, "mixture": 1, "asr": 1})
        self.assertEqual(dict(changed.rebuild_counts), {"rir": 0, "mixture": 0, "asr": 0})

        changed_contract = CacheExpectedManifest(
            infrastructure_contract_sha256="f" * 64,
            expected_rir_keys=(self.rir,),
            expected_mixture_keys=(self.mixture,),
            expected_asr_keys=(self.asr,),
        )
        changed_contract_record = reconcile_cache_manifest(changed_contract, self.store)
        self.assertTrue(changed_contract_record.complete)
        self.assertEqual(changed_contract_record.completion_marker_status, "ABSENT")

    def test_extra_shared_cache_entries_are_harmless(self):
        self._populate_all(include_extra=True)
        result = reconcile_cache_manifest(self.manifest, self.store)
        self.assertTrue(result.complete)
        self.assertEqual(result.expected_counts, {"rir": 1, "mixture": 1, "asr": 1})

    def test_reconciliation_record_is_byte_stable_and_tamper_strict(self):
        self._populate_all()
        first = reconcile_cache_manifest(self.manifest, self.store)
        second = reconcile_cache_manifest(self.manifest, self.store)
        self.assertEqual(first.to_payload(), second.to_payload())
        raw = first.to_payload()
        raw["entries"][0]["reason"] = "tampered"
        with self.assertRaises(CacheResumeError):
            type(first).from_payload(raw)
        raw = first.to_payload()
        raw["complete"] = False
        with self.assertRaises(CacheResumeError):
            type(first).from_payload(raw)

    def test_marker_payload_is_strict_and_contract_scope_is_manifest_scoped(self):
        self._populate_all()
        record = reconcile_cache_manifest(self.manifest, self.store)
        marker = write_completion_marker(self.store, self.manifest)
        raw = marker.to_payload()
        raw["expected_counts"]["asr"] = 99
        with self.assertRaises(CacheResumeError):
            type(marker).from_payload(raw)
        raw = marker.to_payload()
        raw["schema_version"] = "active-asr-a4-cache-completion-marker-v0"
        with self.assertRaises(CacheResumeError):
            type(marker).from_payload(raw)

    def test_resume_module_is_pure_import(self):
        source = Path(__file__).resolve().parents[2] / "active_audition/a4/cache_resume.py"
        imported = []
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(name.startswith("habitat") for name in imported))
        self.assertFalse(any(name.startswith("speechbrain") for name in imported))

    def test_stale_marker_can_be_replaced_when_entries_are_currently_valid(self):
        self._populate_all(include_extra=True)
        write_completion_marker(self.store, self.manifest)
        changed = CacheExpectedManifest(
            infrastructure_contract_sha256=self.manifest.infrastructure_contract_sha256,
            expected_rir_keys=(self.rir, self.rir_extra),
            expected_mixture_keys=(self.mixture,),
            expected_asr_keys=(self.asr,),
        )
        current = reconcile_cache_manifest(changed, self.store)
        self.assertTrue(current.complete)
        self.assertEqual(current.completion_marker_status, "ABSENT")
        write_completion_marker(self.store, changed)
        self.assertEqual(reconcile_cache_manifest(changed, self.store).completion_marker_status, COMPLETION_MARKER_VALID)

    def test_invalid_marker_can_be_recovered(self):
        self._populate_all()
        write_completion_marker(self.store, self.manifest)
        path = completion_marker_path(self.store, self.manifest)
        path.write_text("{not-json", encoding="utf-8")
        current = reconcile_cache_manifest(self.manifest, self.store)
        self.assertTrue(current.complete)
        self.assertEqual(current.completion_marker_status, INVALID_COMPLETION_MARKER)
        write_completion_marker(self.store, self.manifest)
        self.assertEqual(reconcile_cache_manifest(self.manifest, self.store).completion_marker_status, COMPLETION_MARKER_VALID)

    def test_fresh_marker_write_revalidates_after_prior_reconcile(self):
        self._populate_all()
        self.assertTrue(reconcile_cache_manifest(self.manifest, self.store).complete)
        (self.store.entry_dir(self.rir) / "payload.npy").unlink()
        with self.assertRaises(CacheResumeError):
            write_completion_marker(self.store, self.manifest)

    def test_manifest_scoped_markers_do_not_interfere(self):
        self._populate_all(include_extra=True)
        changed = CacheExpectedManifest(
            infrastructure_contract_sha256=self.manifest.infrastructure_contract_sha256,
            expected_rir_keys=(self.rir, self.rir_extra),
            expected_mixture_keys=(self.mixture,),
            expected_asr_keys=(self.asr,),
        )
        first_marker = write_completion_marker(self.store, self.manifest)
        second_marker = write_completion_marker(self.store, changed)
        self.assertNotEqual(completion_marker_path(self.store, self.manifest), completion_marker_path(self.store, changed))
        self.assertEqual(read_completion_marker(self.store, self.manifest).marker_id, first_marker.marker_id)
        self.assertEqual(read_completion_marker(self.store, changed).marker_id, second_marker.marker_id)


if __name__ == "__main__":
    unittest.main()
