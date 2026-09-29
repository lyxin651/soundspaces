import copy
import ast
import hashlib
import importlib
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
    CacheError,
    CacheStore,
    HIT_VALID,
    INVALID_CORRUPT,
    KEY_MISMATCH,
    MISSING_METADATA,
    MISSING_PAYLOAD,
    MISS,
    MixtureCacheKey,
    NOISE_SOURCE_TIME_IDENTITY,
    NONFINITE_PAYLOAD,
    PAYLOAD_HASH_MISMATCH,
    RirCacheKey,
    SEMANTIC_IDENTITY_MISMATCH,
    SHAPE_MISMATCH,
    DTYPE_MISMATCH,
)
from active_audition.a4.contract import contract_sha256, load_contract, validate_contract
from active_audition.a4.identity import stable_id


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs/active_audition/v1/a4_infrastructure_contract.yaml"


def _rir_key(**overrides):
    payload = {
        "scene_resource_identities": {
            "mesh_sha256": "a" * 64,
            "navmesh_sha256": "b" * 64,
            "stage_sha256": "c" * 64,
        },
        "acoustic_contract_identity": A2_ACOUSTIC_CONTRACT_SHA256,
        "renderer_algorithm_identity": "SoundSpaces2_HabitatSim0.2.2_RLRAudioPropagation-v1",
        "materials_policy": "OFF",
        "source_world_transform": {"position_xyz": [1.0, 1.5, -2.0], "yaw_deg": 0.0},
        "receiver_sensor_transform": {"position_xyz": [0.0, 1.5, 0.0], "yaw_deg": 45.0},
        "receiver_yaw_deg": 45.0,
        "replicate_identity": "deterministic-render-replicate-0",
    }
    payload.update(overrides)
    return RirCacheKey(**payload)


def _mixture_key(**overrides):
    payload = {
        "target_rir_cache_key": _rir_key().cache_key,
        "noise_rir_cache_key": _rir_key(source_world_transform={"position_xyz": [4.0, 1.5, -2.0]}).cache_key,
        "target_dry_waveform_sha256": "1" * 64,
        "noise_segment_payload_sha256": "2" * 64,
        "noise_segment_identity": stable_id("noise-segment", {"fixture": "noise"}),
        "noise_source_time_identity": NOISE_SOURCE_TIME_IDENTITY,
        "calibration_artifact_identity": stable_id("calibration", {"fixture": "calibration"}),
        "global_gain_identity": stable_id("global-gain", {"fixture": "gain"}),
        "timeline_identity": stable_id("receiver-timeline", {"fixture": "timeline"}),
        "mixer_contract_identity": stable_id("mixture-contract", {"fixture": "mixer"}),
    }
    payload.update(overrides)
    return MixtureCacheKey(**payload)


def _asr_key(**overrides):
    payload = {
        "mono_payload_sha256": "3" * 64,
        "frontend": "mean_lr",
        "model_identity": {"repo_id": "speechbrain/model", "revision": "r1", "weights_sha256": "4" * 64},
        "language_model_identity": {"name": "lm", "sha256": "5" * 64},
        "tokenizer_identity": {"name": "spm", "sha256": "6" * 64},
        "decoder_identity": {"beam_size": 66, "temperature": 1.15},
        "precision_runtime_identity": {"precision": "float32", "torch": "2.5.1", "device": "cuda:0"},
        "asr_contract_identity": A3_ASR_CONTRACT_SHA256,
    }
    payload.update(overrides)
    return AsrCacheKey(**payload)


def _asr_payload(key):
    return {
        "schema_version": ASR_RESULT_SCHEMA_VERSION,
        "hypothesis": "TEST HYPOTHESIS",
        "score": -1.5,
        "score_semantics": "raw_decoder_sequence_score_not_confidence_or_probability",
        "input_waveform_sha256": key.mono_payload_sha256,
        "frontend": key.frontend,
        "model_identity": key.model_identity,
        "decoder_identity": key.decoder_identity,
        "asr_contract_identity": key.asr_contract_identity,
        "raw_decoder_metadata": {"token_count": 2},
    }


class CacheKeyTests(unittest.TestCase):
    def test_rir_key_is_canonical_and_only_semantic_mutations_change_it(self):
        first = _rir_key(scene_resource_identities={"navmesh_sha256": "b" * 64, "mesh_sha256": "a" * 64, "stage_sha256": "c" * 64})
        second = _rir_key()
        self.assertEqual(first.cache_key, second.cache_key)
        self.assertEqual(first.key_payload_sha256, second.key_payload_sha256)
        self.assertNotEqual(first.cache_key, _rir_key(source_world_transform={"position_xyz": [2.0, 1.5, -2.0]}).cache_key)
        self.assertNotEqual(first.cache_key, _rir_key(receiver_sensor_transform={"position_xyz": [0.0, 1.5, 1.0], "yaw_deg": 45.0}).cache_key)
        self.assertNotEqual(first.cache_key, _rir_key(acoustic_contract_identity="d" * 64).cache_key)
        self.assertNotEqual(first.cache_key, _rir_key(sample_rate_hz=24000).cache_key)
        self.assertNotEqual(first.cache_key, _rir_key(channel_order=("R", "L")).cache_key)
        self.assertNotEqual(first.cache_key, _rir_key(replicate_identity="replicate-1").cache_key)

    def test_rir_key_reuses_across_utterances_and_run_provenance(self):
        self.assertEqual(_rir_key().cache_key, _rir_key().cache_key)
        self.assertNotIn("utterance", _rir_key().to_payload())
        self.assertNotIn("git_head", _rir_key().to_payload())
        self.assertNotIn("output_path", _rir_key().to_payload())

    def test_mixture_key_binds_all_upstream_semantics(self):
        first = _mixture_key()
        self.assertNotEqual(first.cache_key, _mixture_key(target_dry_waveform_sha256="9" * 64).cache_key)
        self.assertNotEqual(first.cache_key, _mixture_key(noise_segment_payload_sha256="9" * 64).cache_key)
        self.assertNotEqual(first.cache_key, _mixture_key(calibration_artifact_identity=stable_id("calibration", {"fixture": "changed"})).cache_key)
        self.assertNotEqual(first.cache_key, _mixture_key(global_gain_identity=stable_id("global-gain", {"fixture": "changed"})).cache_key)
        self.assertNotEqual(first.cache_key, _mixture_key(timeline_identity=stable_id("receiver-timeline", {"fixture": "changed"})).cache_key)
        self.assertNotEqual(first.cache_key, _mixture_key(target_rir_cache_key=_rir_key(replicate_identity="replicate-1").cache_key).cache_key)

    def test_asr_frontends_have_independent_keys_and_result_fields_are_not_key_inputs(self):
        mean = _asr_key(frontend="mean_lr")
        left = _asr_key(frontend="fixed_L")
        right = _asr_key(frontend="fixed_R")
        self.assertEqual(len({mean.cache_key, left.cache_key, right.cache_key}), 3)
        self.assertNotEqual(mean.cache_key, _asr_key(model_identity={"repo_id": "speechbrain/model", "revision": "r2", "weights_sha256": "4" * 64}).cache_key)
        self.assertNotEqual(mean.cache_key, _asr_key(decoder_identity={"beam_size": 67, "temperature": 1.15}).cache_key)
        self.assertNotEqual(mean.cache_key, _asr_key(asr_contract_identity="e" * 64).cache_key)
        self.assertNotIn("hypothesis", mean.to_payload())
        self.assertNotIn("score", mean.to_payload())
        self.assertNotIn("git_head", mean.to_payload())

    def test_authoritative_parent_sha_and_layer_identity_validation(self):
        self.assertEqual(_rir_key().acoustic_contract_identity, A2_ACOUSTIC_CONTRACT_SHA256)
        self.assertEqual(_asr_key().asr_contract_identity, A3_ASR_CONTRACT_SHA256)
        self.assertNotEqual(_rir_key(acoustic_contract_identity="d" * 64).cache_key, _rir_key().cache_key)
        self.assertNotEqual(_asr_key(asr_contract_identity="e" * 64).cache_key, _asr_key().cache_key)
        with self.assertRaises(CacheError):
            _rir_key(acoustic_contract_identity="a2-contract-sha")
        with self.assertRaises(CacheError):
            _asr_key(asr_contract_identity="a3-contract-sha")
        with self.assertRaises(CacheError):
            _asr_key(asr_contract_identity="f" * 63)

    def test_mixture_key_requires_authoritative_stable_id_namespaces(self):
        invalid = (
            "noise_segment_identity",
            "calibration_artifact_identity",
            "global_gain_identity",
            "timeline_identity",
            "mixer_contract_identity",
        )
        for field in invalid:
            with self.subTest(field=field):
                with self.assertRaises(CacheError):
                    _mixture_key(**{field: "placeholder-abc"})
        with self.assertRaises(CacheError):
            _mixture_key(noise_source_time_identity="source-time-placeholder")


class CacheStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = CacheStore(self.tempdir.name)
        self.rir_key = _rir_key()
        self.mixture_key = _mixture_key()
        self.asr_key = _asr_key()
        self.array = np.asarray([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_clean_write_hit_repeat_and_metadata_round_trip(self):
        metadata = self.store.write_rir(self.rir_key, self.array, {"git_head": "deadbeef", "run_id": "run-1"})
        first = self.store.read_rir(self.rir_key)
        second = self.store.read_rir(self.rir_key)
        self.assertEqual(first.status, HIT_VALID)
        self.assertEqual(second.status, HIT_VALID)
        np.testing.assert_array_equal(first.payload, self.array)
        self.assertEqual(first.metadata.to_payload(), metadata.to_payload())
        self.assertEqual(first.metadata.to_payload(), second.metadata.to_payload())
        self.assertEqual(self.store.entry_dir(self.rir_key), self.store.entry_dir(self.rir_key))

    def test_mixture_metadata_and_asr_json_round_trip(self):
        mixture_metadata = self.store.write_mixture(self.mixture_key, self.array)
        mixture_result = self.store.read_mixture(self.mixture_key)
        self.assertEqual(mixture_result.status, HIT_VALID)
        self.assertEqual(mixture_result.metadata.to_payload(), mixture_metadata.to_payload())
        asr_metadata = self.store.write_asr(self.asr_key, _asr_payload(self.asr_key), (2,))
        asr_result = self.store.read_asr(self.asr_key)
        self.assertEqual(asr_result.status, HIT_VALID)
        self.assertEqual(asr_result.metadata.to_payload(), asr_metadata.to_payload())
        self.assertEqual(asr_result.payload["frontend"], "mean_lr")
        asr_metadata_payload = asr_result.metadata.to_payload()
        self.assertEqual(asr_metadata_payload["input_layout"], "mono")
        self.assertEqual(asr_metadata_payload["input_shape"], [2])
        self.assertEqual(asr_metadata_payload["input_dtype"], "float32")
        self.assertNotIn("channel_order", asr_metadata_payload)
        self.assertNotIn("expected_shape", asr_metadata_payload)
        self.assertNotIn("dtype", asr_metadata_payload)
        for frontend in ("mean_lr", "fixed_L", "fixed_R"):
            key = _asr_key(frontend=frontend)
            metadata = self.store.write_asr(key, _asr_payload(key), (2,))
            self.assertEqual(metadata.to_payload()["input_layout"], "mono")
            self.assertEqual(metadata.to_payload()["input_shape"], [2])

    def test_asr_metadata_rejects_binaural_or_tampered_mono_fields(self):
        directory = self.store.entry_dir(self.asr_key)
        self.store.write_asr(self.asr_key, _asr_payload(self.asr_key), (2,))
        path = directory / "metadata.json"

        raw = json.loads(path.read_text())
        raw["input_layout"] = "binaural"
        path.write_text(json.dumps(raw))
        self.assertEqual(self.store.read_asr(self.asr_key).status, INVALID_CORRUPT)

        self.store.write_asr(self.asr_key, _asr_payload(self.asr_key), (2,))
        raw = json.loads(path.read_text())
        raw["channel_order"] = ["L", "R"]
        path.write_text(json.dumps(raw))
        self.assertEqual(self.store.read_asr(self.asr_key).status, INVALID_CORRUPT)

        self.store.write_asr(self.asr_key, _asr_payload(self.asr_key), (2,))
        raw = json.loads(path.read_text())
        raw["input_shape"] = [2, 1]
        path.write_text(json.dumps(raw))
        self.assertEqual(self.store.read_asr(self.asr_key).status, INVALID_CORRUPT)

        self.store.write_asr(self.asr_key, _asr_payload(self.asr_key), (2,))
        raw = json.loads(path.read_text())
        raw["schema_version"] = "active-asr-a4-asr-cache-metadata-v1"
        path.write_text(json.dumps(raw))
        self.assertEqual(self.store.read_asr(self.asr_key).status, INVALID_CORRUPT)

    def test_missing_entries_and_partial_temp_files_are_not_hits(self):
        missing = self.store.read_rir(self.rir_key)
        self.assertEqual((missing.status, missing.reason), (MISS, MISSING_METADATA))
        directory = self.store.entry_dir(self.rir_key)
        directory.mkdir(parents=True)
        (directory / ".tmp-interrupted-payload").write_bytes(b"partial")
        self.assertEqual(self.store.read_rir(self.rir_key).status, MISS)
        self.store.write_rir(self.rir_key, self.array)
        (directory / "metadata.json").unlink()
        result = self.store.read_rir(self.rir_key)
        self.assertEqual((result.status, result.reason), (INVALID_CORRUPT, MISSING_METADATA))
        self.store.write_rir(self.rir_key, self.array)
        (directory / "payload.npy").unlink()
        result = self.store.read_rir(self.rir_key)
        self.assertEqual((result.status, result.reason), (INVALID_CORRUPT, MISSING_PAYLOAD))

    def test_corrupt_bytes_and_metadata_fail_closed(self):
        metadata = self.store.write_rir(self.rir_key, self.array)
        directory = self.store.entry_dir(self.rir_key)
        payload_path = directory / "payload.npy"
        original = payload_path.read_bytes()
        payload_path.write_bytes(original + b"corruption")
        self.assertEqual(self.store.read_rir(self.rir_key).reason, PAYLOAD_HASH_MISMATCH)

        self.store.write_rir(self.rir_key, self.array)
        raw = json.loads((directory / "metadata.json").read_text())
        raw["cache_key"] = _rir_key(replicate_identity="other").cache_key
        (directory / "metadata.json").write_text(json.dumps(raw))
        self.assertEqual(self.store.read_rir(self.rir_key).reason, KEY_MISMATCH)

        self.store.write_rir(self.rir_key, self.array)
        raw = json.loads((directory / "metadata.json").read_text())
        raw["semantic_identities"]["materials_policy"] = "ON"
        (directory / "metadata.json").write_text(json.dumps(raw))
        self.assertEqual(self.store.read_rir(self.rir_key).reason, SEMANTIC_IDENTITY_MISMATCH)

    def test_shape_dtype_and_nonfinite_payload_faults_are_rejected(self):
        directory = self.store.entry_dir(self.rir_key)
        self.store.write_rir(self.rir_key, self.array)
        raw = json.loads((directory / "metadata.json").read_text())
        raw["expected_shape"] = [3, 2]
        (directory / "metadata.json").write_text(json.dumps(raw))
        self.assertEqual(self.store.read_rir(self.rir_key).reason, SHAPE_MISMATCH)

        self.store.write_rir(self.rir_key, self.array)
        wrong_dtype = np.asarray(self.array, dtype=np.float64)
        buffer = __import__("io").BytesIO()
        np.save(buffer, wrong_dtype, allow_pickle=False)
        payload = buffer.getvalue()
        (directory / "payload.npy").write_bytes(payload)
        raw = json.loads((directory / "metadata.json").read_text())
        raw["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        (directory / "metadata.json").write_text(json.dumps(raw))
        self.assertEqual(self.store.read_rir(self.rir_key).reason, DTYPE_MISMATCH)

        self.store.write_rir(self.rir_key, self.array)
        nonfinite = np.asarray(self.array, dtype=np.float32).copy()
        nonfinite[0, 0] = np.nan
        buffer = __import__("io").BytesIO()
        np.save(buffer, nonfinite, allow_pickle=False)
        payload = buffer.getvalue()
        (directory / "payload.npy").write_bytes(payload)
        raw = json.loads((directory / "metadata.json").read_text())
        raw["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        (directory / "metadata.json").write_text(json.dumps(raw))
        self.assertEqual(self.store.read_rir(self.rir_key).reason, NONFINITE_PAYLOAD)

    def test_unknown_metadata_field_is_rejected(self):
        self.store.write_mixture(self.mixture_key, self.array)
        path = self.store.entry_dir(self.mixture_key) / "metadata.json"
        raw = json.loads(path.read_text())
        raw["unexpected"] = True
        path.write_text(json.dumps(raw))
        self.assertEqual(self.store.read_mixture(self.mixture_key).status, INVALID_CORRUPT)

    def test_provenance_changes_do_not_change_key(self):
        first = self.store.write_rir(self.rir_key, self.array, {"git_head": "one", "output_path": "/tmp/one"})
        key = self.rir_key.cache_key
        second = self.store.write_rir(self.rir_key, self.array, {"git_head": "two", "output_path": "/tmp/two"})
        self.assertEqual(key, self.rir_key.cache_key)
        self.assertEqual(first.key.cache_key, second.key.cache_key)


class A4CacheContractTests(unittest.TestCase):
    def test_a4_cache_semantics_are_concrete_including_resume(self):
        contract = load_contract(str(CONTRACT_PATH))
        validate_contract(contract)
        cache = contract["cache_resume"]
        self.assertEqual(cache["algorithm"], "active-asr-a4-content-addressed-cache-v1")
        self.assertEqual(cache["key_serialization"], "active-asr-a4-cache-key-canonical-json-v1")
        self.assertEqual(cache["key_schemas"]["mixture"], "active-asr-a4-mixture-cache-key-v2")
        self.assertEqual(cache["metadata_schema"], "active-asr-a4-cache-metadata-v1")
        self.assertEqual(
            cache["metadata_schemas"],
            {
                "rir": "active-asr-a4-rir-cache-metadata-v1",
                "mixture": "active-asr-a4-mixture-cache-metadata-v1",
                "asr": "active-asr-a4-asr-cache-metadata-v2",
            },
        )
        self.assertEqual(cache["integrity_algorithm"], "active-asr-a4-cache-key-metadata-payload-integrity-v1")
        self.assertIn("finite_payload", cache["integrity"])
        self.assertEqual(cache["resume"], "active-asr-a4-deterministic-resume-reconciliation-v1")
        self.assertEqual(cache["expected_manifest_schema"], "active-asr-a4-cache-expected-manifest-v1")
        self.assertEqual(cache["reconciliation_record_schema"], "active-asr-a4-cache-reconciliation-record-v1")
        self.assertEqual(cache["completion_marker_schema"], "active-asr-a4-cache-completion-marker-v1")
        self.assertEqual(contract["contract"]["state"], "FROZEN")
        self.assertEqual(len(contract_sha256(contract)), 64)

    def test_cache_module_is_pure_import(self):
        module = importlib.import_module("active_audition.a4.cache")
        self.assertTrue(hasattr(module, "CacheStore"))
        source = (ROOT / "active_audition/a4/cache.py").read_text(encoding="utf-8")
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
