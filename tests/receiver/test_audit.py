import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from active_audition.config.loader import load_resolved_config as load_legacy_config
from active_audition.config.v1 import load_resolved_config as load_contract
from active_audition.receiver.audit import (
    RECEIVER_AUDIT_SCHEMA_VERSION,
    RUNTIME_LOCK_SCHEMA_VERSION,
    RuntimeAuditError,
    _audit_status,
    _receiver_observation,
    canonical_json,
    run_runtime_audit,
    validate_receiver_audit,
    validate_runtime_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/active_audition/v1/experiment_contract.yaml"
RUNTIME_CONFIG_PATH = REPO_ROOT / "configs/active_audition/v0_replica_debug.yaml"


class _Enum:
    def __init__(self, name):
        self.name = name


class _FakeAcoustics:
    directRayCount = 500
    sourceRayCount = 200
    indirectRayCount = 5000
    indirectRayDepth = 200
    sourceRayDepth = 10
    maxIRLength = 4.0
    directSHOrder = 3
    indirectSHOrder = 1
    maxDiffractionOrder = 10
    frequencyBands = 4
    threadCount = 1
    unitScale = 1.0
    temporalCoherence = 0
    direct = 1
    indirect = 1
    transmission = 1
    diffraction = 1
    globalVolume = 1.0
    meshSimplification = 0
    sampleRate = 16000.0


class _FakeRotation:
    w = 1.0
    x = 0.0
    y = 0.0
    z = 0.0


class _FakeSensor:
    def __init__(self):
        self.node = SimpleNamespace(translation=[0.0, 1.5, 0.0], rotation=_FakeRotation())
        self._spec = SimpleNamespace(
            sensor_type=_Enum("AUDIO"),
            uuid="audio_sensor",
            channelLayout=SimpleNamespace(type=_Enum("Binaural"), channelCount=2),
            acousticsConfig=_FakeAcoustics(),
            position=[0.0, 1.5, 0.0],
            orientation=[0.0, 0.0, 0.0],
            enableMaterials=False,
        )

    def specification(self):
        return self._spec


class _FakeContextManager:
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        return self.context

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _fake_context():
    scene_asset = REPO_ROOT / "data/scene_datasets/replica/office_0/habitat/mesh_semantic.ply"
    simulator = SimpleNamespace(
        semantic_scene=object(),
        config=SimpleNamespace(sim_cfg=SimpleNamespace(scene_id=str(scene_asset))),
    )
    return SimpleNamespace(
        audio_sensor=_FakeSensor(),
        pathfinder=SimpleNamespace(is_loaded=True),
        simulator=simulator,
    )


class ActiveASRAuditTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract(str(CONTRACT_PATH))
        self.runtime_config = load_legacy_config(str(RUNTIME_CONFIG_PATH))

    def test_canonical_json_is_mapping_order_invariant(self):
        first = {"b": [1, {"z": 0.0, "a": True}], "a": "x"}
        second = {"a": "x", "b": [1, {"a": True, "z": -0.0}]}
        self.assertEqual(canonical_json(first), canonical_json(second))

    def test_requested_effective_and_unknown_are_distinct(self):
        receiver = _receiver_observation(self.contract, self.runtime_config, _fake_context())
        self.assertEqual(receiver["comparisons"]["sample_rate_hz"], "PASS")
        self.assertEqual(receiver["comparisons"]["contract_sample_rate_hz"], "PASS")
        self.assertEqual(receiver["comparisons"]["channel_count"], "PASS")
        self.assertEqual(receiver["effective"]["channel_order"]["status"], "NOT_EXPOSED")
        self.assertEqual(receiver["receiver_geometry"]["ear_spacing_m"]["effective"]["status"], "NOT_EXPOSED")
        status, failures, unknown, not_exposed = _audit_status(receiver)
        self.assertEqual(status, "PASS_WITH_UNKNOWN")
        self.assertEqual(failures, [])
        self.assertEqual(unknown, [])
        self.assertTrue(not_exposed)

    def test_requested_effective_mismatch_is_blocked(self):
        runtime_config = copy.deepcopy(self.runtime_config)
        runtime_config["acoustics"]["sample_rate_hz"] = 24000
        receiver = _receiver_observation(self.contract, runtime_config, _fake_context())
        self.assertEqual(receiver["comparisons"]["sample_rate_hz"], "FAIL")
        status, failures, _, _ = _audit_status(receiver)
        self.assertEqual(status, "BLOCKED")
        self.assertIn("sample_rate_hz", failures)

    def test_artifact_schema_is_strict_at_top_level(self):
        runtime_lock = {
            "schema_version": RUNTIME_LOCK_SCHEMA_VERSION,
            "gate": "A1",
            "contract": {},
            "repository": {},
            "inputs": {},
            "runtime_fingerprint": {},
        }
        self.assertIs(validate_runtime_lock(runtime_lock), runtime_lock)
        with self.assertRaises(RuntimeAuditError):
            validate_runtime_lock(dict(runtime_lock, future_field=True))

        receiver_audit = {
            "schema_version": RECEIVER_AUDIT_SCHEMA_VERSION,
            "gate": "A1",
            "status": "PASS_WITH_UNKNOWN",
            "contract_sha256": "0" * 64,
            "scene": {},
            "receiver": {},
            "findings": {"a2_not_run": True, "a2_boundary": ["ITD/ILD"]},
            "artifacts": {},
        }
        self.assertIs(validate_receiver_audit(receiver_audit), receiver_audit)
        with self.assertRaises(RuntimeAuditError):
            validate_receiver_audit(dict(receiver_audit, gate="A2"))

    def test_full_audit_artifacts_are_reproducible(self):
        fingerprint = {
            "python": {"version": "test"},
            "os": {"system": "test"},
            "git_commit": "test-commit",
            "packages": [],
            "necessary_dependencies": [],
            "rlr_audio_propagation_modules": [],
            "binaries": [],
        }
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            with patch("active_audition.scene.simulator.create_scene_simulator", return_value=_FakeContextManager(_fake_context())), patch(
                "active_audition.receiver.audit._runtime_fingerprint", return_value=fingerprint
            ):
                first = run_runtime_audit(str(CONTRACT_PATH), first_dir, str(RUNTIME_CONFIG_PATH))
            with patch("active_audition.scene.simulator.create_scene_simulator", return_value=_FakeContextManager(_fake_context())), patch(
                "active_audition.receiver.audit._runtime_fingerprint", return_value=fingerprint
            ):
                second = run_runtime_audit(str(CONTRACT_PATH), second_dir, str(RUNTIME_CONFIG_PATH))

            self.assertEqual(first["status"], "PASS_WITH_UNKNOWN")
            self.assertEqual(first["contract_sha256"], second["contract_sha256"])
            for name in ("runtime.lock.json", "receiver_audit.json", "receiver_audit.md"):
                self.assertEqual(
                    (Path(first_dir) / name).read_bytes(),
                    (Path(second_dir) / name).read_bytes(),
                )
            receiver_document = json.loads((Path(first_dir) / "receiver_audit.json").read_text(encoding="utf-8"))
            validate_receiver_audit(receiver_document)
            self.assertEqual(first["artifacts"]["receiver_audit.json"]["sha256"], second["artifacts"]["receiver_audit.json"]["sha256"])


if __name__ == "__main__":
    unittest.main()
