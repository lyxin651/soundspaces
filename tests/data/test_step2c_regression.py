import unittest
from pathlib import Path

from active_audition.datasets.binaural_foa_clsdoa.provenance import (
    ProvenanceError,
    build_provenance_template,
    validate_core_provenance_template,
    validate_final_resources_lock,
)
from active_audition.datasets.binaural_foa_clsdoa.storage import validate_resources_lock


REPO_ROOT = Path(__file__).resolve().parents[2]


class Step2CProvenanceTests(unittest.TestCase):
    def test_core_template_is_deterministic_and_keeps_step2a2b_pending(self):
        first = build_provenance_template(str(REPO_ROOT))
        second = build_provenance_template(str(REPO_ROOT))
        self.assertEqual(first, second)
        validate_core_provenance_template(first)
        self.assertEqual(first["source_registry_sha256"], "PENDING_STEP_2A")
        self.assertEqual(first["scene_registry_sha256"], "PENDING_STEP_2B")

    def test_final_resource_lock_rejects_pending_and_placeholder_hashes(self):
        value = {
            "soundspaces_commit": "fixture",
            "habitat_sim_version": "fixture",
            "rlraudio_propagation": "fixture",
            "hrtf": {"path": "fixture", "sha256": "0" * 64},
            "foa_contract_version": "fixture",
            "ontology_sha256": "a" * 64,
            "source_registry_sha256": "PENDING_STEP_2A",
            "scene_registry_sha256": "b" * 64,
            "source_split_version": "split-v1",
            "scene_split_version": "split-v1",
            "sample_rate_hz": 24000,
            "clip_duration_sec": 5.0,
            "indirectRayCount": 5000,
            "sourceRayCount": 200,
            "materials_mode": "OFF",
            "random_seed": 1,
        }
        with self.assertRaises(ProvenanceError):
            validate_final_resources_lock(value)
        value["source_registry_sha256"] = "c" * 64
        value["hrtf"]["sha256"] = "d" * 64
        validate_final_resources_lock(value)

    def test_core_template_cannot_be_used_as_final_lock(self):
        value = dict(build_provenance_template(str(REPO_ROOT)))
        with self.assertRaises(ProvenanceError):
            validate_final_resources_lock(value)


class Step2CContractHelperTests(unittest.TestCase):
    def test_existing_resource_lock_shape_remains_compatible(self):
        value = {
            "soundspaces_commit": "fixture",
            "habitat_sim_version": "fixture",
            "rlraudio_propagation": "fixture",
            "hrtf": {"path": "fixture", "sha256": "0" * 64},
            "foa_contract_version": "fixture",
            "ontology_sha256": "0" * 64,
            "source_registry_sha256": "0" * 64,
            "scene_registry_sha256": "0" * 64,
            "source_split_version": "UNASSIGNED",
            "scene_split_version": "UNASSIGNED",
            "sample_rate_hz": 24000,
            "clip_duration_sec": 5.0,
            "indirectRayCount": 5000,
            "sourceRayCount": 200,
            "materials_mode": "OFF",
            "random_seed": 1,
        }
        validate_resources_lock(value)

    def test_final_lock_requires_real_sha_not_all_zero(self):
        value = {
            "ontology_sha256": "0" * 64,
            "source_registry_sha256": "a" * 64,
            "scene_registry_sha256": "b" * 64,
        }
        self.assertEqual(len(value["ontology_sha256"]), 64)
        with self.assertRaises(ProvenanceError):
            from active_audition.datasets.binaural_foa_clsdoa.provenance import _require_sha
            _require_sha(value["ontology_sha256"], "ontology_sha256")


if __name__ == "__main__":
    unittest.main()
