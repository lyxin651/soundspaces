import copy
import json
import unittest

from tools.clsdoa_v1.build_final_provenance import build_lock
from tools.clsdoa_v1.validate_resources_lock import LockValidationError, validate


class FinalProvenanceTests(unittest.TestCase):
    def test_generated_lock_validates(self):
        validate(build_lock())

    def test_wrong_sha_is_rejected(self):
        value = copy.deepcopy(build_lock())
        value["source"]["registry_sha256"] = "0" * 64
        with self.assertRaises(LockValidationError):
            validate(value)

    def test_wrong_split_is_rejected(self):
        value = copy.deepcopy(build_lock())
        value["scene"]["split_version"] = "manual"
        with self.assertRaises(LockValidationError):
            validate(value)

    def test_wrong_receiver_and_rays_are_rejected(self):
        for field, replacement in (("receiver_contract", "listener_base"), ("indirect_ray_count", 1), ("source_ray_count", 1)):
            value = copy.deepcopy(build_lock())
            value["acoustics"][field] = replacement
            with self.assertRaises(LockValidationError):
                validate(value)

    def test_superseded_evidence_cannot_be_authoritative(self):
        value = copy.deepcopy(build_lock())
        value["superseded_evidence"] = ["954deee8d3ce5cab070c436deea91118a416977d"]
        with self.assertRaises(LockValidationError):
            validate(value)

    def test_lock_is_json_serializable(self):
        json.dumps(build_lock(), sort_keys=True)


if __name__ == "__main__":
    unittest.main()
