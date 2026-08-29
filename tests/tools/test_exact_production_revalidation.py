import unittest

from tools.clsdoa_v1.validate_exact_production_revalidation import validate_rows


class ExactProductionEvidenceTests(unittest.TestCase):
    def test_contract_rejects_duplicate_probe(self):
        row = {
            "scene_id": "replica.a", "probe_id": "probe_0", "sample_rate_hz": 24000,
            "materials_enabled": False, "indirect_ray_count": 5000, "source_ray_count": 200,
            "normalization_applied": False, "receiver_error_m": 0.0, "foa_receiver_error_m": 0.0,
            "binaural": {"channel_count": 2, "finite": True, "nonzero": True},
            "foa": {"channel_count": 4, "finite": True, "nonzero": True},
        }
        registry = {"scenes": {"replica.a": {"admitted": "PASS"}, "replica.fail": {"admitted": "FAIL"}}}
        with self.assertRaises(AssertionError):
            validate_rows([row] * 206, registry, "SUPERSEDED_BY_CORRECT_RECEIVER_REVALIDATION")


if __name__ == "__main__":
    unittest.main()
