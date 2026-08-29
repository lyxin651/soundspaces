import unittest
from pathlib import Path

import yaml

from tools.clsdoa_v1.validate_pilot_plan import PilotPlanValidationError


class Step3R1ContractTests(unittest.TestCase):
    def test_pilot_002_config_is_split_decoupled_build(self):
        path = Path(__file__).resolve().parents[2] / "configs/active_audition/clsdoa_v1_pilot_002.yaml"
        config = yaml.safe_load(path.read_text())
        self.assertEqual(config["dataset_id"], "clsdoa_v1_pilot_002")
        self.assertEqual(config["plan_version"], "clsdoa_v1_pilot_plan_v2")
        self.assertEqual(config["geometry"]["max_attempts_per_scene"], 1024)
        self.assertTrue(config["storage"]["require_rir"])

    def test_validator_uses_explicit_exception_type(self):
        self.assertTrue(issubclass(PilotPlanValidationError, ValueError))


if __name__ == "__main__":
    unittest.main()
