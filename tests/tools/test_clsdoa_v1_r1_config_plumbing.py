import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import tools.clsdoa_v1.pilot_dataset as pilot_dataset


class ConfigurationPlumbingTests(unittest.TestCase):
    def test_load_config_uses_config_path_without_environment_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.yaml"
            path.write_text("dataset_id: fixture_from_config\nplan_version: fixture_v1\n")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("STEP3_PILOT_CONFIG", None)
                with mock.patch.object(pilot_dataset, "CONFIG_PATH", path):
                    self.assertEqual(pilot_dataset._load_config()["dataset_id"], "fixture_from_config")

    def test_cli_config_reaches_planner_without_environment_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "pilot.yaml"
            root = Path(directory) / "dataset"
            config.write_text(yaml.safe_dump({"dataset_id": "cli_fixture", "plan_version": "v1"}))
            captured = {}
            def fake_write_plan(actual_root, code_commit):
                captured["root"] = actual_root
                captured["config"] = pilot_dataset._load_config()
                captured["code_commit"] = code_commit
                return {"status": "fixture"}
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("STEP3_PILOT_CONFIG", None)
                with mock.patch.object(pilot_dataset, "current_clean_head", return_value="clean-head"), mock.patch.object(pilot_dataset, "write_plan", side_effect=fake_write_plan), mock.patch.object(sys, "argv", ["pilot_dataset.py", "plan", "--config", str(config), "--root", str(root)]):
                    pilot_dataset.main()
            self.assertEqual(captured["config"]["dataset_id"], "cli_fixture")
            self.assertEqual(captured["root"], root)
            self.assertEqual(captured["code_commit"], "clean-head")
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
