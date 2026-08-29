import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/active_audition/clsdoa_v1_contract.yaml"
SOURCE_PREP_PATH = REPO_ROOT / "configs/active_audition/clsdoa_v1_source_prep.yaml"
ONTOLOGY_PATH = REPO_ROOT / "registries/ontology.yaml"


class ClassDOAV1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONTRACT_PATH.open(encoding="utf-8") as handle:
            cls.contract = yaml.safe_load(handle)
        with SOURCE_PREP_PATH.open(encoding="utf-8") as handle:
            cls.source_prep = yaml.safe_load(handle)
        with ONTOLOGY_PATH.open(encoding="utf-8") as handle:
            cls.ontology = yaml.safe_load(handle)["ontology"]

    def test_ontology_is_exactly_twelve_contiguous_unique_classes(self):
        classes = self.ontology["classes"]
        expected = [
            ("coughing", "transient"),
            ("laughing", "repetitive"),
            ("keyboard_typing", "repetitive"),
            ("vacuum_cleaner", "persistent"),
            ("clock_alarm", "repetitive_persistent"),
            ("speech", "persistent"),
            ("running_water", "persistent"),
            ("frying", "persistent"),
            ("mechanical_fan", "persistent"),
            ("microwave_oven", "persistent_repetitive"),
            ("dishes", "repetitive"),
            ("printer", "repetitive"),
        ]
        self.assertEqual([item["id"] for item in classes], list(range(12)))
        self.assertEqual([(item["name"], item["temporal_type"]) for item in classes], expected)
        self.assertEqual(len({item["name"] for item in classes}), 12)

    def test_identity_and_task_contract(self):
        dataset = self.contract["dataset"]
        self.assertEqual(dataset["family"], "soundspaces_binaural_foa_clsdoa_v1")
        self.assertEqual(dataset["namespace"], "binaural_foa_clsdoa_v1")
        self.assertEqual(dataset["schema_version"], "clsdoa_v1.0")
        task = self.contract["task"]
        self.assertEqual(task["name"], "single_target_clip_classification_and_doa")
        self.assertEqual(task["target_polyphony"], 1)
        self.assertTrue(task["classification"]["enabled"])
        self.assertEqual(task["classification"]["num_classes"], 12)
        self.assertTrue(task["localization"]["enabled"])
        self.assertEqual(task["localization"]["target"], "3d_doa_unit_vector")
        for key in ("onset_offset_prediction", "frame_activity_prediction", "distance_prediction", "multi_track_prediction", "pit"):
            self.assertFalse(task[key])
        self.assertEqual(self.contract["legacy_seld_features_in_clsdoa_v1"], "disabled_not_required")

    def test_audio_representations_and_coordinates(self):
        audio = self.contract["audio"]
        self.assertEqual(audio, {"sample_rate_hz": 24000, "clip_duration_sec": 5.0, "num_samples": 120000, "dtype": "float32"})
        binaural = self.contract["representations"]["binaural"]
        self.assertEqual(binaural["channels"], 2)
        self.assertEqual(binaural["channel_order"], ["LEFT", "RIGHT"])
        foa = self.contract["representations"]["foa"]
        self.assertEqual(foa["channels"], 4)
        self.assertEqual(foa["canonical"]["format"], "AmbiX ACN/SN3D")
        self.assertEqual(foa["canonical"]["order"], ["W", "Y_DCASE", "Z_DCASE", "X_DCASE"])
        self.assertEqual(self.contract["coordinate"]["project_axes"], {"up": "+Y", "right": "+X", "forward": "-Z"})

    def test_sources_scenes_acoustics_pairing_and_normalization(self):
        self.assertEqual(
            self.contract["source"]["allowed_families"],
            ["ESC-50", "DESED isolated foreground", "PSELD-selected FSD50K"],
        )
        self.assertEqual(self.contract["scene"]["families"], ["Replica", "MP3D"])
        self.assertEqual(self.contract["paired_render"]["mode"], "sequential_sensor_lifecycle")
        self.assertEqual(self.contract["acoustics"]["indirectRayCount"], 5000)
        self.assertEqual(self.contract["acoustics"]["sourceRayCount"], 200)
        self.assertFalse(self.contract["acoustics"]["materials_enabled"])
        self.assertFalse(self.contract["normalization"]["per_render"])
        self.assertFalse(self.contract["normalization"]["per_viewpoint"])

    def test_pilot_split_uses_stratified_v2_policy(self):
        split = self.source_prep["split"]
        self.assertEqual(split["version"], "clsdoa_source_split_v2_stratified")
        self.assertEqual(split["method"], "deterministic_stratified_sha256_sort")
        self.assertEqual(split["salt"], "clsdoa_v1_source_split_20260828")
        self.assertEqual((split["train_fraction"], split["val_fraction"], split["test_fraction"]),
                         (0.70, 0.15, 0.15))
        self.assertEqual((split["train_minimum"], split["val_minimum"], split["test_minimum"]),
                         (20, 4, 4))
        self.assertNotIn("train_upper_exclusive", split)
        self.assertNotIn("val_upper_exclusive", split)

    def test_contract_has_no_concrete_pilot_identity_or_quota(self):
        forbidden_keys = {
            "pilot", "quota", "source_ids", "scene_ids", "train_ids", "val_ids",
            "test_ids", "split_members", "gain_distribution", "elevation_quota",
            "distance_quota", "episode_count",
        }

        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn(str(key), forbidden_keys)
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(self.contract)


if __name__ == "__main__":
    unittest.main()
