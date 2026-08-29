import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from tools.clsdoa_v1.prepare_source_pool import (
    _normalize,
    _safe_name,
    _sha256_file,
    load_source_prep_config,
    source_offset_contract,
    validate_canonical_files,
    validate_registry_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/active_audition/clsdoa_v1_source_prep.yaml"
ONTOLOGY_PATH = ROOT / "registries/ontology.yaml"


class SourcePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_source_prep_config(CONFIG_PATH)
        with ONTOLOGY_PATH.open(encoding="utf-8") as handle:
            cls.classes = {item["name"] for item in yaml.safe_load(handle)["ontology"]["classes"]}

    def test_normalization_is_one_scalar_gain_with_peak_guard(self):
        waveform = np.ones(24000, dtype=np.float32) * 0.01
        output, stats = _normalize(waveform, 10 ** (-24 / 20), 0.50)
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(float(np.max(np.abs(output))), 0.50)
        self.assertEqual(stats["peak_guard_limited"], "false")
        self.assertAlmostEqual(float(np.sqrt(np.mean(output ** 2))), 10 ** (-24 / 20), places=5)

    def test_safe_name_is_stable(self):
        self.assertEqual(_safe_name("desed:train/a"), "desed__train__a")

    def test_config_drift_fails_before_preparation(self):
        changed = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        changed["normalization"]["peak_guard"] = 0.75
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "changed.yaml"
            path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_source_prep_config(path)

    def test_missing_class_fails_exact_ontology_gate(self):
        rows = [{
            "canonical_class": item, "manual_decision": "ACCEPT",
            "license_status": "DATASET_LEVEL_VERIFIED", "pilot_eligible": "true",
            "peak_after": "0.2",
        } for item in sorted(self.classes) if item != "dishes"]
        with self.assertRaises(ValueError):
            validate_registry_metadata(rows, self.config, self.classes)

    def test_peak_above_configured_guard_fails(self):
        rows = [{
            "canonical_class": item, "manual_decision": "ACCEPT",
            "license_status": "DATASET_LEVEL_VERIFIED", "pilot_eligible": "true",
            "peak_after": "0.6",
        } for item in sorted(self.classes)]
        with self.assertRaises(ValueError):
            validate_registry_metadata(rows, self.config, self.classes)

    def test_pcm_wav_subtype_is_checked_not_inferred_from_read_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pcm.wav"
            sf.write(str(path), np.ones(2400, dtype=np.float32) * 0.1, 24000, subtype="PCM_16")
            row = {
                "canonical_path": str(path),
                "canonical_wav_sha256": _sha256_file(path),
            }
            with self.assertRaises(ValueError):
                validate_canonical_files([row], self.config)

    def test_short_source_offset_is_deferred_to_step3(self):
        self.assertEqual(
            source_offset_contract(2.0, self.config),
            ("episode_deterministic_random", None),
        )
        self.assertEqual(source_offset_contract(5.0, self.config), ("fixed_zero", 0.0))


if __name__ == "__main__":
    unittest.main()
