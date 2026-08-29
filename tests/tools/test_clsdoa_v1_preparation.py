import csv
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
    prepare_once,
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

    def test_offset_and_crop_policy_drift_fails_fast(self):
        for key, value in (
            ("short_source_offset_policy", "fixed_zero"),
            ("overlong_crop_policy", "left_aligned"),
        ):
            changed = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            if key == "short_source_offset_policy":
                changed["canonical"][key] = value
            else:
                changed["canonical"][key] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "changed.yaml"
                path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_source_prep_config(path)

        changed = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        changed["canonical"]["overlong_crop_policy"] = "left_aligned"
        with self.assertRaises(ValueError):
            prepare_once(Path("missing-membership.csv"), Path("missing-split.csv"),
                         Path(tempfile.gettempdir()) / "clsdoa_unused", changed)

    def test_prepare_once_synthetic_float_wav_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.wav"
            waveform = np.sin(np.linspace(0, 20 * np.pi, 24000)).astype(np.float32) * 0.1
            sf.write(str(raw), waveform, 24000, subtype="FLOAT")
            membership = root / "membership.csv"
            fields = [
                "canonical_class", "source_dataset", "source_label", "original_id",
                "base_clip_id", "identity_key", "audio_path", "manual_decision",
                "license_status", "provenance_source", "raw_sha256", "mapping_type",
                "auto_qc_status", "auto_qc_reasons", "pretrain_seen_status",
                "resource_status",
            ]
            with membership.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerow({
                    "canonical_class": "dishes", "source_dataset": "synthetic",
                    "source_label": "dishes", "original_id": "one",
                    "base_clip_id": "synthetic:one", "identity_key": "synthetic:one",
                    "audio_path": str(raw), "manual_decision": "ACCEPT",
                    "license_status": "DATASET_LEVEL_VERIFIED",
                    "provenance_source": "synthetic-test", "raw_sha256": _sha256_file(raw),
                    "mapping_type": "EXACT", "auto_qc_status": "AUTO_PASS",
                    "auto_qc_reasons": "", "pretrain_seen_status": "unknown",
                    "resource_status": "PRESENT",
                })
            split = root / "split.csv"
            with split.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "canonical_class", "base_clip_id", "identity_key", "split",
                    "split_sort_key", "split_version",
                ], lineterminator="\n")
                writer.writeheader()
                writer.writerow({
                    "canonical_class": "dishes", "base_clip_id": "synthetic:one",
                    "identity_key": "synthetic:one", "split": "train",
                    "split_sort_key": "0" * 64,
                    "split_version": "clsdoa_source_split_v2_stratified",
                })
            records, _ = prepare_once(membership, split, root / "prepared", self.config)
            self.assertEqual(len(records), 1)
            output = Path(records[0]["canonical_path"])
            self.assertTrue(output.is_file())
            info = sf.info(str(output))
            data, _ = sf.read(str(output), dtype="float32")
            self.assertEqual(info.samplerate, 24000)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.subtype, "FLOAT")
            self.assertGreater(len(data), 0)
            self.assertTrue(np.isfinite(data).all())
            self.assertLessEqual(float(np.max(np.abs(data))), 0.50 + 1e-7)

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
