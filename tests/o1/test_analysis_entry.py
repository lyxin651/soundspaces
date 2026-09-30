import json
import tempfile
import unittest
from pathlib import Path

from active_audition.a4.smoke_manifest import EngineeringSmokeManifest
from active_audition.a4.cache import CacheStore, MixtureCacheKey
from active_audition.a4.cache_resume import CacheExpectedManifest
from active_audition.asr.frontends import apply_frontend
from active_audition.o1.analysis_entry import (
    O1AnalysisEntryError,
    load_manifest_by_schema,
    resolve_analysis_kind,
    validate_scientific_run_root,
)
from active_audition.o1.landscape import O1_REAL_KIND, O1_DRY_RUN_KIND
from scripts.run_o1_production import _asr_cache_key_for_mono


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC = ROOT / "runs/active_asr_v1/o1_replica_apartment_2_3155ddba31ea/o1_scientific_manifest.json"
PRE_AUDIT = ROOT / "runs/active_asr_v1/o1_replica_apartment_2_c8c821eae922/o1_exploratory_manifest.json"
ENGINEERING = ROOT / "configs/active_audition/v1/a4_engineering_smoke_manifest.json"
RUN_ROOT = SCIENTIFIC.parent


class O1AnalysisEntryTests(unittest.TestCase):
    def test_explicit_manifest_dispatch(self):
        scientific, kind = load_manifest_by_schema(SCIENTIFIC)
        self.assertEqual(kind, "o1")
        self.assertEqual(resolve_analysis_kind(scientific, kind, None), O1_REAL_KIND)
        engineering, kind = load_manifest_by_schema(ENGINEERING)
        self.assertEqual(kind, "engineering")
        self.assertEqual(resolve_analysis_kind(engineering, kind, None), O1_DRY_RUN_KIND)

    def test_v1_and_engineering_cannot_enter_real_landscape(self):
        old, kind = load_manifest_by_schema(PRE_AUDIT)
        with self.assertRaises(O1AnalysisEntryError):
            resolve_analysis_kind(old, kind, O1_REAL_KIND)
        engineering, kind = load_manifest_by_schema(ENGINEERING)
        with self.assertRaises(O1AnalysisEntryError):
            resolve_analysis_kind(engineering, kind, O1_REAL_KIND)
        self.assertEqual(resolve_analysis_kind(old, kind, O1_DRY_RUN_KIND), O1_DRY_RUN_KIND)

    def test_incompatible_explicit_kind_is_rejected(self):
        scientific, kind = load_manifest_by_schema(SCIENTIFIC)
        with self.assertRaises(O1AnalysisEntryError):
            resolve_analysis_kind(scientific, kind, O1_DRY_RUN_KIND)

    def test_unknown_schema_is_rejected_without_loader_probing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.json"
            path.write_text(json.dumps({"schema_version": "unknown"}), encoding="utf-8")
            with self.assertRaises(O1AnalysisEntryError):
                load_manifest_by_schema(path)

    def test_scientific_run_root_is_strictly_bound(self):
        manifest, kind = load_manifest_by_schema(SCIENTIFIC)
        self.assertEqual(kind, "o1")
        evidence = validate_scientific_run_root(manifest, SCIENTIFIC, RUN_ROOT, ROOT / "data/active_asr_a4/cache")
        self.assertEqual(evidence["expected_counts"], {"rir": 384, "mixture": 768, "asr": 2304})
        self.assertEqual(evidence["valid_counts"], evidence["expected_counts"])
        self.assertEqual(evidence["completion_marker_status"], "VALID")

    def test_engineering_manifest_remains_strict_type(self):
        payload = json.loads(ENGINEERING.read_text(encoding="utf-8"))
        parsed = EngineeringSmokeManifest.from_payload(payload)
        self.assertTrue(parsed.engineering_only)

    def test_diagnostics_key_reconstruction_is_exact_and_frontend_bound(self):
        index = json.loads((RUN_ROOT / "o1_mixture_index.json").read_text(encoding="utf-8"))["entries"]
        entry = index[0]
        mixture_key = MixtureCacheKey.from_payload(entry["mixture_key_payload"])
        mixture = CacheStore(str(ROOT / "data/active_asr_a4/cache")).read_mixture(mixture_key)
        self.assertEqual(mixture.status, "HIT_VALID")
        expected = CacheExpectedManifest.from_payload(json.loads((RUN_ROOT / "final_cache_expected_manifest.json").read_text(encoding="utf-8")))
        expected_keys = {key.cache_key: key for key in expected.expected_asr_keys}
        for frontend in ("mean_lr", "fixed_L", "fixed_R"):
            mono = apply_frontend(mixture.payload, frontend)
            key = _asr_cache_key_for_mono(mono, frontend)
            self.assertIn(key.cache_key, expected_keys)
            self.assertEqual(expected_keys[key.cache_key].frontend, frontend)
        mean_key = _asr_cache_key_for_mono(apply_frontend(mixture.payload, "mean_lr"), "mean_lr")
        fixed_key = _asr_cache_key_for_mono(apply_frontend(mixture.payload, "fixed_L"), "fixed_L")
        self.assertNotEqual(mean_key.cache_key, fixed_key.cache_key)
