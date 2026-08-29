"""R5 named negative ledger; every case executes a production validation helper."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.clsdoa_v1.integrity import IntegrityError, verify_plan_integrity
from tools.clsdoa_v1.scheduler import schedule_block
from tools.clsdoa_v1.validate_pilot_plan import (
    PilotPlanValidationError,
    validate_episode_ids,
    validate_episode_membership,
    validate_family_block,
    validate_render_payload,
    validate_render_record_payload,
    validate_render_rows,
    validate_plan_payload_absence,
    _payload_expected_keys,
    validate_scene_set,
    validate_source_reuse,
)


CASE_LEDGER = {
    "N01": "wrong family distance quota", "N02": "wrong family elevation quota",
    "N03": "wrong family azimuth micro-pattern", "N04": "wrong family gain micro-pattern",
    "N05": "missing one PASS scene", "N06": "injected FAIL scene",
    "N07": "source split mismatch", "N08": "scene split mismatch", "N09": "scene family mismatch",
    "N10": "duplicate episode ID", "N11": "source reuse imbalance",
    "N12": "episodes SHA drift", "N13": "config_resolved SHA drift", "N14": "resources.lock SHA drift",
    "N15": "fake cache/rir/*.npy in metadata PLAN", "N16": "missing RIR", "N17": "wrong WAV dtype",
    "N18": "wrong Binaural WAV shape", "N19": "wrong FOA WAV shape", "N20": "wrong Binaural RIR shape",
    "N21": "wrong FOA RIR shape", "N22": "wrong RIR dtype", "N23": "duplicate representation",
    "N24": "fewer than 960 episodes", "N25": "extra failed/orphan render record", "N26": "premature _SUCCESS",
}


class R5ExactNegativeMatrixTests(unittest.TestCase):
    def test_case_ledger_has_exact_26_ids(self):
        self.assertEqual(list(CASE_LEDGER), ["N{:02d}".format(i) for i in range(1, 27)])

    def test_N01_to_N04_family_quota(self):
        expected = schedule_block("train", 0, "Replica")
        for field in ("distance", "elevation", "azimuth", "gain"):
            block = [{"diagnostics": {name + "_bin": expected[name][i] for name in expected}} for i in range(28)]
            block[0]["diagnostics"][field + "_bin"] = "invalid"
            with self.subTest(case="N" + str(("distance", "elevation", "azimuth", "gain").index(field) + 1).zfill(2)):
                with self.assertRaises(PilotPlanValidationError):
                    validate_family_block(block, 0, "train", "Replica")

    def test_N05_N06_scene_set(self):
        scenes = {"scenes": {"pass": {"admitted": "PASS"}, "fail": {"admitted": "FAIL"}, "pass2": {"admitted": "PASS"}}}
        for used in (("pass",), ("pass", "fail", "pass2")):
            with self.assertRaises(PilotPlanValidationError):
                validate_scene_set(used, scenes)

    def test_N07_N08_N09_membership(self):
        episode = {"split": "train", "scene": {"scene_id": "scene", "scene_family": "Replica"}, "source": {"base_clip_id": "base"}}
        for case, source, scene in (("N07", [{"base_clip_id": "base", "split": "val"}], {"split": "train", "scene_family": "Replica", "admitted": "PASS"}), ("N08", [{"base_clip_id": "base", "split": "train"}], {"split": "val", "scene_family": "Replica", "admitted": "PASS"}), ("N09", [{"base_clip_id": "base", "split": "train"}], {"split": "train", "scene_family": "Replica", "admitted": "PASS"})):
            row = dict(episode)
            if case == "N09":
                row["scene"] = dict(episode["scene"], scene_family="MP3D")
            with self.subTest(case=case), self.assertRaises(PilotPlanValidationError):
                validate_episode_membership([row], source, {"scenes": {"scene": scene}})

    def test_N10_N11_ids_and_reuse(self):
        with self.assertRaises(PilotPlanValidationError):
            validate_episode_ids([{"episode_id": "x"}, {"episode_id": "x"}])
        rows = [{"label": {"class_id": 0}, "split": "train", "source": {"base_clip_id": "a"}}] * 3 + [{"label": {"class_id": 0}, "split": "train", "source": {"base_clip_id": "b"}}]
        with self.assertRaises(PilotPlanValidationError):
            validate_source_reuse(rows, 0, "train")

    def test_N12_N14_sha_drift(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "manifests").mkdir()
            for name, data in (("episodes.jsonl", "e\n"), ("config_resolved.yaml", "storage:\n  require_rir: true\n"), ("resources.lock.json", "{}\n")): (root / "manifests" / name if name == "episodes.jsonl" else root / name).write_text(data)
            def sha(path):
                import hashlib
                return hashlib.sha256(path.read_bytes()).hexdigest()
            source, scene = sha(repo / "registries/source_audio.csv"), sha(repo / "registries/clsdoa_v1_scenes.yaml")
            lock = {"episodes_sha256": sha(root / "manifests/episodes.jsonl"), "config_resolved_sha256": sha(root / "config_resolved.yaml"), "resources_lock_sha256": sha(root / "resources.lock.json"), "source_registry_sha256": source, "scene_registry_sha256": scene, "plan_generation_code_commit": "clean"}
            (root / "identity.json").write_text(json.dumps({"generation_code_commit": "clean", "source_registry_sha256": source, "scene_registry_sha256": scene}))
            for case, field in zip(("N12", "N13", "N14"), ("episodes_sha256", "config_resolved_sha256", "resources_lock_sha256")):
                broken = dict(lock, **{field: "0" * 64}); (root / "manifests/plan.lock.json").write_text(json.dumps(broken))
                with self.subTest(case=case), self.assertRaises(IntegrityError): verify_plan_integrity(root, repo)

    def test_N15_metadata_plan_cache_and_N24_episode_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "cache/rir").mkdir(parents=True); np.save(root / "cache/rir/test.npy", np.ones((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(PilotPlanValidationError, "^plan contains render payload$"):
                validate_plan_payload_absence(root)
            (root / "manifests").mkdir(); (root / "manifests/episodes.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(PilotPlanValidationError, "^payload requires all 960 unique episodes$"):
                _payload_expected_keys(root)

    def _payload_fixture(self, representation, wav_shape, wav_dtype=np.float32, rir_shape=(10, 2), rir_dtype=np.float32):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name); channels = 2 if representation == "binaural" else 4
        wavfile.write(str(root / "a.wav"), 24000, np.ones(wav_shape, dtype=wav_dtype)); np.save(root / "r.npy", np.ones(rir_shape, dtype=rir_dtype))
        return directory, root, {"representation": representation, "sample_rate_hz": 24000, "num_samples": 120000, "num_channels": channels, "dtype": "float32", "audio_path": "a.wav", "rir_path": "r.npy"}

    def test_N16_to_N22_payload_errors_are_isolated_and_exact(self):
        cases = [("N16", "binaural", (120000, 2), np.float32, (10, 2), np.float32, "render RIR missing", "missing.npy"), ("N17", "binaural", (120000, 2), np.int16, (10, 2), np.float32, "WAV dtype mismatch", "r.npy"), ("N18", "binaural", (120000, 1), np.float32, (10, 2), np.float32, "WAV payload shape/rate mismatch", "r.npy"), ("N19", "foa", (120000, 2), np.float32, (4, 10), np.float32, "WAV payload shape/rate mismatch", "r.npy"), ("N20", "binaural", (120000, 2), np.float32, (2, 10), np.float32, "RIR channel payload mismatch", "r.npy"), ("N21", "foa", (120000, 4), np.float32, (10, 4), np.float32, "RIR channel payload mismatch", "r.npy"), ("N22", "binaural", (120000, 2), np.float32, (10, 2), np.float64, "RIR dtype mismatch", "r.npy")]
        for case, representation, wav_shape, wav_dtype, rir_shape, rir_dtype, message, rir_path in cases:
            with self.subTest(case=case):
                directory, root, row = self._payload_fixture(representation, wav_shape, wav_dtype, rir_shape, rir_dtype)
                try:
                    row["rir_path"] = rir_path if case != "N16" else "missing.npy"
                    with self.assertRaisesRegex(PilotPlanValidationError, "^" + message + "$" ):
                        validate_render_record_payload(root, row)
                finally:
                    directory.cleanup()

    def test_N23_N25_strict_journal_and_N24_N26_payload_entry(self):
        expected = {("ep{:03d}".format(i), representation) for i in range(960) for representation in ("binaural", "foa")}
        valid_rows = [{"episode_id": episode, "representation": representation, "render_status": "complete"} for episode, representation in sorted(expected)]
        duplicate = list(valid_rows); duplicate[-1] = dict(duplicate[-2])
        orphan = list(valid_rows); orphan[-1] = {"episode_id": "orphan", "representation": "foa", "render_status": "failed"}
        for case, rows in (("N23", duplicate), ("N25", orphan)):
            with self.subTest(case=case), self.assertRaises(PilotPlanValidationError): validate_render_rows(rows, expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "manifests").mkdir(); (root / "manifests/episodes.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(PilotPlanValidationError, "^payload requires all 960 unique episodes$"):
                _payload_expected_keys(root)
            (root / "_SUCCESS").write_text("premature\n")
            with self.assertRaisesRegex(PilotPlanValidationError, "^dataset already finalized$"):
                validate_render_payload(root)

    def test_good_fixture_normal_and_optimized(self):
        code = "from tools.clsdoa_v1.scheduler import schedule_block; b=schedule_block('train',0,'Replica');\nif len(b['distance']) != 28: raise SystemExit(1)"
        for flag in ([], ["-O"]): self.assertEqual(subprocess.run([sys.executable] + flag + ["-c", code], capture_output=True).returncode, 0)


if __name__ == "__main__": unittest.main()
