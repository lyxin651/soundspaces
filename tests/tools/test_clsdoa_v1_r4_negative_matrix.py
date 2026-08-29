import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.io import wavfile

from tools.clsdoa_v1.integrity import IntegrityError, verify_plan_integrity
from tools.clsdoa_v1.orchestration import upsert_render_record
from tools.clsdoa_v1.scheduler import schedule_block
from tools.clsdoa_v1.validate_pilot_plan import (
    PilotPlanValidationError,
    validate_episode_ids,
    validate_episode_membership,
    validate_family_block,
    validate_render_payload,
    validate_render_record_payload,
    validate_render_rows,
    validate_scene_set,
    validate_source_reuse,
)


class R4NegativeMatrixTests(unittest.TestCase):
    def test_family_quota_negative_matrix(self):
        good = schedule_block("train", 0, "Replica")
        for field in ("distance", "elevation", "azimuth", "gain"):
            block = [{"diagnostics": {"distance_bin": good["distance"][i], "elevation_bin": good["elevation"][i], "azimuth_bin": good["azimuth"][i], "gain_bin": good["gain"][i]}} for i in range(28)]
            block[0]["diagnostics"][{"distance": "distance", "elevation": "elevation", "azimuth": "azimuth", "gain": "gain"}[field] + "_bin"] = "invalid"
            with self.assertRaises(PilotPlanValidationError):
                validate_family_block(block, 0, "train", "Replica")

    def test_scene_and_source_negative_matrix_are_independent(self):
        episode = {"split": "train", "scene": {"scene_id": "scene", "scene_family": "Replica"}, "source": {"base_clip_id": "base"}}
        sources = [{"base_clip_id": "base", "split": "train"}]
        scenes = {"scenes": {"scene": {"split": "train", "scene_family": "Replica", "admitted": "PASS"}}}
        cases = []
        source_bad = dict(episode, source={"base_clip_id": "base"})
        sources_bad = [{"base_clip_id": "base", "split": "val"}]
        cases.append((source_bad, sources_bad, scenes))
        scene_bad = dict(episode, scene={"scene_id": "scene", "scene_family": "Replica"})
        scenes_bad = {"scenes": {"scene": {"split": "val", "scene_family": "Replica", "admitted": "PASS"}}}
        cases.append((scene_bad, sources, scenes_bad))
        family_bad = dict(episode, scene={"scene_id": "scene", "scene_family": "MP3D"})
        cases.append((family_bad, sources, scenes))
        for row, source_rows, scene_rows in cases:
            with self.assertRaises(PilotPlanValidationError):
                validate_episode_membership([row], source_rows, scene_rows)

    def test_scene_set_ids_duplicate_and_reuse_negatives(self):
        scenes = {"scenes": {"pass": {"admitted": "PASS"}, "fail": {"admitted": "FAIL"}}}
        validate_scene_set(("pass",), scenes)
        with self.assertRaises(PilotPlanValidationError):
            validate_scene_set(("pass", "fail"), scenes)
        with self.assertRaises(PilotPlanValidationError):
            validate_episode_ids([{"episode_id": "same"}, {"episode_id": "same"}])
        rows = [{"label": {"class_id": 0}, "split": "train", "source": {"base_clip_id": "a"}}] * 3
        rows += [{"label": {"class_id": 0}, "split": "train", "source": {"base_clip_id": "b"}}]
        with self.assertRaises(PilotPlanValidationError):
            validate_source_reuse(rows, 0, "train")

    def test_integrity_sha_drift_matrix(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            (root / "manifests/episodes.jsonl").write_text("episodes\n")
            (root / "config_resolved.yaml").write_text("storage:\n  require_rir: true\n")
            (root / "resources.lock.json").write_text("{}\n")
            def sha(path):
                import hashlib
                return hashlib.sha256(path.read_bytes()).hexdigest()
            source_sha, scene_sha = sha(repo / "registries/source_audio.csv"), sha(repo / "registries/clsdoa_v1_scenes.yaml")
            base = {"episodes_sha256": sha(root / "manifests/episodes.jsonl"), "config_resolved_sha256": sha(root / "config_resolved.yaml"), "resources_lock_sha256": sha(root / "resources.lock.json"), "source_registry_sha256": source_sha, "scene_registry_sha256": scene_sha, "plan_generation_code_commit": "clean"}
            (root / "identity.json").write_text(json.dumps({"generation_code_commit": "clean", "source_registry_sha256": source_sha, "scene_registry_sha256": scene_sha}))
            for field in ("episodes_sha256", "config_resolved_sha256", "resources_lock_sha256"):
                broken = dict(base, **{field: "0" * 64})
                (root / "manifests/plan.lock.json").write_text(json.dumps(broken))
                with self.assertRaises(IntegrityError):
                    verify_plan_integrity(root, repo)

    def test_payload_negative_matrix_and_strict_journal(self):
        from tools.clsdoa_v1.validate_pilot_plan import validate_render_record_payload
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wavfile.write(str(root / "a.wav"), 24000, np.ones((120000, 2), dtype=np.float32))
            np.save(root / "b.npy", np.ones((12, 2), dtype=np.float32))
            row = {"representation": "binaural", "sample_rate_hz": 24000, "num_samples": 120000, "num_channels": 2, "dtype": "float32", "audio_path": "a.wav", "rir_path": "b.npy"}
            for bad in ("missing", "dtype", "shape"):
                if bad == "missing":
                    row["rir_path"] = "missing.npy"
                elif bad == "dtype":
                    np.save(root / "b.npy", np.ones((12, 2), dtype=np.float64)); row["rir_path"] = "b.npy"
                else:
                    np.save(root / "b.npy", np.ones((2, 12), dtype=np.float32)); row["rir_path"] = "b.npy"
                with self.assertRaises(PilotPlanValidationError):
                    validate_render_record_payload(root, row)
            with self.assertRaises(PilotPlanValidationError):
                validate_render_payload(root)

    def test_good_and_bad_payload_gate_subprocesses(self):
        code = "from tools.clsdoa_v1.validate_pilot_plan import PilotPlanValidationError; raise PilotPlanValidationError('expected')"
        for optimize in (False, True):
            command = [sys.executable] + (["-O"] if optimize else []) + ["-c", code]
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_good_fixture_passes_in_normal_and_optimized_python(self):
        code = "from tools.clsdoa_v1.scheduler import schedule_block; b=schedule_block('train',0,'Replica');\nif len(b['distance']) != 28: raise SystemExit(1)"
        for optimize in (False, True):
            command = [sys.executable] + (["-O"] if optimize else []) + ["-c", code]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_strict_journal_negative_cases_are_individually_rejected(self):
        expected = {("ep", "binaural"), ("ep", "foa")}
        cases = {
            "extra_failed_orphan": [{"episode_id": "ep", "representation": "binaural", "render_status": "complete"}, {"episode_id": "orphan", "representation": "foa", "render_status": "failed"}],
            "duplicate_representation": [{"episode_id": "ep", "representation": "binaural", "render_status": "complete"}] * 2,
            "fewer_than_960": [{"episode_id": "ep", "representation": "binaural", "render_status": "complete"}],
        }
        for name, rows in cases.items():
            with self.subTest(name=name), self.assertRaises(PilotPlanValidationError):
                validate_render_rows(rows, expected)


if __name__ == "__main__":
    unittest.main()
