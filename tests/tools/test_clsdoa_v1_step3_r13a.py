import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from tools.clsdoa_v1.git_identity import GitIdentityError, current_clean_head
from tools.clsdoa_v1.scheduler import (
    azimuth_schedule,
    distance_schedule,
    elevation_schedule,
    gain_schedule,
    schedule_block,
)


class Pilot004SchedulerTests(unittest.TestCase):
    def test_all_family_micro_patterns_are_deterministic_and_counted(self):
        for class_id in range(12):
            for split in ("train", "val", "test"):
                for family in ("Replica", "MP3D"):
                    first = schedule_block(split, class_id, family)
                    second = schedule_block(split, class_id, family)
                    self.assertEqual(first, second)
                    expected = 28 if split == "train" else 6
                    self.assertEqual({key: len(value) for key, value in first.items()}, {key: expected for key in first})
                    self.assertNotEqual(first["distance"], first["elevation"])
        self.assertEqual(sorted(distance_schedule("train", 0, "Replica")).count("near"), 11)
        self.assertEqual(sorted(elevation_schedule("train", 0, "MP3D")).count("small"), 21)
        self.assertEqual(set(azimuth_schedule("val", 0, "Replica")) | set(azimuth_schedule("val", 0, "MP3D")), set(range(8)))
        self.assertEqual(set(gain_schedule("test", 4, "Replica")) | set(gain_schedule("test", 4, "MP3D")), set(range(8)))

    def test_duplicate_occurrences_are_stably_permuted_and_gain_pattern_is_independent(self):
        from tools.clsdoa_v1.scheduler import stable_permutation
        values = ["near"] * 8 + ["mid"] * 8
        first = stable_permutation(values, "distance_schedule_v2", 1, "train", "Replica")
        self.assertEqual(first, stable_permutation(values, "distance_schedule_v2", 1, "train", "Replica"))
        self.assertNotEqual(first, values)
        self.assertNotEqual(azimuth_schedule("train", 1, "Replica"), gain_schedule("train", 1, "Replica"))


class Pilot004IdentityTests(unittest.TestCase):
    def test_clean_and_dirty_git_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked").write_text("clean\n")
            subprocess.run(["git", "add", "tracked"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            self.assertEqual(len(current_clean_head(root)), 40)
            (root / "tracked").write_text("dirty\n")
            with self.assertRaises(GitIdentityError):
                current_clean_head(root)


class Pilot004CommandSurfaceTests(unittest.TestCase):
    def test_validate_and_finalize_plan_only_are_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            (root / "manifests/episodes.jsonl").write_text("")
            command = [sys.executable, "tools/clsdoa_v1/pilot_dataset.py", "finalize", "--root", str(root)]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_resume_upsert_preserves_completed_other_representation(self):
        from tools.clsdoa_v1.orchestration import upsert_render_record
        binaural = {"episode_id": "ep", "representation": "binaural", "render_status": "complete"}
        foa = {"episode_id": "ep", "representation": "foa", "render_status": "complete"}
        rows = upsert_render_record([binaural], foa)
        self.assertEqual({(row["episode_id"], row["representation"]) for row in rows}, {("ep", "binaural"), ("ep", "foa")})

    def test_membership_split_and_family_gates(self):
        from tools.clsdoa_v1.validate_pilot_plan import PilotPlanValidationError, validate_episode_membership
        episode = {"split": "train", "scene": {"scene_id": "s", "scene_family": "Replica"}, "source": {"base_clip_id": "b"}}
        source = [{"base_clip_id": "b", "split": "train"}]
        scenes = {"scenes": {"s": {"split": "train", "scene_family": "Replica"}}}
        validate_episode_membership([episode], source, scenes)
        for bad in ({"split": "val"}, {"scene_family": "MP3D"}):
            broken = dict(episode)
            if "split" in bad:
                broken["split"] = bad["split"]
            else:
                broken["scene"] = dict(episode["scene"], scene_family=bad["scene_family"])
            with self.assertRaises(PilotPlanValidationError):
                validate_episode_membership([broken], source, scenes)

    def test_payload_record_rejects_mono_wav_and_wrong_rir_axis(self):
        from tools.clsdoa_v1.validate_pilot_plan import PilotPlanValidationError, validate_render_record_payload
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wavfile.write(str(root / "audio.wav"), 24000, np.ones((120000, 1), dtype=np.float32))
            np.save(str(root / "rir.npy"), np.ones((2, 12), dtype=np.float32))
            row = {"representation": "binaural", "sample_rate_hz": 24000, "num_samples": 120000, "num_channels": 2, "dtype": "float32", "audio_path": "audio.wav", "rir_path": "rir.npy"}
            with self.assertRaises(PilotPlanValidationError):
                validate_render_record_payload(root, row)
            wavfile.write(str(root / "audio.wav"), 24000, np.ones((120000, 2), dtype=np.float32))
            with self.assertRaises(PilotPlanValidationError):
                validate_render_record_payload(root, row)

    def test_plan_rejects_dirty_repo_before_creating_root_and_nonempty_root(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new-root"
            command = [sys.executable, "tools/clsdoa_v1/pilot_dataset.py", "plan", "--config", "configs/active_audition/clsdoa_v1_pilot_004.yaml", "--root", str(root)]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, env=dict(os.environ))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(root.exists())
            root.mkdir()
            (root / "sentinel").write_text("keep\n")
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, env=dict(os.environ))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((root / "sentinel").read_text(), "keep\n")
if __name__ == "__main__":
    unittest.main()
