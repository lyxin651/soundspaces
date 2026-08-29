import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
