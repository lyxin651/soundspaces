import json
import tempfile
import unittest
from pathlib import Path

from active_audition.data.manifest import ManifestError, write_plan_manifests
from active_audition.types import Candidate, EpisodeSpec, ListenerPose, SourceSpec


def fixture_episode():
    return EpisodeSpec(
        "v0.1",
        "ep_000001",
        "replica.office_0",
        1,
        SourceSpec((1.0, 0.5, 1.0), "golden_probe_v0", 0.0, 5.0, 0.0),
        ListenerPose((0.0, 0.0, 0.0), (0.0, 1.5, 0.0), 0.0),
    )


def fixture_candidate(candidate_id):
    return Candidate(
        "ep_000001", candidate_id, "rotation", None, 0.0, 45.0,
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.5, 0.0), 45.0,
        0.0, 0.0, None, True, False, None, True, None,
    )


class ManifestTests(unittest.TestCase):
    def test_jsonl_is_sorted_utf8_atomic_and_unique(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_plan_manifests(
                temp_dir, fixture_episode(), [fixture_candidate("rot_right_45"), fixture_candidate("rot_left_45")]
            )
            candidates = Path(paths["candidates"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(row)["candidate_id"] for row in candidates], ["rot_left_45", "rot_right_45"])
            self.assertTrue(Path(paths["episodes"]).read_bytes().endswith(b"\n"))
            self.assertFalse((Path(temp_dir) / "_SUCCESS").exists())

    def test_duplicate_candidate_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ManifestError):
                write_plan_manifests(
                    temp_dir, fixture_episode(), [fixture_candidate("rot_left_45"), fixture_candidate("rot_left_45")]
                )


if __name__ == "__main__":
    unittest.main()
