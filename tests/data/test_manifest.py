import json
import tempfile
import unittest
from pathlib import Path

from active_audition.data.manifest import (
    ManifestError,
    read_viewpoint_manifest,
    write_plan_manifests,
    write_viewpoint_manifest,
)
from active_audition.types import Candidate, EpisodeSpec, ListenerPose, SourceSpec


def fixture_episode(episode_id):
    return EpisodeSpec(
        "v0.1",
        episode_id,
        "replica.office_0",
        1,
        SourceSpec((1.0, 0.5, 1.0), "golden_probe_v0", 0.0, 5.0, 0.0),
        ListenerPose((0.0, 0.0, 0.0), (0.0, 1.5, 0.0), 0.0),
    )


def fixture_candidate(episode_id, candidate_id):
    return Candidate(
        episode_id, candidate_id, "rotation", None, 0.0, 45.0,
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.5, 0.0), 45.0,
        0.0, 0.0, 0.0, True, True, None, True, None,
    )


class ManifestTests(unittest.TestCase):
    def test_multi_episode_jsonl_is_sorted_unique_and_atomic(self):
        episodes = [fixture_episode("ep_000002"), fixture_episode("ep_000001")]
        candidates = [
            fixture_candidate(episode_id, "rot_right_45")
            for episode_id in ("ep_000002", "ep_000001")
        ]
        candidates += [
            fixture_candidate(episode_id, "rot_left_45")
            for episode_id in ("ep_000002", "ep_000001")
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_plan_manifests(temp_dir, reversed(episodes), reversed(candidates))
            episode_rows = Path(paths["episodes"]).read_text(encoding="utf-8").splitlines()
            candidate_rows = Path(paths["candidates"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                [json.loads(row)["episode_id"] for row in episode_rows],
                ["ep_000001", "ep_000002"],
            )
            self.assertEqual(
                [(json.loads(row)["episode_id"], json.loads(row)["candidate_id"])
                 for row in candidate_rows],
                [
                    ("ep_000001", "rot_left_45"),
                    ("ep_000001", "rot_right_45"),
                    ("ep_000002", "rot_left_45"),
                    ("ep_000002", "rot_right_45"),
                ],
            )
            before = Path(paths["candidates"]).read_bytes()
            write_plan_manifests(temp_dir, reversed(episodes), reversed(candidates))
            self.assertEqual(before, Path(paths["candidates"]).read_bytes())
            self.assertTrue(Path(paths["episodes"]).read_bytes().endswith(b"\n"))
            self.assertFalse((Path(temp_dir) / "_SUCCESS").exists())

    def test_two_episode_batch_has_twelve_candidate_keys(self):
        episodes = [fixture_episode("ep_000001"), fixture_episode("ep_000002")]
        candidates = [
            fixture_candidate(episode.episode_id, "candidate_{:02d}".format(index))
            for episode in episodes
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_plan_manifests(temp_dir, episodes, candidates)
            rows = [
                json.loads(line)
                for line in Path(paths["candidates"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 12)
            self.assertEqual(len({(row["episode_id"], row["candidate_id"]) for row in rows}), 12)

    def test_duplicate_candidate_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ManifestError):
                write_plan_manifests(
                    temp_dir,
                    [fixture_episode("ep_000001")],
                    [
                        fixture_candidate("ep_000001", "rot_left_45"),
                        fixture_candidate("ep_000001", "rot_left_45"),
                    ],
                )

    def test_multi_episode_viewpoint_manifest_is_sorted_unique_and_atomic(self):
        rows = [
            {
                "episode_id": episode_id,
                "viewpoint_id": "viewpoint_{:02d}".format(index),
                "audio_path": "audio/{}/{}.wav".format(episode_id, index),
            }
            for episode_id in ("ep_000002", "ep_000001")
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_viewpoint_manifest(temp_dir, reversed(rows))
            before = path.read_bytes()
            self.assertEqual(len(read_viewpoint_manifest(temp_dir)), 12)
            self.assertEqual(
                [(row["episode_id"], row["viewpoint_id"]) for row in read_viewpoint_manifest(temp_dir)[:2]],
                [("ep_000001", "viewpoint_00"), ("ep_000001", "viewpoint_01")],
            )
            write_viewpoint_manifest(temp_dir, reversed(rows))
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
