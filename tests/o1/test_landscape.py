import unittest
from dataclasses import replace

from active_audition.o1.landscape import (
    BASELINES,
    O1PoseScore,
    _baseline_rows,
    _feasible,
    _nearest_position_ids,
    _selection,
    _transfer_selection,
    _uniform,
)


def _score(pose_id, error_count=0, motion=1.0, episode="episode"):
    return O1PoseScore(
        schema_version="active-asr-o1-landscape-v1",
        score_id="unused",
        block_id="block-1",
        episode_id=episode,
        role="selection" if episode.startswith("selection") else "evaluation",
        utterance_id=episode,
        frontend="mean_lr",
        pose_id=pose_id,
        position_id="position-1" if pose_id == "pose-a" else "position-2",
        yaw_id="yaw-" + pose_id,
        geometry_id="geometry-1",
        geometry_legality="LEGAL",
        actual_snapped_base_xyz=(0.0, 0.0, 0.0),
        sensor_xyz=(0.0, 1.5, 0.0),
        yaw_deg=0.0,
        motion_cost_sec=motion,
        budget_feasible_at_manifest_budget=True,
        S=error_count,
        D=0,
        I=0,
        N=10,
        WER=error_count / 10.0,
        reference="REFERENCE",
        hypothesis="HYPOTHESIS",
    )


class O1LandscapeUnitTests(unittest.TestCase):
    def test_uniform_is_exact_arithmetic_mean_and_has_no_selected_pose(self):
        rows = [_score("pose-a", error_count=1), _score("pose-b", error_count=3)]
        record = _uniform("block-1", "mean_lr", 2.0, rows, rows, "episode-1")
        self.assertEqual(record.status, "SELECTED")
        self.assertIsNone(record.selected_pose_id)
        self.assertEqual(record.expected_WER, 0.2)
        self.assertEqual(record.action_distribution, {"pose-a": 0.5, "pose-b": 0.5})

    def test_error_rate_then_motion_then_pose_id_tie_break(self):
        rows = [_score("pose-z", error_count=1, motion=2.0), _score("pose-a", error_count=1, motion=1.0)]
        record = _selection("block-1", "mean_lr", 2.0, "Local-Oracle(B)", "episode", rows, rows, episode_id="episode-1")
        self.assertEqual(record.selected_pose_id, "pose-a")
        slower = [_score("pose-z", error_count=1, motion=1.0), _score("pose-a", error_count=1, motion=1.0)]
        record = _selection("block-1", "mean_lr", 2.0, "Local-Oracle(B)", "episode", slower, slower, episode_id="episode-1")
        self.assertEqual(record.selected_pose_id, "pose-a")

    def test_transfer_selection_is_frozen_before_evaluation(self):
        selection = [_score("pose-a", error_count=0, motion=1.0, episode="selection-0"),
                     _score("pose-b", error_count=2, motion=1.0, episode="selection-0")]
        selection += [_score("pose-a", error_count=0, motion=1.0, episode="selection-1"),
                      _score("pose-b", error_count=2, motion=1.0, episode="selection-1")]
        evaluation = [_score("pose-a", error_count=4, motion=1.0, episode="evaluation-0"),
                      _score("pose-b", error_count=0, motion=1.0, episode="evaluation-0")]
        evaluation += [_score("pose-a", error_count=4, motion=1.0, episode="evaluation-1"),
                       _score("pose-b", error_count=0, motion=1.0, episode="evaluation-1")]
        block = {
            "block_record": {"block_id": "block-1"},
            "episodes": [
                {"episode_id": "selection-0", "role": "selection"},
                {"episode_id": "selection-1", "role": "selection"},
                {"episode_id": "evaluation-0", "role": "evaluation"},
                {"episode_id": "evaluation-1", "role": "evaluation"},
            ],
            "geometry_record": {"target_world_pose": {"position_xyz": [0.0, 1.5, -1.0]}},
        }
        selection_record, evaluation_records = _transfer_selection(
            block, "mean_lr", 2.0, selection, evaluation, "O_transfer"
        )
        self.assertEqual(selection_record.selected_pose_id, "pose-a")
        self.assertEqual([row.selected_pose_id for row in evaluation_records], ["pose-a", "pose-a"])

    def test_budget_filtering_and_stay_only_at_zero_seconds(self):
        rows = [_score("pose-a", motion=0.0), _score("pose-b", motion=2.01)]
        self.assertEqual([row.pose_id for row in _feasible(rows, 0.0)], ["pose-a"])
        self.assertEqual([row.pose_id for row in _feasible(rows, 2.0)], ["pose-a"])
        stay_rows, reason = _baseline_rows(
            {"geometry_record": {"target_world_pose": {"position_xyz": [0.0, 1.5, -1.0]}}},
            rows,
            0.0,
            "Stay",
        )
        self.assertIsNone(reason)
        self.assertEqual([row.pose_id for row in stay_rows], ["pose-a"])

    def test_nearest_position_tolerance_is_set_based(self):
        rows = [
            _score("pose-a", motion=1.0),
            _score("pose-b", motion=1.0),
            _score("pose-c", motion=1.0),
        ]
        rows[0] = replace(rows[0], actual_snapped_base_xyz=(1.0, 0.0, 0.0))
        rows[1] = replace(rows[1], actual_snapped_base_xyz=(1.005, 0.0, 0.0))
        rows[2] = replace(rows[2], actual_snapped_base_xyz=(2.0, 0.0, 0.0))
        self.assertEqual(
            _nearest_position_ids(rows, (0.0, 0.0, 0.0), tolerance_m=0.01),
            ("position-1", "position-2"),
        )

    def test_baseline_surface_and_frontend_policy_do_not_offer_best_ear(self):
        self.assertIn("Stay", BASELINES)
        self.assertIn("Uniform-random feasible", BASELINES)
        self.assertIn("Nearest-best-heading", BASELINES)
        self.assertNotIn("best-ear", BASELINES)
