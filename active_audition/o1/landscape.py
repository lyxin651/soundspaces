"""Strict, result-only O1 landscape analysis.

This module consumes frozen pose metadata and already-decoded ASR records.  It
does not render, decode, select sources, or modify A4 artifacts.  The analysis
is intentionally descriptive: its Oracle records are exploratory diagnostics,
not policy or held-out validation results.
"""

from dataclasses import dataclass
from fractions import Fraction
import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from active_audition.a4.identity import identity_sha256, stable_id
from active_audition.evaluation.asr_metrics import error_counts


O1_SCHEMA_VERSION = "active-asr-o1-landscape-v1"
O1_ALGORITHM_IDENTITY = "active-asr-o1-descriptive-landscape-v1"
O1_TIE_BREAK_IDENTITY = "active-asr-o1-error-rate-motion-pose-id-v1"
O1_FRONTEND_POLICY_IDENTITY = "active-asr-o1-mean-lr-primary-fixed-channel-sensitivity-v1"
O1_DRY_RUN_KIND = "FAMILIAR_ENGINEERING_DRY_RUN"
O1_REAL_KIND = "O1_EXPLORATORY_SINGLE_SCENE"
FRONTENDS = ("mean_lr", "fixed_L", "fixed_R")
BUDGETS_SEC = (0.0, 2.0, 5.0, 10.0)
BASELINES = (
    "Stay",
    "Uniform-random feasible",
    "Rotate-best",
    "Nearest-face-target",
    "Nearest-best-heading",
    "Translation-fixed-yaw",
    "Local-Oracle(B)",
    "Nearest-geodesic + best-heading",
    "Max-SNR",
)
TRANSFER_BASELINES = ("O_transfer", "Rotate-transfer", "Nearest-best-heading-transfer")
SELECTION_LABELS = BASELINES + TRANSFER_BASELINES


class O1AnalysisError(ValueError):
    """Raised for malformed or semantically inconsistent O1 evidence."""


def o1_canonical_json_bytes(value: Any) -> bytes:
    """Canonical O1 serialization permits result fields as analysis evidence."""

    return json.dumps(
        _plain(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _o1_result_id(namespace: str, payload: Mapping[str, Any]) -> str:
    """Hash O1 evidence without applying A4 semantic-input restrictions.

    A4 stable IDs intentionally reject result-dependent fields such as WER.
    O1 landscape summaries are result artifacts, so their identity must be
    content-addressed by the O1 canonical serializer instead of borrowing the
    A4 constructor-input identity boundary.
    """

    envelope = {"namespace": namespace, "payload": _plain(payload)}
    return "{}-{}".format(namespace, hashlib.sha256(o1_canonical_json_bytes(envelope)).hexdigest())


def _exact(value: Mapping[str, Any], expected: Iterable[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise O1AnalysisError("{} must be a mapping".format(path))
    expected_set = set(expected)
    unknown = sorted(set(value) - expected_set)
    missing = sorted(expected_set - set(value))
    if unknown or missing:
        raise O1AnalysisError("{} fields invalid: unknown={}, missing={}".format(path, unknown, missing))


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise O1AnalysisError("{} must be a non-empty string".format(path))
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise O1AnalysisError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise O1AnalysisError("{} must be finite".format(path))
    return result


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise O1AnalysisError("{} must be a non-negative integer".format(path))
    return value


def _positive_int(value: Any, path: str) -> int:
    value = _nonnegative_int(value, path)
    if value <= 0:
        raise O1AnalysisError("{} must be positive".format(path))
    return value


def _xyz(value: Any, path: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise O1AnalysisError("{} must contain three coordinates".format(path))
    return tuple(_finite(item, "{}[{}]".format(path, index)) for index, item in enumerate(value))  # type: ignore[return-value]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _yaw(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _yaw_delta(a: float, b: float) -> float:
    return _yaw(float(b) - float(a))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def _heading_to_target(listener: Sequence[float], target: Sequence[float]) -> float:
    # Frozen A0 convention: forward=-Z, positive yaw is left.
    return _yaw(math.degrees(math.atan2(-(target[0] - listener[0]), -(target[2] - listener[2]))))


def _error_fraction(row: "O1PoseScore") -> Fraction:
    return Fraction(row.error_count, row.N)


def _score_compare(left: "O1PoseScore", right: "O1PoseScore") -> int:
    left_fraction = _error_fraction(left)
    right_fraction = _error_fraction(right)
    if left_fraction != right_fraction:
        return -1 if left_fraction < right_fraction else 1
    if left.motion_cost_sec != right.motion_cost_sec:
        return -1 if left.motion_cost_sec < right.motion_cost_sec else 1
    return -1 if left.pose_id < right.pose_id else (1 if left.pose_id > right.pose_id else 0)


def _best(rows: Sequence["O1PoseScore"]) -> Optional["O1PoseScore"]:
    return min(rows, key=functools.cmp_to_key(_score_compare)) if rows else None


def _motion_best(rows: Sequence["O1PoseScore"]) -> Optional["O1PoseScore"]:
    """Select a geometry-defined candidate without consulting ASR results."""

    return min(rows, key=lambda row: (row.motion_cost_sec, row.pose_id)) if rows else None


def _aggregate(rows: Sequence["O1PoseScore"]) -> Dict[str, Any]:
    if not rows:
        raise O1AnalysisError("cannot aggregate an empty score set")
    result = {field: sum(getattr(row, field) for row in rows) for field in ("S", "D", "I", "N")}
    result["error_count"] = result["S"] + result["D"] + result["I"]
    result["WER"] = float(result["error_count"]) / float(result["N"])
    return result


def _descriptive_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "iqr": None}
    ordered = sorted(float(value) for value in values)
    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / float(len(ordered)),
        "median": percentile(0.5),
        "iqr": percentile(0.75) - percentile(0.25),
    }


@dataclass(frozen=True)
class O1PoseScore:
    schema_version: str
    score_id: str
    block_id: str
    episode_id: str
    role: str
    utterance_id: str
    frontend: str
    pose_id: str
    position_id: str
    yaw_id: str
    geometry_id: str
    geometry_legality: str
    actual_snapped_base_xyz: Tuple[float, float, float]
    sensor_xyz: Tuple[float, float, float]
    yaw_deg: float
    motion_cost_sec: float
    budget_feasible_at_manifest_budget: bool
    S: int
    D: int
    I: int
    N: int
    WER: float
    reference: str
    hypothesis: str
    component_snr_db: Optional[float] = None

    @property
    def error_count(self) -> int:
        return self.S + self.D + self.I

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "episode_id": self.episode_id,
            "role": self.role,
            "utterance_id": self.utterance_id,
            "frontend": self.frontend,
            "pose_id": self.pose_id,
            "position_id": self.position_id,
            "yaw_id": self.yaw_id,
            "geometry_id": self.geometry_id,
            "geometry_legality": self.geometry_legality,
            "actual_snapped_base_xyz": list(self.actual_snapped_base_xyz),
            "sensor_xyz": list(self.sensor_xyz),
            "yaw_deg": self.yaw_deg,
            "motion_cost_sec": self.motion_cost_sec,
            "budget_feasible_at_manifest_budget": self.budget_feasible_at_manifest_budget,
            "S": self.S,
            "D": self.D,
            "I": self.I,
            "N": self.N,
            "component_snr_db": self.component_snr_db,
        }

    def to_payload(self) -> Dict[str, Any]:
        return dict(self.identity_payload(), score_id=self.score_id, WER=self.WER)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1PoseScore":
        expected = tuple(field.name for field in cls.__dataclass_fields__.values())
        _exact(payload, expected, "o1_pose_score")
        value = cls(**dict(payload))
        if value.schema_version != O1_SCHEMA_VERSION:
            raise O1AnalysisError("unsupported O1 pose score schema")
        if value.frontend not in FRONTENDS or value.role not in ("selection", "evaluation"):
            raise O1AnalysisError("invalid frontend or role")
        _xyz(value.actual_snapped_base_xyz, "o1_pose_score.actual_snapped_base_xyz")
        _xyz(value.sensor_xyz, "o1_pose_score.sensor_xyz")
        for name in ("S", "D", "I"):
            _nonnegative_int(getattr(value, name), "o1_pose_score." + name)
        N = _positive_int(value.N, "o1_pose_score.N")
        expected_wer = float(value.error_count) / float(N)
        if not math.isclose(value.WER, expected_wer, rel_tol=0.0, abs_tol=1.0e-15):
            raise O1AnalysisError("WER is not the unclamped (S+D+I)/N value")
        if value.score_id != stable_id("o1-pose-score", value.identity_payload()):
            raise O1AnalysisError("score_id does not match canonical score payload")
        return value

    @property
    def identity_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


def score_from_diagnostic(row: Mapping[str, Any], pose: Mapping[str, Any], episode: Mapping[str, Any]) -> O1PoseScore:
    """Recompute counts from the frozen reference/hypothesis, never trust summary fields."""

    required = ("block_id", "episode_id", "frontend", "pose_id", "role", "utterance_id", "reference", "hypothesis")
    for field in required:
        _string(row.get(field), "diagnostic." + field)
    metrics = error_counts(row["reference"], row["hypothesis"])
    for field in ("S", "D", "I", "N"):
        if field in row and int(row[field]) != int(metrics[field]):
            raise O1AnalysisError("diagnostic {} disagrees with recomputed error count".format(field))
    if "WER" in row and not math.isclose(float(row["WER"]), float(metrics["WER"]), rel_tol=0.0, abs_tol=1.0e-15):
        raise O1AnalysisError("diagnostic WER disagrees with recomputation")
    payload = {
        "schema_version": O1_SCHEMA_VERSION,
        "block_id": row["block_id"],
        "episode_id": row["episode_id"],
        "role": row["role"],
        "utterance_id": row["utterance_id"],
        "frontend": row["frontend"],
        "pose_id": row["pose_id"],
        "position_id": pose["position_id"],
        "yaw_id": pose["yaw_id"],
        "geometry_id": pose["geometry_id"],
        "geometry_legality": pose["geometry_legality"],
        "actual_snapped_base_xyz": tuple(pose["actual_snapped_base_xyz"]),
        "sensor_xyz": tuple(pose["sensor_xyz"]),
        "yaw_deg": float(pose["yaw_deg"]),
        "motion_cost_sec": float(pose["total_cost_sec"]),
        "budget_feasible_at_manifest_budget": bool(pose["budget_feasible"]),
        "S": int(metrics["S"]),
        "D": int(metrics["D"]),
        "I": int(metrics["I"]),
        "N": int(metrics["N"]),
        "WER": float(metrics["WER"]),
        "reference": row["reference"],
        "hypothesis": row["hypothesis"],
        "component_snr_db": row.get("component_snr_db"),
    }
    payload["score_id"] = stable_id(
        "o1-pose-score",
        {key: value for key, value in payload.items() if key not in ("score_id", "WER", "reference", "hypothesis")},
    )
    return O1PoseScore.from_payload(payload)


@dataclass(frozen=True)
class O1BaselineSelection:
    schema_version: str
    selection_id: str
    block_id: str
    frontend: str
    budget_sec: float
    baseline: str
    scope: str
    episode_id: Optional[str]
    selection_episode_ids: Tuple[str, ...]
    evaluation_episode_id: Optional[str]
    selected_pose_id: Optional[str]
    candidate_pose_ids: Tuple[str, ...]
    expected_WER: Optional[float]
    selected_error_count: Optional[int]
    selected_reference_words: Optional[int]
    action_distribution: Mapping[str, float]
    status: str
    reason: Optional[str]

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "frontend": self.frontend,
            "budget_sec": self.budget_sec,
            "baseline": self.baseline,
            "scope": self.scope,
            "episode_id": self.episode_id,
            "selection_episode_ids": list(self.selection_episode_ids),
            "evaluation_episode_id": self.evaluation_episode_id,
            "selected_pose_id": self.selected_pose_id,
            "candidate_pose_ids": list(self.candidate_pose_ids),
            "expected_WER": self.expected_WER,
            "selected_error_count": self.selected_error_count,
            "selected_reference_words": self.selected_reference_words,
            "action_distribution": dict(self.action_distribution),
            "status": self.status,
            "reason": self.reason,
        }

    def to_payload(self) -> Dict[str, Any]:
        return dict(self.identity_payload(), selection_id=self.selection_id)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1BaselineSelection":
        expected = tuple(field.name for field in cls.__dataclass_fields__.values())
        _exact(payload, expected, "o1_baseline_selection")
        value = cls(**dict(payload))
        if value.schema_version != O1_SCHEMA_VERSION:
            raise O1AnalysisError("unsupported O1 selection schema")
        if value.frontend not in FRONTENDS or value.baseline not in SELECTION_LABELS:
            raise O1AnalysisError("invalid O1 selection label")
        if value.scope not in ("episode", "block_selection", "block_evaluation"):
            raise O1AnalysisError("invalid O1 selection scope")
        _finite(value.budget_sec, "o1_baseline_selection.budget_sec")
        if value.status not in ("SELECTED", "NOT_AVAILABLE"):
            raise O1AnalysisError("invalid O1 selection status")
        if value.status == "SELECTED" and not value.selected_pose_id and value.baseline != "Uniform-random feasible":
            raise O1AnalysisError("selected result lacks selected_pose_id")
        if value.expected_WER is not None:
            _finite(value.expected_WER, "o1_baseline_selection.expected_WER")
        if value.selected_error_count is not None:
            _nonnegative_int(value.selected_error_count, "o1_baseline_selection.selected_error_count")
        if value.selected_reference_words is not None:
            _positive_int(value.selected_reference_words, "o1_baseline_selection.selected_reference_words")
        distribution = dict(value.action_distribution)
        if value.status == "SELECTED":
            total = sum(float(probability) for probability in distribution.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise O1AnalysisError("action distribution must sum to one")
        if value.selection_id != stable_id("o1-selection", value.identity_payload()):
            raise O1AnalysisError("selection_id does not match canonical payload")
        return value


@dataclass(frozen=True)
class O1BlockLandscape:
    schema_version: str
    landscape_id: str
    block_id: str
    frontend: str
    budget_sec: float
    pose_score_count: int
    baseline_selection_ids: Tuple[str, ...]
    same_oracle_selection_ids: Tuple[str, ...]
    transfer_selection_ids: Tuple[str, ...]
    diagnostic_flags: Tuple[str, ...]

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "frontend": self.frontend,
            "budget_sec": self.budget_sec,
            "pose_score_count": self.pose_score_count,
            "baseline_selection_ids": list(self.baseline_selection_ids),
            "same_oracle_selection_ids": list(self.same_oracle_selection_ids),
            "transfer_selection_ids": list(self.transfer_selection_ids),
            "diagnostic_flags": list(self.diagnostic_flags),
        }

    def to_payload(self) -> Dict[str, Any]:
        return dict(self.identity_payload(), schema_version=self.schema_version, landscape_id=self.landscape_id)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1BlockLandscape":
        expected = tuple(field.name for field in cls.__dataclass_fields__.values())
        _exact(payload, expected, "o1_block_landscape")
        value = cls(**dict(payload))
        if value.schema_version != O1_SCHEMA_VERSION or value.landscape_id != stable_id("o1-landscape", value.identity_payload()):
            raise O1AnalysisError("block landscape identity/schema mismatch")
        return value


@dataclass(frozen=True)
class O1LandscapeSummary:
    schema_version: str
    summary_id: str
    analysis_kind: str
    algorithm_identity: str
    frontend_policy_identity: str
    infrastructure_contract_sha256: str
    source_manifest_identity: str
    blocks: int
    episodes: int
    poses: int
    attempted: int
    valid: int
    incomplete: int
    summary: Mapping[str, Any]

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "analysis_kind": self.analysis_kind,
            "algorithm_identity": self.algorithm_identity,
            "frontend_policy_identity": self.frontend_policy_identity,
            "infrastructure_contract_sha256": self.infrastructure_contract_sha256,
            "source_manifest_identity": self.source_manifest_identity,
            "blocks": self.blocks,
            "episodes": self.episodes,
            "poses": self.poses,
            "attempted": self.attempted,
            "valid": self.valid,
            "incomplete": self.incomplete,
            "summary": _plain(self.summary),
        }

    def to_payload(self) -> Dict[str, Any]:
        return dict(self.identity_payload(), summary_id=self.summary_id)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "O1LandscapeSummary":
        expected = tuple(field.name for field in cls.__dataclass_fields__.values())
        _exact(payload, expected, "o1_landscape_summary")
        value = cls(**dict(payload))
        if value.schema_version != O1_SCHEMA_VERSION or value.algorithm_identity != O1_ALGORITHM_IDENTITY:
            raise O1AnalysisError("summary schema/algorithm mismatch")
        if value.summary_id != _o1_result_id("o1-summary", value.identity_payload()):
            raise O1AnalysisError("summary_id does not match canonical summary")
        return value


def _selection(
    block_id: str,
    frontend: str,
    budget: float,
    baseline: str,
    scope: str,
    rows: Sequence[O1PoseScore],
    candidate_rows: Sequence[O1PoseScore],
    episode_id: Optional[str] = None,
    selection_episode_ids: Sequence[str] = (),
    evaluation_episode_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> O1BaselineSelection:
    candidate_ids = tuple(row.pose_id for row in candidate_rows)
    if baseline == "Max-SNR":
        selected = min(
            rows,
            key=lambda row: (-float(row.component_snr_db), row.motion_cost_sec, row.pose_id),
        ) if rows and all(row.component_snr_db is not None for row in rows) else None
    else:
        selected = _best(rows)
    status = "SELECTED" if selected is not None else "NOT_AVAILABLE"
    expected = None if selected is None else selected.WER
    selected_error = None if selected is None else selected.error_count
    selected_N = None if selected is None else selected.N
    if selected is not None:
        distribution = {selected.pose_id: 1.0}
    else:
        distribution = {}
    payload = {
        "schema_version": O1_SCHEMA_VERSION,
        "block_id": block_id,
        "frontend": frontend,
        "budget_sec": float(budget),
        "baseline": baseline,
        "scope": scope,
        "episode_id": episode_id,
        "selection_episode_ids": tuple(selection_episode_ids),
        "evaluation_episode_id": evaluation_episode_id,
        "selected_pose_id": None if selected is None else selected.pose_id,
        "candidate_pose_ids": candidate_ids,
        "expected_WER": expected,
        "selected_error_count": selected_error,
        "selected_reference_words": selected_N,
        "action_distribution": distribution,
        "status": status,
        "reason": reason if reason else (None if selected is not None else "NO_FEASIBLE_CANDIDATE"),
    }
    payload["selection_id"] = stable_id("o1-selection", {key: value for key, value in payload.items() if key != "selection_id"})
    return O1BaselineSelection.from_payload(payload)


def _not_available(block_id: str, frontend: str, budget: float, baseline: str, scope: str, reason: str, **kwargs: Any) -> O1BaselineSelection:
    return _selection(block_id, frontend, budget, baseline, scope, (), (), reason=reason, **kwargs)


def _feasible(rows: Sequence[O1PoseScore], budget: float) -> List[O1PoseScore]:
    # A4 PoseRecord legality is authoritative; budget is recomputed for each O1 curve.
    return [row for row in rows if row.geometry_legality == "LEGAL" and row.motion_cost_sec <= budget + 1.0e-12]


def _uniform(block_id: str, frontend: str, budget: float, rows: Sequence[O1PoseScore], candidate_rows: Sequence[O1PoseScore], episode_id: str) -> O1BaselineSelection:
    if not candidate_rows:
        return _not_available(block_id, frontend, budget, "Uniform-random feasible", "episode", "NO_FEASIBLE_CANDIDATE", episode_id=episode_id)
    errors = sum(row.error_count for row in candidate_rows)
    words = sum(row.N for row in candidate_rows)
    # Uniform expected WER is the arithmetic mean of per-pose WER, not a pooled WER.
    expected = sum(row.WER for row in candidate_rows) / float(len(candidate_rows))
    distribution = {row.pose_id: 1.0 / float(len(candidate_rows)) for row in candidate_rows}
    payload = {
        "schema_version": O1_SCHEMA_VERSION,
        "block_id": block_id,
        "frontend": frontend,
        "budget_sec": float(budget),
        "baseline": "Uniform-random feasible",
        "scope": "episode",
        "episode_id": episode_id,
        "selection_episode_ids": (),
        "evaluation_episode_id": None,
        "selected_pose_id": None,
        "candidate_pose_ids": tuple(row.pose_id for row in candidate_rows),
        "expected_WER": float(expected),
        "selected_error_count": None,
        "selected_reference_words": None,
        "action_distribution": distribution,
        "status": "SELECTED",
        "reason": None,
    }
    payload["selection_id"] = stable_id("o1-selection", {key: value for key, value in payload.items() if key != "selection_id"})
    return O1BaselineSelection.from_payload(payload)


def _nearest_position_ids(rows: Sequence[O1PoseScore], target: Sequence[float], tolerance_m: float = 0.01) -> Tuple[str, ...]:
    positions = {}
    for row in rows:
        positions.setdefault(row.position_id, row.actual_snapped_base_xyz)
    if not positions:
        return ()
    distances = {position_id: _distance(point, target) for position_id, point in positions.items()}
    minimum = min(distances.values())
    return tuple(sorted(position_id for position_id, value in distances.items() if value <= minimum + tolerance_m))


def _aggregate_rows(rows_by_pose: Mapping[str, Sequence[O1PoseScore]], candidate_ids: Sequence[str], motion_rows: Mapping[str, O1PoseScore]) -> List[O1PoseScore]:
    result = []
    for pose_id in candidate_ids:
        rows = rows_by_pose.get(pose_id, ())
        if not rows:
            continue
        aggregate = _aggregate(rows)
        base = motion_rows[pose_id]
        result.append(O1PoseScore(
            schema_version=O1_SCHEMA_VERSION,
            score_id=stable_id("o1-pose-score", {"aggregate": True, "pose_id": pose_id, "rows": [row.score_id for row in rows]}),
            block_id=base.block_id,
            episode_id="aggregate-selection",
            role="selection",
            utterance_id="aggregate-selection",
            frontend=base.frontend,
            pose_id=pose_id,
            position_id=base.position_id,
            yaw_id=base.yaw_id,
            geometry_id=base.geometry_id,
            geometry_legality=base.geometry_legality,
            actual_snapped_base_xyz=base.actual_snapped_base_xyz,
            sensor_xyz=base.sensor_xyz,
            yaw_deg=base.yaw_deg,
            motion_cost_sec=base.motion_cost_sec,
            budget_feasible_at_manifest_budget=base.budget_feasible_at_manifest_budget,
            S=aggregate["S"], D=aggregate["D"], I=aggregate["I"], N=aggregate["N"], WER=aggregate["WER"],
            reference="<aggregate>", hypothesis="<aggregate>", component_snr_db=None,
        ))
    return result


def _baseline_rows(block: Mapping[str, Any], episode_rows: Sequence[O1PoseScore], budget: float, baseline: str) -> Tuple[List[O1PoseScore], Optional[str]]:
    feasible = _feasible(episode_rows, budget)
    if budget == 0.0:
        feasible = [row for row in feasible if row.motion_cost_sec == 0.0]
    if not feasible:
        return [], "NO_FEASIBLE_CANDIDATE"
    initial = min(feasible, key=lambda row: (row.motion_cost_sec, row.pose_id))
    if baseline == "Stay":
        rows = [row for row in feasible if row.motion_cost_sec == 0.0 and row.position_id == initial.position_id]
        return rows[:1], None if rows else "STAY_NOT_AVAILABLE"
    if baseline == "Rotate-best":
        rows = [row for row in feasible if row.position_id == initial.position_id]
        return rows, None if rows else "NO_FEASIBLE_ROTATION"
    if baseline == "Translation-fixed-yaw":
        initial_yaw = block["geometry_record"]["initial_yaw_deg"]
        rows = [row for row in feasible if abs(_yaw_delta(initial_yaw, row.yaw_deg)) <= 1.0e-9]
        return rows, None if rows else "NO_FEASIBLE_FIXED_YAW"
    if baseline in ("Nearest-face-target", "Nearest-best-heading"):
        target = block["geometry_record"]["target_world_pose"]["position_xyz"]
        nearest_ids = _nearest_position_ids(feasible, target)
        rows = [row for row in feasible if row.position_id in nearest_ids]
        if baseline == "Nearest-face-target" and rows:
            desired_by_pose = {
                row.pose_id: abs(_yaw_delta(row.yaw_deg, _heading_to_target(row.sensor_xyz, target)))
                for row in rows
            }
            best_heading = min(desired_by_pose.values())
            rows = [row for row in rows if desired_by_pose[row.pose_id] <= best_heading + 1.0e-12]
            # Face-target is a geometry/action baseline.  Once the nearest
            # position and closest target-facing yaw are fixed, its tie-break
            # is motion cost then stable pose ID; WER must not choose among
            # geometrically tied candidates.
            selected = _motion_best(rows)
            rows = [] if selected is None else [selected]
        return rows, None if rows else "NO_FEASIBLE_NEAREST_POSITION"
    if baseline == "Local-Oracle(B)":
        return feasible, None
    if baseline == "Nearest-geodesic + best-heading":
        return [], "SOURCE_ANCHOR_NOT_PROVIDED"
    if baseline == "Max-SNR":
        with_snr = [row for row in feasible if row.component_snr_db is not None]
        return with_snr, None if with_snr else "COMPONENT_SNR_NOT_AVAILABLE"
    raise O1AnalysisError("unsupported baseline {}".format(baseline))


def load_asr_diagnostics(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise O1AnalysisError("invalid diagnostics JSON at line {}".format(line_number)) from exc
            rows.append(row)
    if not rows:
        raise O1AnalysisError("diagnostics file is empty")
    return rows


def _make_scores(blocks: Sequence[Mapping[str, Any]], diagnostics: Sequence[Mapping[str, Any]]) -> List[O1PoseScore]:
    block_by_id = {block["block_record"]["block_id"]: block for block in blocks}
    pose_by_block = {
        block["block_record"]["block_id"]: {pose["pose_id"]: pose for pose in block["poses"]}
        for block in blocks
    }
    episode_by_id = {
        episode["episode_id"]: episode
        for block in blocks for episode in block["episodes"]
    }
    scores = []
    seen = set()
    for row in diagnostics:
        block = block_by_id.get(row.get("block_id"))
        pose = pose_by_block.get(row.get("block_id"), {}).get(row.get("pose_id"))
        episode = episode_by_id.get(row.get("episode_id"))
        if block is None or pose is None or episode is None:
            raise O1AnalysisError("diagnostic references a non-frozen block/episode/pose")
        key = (row["block_id"], row["episode_id"], row["pose_id"], row["frontend"])
        if key in seen:
            raise O1AnalysisError("duplicate diagnostic record {}".format(key))
        seen.add(key)
        if row["role"] != episode["role"] or row["utterance_id"] != episode["utterance_identity"]["utterance_id"]:
            raise O1AnalysisError("diagnostic episode binding mismatch")
        scores.append(score_from_diagnostic(row, pose, episode))
    expected = len(blocks) * 4 * 48 * len(FRONTENDS)
    if len(scores) != expected:
        raise O1AnalysisError("diagnostics cardinality {} != expected {}".format(len(scores), expected))
    return sorted(scores, key=lambda row: (row.block_id, row.episode_id, row.frontend, row.pose_id))


def _transfer_selection(
    block: Mapping[str, Any],
    frontend: str,
    budget: float,
    selection_scores: Sequence[O1PoseScore],
    evaluation_scores: Sequence[O1PoseScore],
    baseline: str,
) -> Tuple[O1BaselineSelection, List[O1BaselineSelection]]:
    block_id = block["block_record"]["block_id"]
    episodes = [episode["episode_id"] for episode in block["episodes"] if episode["role"] == "selection"]
    by_pose: Dict[str, List[O1PoseScore]] = {}
    motion_rows = {}
    for row in selection_scores:
        by_pose.setdefault(row.pose_id, []).append(row)
        motion_rows[row.pose_id] = row
    all_ids = tuple(sorted(by_pose))
    aggregates = _aggregate_rows(by_pose, all_ids, motion_rows)
    feasible = _feasible(aggregates, budget)
    if baseline == "Rotate-transfer":
        initial_position = min(aggregates, key=lambda row: (row.motion_cost_sec, row.pose_id)).position_id if aggregates else None
        candidates = [row for row in feasible if row.position_id == initial_position]
    elif baseline == "Nearest-best-heading-transfer":
        target = block["geometry_record"]["target_world_pose"]["position_xyz"]
        nearest = _nearest_position_ids(feasible, target)
        candidates = [row for row in feasible if row.position_id in nearest]
    else:
        candidates = feasible
    selected = _best(candidates)
    selection = _selection(
        block_id, frontend, budget, baseline, "block_selection", candidates, candidates,
        selection_episode_ids=episodes, reason=None if selected else "NO_FEASIBLE_SELECTION_CANDIDATE",
    )
    evaluation_records = []
    if selected is not None:
        evaluation_ids = [episode["episode_id"] for episode in block["episodes"] if episode["role"] == "evaluation"]
        for evaluation_id in evaluation_ids:
            rows = [row for row in evaluation_scores if row.episode_id == evaluation_id and row.pose_id == selected.pose_id]
            evaluation_records.append(_selection(
                block_id, frontend, budget, baseline, "block_evaluation", rows, rows,
                evaluation_episode_id=evaluation_id, selection_episode_ids=episodes,
                reason=None if rows else "EVALUATION_SCORE_MISSING",
            ))
    return selection, evaluation_records


def _paired_improvement(records: Sequence[O1BaselineSelection], baseline: str, frontend: str, budget: float) -> Dict[str, Any]:
    """Compare each action to Stay for the same block/episode/frontend/budget."""

    stay = {
        (record.block_id, record.episode_id): record.expected_WER
        for record in records
        if record.baseline == "Stay" and record.frontend == frontend
        and record.budget_sec == float(budget) and record.expected_WER is not None
    }
    action = {
        (record.block_id, record.episode_id): record.expected_WER
        for record in records
        if record.baseline == baseline and record.frontend == frontend
        and record.budget_sec == float(budget) and record.expected_WER is not None
    }
    deltas = [float(stay[key]) - float(action[key]) for key in sorted(set(stay) & set(action))]
    improved = sum(delta > 0.0 for delta in deltas)
    tied = sum(delta == 0.0 for delta in deltas)
    worsened = sum(delta < 0.0 for delta in deltas)
    ordered = sorted(deltas)
    median = None if not ordered else ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
    return {
        "paired_episode_count": len(deltas),
        "mean_delta_WER": sum(deltas) / len(deltas) if deltas else None,
        "median_delta_WER": median,
        "improved_count": improved,
        "tied_count": tied,
        "worsened_count": worsened,
    }


def analyze_landscape(
    blocks: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    infrastructure_contract_sha256: str,
    source_manifest_identity: str,
    analysis_kind: str = O1_DRY_RUN_KIND,
    budgets: Sequence[float] = BUDGETS_SEC,
) -> Dict[str, Any]:
    """Return strict O1 records and descriptive summary for frozen evidence."""

    scores = _make_scores(blocks, diagnostics)
    scores_by_key: Dict[Tuple[str, str, str], List[O1PoseScore]] = {}
    for score in scores:
        scores_by_key.setdefault((score.block_id, score.episode_id, score.frontend), []).append(score)
    baseline_records: List[O1BaselineSelection] = []
    same_records: List[O1BaselineSelection] = []
    transfer_records: List[O1BaselineSelection] = []
    landscapes: List[O1BlockLandscape] = []
    by_frontend: Dict[str, Dict[str, Any]] = {}
    all_flags = set()
    for block in blocks:
        block_id = block["block_record"]["block_id"]
        episodes = [episode["episode_id"] for episode in block["episodes"]]
        for frontend in FRONTENDS:
            block_scores = [row for row in scores if row.block_id == block_id and row.frontend == frontend]
            for budget in budgets:
                budget_baseline_ids = []
                budget_same_ids = []
                budget_transfer_ids = []
                for episode in block["episodes"]:
                    episode_id = episode["episode_id"]
                    episode_scores = scores_by_key[(block_id, episode_id, frontend)]
                    feasible = _feasible(episode_scores, budget)
                    if not feasible:
                        all_flags.add("INCOMPLETE_NO_FEASIBLE_CANDIDATE")
                    for baseline in BASELINES:
                        if baseline == "Nearest-geodesic + best-heading":
                            rows, reason = _baseline_rows(block, episode_scores, budget, baseline)
                            record = _not_available(block_id, frontend, budget, baseline, "episode", reason, episode_id=episode_id)
                        elif baseline == "Max-SNR":
                            candidates, reason = _baseline_rows(block, episode_scores, budget, baseline)
                            record = _selection(block_id, frontend, budget, baseline, "episode", candidates, candidates, episode_id=episode_id, reason=reason)
                        elif baseline == "Uniform-random feasible":
                            candidates = _feasible(episode_scores, budget)
                            if budget == 0.0:
                                candidates = [row for row in candidates if row.motion_cost_sec == 0.0]
                            record = _uniform(block_id, frontend, budget, episode_scores, candidates, episode_id)
                        else:
                            candidates, reason = _baseline_rows(block, episode_scores, budget, baseline)
                            record = _selection(block_id, frontend, budget, baseline, "episode", candidates if baseline in ("Stay", "Rotate-best", "Translation-fixed-yaw", "Nearest-face-target", "Nearest-best-heading", "Local-Oracle(B)") else (), candidates, episode_id=episode_id, reason=reason)
                        baseline_records.append(record)
                        budget_baseline_ids.append(record.selection_id)
                    # O_same is an explicitly named per-utterance oracle.
                    candidates = _feasible(episode_scores, budget)
                    if budget == 0.0:
                        candidates = [row for row in candidates if row.motion_cost_sec == 0.0]
                    record = _selection(block_id, frontend, budget, "Local-Oracle(B)", "episode", candidates, candidates, episode_id=episode_id)
                    same_records.append(record)
                    budget_same_ids.append(record.selection_id)
                selection_ids = {episode["episode_id"] for episode in block["episodes"] if episode["role"] == "selection"}
                selection_scores = [row for row in block_scores if row.episode_id in selection_ids]
                for transfer_name in ("O_transfer", "Rotate-transfer", "Nearest-best-heading-transfer"):
                    evaluation_scores = [row for row in block_scores if row.episode_id not in selection_ids]
                    selection, evaluations = _transfer_selection(block, frontend, budget, selection_scores, evaluation_scores, transfer_name)
                    transfer_records.append(selection)
                    transfer_records.extend(evaluations)
                    budget_transfer_ids.append(selection.selection_id)
                landscapes.append(O1BlockLandscape(
                    schema_version=O1_SCHEMA_VERSION,
                    landscape_id="",
                    block_id=block_id,
                    frontend=frontend,
                    budget_sec=float(budget),
                    pose_score_count=len(block_scores),
                    baseline_selection_ids=tuple(budget_baseline_ids),
                    same_oracle_selection_ids=tuple(budget_same_ids),
                    transfer_selection_ids=tuple(budget_transfer_ids),
                    diagnostic_flags=tuple(sorted(all_flags)),
                ))
                landscape = landscapes[-1]
                object.__setattr__(landscape, "landscape_id", stable_id("o1-landscape", landscape.identity_payload()))
            frontend_summary = by_frontend.setdefault(frontend, {"baseline_curves": {}, "pose_score_count": 0})
            frontend_summary["pose_score_count"] += len(block_scores)
    # Compact descriptive curves are derived from the strict selection records.
    for frontend in FRONTENDS:
        curves = {}
        for budget in budgets:
            entries = [record for record in baseline_records if record.frontend == frontend and record.budget_sec == float(budget)]
            curves[str(budget)] = {}
            for baseline in BASELINES:
                selected_values = [record.expected_WER for record in entries if record.baseline == baseline and record.expected_WER is not None]
                count = sum(record.status == "SELECTED" for record in entries if record.baseline == baseline)
                curves[str(budget)][baseline] = {
                    "selected_count": count,
                    "record_count": sum(record.baseline == baseline for record in entries),
                    "expected_WER_mean": (sum(selected_values) / float(len(selected_values))) if selected_values else None,
                }
        by_frontend[frontend]["baseline_curves"] = curves
        frontend_scores = [row for row in scores if row.frontend == frontend]
        by_frontend[frontend]["pose_wer_distribution"] = _descriptive_stats([row.WER for row in frontend_scores])
        by_frontend[frontend]["cost_vs_wer"] = [
            {"block_id": row.block_id, "episode_id": row.episode_id, "pose_id": row.pose_id,
             "motion_cost_sec": row.motion_cost_sec, "WER": row.WER}
            for row in sorted(frontend_scores, key=lambda item: (item.block_id, item.episode_id, item.pose_id))
        ]
        rotation_opportunity = {}
        translation_opportunity = {}
        nearest_gap = {}
        for budget in budgets:
            entries = [record for record in baseline_records if record.frontend == frontend and record.budget_sec == float(budget)]
            stay = [record.expected_WER for record in entries if record.baseline == "Stay" and record.expected_WER is not None]
            rotate = [record.expected_WER for record in entries if record.baseline == "Rotate-best" and record.expected_WER is not None]
            translation = [record.expected_WER for record in entries if record.baseline == "Translation-fixed-yaw" and record.expected_WER is not None]
            nearest = [record.expected_WER for record in entries if record.baseline == "Nearest-best-heading" and record.expected_WER is not None]
            oracle = [record.expected_WER for record in entries if record.baseline == "Local-Oracle(B)" and record.expected_WER is not None]
            stay_mean = sum(stay) / len(stay) if stay else None
            rotate_pair = _paired_improvement(baseline_records, "Rotate-best", frontend, budget)
            translation_pair = _paired_improvement(baseline_records, "Translation-fixed-yaw", frontend, budget)
            rotation_opportunity[str(budget)] = {
                "stay_mean_WER": stay_mean,
                "rotate_best_mean_WER": sum(rotate) / len(rotate) if rotate else None,
                "episodes_improved_over_stay": rotate_pair["improved_count"],
                "paired": rotate_pair,
            }
            translation_opportunity[str(budget)] = {
                "stay_mean_WER": stay_mean,
                "translation_fixed_yaw_mean_WER": sum(translation) / len(translation) if translation else None,
                "episodes_improved_over_stay": translation_pair["improved_count"],
                "paired": translation_pair,
            }
            nearest_gap[str(budget)] = {
                "nearest_best_heading_mean_WER": sum(nearest) / len(nearest) if nearest else None,
                "local_oracle_mean_WER": sum(oracle) / len(oracle) if oracle else None,
                "mean_gap_nearest_minus_oracle": (sum(nearest) / len(nearest) - sum(oracle) / len(oracle)) if nearest and oracle else None,
            }
        by_frontend[frontend]["rotation_only_opportunity"] = rotation_opportunity
        by_frontend[frontend]["translation_fixed_yaw_opportunity"] = translation_opportunity
        by_frontend[frontend]["nearest_vs_oracle_gap"] = nearest_gap
    attempted = len(scores)
    valid = sum(row.geometry_legality == "LEGAL" for row in scores)
    incomplete = attempted - valid
    summary_body = {
        "budgets_sec": [float(value) for value in budgets],
        "frontends": by_frontend,
        "baseline_names": list(BASELINES),
        "tie_break_identity": O1_TIE_BREAK_IDENTITY,
        "primary_frontend": "mean_lr",
        "frontend_sensitivity_only": ["fixed_L", "fixed_R"],
        "oracle_semantics": {"O_same": "per_utterance_independent_minimum", "O_transfer": "selection_aggregate_fixed_pose_then_evaluation"},
        "diagnostic_flags": sorted(all_flags),
        "scientific_status": "EXPLORATORY_DIAGNOSTIC",
        "no_policy_or_heldout_claim": True,
        "flat_attribution": {
            "status": "EXPLORATORY_DIAGNOSTIC",
            "stay_WER_level_by_frontend": {
                frontend: by_frontend[frontend]["baseline_curves"]["10.0"]["Stay"]["expected_WER_mean"]
                for frontend in FRONTENDS
            },
            "candidate_acoustic_diversity": "NOT_AVAILABLE_TO_RESULT_ONLY_ANALYZER",
            "component_snr_spread": {
                frontend: _descriptive_stats([row.component_snr_db for row in scores if row.frontend == frontend and row.component_snr_db is not None])
                for frontend in FRONTENDS
            },
            "yaw_spread_deg": {
                frontend: _descriptive_stats([row.yaw_deg for row in scores if row.frontend == frontend])
                for frontend in FRONTENDS
            },
            "translation_motion_spread_sec": {
                frontend: _descriptive_stats([row.motion_cost_sec for row in scores if row.frontend == frontend])
                for frontend in FRONTENDS
            },
            "budget_feasible_fraction": {
                frontend: {
                    str(budget): sum(row.geometry_legality == "LEGAL" and row.motion_cost_sec <= budget + 1.0e-12 for row in scores if row.frontend == frontend) / float(sum(row.frontend == frontend for row in scores))
                    for budget in budgets
                }
                for frontend in FRONTENDS
            },
            "frontend_spread_at_10s_mean_WER": max(
                by_frontend[frontend]["baseline_curves"]["10.0"]["Local-Oracle(B)"]["expected_WER_mean"] or 0.0
                for frontend in FRONTENDS
            ) - min(
                by_frontend[frontend]["baseline_curves"]["10.0"]["Local-Oracle(B)"]["expected_WER_mean"] or 0.0
                for frontend in FRONTENDS
            ),
        },
    }
    summary_payload = {
        "schema_version": O1_SCHEMA_VERSION,
        "analysis_kind": analysis_kind,
        "algorithm_identity": O1_ALGORITHM_IDENTITY,
        "frontend_policy_identity": O1_FRONTEND_POLICY_IDENTITY,
        "infrastructure_contract_sha256": infrastructure_contract_sha256,
        "source_manifest_identity": source_manifest_identity,
        "blocks": len(blocks),
        "episodes": len({row.episode_id for row in scores}),
        "poses": len({(row.block_id, row.pose_id) for row in scores}),
        "attempted": attempted,
        "valid": valid,
        "incomplete": incomplete,
        "summary": summary_body,
    }
    summary_payload["summary_id"] = _o1_result_id("o1-summary", summary_payload)
    summary = O1LandscapeSummary.from_payload(summary_payload)
    return {
        "pose_scores": scores,
        "same_oracle": same_records,
        "transfer_selections": transfer_records,
        "baseline_selections": baseline_records,
        "block_landscapes": landscapes,
        "summary": summary,
    }


def write_analysis_outputs(result: Mapping[str, Any], output_dir: str) -> None:
    """Write only O1 analysis artifacts; callers own the run namespace."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    def write_jsonl(name: str, values: Sequence[Any]) -> None:
        payload = b"\n".join(o1_canonical_json_bytes(value.to_payload()) for value in values) + b"\n"
        (root / name).write_bytes(payload)

    write_jsonl("o1_pose_scores.jsonl", result["pose_scores"])
    write_jsonl("o1_same_oracle.jsonl", result["same_oracle"])
    write_jsonl("o1_transfer_selections.jsonl", result["transfer_selections"])
    write_jsonl("o1_baseline_selections.jsonl", result["baseline_selections"])
    write_jsonl("o1_block_landscapes.jsonl", result["block_landscapes"])
    (root / "o1_landscape_summary.json").write_bytes(o1_canonical_json_bytes(result["summary"].to_payload()) + b"\n")


__all__ = [
    "BASELINES",
    "BUDGETS_SEC",
    "FRONTENDS",
    "O1_ALGORITHM_IDENTITY",
    "O1AnalysisError",
    "O1BaselineSelection",
    "O1BlockLandscape",
    "O1LandscapeSummary",
    "O1PoseScore",
    "analyze_landscape",
    "load_asr_diagnostics",
    "o1_canonical_json_bytes",
    "score_from_diagnostic",
    "write_analysis_outputs",
]
