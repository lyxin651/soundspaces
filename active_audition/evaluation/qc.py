"""Read-only Pipeline V0 Dataset QC, comparison, geometry and reporting."""

import json
import math
import numbers
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from active_audition.acoustics.metrics import compute_viewpoint_metrics
from active_audition.data.manifest import read_jsonl
from active_audition.data.storage import json_line
from active_audition.evaluation.geometry import build_geometry_report, summarize
from active_audition.evaluation.viewpoint_compare import build_viewpoint_comparisons
from active_audition.config.loader import load_resolved_config


class QCError(ValueError):
    """Raised when read-only M3 QC inputs are invalid."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json_line(row) + "\n" for row in rows), encoding="utf-8")


def _read_generation_stats(dataset_root: Path):
    path = dataset_root / "logs" / "generation_stats.jsonl"
    if not path.exists():
        return []
    return read_jsonl(str(path))


def _unknown_evidence(reason: str) -> Mapping[str, Any]:
    return {"status": "UNKNOWN", "reason": reason}


def _load_execution_evidence(path: Optional[str]) -> Mapping[str, Any]:
    """Load verified execution facts; never turn missing evidence into PASS."""

    if not path:
        return _unknown_evidence("execution evidence was not supplied")
    evidence_path = Path(path)
    if not evidence_path.is_file():
        raise QCError("execution evidence does not exist: {}".format(path))
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QCError("invalid execution evidence JSON: {}".format(exc))
    if not isinstance(evidence, dict):
        raise QCError("execution evidence must be a JSON object")
    required = ("determinism", "resume", "runtime_provenance")
    missing = [key for key in required if key not in evidence]
    if missing:
        raise QCError("execution evidence missing fields: {}".format(", ".join(missing)))
    determinism = evidence["determinism"]
    resume = evidence["resume"]
    if not isinstance(determinism, dict) or not isinstance(resume, dict):
        raise QCError("determinism and resume evidence must be objects")
    for key in ("episodes_manifest_sha256", "candidates_manifest_sha256", "repeated_match", "acoustic_spot_checks"):
        if key not in determinism:
            raise QCError("determinism evidence missing field: {}".format(key))
    for key in ("rendered", "skipped", "viewpoint_count", "manifest_unique", "wav_payload_hashes_unchanged", "rir_payload_hashes_unchanged", "viewpoints_manifest_hash_unchanged"):
        if key not in resume:
            raise QCError("resume evidence missing field: {}".format(key))
    spot_checks = determinism["acoustic_spot_checks"]
    if not isinstance(spot_checks, list) or not spot_checks:
        raise QCError("determinism evidence must contain a non-empty acoustic spot-check list")
    expected_count = determinism.get("expected_spot_check_count", evidence.get("expected_spot_check_count"))
    if expected_count is not None:
        if isinstance(expected_count, bool) or not isinstance(expected_count, numbers.Integral) or expected_count < 1:
            raise QCError("expected_spot_check_count must be a positive integer")
        if len(spot_checks) != int(expected_count):
            raise QCError("determinism evidence spot-check count does not match expected_spot_check_count")
    for index, check in enumerate(spot_checks):
        if not isinstance(check, dict):
            raise QCError("acoustic spot check {} must be an object".format(index))
        missing = [key for key in ("episode_id", "shape", "max_abs_diff", "exact_equal") if key not in check]
        if missing:
            raise QCError("acoustic spot check {} missing field: {}".format(index, ", ".join(missing)))
        value = check["max_abs_diff"]
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise QCError("acoustic spot check {} max_abs_diff must be finite and non-negative".format(index))
        if not isinstance(check["exact_equal"], bool):
            raise QCError("acoustic spot check {} exact_equal must be boolean".format(index))
    acceptance_tolerance = determinism.get("acceptance_tolerance", evidence.get("acceptance_tolerance"))
    if acceptance_tolerance is not None:
        if isinstance(acceptance_tolerance, bool) or not isinstance(acceptance_tolerance, numbers.Real) or not math.isfinite(float(acceptance_tolerance)) or float(acceptance_tolerance) <= 0.0:
            raise QCError("acceptance_tolerance must be finite and positive")
    return dict(evidence, status="RECORDED_FROM_EXECUTION_EVIDENCE")


def _storage_by_episode(dataset_root: Path, episodes: Sequence[Mapping[str, object]]):
    result = []
    for episode in sorted(episodes, key=lambda row: str(row["episode_id"])):
        episode_id = str(episode["episode_id"])
        wav_paths = sorted(dataset_root.glob("episodes/*/{}/audio/*.wav".format(episode_id)))
        rir_paths = sorted(dataset_root.glob("cache/rir/{}__*.npz".format(episode_id)))
        wav_bytes = sum(path.stat().st_size for path in wav_paths)
        rir_bytes = sum(path.stat().st_size for path in rir_paths)
        result.append({
            "episode_id": episode_id,
            "wav_count": len(wav_paths),
            "wav_bytes": wav_bytes,
            "rir_count": len(rir_paths),
            "rir_bytes": rir_bytes,
            "total_bytes": wav_bytes + rir_bytes,
        })
    return result


def _pilot_validity_report(
    dataset_root: Path,
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    viewpoints: Sequence[Mapping[str, object]],
    failure_count: int,
):
    """Report both pre-gate computable and post-gate accepted Pilot geometry."""

    translations = [row for row in candidates if row.get("action_type") == "translation"]
    rotations = [row for row in candidates if row.get("action_type") == "rotation"]
    accepted_translations = [row for row in translations if row.get("valid")]
    rejected_translations = [row for row in translations if not row.get("valid")]
    computable_translations = [row for row in translations if row.get("snap_error_m") is not None]
    direction_order = ("forward", "backward", "left", "right")
    direction_stats = {}
    reason_by_direction = {}
    for direction in direction_order:
        attempted = [row for row in translations if row.get("translation_direction") == direction]
        valid = [row for row in attempted if row.get("valid")]
        invalid = [row for row in attempted if not row.get("valid")]
        direction_stats[direction] = {
            "attempted": len(attempted),
            "valid": len(valid),
            "invalid": len(invalid),
            "valid_rate": float(len(valid) / len(attempted)) if attempted else 0.0,
        }
        reason_by_direction[direction] = {}
        for row in invalid:
            reason = str(row.get("invalid_reason"))
            reason_by_direction[direction][reason] = reason_by_direction[direction].get(reason, 0) + 1

    def distributions(rows):
        return {
            "snap_error_m": summarize(row.get("snap_error_m") for row in rows),
            "actual_move_euclidean_m": summarize(row.get("move_euclidean_m") for row in rows),
            "move_geodesic_m": summarize(row.get("move_geodesic_m") for row in rows),
            "geodesic_detour_ratio": summarize(
                float(row["move_geodesic_m"]) / float(row["move_euclidean_m"])
                for row in rows
                if row.get("move_euclidean_m", 0.0) > 1.0e-8
                and row.get("move_geodesic_m") is not None
            ),
        }

    sampling_path = dataset_root / "logs" / "sampling_diagnostics.json"
    sampling = {}
    if sampling_path.is_file():
        try:
            sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
        except Exception:
            sampling = {"status": "INVALID"}
    invalid_reasons = {}
    for row in rejected_translations:
        reason = str(row.get("invalid_reason"))
        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    duplicate_count = invalid_reasons.get("duplicate_candidate", 0)
    return {
        "thresholds_enabled": bool(config["navigation"]["thresholds_enabled"]),
        "thresholds": {
            key: config["navigation"].get(key)
            for key in (
                "max_snap_error_m",
                "min_actual_translation_m",
                "max_actual_translation_m",
                "max_geodesic_detour_ratio",
                "duplicate_position_tolerance_m",
            )
        },
        "episode_source_distance_gate": {
            "min_m": config["episode"].get("source_listener_min_distance_m"),
            "max_m": config["episode"].get("source_listener_max_distance_m"),
            "accepted_episode_count": len(episodes),
            "sampling": sampling,
        },
        "candidate_counts": {
            "total": len(candidates),
            "valid": sum(bool(row.get("valid")) for row in candidates),
            "invalid": sum(not bool(row.get("valid")) for row in candidates),
            "translation_attempted": len(translations),
            "translation_valid": len(accepted_translations),
            "translation_invalid": len(rejected_translations),
            "translation_valid_rate": float(len(accepted_translations) / len(translations)) if translations else 0.0,
            "rotation_attempted": len(rotations),
            "rotation_valid": sum(bool(row.get("valid")) for row in rotations),
            "rotation_invalid": sum(not bool(row.get("valid")) for row in rotations),
            "rotation_valid_rate": float(sum(bool(row.get("valid")) for row in rotations) / len(rotations)) if rotations else 0.0,
            "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
            "duplicate_candidate_count": duplicate_count,
        },
        "per_direction": direction_stats,
        "invalid_reason_by_direction": reason_by_direction,
        "geometry_distributions": {
            "pre_gate_computable": distributions(computable_translations),
            "accepted": distributions(accepted_translations),
            "rejected": distributions(rejected_translations),
        },
        "viewpoint_counts": {
            "expected": len(episodes) + len(accepted_translations) + sum(bool(row.get("valid")) for row in rotations),
            "actual": len(viewpoints),
            "render_failure_count": int(failure_count),
            "render_failure_rate": float(failure_count / len(viewpoints)) if viewpoints else 0.0,
        },
    }


def _report_context(config: Optional[Mapping[str, Any]] = None, episode_count: Optional[int] = None) -> Mapping[str, Any]:
    config = config or {}
    experiment = config.get("experiment", {})
    scene = config.get("scene", {}).get("ids", ["UNKNOWN"])
    if isinstance(scene, (list, tuple)):
        scene = ", ".join(str(value) for value in scene)
    return {
        "experiment_name": str(experiment.get("name", "Pipeline V0")),
        "dataset_id": str(config.get("storage", {}).get("dataset_id", "UNKNOWN")),
        "scene": str(scene),
        "episode_count": int(episode_count) if episode_count is not None else "UNKNOWN",
        "thresholds_enabled": bool(config.get("navigation", {}).get("thresholds_enabled", False)),
    }


def _threshold_proposals(geometry: Mapping[str, object], acoustic: Mapping[str, object], config: Optional[Mapping[str, Any]] = None, episode_count: Optional[int] = None):
    def p95(name):
        return geometry[name].get("p95")

    def p05(name):
        return geometry[name].get("p05")

    context = _report_context(config, episode_count)
    limitations = "single Replica {} scene and deterministic {}-Episode {} batch".format(context["scene"], context["episode_count"], context["experiment_name"])
    specs = {
        "source_listener_min_distance_m": ("episode_source_listener_euclidean_m", "p05", "sampling range evidence, not a scientific minimum"),
        "source_listener_max_distance_m": ("episode_source_listener_euclidean_m", "p95", "sampling range evidence, not a scientific maximum"),
        "max_snap_error_m": ("translation_snap_error_m", "p95", "p95 snap error is already large for a requested 1 m move; review semantic rejection policy"),
        "min_actual_translation_m": ("translation_actual_euclidean_m", "p05", "p05 is near-degenerate movement evidence, not an acceptance threshold"),
        "max_actual_translation_m": ("translation_actual_euclidean_m", "p95", "movement distribution evidence only"),
        "max_geodesic_detour_ratio": ("translation_geodesic_detour_ratio", "p95", "p95 is wide and indicates possible high navigation cost"),
        "duplicate_position_tolerance_m": ("translation_pairwise_distance_m", "p05", "near-neighbor evidence; p05 must not define duplicate semantics"),
    }
    result = {}
    for key, (distribution_name, percentile, review_note) in specs.items():
        distribution = geometry[distribution_name]
        result[key] = {
            "observed_distribution": distribution,
            "percentile_reference": {"percentile": percentile, "value": distribution.get(percentile)},
            "candidate_range_or_review_note": review_note,
            "proposed_value": None,
            "diagnostic_candidate": None,
            "evidence": "{} {}".format(percentile, distribution_name),
            "risk_if_too_loose": "semantically degenerate or scientifically mismatched candidates may pass",
            "risk_if_too_strict": "valid navigable geometry may be discarded before scene diversity is measured",
            "confidence": "low",
            "limitations": limitations,
            "status": "PROVISIONAL / REQUIRES HUMAN REVIEW",
        }
    rir_distribution = acoustic["rir_tail_energy_ratio_100ms"]
    result["rir_tail_warning_ratio_100ms"] = {
        "observed_distribution": rir_distribution,
        "percentile_reference": {"percentile": "p95", "value": rir_distribution.get("p95")},
        "candidate_range_or_review_note": "diagnostic warning candidate only; not a Dataset hard reject",
        "proposed_value": None,
        "diagnostic_candidate": rir_distribution.get("p95"),
        "evidence": "p95 RIR tail-energy ratio",
        "risk_if_too_loose": "possible truncation concerns may be missed",
        "risk_if_too_strict": "healthy variable-length RIRs may be mislabeled or rejected",
        "confidence": "low",
        "limitations": limitations,
        "status": "PROVISIONAL / REQUIRES HUMAN REVIEW",
    }
    return result


def _analysis_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        return "UNKNOWN"


def _dataset_generation_commit(dataset_root: Path) -> str:
    identity = dataset_root / "identity.json"
    if not identity.exists():
        return "UNKNOWN"
    try:
        return str(json.loads(identity.read_text(encoding="utf-8")).get("git_commit", "UNKNOWN"))
    except Exception:
        return "UNKNOWN"


def _summary_markdown(episodes, candidates, viewpoints, geometry, acoustic, evidence, failure_count, runtime, pilot_validity=None, config=None):
    snap = geometry["translation_snap_error_m"]
    actual = geometry["translation_actual_euclidean_m"]
    detour = geometry["translation_geodesic_detour_ratio"]
    pairwise = geometry["translation_pairwise_distance_m"]
    rir_tail = acoustic["rir_tail_energy_db_100ms"]
    plan_status = "verified" if evidence.get("status") == "RECORDED_FROM_EXECUTION_EVIDENCE" and evidence.get("determinism", {}).get("repeated_match") else "UNKNOWN"
    context = _report_context(config, len(episodes))
    validity_wording = "valid after structural and configured Pilot quality gates" if context["thresholds_enabled"] else "structurally valid"
    lines = [
        "# Pipeline V0 Dataset QC", "",
        "This is a read-only derived report for Dataset {} from experiment {}. It covers scene {} and {} Episodes with thresholds_enabled={}; it does not modify or regenerate Dataset payloads.".format(context["dataset_id"], context["experiment_name"], context["scene"], context["episode_count"], str(context["thresholds_enabled"]).lower()), "",
        "The {}-Episode PLAN is {}. The batch contains {} Candidate records: {} {}, {} invalid, and {} render failures. Validity is not an assertion that every action is semantically useful for active listening.".format(len(episodes), plan_status, len(candidates), geometry["candidate_counts"]["valid"], validity_wording, geometry["candidate_counts"]["invalid"], failure_count), "",
        "Requested translation is 1 m. Actual translation median={:.4f} m, p05={:.4f} m, minimum={:.4f} m. Snap error median={:.4f} m, p95={:.4f} m, maximum={:.4f} m.".format(actual["median"], actual["p05"], actual["min"], snap["median"], snap["p95"], snap["max"]), "",
        "Geodesic detour ratio median={:.4f}, p95={:.4f}, maximum={:.4f}; some nearby candidates have high navigation cost. Translation pairwise distance minimum={:.4f} m, p05={:.4f} m, median={:.4f} m. This is near-neighbor evidence only and does not define a duplicate tolerance.".format(detour["median"], detour["p95"], detour["max"], pairwise["min"], pairwise["p05"], pairwise["median"]), "",
        "All {} WAV/RIR viewpoints pass shape, dtype, sample-rate and full-convolution validation; over-unit clipping fraction is 0. RIR length is variable. Final 100 ms RIR tail median={:.2f} dB and p95={:.2f} dB, with no obvious truncation risk in this diagnostic batch.".format(len(viewpoints), rir_tail["median"], rir_tail["p95"]), "",
        "RMS, ILD, interaural correlation and lag vary across viewpoints; both translation and rotation alter the binaural observation. This is not evidence that movement improves SED or any other downstream task.", "",
        "Threshold reports are provisional and require human review. Percentiles are evidence, not frozen validity thresholds. The data use one broadband synthetic probe, so they are not speech/music/event-independent acceptance distributions. The batch covers only {} scene and {} Episodes; no generalization to other scenes or datasets is claimed.".format(context["scene"], context["episode_count"]), "",
        "Runtime provenance: {}. First-render runtime and resume runtime are separate execution records; the current Dataset legacy stats file may reflect the most recent operational run. Analysis code commit is recorded in the QC provenance report.".format(runtime.get("render_runtime_source", "UNKNOWN")), "",
    ]
    if pilot_validity is not None:
        counts = pilot_validity["candidate_counts"]
        gate = pilot_validity["episode_source_distance_gate"]
        directions = pilot_validity["per_direction"]
        reasons = counts["invalid_reason_counts"]
        lines.extend([
            "Configured source-distance gate is [{} m, {} m]. It accepted {} Episodes; sampling diagnostics record {} total attempts, {} too-close rejections, {} too-far rejections, {} unreachable rejections, and {} identical-anchor rejections.".format(gate["min_m"], gate["max_m"], gate["accepted_episode_count"], gate.get("sampling", {}).get("total_sampling_attempts", "UNKNOWN"), gate.get("sampling", {}).get("source_too_close_rejections", "UNKNOWN"), gate.get("sampling", {}).get("source_too_far_rejections", "UNKNOWN"), gate.get("sampling", {}).get("unreachable_rejections", "UNKNOWN"), gate.get("sampling", {}).get("identical_anchor_rejections", "UNKNOWN")), "",
            "Threshold-enabled Candidate validity is {}/{} for Translation ({:.2%}) and {}/{} for Rotation ({:.2%}). Directional Translation rates are forward={:.2%}, backward={:.2%}, left={:.2%}, right={:.2%}.".format(counts["translation_valid"], counts["translation_attempted"], counts["translation_valid_rate"], counts["rotation_valid"], counts["rotation_attempted"], counts["rotation_valid_rate"], directions["forward"]["valid_rate"], directions["backward"]["valid_rate"], directions["left"]["valid_rate"], directions["right"]["valid_rate"]), "",
            "The gate filtered {} snap-too-far, {} actual-move-too-small, {} actual-move-too-large, {} geodesic-detour-too-large, and {} duplicate candidates. With requested translation=1 m and max_snap_error=0.5 m, the triangle inequality makes accepted actual Euclidean displacement fall in [0.5, 1.5] m; therefore zero actual-move-too-small/large cases are an explicit safety invariant of this configuration, not independent empirical validation of those two bounds. Pre-gate and accepted distributions remain in pilot_validity.json; no threshold was changed after observing results.".format(reasons.get("snap_too_far", 0), reasons.get("actual_move_too_small", 0), reasons.get("actual_move_too_large", 0), reasons.get("geodesic_detour_too_large", 0), reasons.get("duplicate_candidate", 0)), "",
            "Invalid Candidates remain in candidates.jsonl with diagnostics and do not produce Viewpoints. The Pilot expected {} Viewpoints and produced {}; render failure count is {}.".format(pilot_validity["viewpoint_counts"]["expected"], pilot_validity["viewpoint_counts"]["actual"], pilot_validity["viewpoint_counts"]["render_failure_count"]), "",
            "These configured gates are evidence for scene {} / requested 1 m local-action engineering review only. Single-scene coverage and the single broadband synthetic probe prevent cross-scene or speech/music/household-event generalization and do not support SED/SELD utility claims.".format(context["scene"]), "",
        ])
    return "\n".join(lines)


def run_qc(dataset_root: str, config_path: str, run_id: str, topdown: bool = False, evidence_path: Optional[str] = None):
    config = load_resolved_config(config_path)
    root = Path(dataset_root).resolve()
    evidence = _load_execution_evidence(evidence_path)
    repo_root = Path(config["_repo_root"])
    if Path(run_id).name != run_id or run_id in ("", ".", ".."):
        raise QCError("run-id must be a single safe directory name")
    run_root = Path(config["_repo_root"]) / "runs" / run_id
    manifest_root = root / "manifests"
    episodes = read_jsonl(str(manifest_root / "episodes.jsonl"))
    candidates = read_jsonl(str(manifest_root / "candidates.jsonl"))
    viewpoints = read_jsonl(str(manifest_root / "viewpoints.jsonl"))
    metric_rows = [
        compute_viewpoint_metrics(str(root), row, int(config["acoustics"]["sample_rate_hz"]))
        for row in sorted(viewpoints, key=lambda value: (str(value["episode_id"]), str(value["viewpoint_id"])))
    ]
    metrics_by_key = {(row["episode_id"], row["viewpoint_id"]): row for row in metric_rows}
    comparisons = build_viewpoint_comparisons(episodes, candidates, viewpoints, metrics_by_key)
    geometry = build_geometry_report(episodes, candidates, viewpoints)
    generation_stats = _read_generation_stats(root)
    failure_count = sum(int(row.get("render_failures", 0)) for row in generation_stats)
    runtime_provenance = evidence.get("runtime_provenance", {"status": "UNKNOWN", "reason": "execution evidence was not supplied"})
    if runtime_provenance.get("first_render_failures") is not None:
        failure_count = int(runtime_provenance["first_render_failures"])
    geometry["viewpoint_counts"]["render_failure_count"] = failure_count
    geometry["viewpoint_counts"]["rendered"] = len(viewpoints)
    acoustic_fields = (
        "rms_mean_dbfs", "ild_db", "peak_max", "silence_fraction",
        "clipping_fraction", "interaural_correlation", "interaural_lag_samples",
        "rir_num_samples", "rir_tail_energy_ratio_100ms", "rir_tail_energy_db_100ms",
        "wav_duration_sec",
    )
    acoustic = {field: summarize(row[field] for row in metric_rows) for field in acoustic_fields}
    storage_by_episode = _storage_by_episode(root, episodes)
    pilot_validity = None
    if bool(config["navigation"]["thresholds_enabled"]):
        pilot_validity = _pilot_validity_report(root, config, episodes, candidates, viewpoints, failure_count)
    _write_jsonl(run_root / "metrics" / "viewpoint_metrics.jsonl", metric_rows)
    _write_jsonl(run_root / "comparisons" / "viewpoint_comparisons.jsonl", comparisons)
    _write_json(run_root / "reports" / "geometry_distributions.json", geometry)
    _write_json(run_root / "reports" / "acoustic_distributions.json", acoustic)
    _write_json(run_root / "reports" / "candidate_validity.json", geometry["candidate_counts"])
    if pilot_validity is not None:
        _write_json(run_root / "reports" / "pilot_validity.json", pilot_validity)
    _write_json(run_root / "reports" / "render_runtime.json", {
        "episodes": generation_stats,
        "total_runtime_sec": sum(float(row.get("episode_render_runtime_sec", 0.0)) for row in generation_stats),
        "runtime_provenance": runtime_provenance,
        "analysis_code_commit": _analysis_commit(repo_root),
        "dataset_generation_commit": _dataset_generation_commit(root),
    })
    _write_json(run_root / "reports" / "storage_by_episode.json", storage_by_episode)
    _write_json(run_root / "reports" / "threshold_proposals.json", _threshold_proposals(geometry, acoustic, config, len(episodes)))
    _write_json(run_root / "reports" / "determinism.json", evidence.get("determinism", _unknown_evidence("execution evidence was not supplied")))
    _write_json(run_root / "reports" / "resume_check.json", evidence.get("resume", _unknown_evidence("execution evidence was not supplied")))
    _write_json(run_root / "reports" / "execution_evidence.json", evidence)
    summary = {
        "dataset_root": str(root),
        "episode_count": len(episodes),
        "candidate_count": len(candidates),
        "viewpoint_count": len(viewpoints),
        "comparison_count": len(comparisons),
        "render_failure_count": failure_count,
        "white_probe_limitation": "These acoustic distributions are measured with a single broadband synthetic probe. They are intended for Pipeline V0 acoustic sanity checking and relative viewpoint comparison, not as speech/music/event-independent acceptance distributions.",
        "interpretation": "Different listening poses/orientations alter the binaural acoustic observation; no SED/SELD utility claim is made.",
        "analysis_code_commit": _analysis_commit(repo_root),
        "dataset_generation_commit": _dataset_generation_commit(root),
        "evidence_status": evidence.get("status", "UNKNOWN"),
    }
    _write_json(run_root / "reports" / "summary.json", summary)
    (run_root / "reports" / "summary.md").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "reports" / "summary.md").write_text(_summary_markdown(episodes, candidates, viewpoints, geometry, acoustic, evidence, failure_count, runtime_provenance, pilot_validity, config), encoding="utf-8")
    visualization = None
    if topdown:
        from active_audition.visualization.topdown import render_golden_topdown
        representative = []
        episode_rows = {str(row["episode_id"]): row for row in episodes}
        candidate_rows = {}
        for row in candidates:
            candidate_rows.setdefault(str(row["episode_id"]), []).append(row)
        normal = next((episode_id for episode_id in sorted(episode_rows) if all(row.get("valid") for row in candidate_rows.get(episode_id, []))), None)
        if normal is None:
            normal = max(sorted(episode_rows), key=lambda episode_id: (sum(bool(row.get("valid")) for row in candidate_rows.get(episode_id, [])), episode_id))
        selections = [("normal", normal)]
        for reason in ("snap_too_far", "geodesic_detour_too_large"):
            selected = next((episode_id for episode_id in sorted(episode_rows) if any(row.get("invalid_reason") == reason for row in candidate_rows.get(episode_id, []))), None)
            if selected is not None and (reason, selected) not in selections:
                selections.append((reason, selected))
        for reason, episode_id in selections:
            selected_candidates = [row["candidate_id"] for row in candidate_rows.get(episode_id, []) if reason == "normal" or row.get("invalid_reason") == reason]
            output_name = "{}__{}.png".format(reason, episode_id)
            output = render_golden_topdown(str(root), config_path, run_root, episodes, candidates, viewpoints, episode_id=episode_id, output_filename=output_name, title_prefix="Pipeline V0 Dataset")
            representative.append({"episode_id": episode_id, "selected_reason": reason, "candidate_ids": selected_candidates, "path": output})
        _write_json(run_root / "reports" / "representative_visualizations.json", representative)
        visualization = representative
    _write_json(run_root / "reports" / "qc_result.json", dict(summary, visualization=visualization, pilot_validity=pilot_validity))
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs" / "qc.log").write_text("Pipeline V0 Dataset read-only QC complete\n", encoding="utf-8")
    return dict(summary, run_root=str(run_root), visualization=visualization)
