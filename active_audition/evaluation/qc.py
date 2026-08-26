"""Read-only M3 acoustic QC, comparison, geometry and report orchestration."""

import json
from pathlib import Path
from typing import Mapping, Sequence

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


def _threshold_proposals(geometry: Mapping[str, object], acoustic: Mapping[str, object]):
    def p95(name):
        return geometry[name].get("p95")

    def p05(name):
        return geometry[name].get("p05")

    specs = {
        "source_listener_min_distance_m": (p05("episode_source_listener_euclidean_m"), "p05 source-listener Euclidean distance"),
        "source_listener_max_distance_m": (p95("episode_source_listener_euclidean_m"), "p95 source-listener Euclidean distance"),
        "max_snap_error_m": (p95("translation_snap_error_m"), "p95 translation snap error"),
        "min_actual_translation_m": (p05("translation_actual_euclidean_m"), "p05 actual translation"),
        "max_actual_translation_m": (p95("translation_actual_euclidean_m"), "p95 actual translation"),
        "max_geodesic_detour_ratio": (p95("translation_geodesic_detour_ratio"), "p95 finite geodesic detour ratio"),
        "duplicate_position_tolerance_m": (p05("translation_pairwise_distance_m"), "p05 translation-only pairwise distance"),
        "rir_tail_warning_ratio_100ms": (acoustic["rir_tail_energy_ratio_100ms"].get("p95"), "p95 RIR tail-energy ratio; diagnostic only"),
    }
    return {
        key: {
            "observed_value": value,
            "proposed_value": value,
            "evidence": evidence,
            "confidence": "low",
            "limitations": "single Replica office_0 scene and deterministic 20-Episode diagnostic batch",
            "status": "PROVISIONAL / REQUIRES HUMAN REVIEW",
        }
        for key, (value, evidence) in specs.items()
    }


def run_qc(dataset_root: str, config_path: str, run_id: str, topdown: bool = False):
    config = load_resolved_config(config_path)
    root = Path(dataset_root).resolve()
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
    _write_jsonl(run_root / "metrics" / "viewpoint_metrics.jsonl", metric_rows)
    _write_jsonl(run_root / "comparisons" / "viewpoint_comparisons.jsonl", comparisons)
    _write_json(run_root / "reports" / "geometry_distributions.json", geometry)
    _write_json(run_root / "reports" / "acoustic_distributions.json", acoustic)
    _write_json(run_root / "reports" / "candidate_validity.json", geometry["candidate_counts"])
    _write_json(run_root / "reports" / "render_runtime.json", {"episodes": generation_stats, "total_runtime_sec": sum(float(row.get("episode_render_runtime_sec", 0.0)) for row in generation_stats)})
    _write_json(run_root / "reports" / "storage_by_episode.json", storage_by_episode)
    _write_json(run_root / "reports" / "threshold_proposals.json", _threshold_proposals(geometry, acoustic))
    _write_json(run_root / "reports" / "determinism.json", {"status": "RECORDED_BY_EXECUTION"})
    _write_json(run_root / "reports" / "resume_check.json", {"status": "RECORDED_BY_EXECUTION"})
    summary = {
        "dataset_root": str(root),
        "episode_count": len(episodes),
        "candidate_count": len(candidates),
        "viewpoint_count": len(viewpoints),
        "comparison_count": len(comparisons),
        "render_failure_count": failure_count,
        "white_probe_limitation": "These acoustic distributions are measured with a single broadband synthetic probe. They are intended for Pipeline V0 acoustic sanity checking and relative viewpoint comparison, not as speech/music/event-independent acceptance distributions.",
        "interpretation": "Different listening poses/orientations alter the binaural acoustic observation; no SED/SELD utility claim is made.",
    }
    _write_json(run_root / "reports" / "summary.json", summary)
    (run_root / "reports" / "summary.md").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "reports" / "summary.md").write_text(
        "# Pipeline V0 M3 QC\n\n"
        "- Episodes: {}\n- Candidates: {}\n- Viewpoints: {}\n- Render failures: {}\n\n"
        "> These acoustic distributions are measured with a single broadband synthetic probe. They are intended for Pipeline V0 acoustic sanity checking and relative viewpoint comparison, not as speech/music/event-independent acceptance distributions.\n\n"
        "Different listening poses/orientations alter the binaural acoustic observation; this report makes no SED/SELD utility claim.\n".format(
            len(episodes), len(candidates), len(viewpoints), failure_count
        ), encoding="utf-8",
    )
    visualization = None
    if topdown:
        from active_audition.visualization.topdown import render_golden_topdown
        visualization = render_golden_topdown(str(root), config_path, run_root, episodes, candidates, viewpoints)
    _write_json(run_root / "reports" / "qc_result.json", dict(summary, visualization=visualization))
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs" / "qc.log").write_text("M3 read-only QC complete\n", encoding="utf-8")
    return dict(summary, run_root=str(run_root), visualization=visualization)
