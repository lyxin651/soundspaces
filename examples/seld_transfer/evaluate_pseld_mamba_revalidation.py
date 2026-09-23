"""Evaluate PSELD_Mamba SoundSpaces revalidation predictions.

This evaluator consumes the project JSONL emitted by run_pseld_mamba.py and a
SoundSpaces-style manifest.  It reports target-class detection and target-class
localization in the project azimuth convention, plus a global sign audit for
DCASE azimuths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(row) for row in data]
        if isinstance(data, dict) and "samples" in data:
            return [dict(row) for row in data["samples"]]
        raise ValueError("JSON manifest must be a list or contain a 'samples' list")
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_sample_id(row: Mapping[str, Any], index: int, sample_id_field: str) -> str:
    if sample_id_field in row and str(row[sample_id_field]):
        return str(row[sample_id_field])
    if row.get("episode_id") is not None and row.get("viewpoint_id") is not None:
        return "{}__{}".format(row["episode_id"], row["viewpoint_id"])
    if row.get("viewpoint_id") is not None:
        return str(row["viewpoint_id"])
    return "sample_{:06d}".format(index)


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def circular_error(pred: float, target: float) -> float:
    return abs((float(pred) - float(target) + 180.0) % 360.0 - 180.0)


def signed_project_azimuth(theta_dcase: Optional[float], sign: int) -> Optional[float]:
    if theta_dcase is None:
        return None
    return float(sign) * float(theta_dcase)


def percentile_summary(values: Sequence[float]) -> Dict[str, Any]:
    items = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=np.float64)
    if items.size == 0:
        return {"count": 0, "min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(items.size),
        "min": float(np.min(items)),
        "p05": float(np.percentile(items, 5)),
        "p25": float(np.percentile(items, 25)),
        "median": float(np.percentile(items, 50)),
        "p75": float(np.percentile(items, 75)),
        "p95": float(np.percentile(items, 95)),
        "max": float(np.max(items)),
        "mean": float(np.mean(items)),
    }


def safe_ratio(num: int, den: int) -> Optional[float]:
    return float(num) / float(den) if den else None


def spearman_rho(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def regression_summary(samples: Sequence[Mapping[str, Any]], sign: int) -> Dict[str, Any]:
    active = [sample for sample in samples if sample.get("target_active")]
    gt = [float(sample["gt_azimuth_project_deg"]) for sample in active if sample.get("target_azimuth_dcase_deg") is not None]
    pred = [signed_project_azimuth(sample.get("target_azimuth_dcase_deg"), sign) for sample in active if sample.get("target_azimuth_dcase_deg") is not None]
    pred = [float(value) for value in pred if value is not None]
    if len(gt) >= 2 and len(pred) >= 2 and np.ptp(np.asarray(gt)) > 0:
        beta, alpha = np.polyfit(np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64), 1)
        return {
            "n_active_samples": len(pred),
            "alpha_deg": float(alpha),
            "beta": float(beta),
            "spearman_rho": spearman_rho(gt, pred),
            "gt_range_deg": [float(min(gt)), float(max(gt))],
            "pred_range_deg": [float(min(pred)), float(max(pred))],
        }
    return {"n_active_samples": len(pred), "alpha_deg": None, "beta": None, "spearman_rho": None, "gt_range_deg": None, "pred_range_deg": None}


def metric_block(samples: Sequence[Mapping[str, Any]], sign: int, model_frame_count: Optional[int]) -> Dict[str, Any]:
    total = len(samples)
    target_active = [sample for sample in samples if sample.get("target_active")]
    errors = []
    lr_flags = []
    target_active_ratios = []
    top_matches = []
    for sample in samples:
        target_class_id = sample.get("target_class_id")
        predicted_class_id = sample.get("predicted_class_id")
        if target_class_id is not None and predicted_class_id is not None:
            top_matches.append(int(target_class_id) == int(predicted_class_id))
        if model_frame_count and sample.get("target_active_frame_count") is not None:
            target_active_ratios.append(float(sample["target_active_frame_count"]) / float(model_frame_count))
        if not sample.get("target_active"):
            continue
        theta_dcase = sample.get("target_azimuth_dcase_deg")
        gt = sample.get("gt_azimuth_project_deg")
        pred = signed_project_azimuth(theta_dcase, sign)
        if pred is None or gt is None:
            continue
        error = circular_error(pred, float(gt))
        errors.append(error)
        if abs(float(gt)) >= 1.0:
            lr_flags.append(np.sign(pred) == np.sign(float(gt)))
    return {
        "n_samples": total,
        "n_target_active_samples": len(target_active),
        "target_class_detection_rate": safe_ratio(len(target_active), total),
        "miss_rate": 1.0 - safe_ratio(len(target_active), total) if total else None,
        "mean_target_active_frame_ratio": float(np.mean(target_active_ratios)) if target_active_ratios else None,
        "top_class_match_rate": float(np.mean(top_matches)) if top_matches else None,
        "sample_balanced_folded_mae_deg": float(np.mean(errors)) if errors else None,
        "sample_balanced_median_ae_deg": float(np.median(errors)) if errors else None,
        "left_right_accuracy": float(np.mean(lr_flags)) if lr_flags else None,
        "error_distribution_deg": percentile_summary(errors),
        "target_confidence_distribution": percentile_summary([sample["target_confidence"] for sample in target_active if sample.get("target_confidence") is not None]),
        "target_active_frame_count_distribution": percentile_summary([sample["target_active_frame_count"] for sample in target_active if sample.get("target_active_frame_count") is not None]),
        "angle_trend": regression_summary(samples, sign),
    }


def grouped_metrics(samples: Sequence[Mapping[str, Any]], group_key: str, sign: int, model_frame_count: Optional[int]) -> Dict[str, Any]:
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.get(group_key, "")].append(sample)
    return {str(key): metric_block(values, sign, model_frame_count) for key, values in sorted(groups.items(), key=lambda item: str(item[0]))}


def build_samples(manifest_rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]], sample_id_field: str) -> List[Dict[str, Any]]:
    samples = []
    for index, row in enumerate(manifest_rows):
        sample_id = row_sample_id(row, index, sample_id_field)
        pred = predictions.get(sample_id, {})
        target = pred.get("target") or {}
        target_loc = target.get("localization") or None
        target_active = bool(target_loc and as_int(target.get("active_frame_count"), 0) and as_int(target.get("active_frame_count"), 0) > 0)
        class_id = as_int(row.get("class_id"), None)
        sample = {
            "sample_id": sample_id,
            "condition": row.get("condition", ""),
            "scene": row.get("scene", row.get("scene_id", "")),
            "class_id": class_id,
            "class_name": row.get("class_name", ""),
            "dry_name": row.get("dry_name", ""),
            "gt_azimuth_project_deg": as_float(row.get("gt_azimuth_project_deg"), as_float(row.get("gt_azimuth_deg"), None)),
            "gt_azimuth_dcase_deg": as_float(row.get("gt_azimuth_dcase_deg"), None),
            "distance_m": as_float(row.get("distance_m"), None),
            "materials_on": str(row.get("materials_on", "")).lower() == "true",
            "prediction_found": bool(pred),
            "predicted_class_id": pred.get("predicted_class_id"),
            "predicted_class_name": pred.get("predicted_class_name"),
            "predicted_azimuth_deg_from_file": pred.get("predicted_azimuth_deg"),
            "target_class_id": target.get("class_id", class_id),
            "target_class_name": target.get("class_name", row.get("class_name", "")),
            "target_active": target_active,
            "target_max_score": as_float(target.get("max_score"), None),
            "target_active_frame_count": as_int(target.get("active_frame_count"), 0),
            "target_confidence": as_float(target_loc.get("confidence") if target_loc else None, None),
            "target_azimuth_dcase_deg": as_float(target_loc.get("azimuth_dcase_deg") if target_loc else None, None),
            "target_distance_m": as_float(target_loc.get("distance_m") if target_loc else None, None),
        }
        samples.append(sample)
    return samples


def write_sample_csv(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "condition", "scene", "class_id", "class_name", "dry_name",
        "gt_azimuth_project_deg", "gt_azimuth_dcase_deg", "distance_m", "materials_on",
        "prediction_found", "predicted_class_id", "predicted_class_name", "target_active",
        "target_max_score", "target_active_frame_count", "target_confidence",
        "target_azimuth_dcase_deg", "target_project_azimuth_sign_minus", "target_project_azimuth_sign_plus",
        "error_sign_minus_deg", "error_sign_plus_deg", "target_distance_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = dict(sample)
            minus = signed_project_azimuth(sample.get("target_azimuth_dcase_deg"), -1)
            plus = signed_project_azimuth(sample.get("target_azimuth_dcase_deg"), 1)
            gt = sample.get("gt_azimuth_project_deg")
            row["target_project_azimuth_sign_minus"] = minus
            row["target_project_azimuth_sign_plus"] = plus
            row["error_sign_minus_deg"] = circular_error(minus, gt) if minus is not None and gt is not None else None
            row["error_sign_plus_deg"] = circular_error(plus, gt) if plus is not None and gt is not None else None
            writer.writerow({field: row.get(field) for field in fields})


def write_summary(path: Path, metrics: Mapping[str, Any], primary_sign: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary_key = "sign_minus" if primary_sign == -1 else "sign_plus"
    primary = metrics[primary_key]["overall"]
    other_key = "sign_plus" if primary_sign == -1 else "sign_minus"
    other = metrics[other_key]["overall"]
    lines = [
        "# PSELD_Mamba Revalidation",
        "",
        "Primary convention: `theta_project = {} * theta_dcase`.".format(primary_sign),
        "",
        "## Overall",
        "",
        "- Samples: {}".format(primary["n_samples"]),
        "- Target detection rate: {:.4f}".format(primary["target_class_detection_rate"] or 0.0),
        "- Mean target active-frame ratio: {:.4f}".format(primary["mean_target_active_frame_ratio"] or 0.0),
        "- Sample-balanced MAE: {}".format(primary["sample_balanced_folded_mae_deg"]),
        "- Median AE: {}".format(primary["sample_balanced_median_ae_deg"]),
        "- Left/right accuracy: {}".format(primary["left_right_accuracy"]),
        "- Angle trend beta: {}".format(primary["angle_trend"].get("beta")),
        "- Spearman rho: {}".format(primary["angle_trend"].get("spearman_rho")),
        "",
        "## Sign Audit",
        "",
        "- Primary left/right: {}".format(primary["left_right_accuracy"]),
        "- Opposite sign left/right: {}".format(other["left_right_accuracy"]),
        "- Primary MAE: {}".format(primary["sample_balanced_folded_mae_deg"]),
        "- Opposite sign MAE: {}".format(other["sample_balanced_folded_mae_deg"]),
        "",
        "Use target-class localization only; inactive target-class samples do not contribute to spatial error.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSELD_Mamba revalidation predictions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sanity", default="")
    parser.add_argument("--sample-id-field", default="sample_id")
    parser.add_argument("--primary-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    prediction_path = Path(args.predictions)
    output_root = Path(args.output_root)
    manifest_rows = load_manifest(manifest_path)
    prediction_rows = load_jsonl(prediction_path)
    predictions = {str(row["sample_id"]): row for row in prediction_rows}
    samples = build_samples(manifest_rows, predictions, args.sample_id_field)

    model_frame_count = None
    if args.sanity:
        sanity = json.loads(Path(args.sanity).read_text(encoding="utf-8"))
        shape = sanity.get("output_logits", {}).get("shape")
        if shape and len(shape) >= 2:
            model_frame_count = int(shape[1])

    missing = [sample["sample_id"] for sample in samples if not sample["prediction_found"]]
    metrics = {
        "inputs": {
            "manifest": str(manifest_path),
            "predictions": str(prediction_path),
            "sanity": args.sanity or None,
            "manifest_samples": len(manifest_rows),
            "prediction_rows": len(prediction_rows),
            "missing_prediction_count": len(missing),
            "missing_prediction_examples": missing[:10],
            "model_frame_count": model_frame_count,
        },
        "sign_minus": {
            "azimuth_rule": "theta_project = -theta_dcase",
            "overall": metric_block(samples, -1, model_frame_count),
            "by_condition": grouped_metrics(samples, "condition", -1, model_frame_count),
            "by_class": grouped_metrics(samples, "class_name", -1, model_frame_count),
            "by_angle": grouped_metrics(samples, "gt_azimuth_project_deg", -1, model_frame_count),
        },
        "sign_plus": {
            "azimuth_rule": "theta_project = +theta_dcase",
            "overall": metric_block(samples, 1, model_frame_count),
            "by_condition": grouped_metrics(samples, "condition", 1, model_frame_count),
            "by_class": grouped_metrics(samples, "class_name", 1, model_frame_count),
            "by_angle": grouped_metrics(samples, "gt_azimuth_project_deg", 1, model_frame_count),
        },
    }

    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "revalidation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sample_csv(metrics_dir / "sample_predictions.csv", samples)
    write_summary(output_root / "report" / "summary.md", metrics, args.primary_sign)
    print(json.dumps(metrics["sign_minus"]["overall"], ensure_ascii=False, indent=2))
    print(json.dumps({"sign_plus_overall": metrics["sign_plus"]["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
