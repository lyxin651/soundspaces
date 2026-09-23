"""Evaluate Candidate A revalidation data in the project azimuth convention."""

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def dcase_to_project_azimuth(theta_dcase):
    """DCASE stereo output uses the opposite positive direction to this project."""
    return float(-float(theta_dcase))


def circular_error(pred, gt):
    return abs((float(pred) - float(gt) + 180.0) % 360.0 - 180.0)


def circular_median(values):
    if not values:
        return None
    values = [float(value) for value in values]
    return min(values, key=lambda candidate: sum(circular_error(candidate, other) for other in values))


def load_rows(path, repo_root):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wav = Path(row["wav_path"])
            row["_wav_path"] = str(wav if wav.is_absolute() else repo_root / wav)
            rows.append(row)
    return rows


def load_model(repo_root, candidate_root, config_path, checkpoint_path):
    sys.path.insert(0, str(repo_root / "examples" / "seld_transfer"))
    sys.path.insert(0, str(candidate_root))
    from run_dcase25_baseline import load_audio_features
    from model import SELDModel
    import utils as feature_utils

    with Path(config_path).open("rb") as handle:
        params = dict(pickle.load(handle))
    params["modality"] = "audio"
    params["multiACCDOA"] = True
    model = SELDModel(params)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(checkpoint["seld_model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("checkpoint mismatch")
    model.eval()
    return model, params, feature_utils, load_audio_features


def decode_sample(raw, class_id, gt_project):
    shaped = raw.reshape(50, 3, 3, 13)
    x = shaped[:, :, 0, class_id]
    y = shaped[:, :, 1, class_id]
    norm = np.sqrt(x * x + y * y)
    active = norm > 0.5
    frame_active = active.any(axis=1)
    frame_predictions = []
    for frame in np.flatnonzero(frame_active):
        track = int(np.argmax(norm[frame]))
        theta_dcase = float(np.degrees(np.arctan2(y[frame, track], x[frame, track])))
        theta_project = dcase_to_project_azimuth(theta_dcase)
        frame_predictions.append({
            "frame": int(frame),
            "track": track,
            "theta_dcase_deg": theta_dcase,
            "theta_project_deg": theta_project,
            "activity_norm": float(norm[frame, track]),
            "error_project_deg": circular_error(theta_project, gt_project),
        })
    sample_pred = circular_median([item["theta_project_deg"] for item in frame_predictions])
    return {
        "target_max_norm": float(norm.max()),
        "active_frame_count": int(frame_active.sum()),
        "active_frame_ratio": float(frame_active.mean()),
        "active": bool(frame_active.any()),
        "predicted_theta_project_deg": sample_pred,
        "predicted_theta_dcase_deg": dcase_to_project_azimuth(sample_pred) if sample_pred is not None else None,
        "frame_predictions": frame_predictions,
    }


def sample_metrics(samples):
    active = [sample for sample in samples if sample["active"] and sample["predicted_theta_project_deg"] is not None]
    errors = [circular_error(sample["predicted_theta_project_deg"], sample["gt_azimuth_project_deg"]) for sample in active]
    sign_samples = [sample for sample in active if abs(sample["gt_azimuth_project_deg"]) >= 1.0]
    left_right = (
        float(np.mean([
            np.sign(sample["predicted_theta_project_deg"]) == np.sign(sample["gt_azimuth_project_deg"])
            for sample in sign_samples
        ]))
        if sign_samples else None
    )
    return {
        "n_samples": len(samples),
        "n_active_samples": len(active),
        "clip_detection_rate": float(len(active) / len(samples)) if samples else None,
        "miss_rate": float(1.0 - len(active) / len(samples)) if samples else None,
        "mean_active_frame_ratio": float(np.mean([sample["active_frame_ratio"] for sample in samples])) if samples else None,
        "sample_balanced_folded_mae_deg": float(np.mean(errors)) if errors else None,
        "sample_balanced_median_ae_deg": float(np.median(errors)) if errors else None,
        "left_right_accuracy": left_right,
    }


def grouped_metrics(samples, key):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample[key]].append(sample)
    output = {}
    for group, group_samples in sorted(groups.items(), key=lambda item: str(item[0])):
        output[str(group)] = sample_metrics(group_samples)
    return output


def compression(samples):
    active = [sample for sample in samples if sample["active"] and sample["predicted_theta_project_deg"] is not None]
    by_class = defaultdict(list)
    for sample in active:
        by_class[sample["class_name"]].append(sample)
    result = {}
    for class_name, values in sorted(by_class.items()):
        gt = np.asarray([sample["gt_azimuth_project_deg"] for sample in values], dtype=np.float64)
        pred = np.asarray([sample["predicted_theta_project_deg"] for sample in values], dtype=np.float64)
        if len(values) >= 2 and np.ptp(gt) > 0:
            beta, alpha = np.polyfit(gt, pred, 1)
            rho, pvalue = spearmanr(gt, pred)
            result[class_name] = {
                "n_active_samples": len(values),
                "alpha_deg": float(alpha),
                "beta": float(beta),
                "spearman_rho": float(rho),
                "spearman_pvalue": float(pvalue),
                "gt_range_deg": [float(gt.min()), float(gt.max())],
                "pred_range_deg": [float(pred.min()), float(pred.max())],
            }
        else:
            result[class_name] = {"n_active_samples": len(values), "alpha_deg": None, "beta": None, "spearman_rho": None, "spearman_pvalue": None}
    return result


def make_plots(samples, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    active = [sample for sample in samples if sample["active"] and sample["predicted_theta_project_deg"] is not None]
    width, height, left, top, plot_w, plot_h = 820, 620, 70, 45, 700, 500
    def sx(value):
        return left + (float(value) + 90.0) / 180.0 * plot_w
    def sy(value):
        return top + (90.0 - float(value)) / 180.0 * plot_h
    colors = {"Male speech": "#1f77b4", "Music": "#d62728", "Domestic sounds": "#2ca02c"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<text x="410" y="22" text-anchor="middle" font-size="16">Candidate A project-coordinate GT to prediction</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>')
    for tick in (-90, -60, -30, 0, 30, 60, 90):
        parts.append(f'<line x1="{sx(tick):.1f}" y1="{top + plot_h}" x2="{sx(tick):.1f}" y2="{top + plot_h + 5}" stroke="black"/><text x="{sx(tick):.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="11">{tick}</text>')
        parts.append(f'<line x1="{left - 5}" y1="{sy(tick):.1f}" x2="{left}" y2="{sy(tick):.1f}" stroke="black"/><text x="{left - 10}" y="{sy(tick) + 4:.1f}" text-anchor="end" font-size="11">{tick}</text>')
    parts.append(f'<line x1="{sx(-90):.1f}" y1="{sy(-90):.1f}" x2="{sx(90):.1f}" y2="{sy(90):.1f}" stroke="#777" stroke-dasharray="5,4"/>')
    for sample in active:
        color = colors.get(sample["class_name"], "#555")
        parts.append(f'<circle cx="{sx(sample["gt_azimuth_project_deg"]):.1f}" cy="{sy(sample["predicted_theta_project_deg"]):.1f}" r="4" fill="{color}"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-size="13">GT project azimuth (deg)</text><text x="16" y="{top + plot_h / 2}" transform="rotate(-90 16 {top + plot_h / 2})" text-anchor="middle" font-size="13">Predicted project azimuth (deg)</text>')
    legend_x = left + plot_w - 170
    for index, class_name in enumerate(sorted({sample["class_name"] for sample in active})):
        y = top + 20 + index * 18
        parts.append(f'<circle cx="{legend_x}" cy="{y - 4}" r="4" fill="{colors.get(class_name, "#555")}"/><text x="{legend_x + 10}" y="{y}" font-size="11">{class_name}</text>')
    parts.append('</svg>')
    (plots_dir / "gt_pred_project_by_class.svg").write_text("".join(parts), encoding="utf-8")

    angle_groups = grouped_metrics(samples, "gt_azimuth_project_deg")
    angles = sorted(float(angle) for angle in angle_groups)
    mae = [angle_groups[str(angle)]["sample_balanced_folded_mae_deg"] for angle in angles]
    valid_mae = [value for value in mae if value is not None]
    ymax = max(valid_mae + [1.0]) * 1.15
    def sy_mae(value):
        return top + plot_h - float(value) / ymax * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="410" y="22" text-anchor="middle" font-size="16">Sample-balanced error by project azimuth</text>']
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>')
    points = []
    for angle, value in zip(angles, mae):
        if value is not None:
            points.append(f'{sx(angle):.1f},{sy_mae(value):.1f}')
            parts.append(f'<circle cx="{sx(angle):.1f}" cy="{sy_mae(value):.1f}" r="4" fill="#1f77b4"/>')
        parts.append(f'<text x="{sx(angle):.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="11">{int(angle)}</text>')
    if points:
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1f77b4" stroke-width="2"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-size="13">GT project azimuth (deg)</text><text x="16" y="{top + plot_h / 2}" transform="rotate(-90 16 {top + plot_h / 2})" text-anchor="middle" font-size="13">Sample-balanced MAE (deg)</text></svg>')
    (plots_dir / "sample_balanced_error_by_angle.svg").write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    rows = load_rows(args.manifest, repo_root)
    model, params, feature_utils, load_audio_features = load_model(
        repo_root, Path(args.candidate_root).resolve(), args.config, args.checkpoint
    )
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    all_sample_records = []
    all_frame_records = []
    condition_results = {}
    predictions_dir = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for condition, condition_rows in sorted(by_condition.items()):
        features = [load_audio_features(row["_wav_path"], feature_utils, params) for row in condition_rows]
        with torch.inference_mode():
            raw = model(torch.from_numpy(np.stack(features))).cpu().numpy()
        np.save(predictions_dir / f"raw_{condition}.npy", raw)
        samples = []
        for index, row in enumerate(condition_rows):
            record = {
                "sample_id": row["sample_id"],
                "condition": condition,
                "scene": row["scene"],
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "dry_name": row["dry_name"],
                "gt_azimuth_project_deg": float(row["gt_azimuth_project_deg"]),
                "gt_azimuth_dcase_deg": float(row["gt_azimuth_dcase_deg"]),
                "distance_m": float(row["distance_m"]),
                "materials_on": row["materials_on"].lower() == "true",
            }
            record.update(decode_sample(raw[index], int(row["class_id"]), record["gt_azimuth_project_deg"]))
            samples.append(record)
            all_sample_records.append(record)
            for frame in record["frame_predictions"]:
                all_frame_records.append({
                    "sample_id": record["sample_id"],
                    "condition": condition,
                    "class_name": record["class_name"],
                    "gt_azimuth_project_deg": record["gt_azimuth_project_deg"],
                    **frame,
                })
        condition_results[condition] = {
            "sample_metrics": sample_metrics(samples),
            "by_class": grouped_metrics(samples, "class_name"),
            "by_angle": grouped_metrics(samples, "gt_azimuth_project_deg"),
            "compression": compression(samples),
        }

    frame_path = predictions_dir / "frame_predictions.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_frame_records[0]) if all_frame_records else ["sample_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_frame_records)
    sample_path = metrics_dir / "sample_predictions.csv"
    sample_fields = [key for key in all_sample_records[0] if key != "frame_predictions"]
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows([{key: record[key] for key in sample_fields} for record in all_sample_records])

    summary = {
        "coordinate_convention": {
            "project": "0=front,+90=right,-90=left",
            "conversion_function": "dcase_to_project_azimuth(theta_dcase) = -theta_dcase",
            "formal_doa_threshold": 0.5,
            "sample_balanced_primary_metric": True,
        },
        "n_samples": len(all_sample_records),
        "conditions": condition_results,
        "overall_by_condition": {condition: result["sample_metrics"] for condition, result in condition_results.items()},
        "cross_scene": {"status": "BLOCKED", "reason": "Only office_0 Replica scene is available locally; no additional Replica scene asset found."},
        "dry_inventory": "dry_inventory.csv",
        "predictions": "predictions/",
        "plots": "plots/",
    }
    (metrics_dir / "revalidation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(all_sample_records, output_root / "plots")
    print(json.dumps(summary["overall_by_condition"], indent=2))


if __name__ == "__main__":
    main()
