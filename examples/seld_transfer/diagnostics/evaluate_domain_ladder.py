"""Evaluate C4 ladder using only official activity threshold for formal DOA."""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root / "examples" / "seld_transfer"))
    sys.path.insert(0, str(Path(args.candidate_root).resolve()))
    from run_dcase25_baseline import load_audio_features
    from model import SELDModel
    import utils as feature_utils

    with Path(args.config).open("rb") as handle:
        params = dict(pickle.load(handle))
    params["modality"] = "audio"
    params["multiACCDOA"] = True
    model = SELDModel(params)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = model.load_state_dict(checkpoint["seld_model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("checkpoint mismatch")
    model.eval()

    rows = []
    with Path(args.manifest).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    features = [load_audio_features(row["wav_path"], feature_utils, params) for row in rows]
    with torch.inference_mode():
        raw = model(torch.from_numpy(np.stack(features, axis=0))).cpu().numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "raw_output.npy", raw)

    sample_results = []
    for index, row in enumerate(rows):
        shaped = raw[index].reshape(50, 3, 3, 13)
        x = shaped[:, :, 0, int(row["class_id"])]
        y = shaped[:, :, 1, int(row["class_id"])]
        norm = np.sqrt(x * x + y * y)
        active = norm > 0.5
        frame_active = active.any(axis=1)
        predictions = []
        for frame in np.flatnonzero(frame_active):
            track = int(np.argmax(norm[frame]))
            pred_angle = float(np.degrees(np.arctan2(y[frame, track], x[frame, track])))
            gt = float(row["gt_azimuth_folded_deg"])
            error = abs((pred_angle - gt + 180.0) % 360.0 - 180.0)
            predictions.append({"frame": int(frame), "track": track, "predicted_azimuth_deg": pred_angle, "error_deg": error})
        sample_results.append({
            "sample_id": row["sample_id"], "class_id": int(row["class_id"]), "class_name": row["class_name"],
            "gt_azimuth_full_deg": float(row["gt_azimuth_full_deg"]), "gt_azimuth_folded_deg": float(row["gt_azimuth_folded_deg"]),
            "distance_m": float(row["distance_m"]), "max_norm": float(norm.max()), "mean_norm": float(norm.mean()),
            "active_frame_count": int(frame_active.sum()), "active_frame_ratio": float(frame_active.mean()),
            "active": bool(frame_active.any()), "active_predictions": predictions,
        })
    active_rows = [row for row in sample_results if row["active"]]
    active_predictions = [item for row in active_rows for item in row["active_predictions"]]
    result_json = {
        "definition": "Formal spatial metrics use only frames with official ACCDOA norm > 0.5; no subthreshold atan2 is included.",
        "raw_shape": list(raw.shape),
        "n_samples": len(sample_results),
        "active_sample_rate": len(active_rows) / len(sample_results) if sample_results else None,
        "active_frame_rate": float(np.mean([row["active_frame_ratio"] for row in sample_results])) if sample_results else None,
        "active_rows": len(active_rows),
        "active_prediction_count": len(active_predictions),
        "formal_folded_mae_deg": float(np.mean([item["error_deg"] for item in active_predictions])) if active_predictions else None,
        "formal_median_error_deg": float(np.median([item["error_deg"] for item in active_predictions])) if active_predictions else None,
        "left_right_accuracy": (float(np.mean([np.sign(item["predicted_azimuth_deg"]) == np.sign(next(row["gt_azimuth_folded_deg"] for row in active_rows if item in row["active_predictions"])) for item in active_predictions if abs(next(row["gt_azimuth_folded_deg"] for row in active_rows if item in row["active_predictions"])) >= 1.0])) if active_predictions else None),
        "per_sample": sample_results,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    print(json.dumps(result_json, indent=2))


if __name__ == "__main__":
    main()
