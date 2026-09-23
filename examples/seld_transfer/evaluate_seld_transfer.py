"""评价迁移预测；prediction JSONL 每行对应 manifest 的一个 sample_id。"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def circular_error(pred, target):
    return abs((float(pred) - float(target) + 180.0) % 360.0 - 180.0)


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/experiments/seld_transfer_gate_20260824/manifest.csv")
    parser.add_argument("--predictions", required=True, help="JSONL: sample_id, predicted_class_id, activity/confidence, predicted_azimuth_deg")
    parser.add_argument("--output", default="data/experiments/seld_transfer_gate_20260824/metrics/metrics.json")
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle)}
    predictions = {}
    with Path(args.predictions).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[row["sample_id"]] = row

    detection = {"matched": 0, "wrong_class": 0, "miss": 0}
    spatial = []
    front_spatial = []
    rear_spatial = []
    left_right = []
    by_angle = defaultdict(list)
    by_scene = defaultdict(list)
    by_class = defaultdict(list)
    failures = []
    for sample_id, row in manifest.items():
        prediction = predictions.get(sample_id, {})
        active = bool(prediction.get("active", prediction.get("activity", False)))
        pred_class = prediction.get("predicted_class_id")
        if not active:
            detection["miss"] += 1
            failures.append((sample_id, "miss"))
            continue
        if str(pred_class) == row["class_id"]:
            detection["matched"] += 1
        else:
            detection["wrong_class"] += 1
            failures.append((sample_id, "wrong_class"))
        pred_azimuth = prediction.get("predicted_azimuth_deg")
        if pred_azimuth is None or not math.isfinite(float(pred_azimuth)):
            failures.append((sample_id, "invalid_azimuth"))
            continue
        full_gt = float(row["gt_azimuth_full_deg"])
        folded_gt = float(row["gt_azimuth_folded_deg"])
        error = circular_error(pred_azimuth, folded_gt)
        spatial.append(error)
        bucket = round(full_gt / 30.0) * 30
        by_angle[str(bucket)].append(error)
        by_scene[row["scene"]].append(error)
        by_class[row["class_name"]].append(error)
        if abs(full_gt) <= 90.0:
            front_spatial.append(error)
        else:
            rear_spatial.append(error)
        if abs(folded_gt) >= 1.0:
            left_right.append(np.sign(float(pred_azimuth)) == np.sign(folded_gt))

    total = len(manifest)
    metrics = {
        "n_manifest": total, "n_predictions": len(predictions),
        "detection": {
            "target_class_detection_rate": detection["matched"] / total if total else None,
            "wrong_class_rate": detection["wrong_class"] / total if total else None,
            "miss_rate": detection["miss"] / total if total else None,
            **detection,
        },
        "spatial": {
            "folded_azimuth_mae_deg": mean_or_none(spatial),
            "left_right_accuracy": float(np.mean(left_right)) if left_right else None,
            "front_hemisphere_mae_deg": mean_or_none(front_spatial),
            "rear_hemisphere_folded_mae_deg": mean_or_none(rear_spatial),
            "per_azimuth_mae_deg": {key: mean_or_none(value) for key, value in sorted(by_angle.items())},
            "per_scene_mae_deg": {key: mean_or_none(value) for key, value in sorted(by_scene.items())},
            "per_class_mae_deg": {key: mean_or_none(value) for key, value in sorted(by_class.items())},
            "n_active_spatial": len(spatial),
        },
        "failure_cases": [{"sample_id": sample_id, "reason": reason} for sample_id, reason in failures[:20]],
        "note": "Rear hemisphere is evaluated against folded GT only; this cannot establish full-360 localization.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
