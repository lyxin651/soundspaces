"""Exploratory spatial diagnostic for subthreshold Candidate A outputs.

This is deliberately separate from the official active-prediction evaluator:
all rows are measured from the target class x/y peak even when ACCDOA activity
is below 0.5.  It must not be interpreted as SELD detection or valid DOA.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def circular_error(pred, target):
    return abs((float(pred) - float(target) + 180.0) % 360.0 - 180.0)


def circular_std(values):
    if not values:
        return None
    radians = np.deg2rad(np.asarray(values, dtype=float))
    resultant = float(np.hypot(np.mean(np.cos(radians)), np.mean(np.sin(radians))))
    return float(np.rad2deg(np.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-12))))))


def mean(values):
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle)}
    predictions = {}
    with Path(args.predictions).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[row["sample_id"]] = row

    rows = []
    by_angle = defaultdict(list)
    rear_folded = []
    rear_full = []
    front = []
    left_right = []
    for sample_id, gt in manifest.items():
        peak = predictions.get(sample_id, {}).get("target_latent_peak")
        if not peak or peak.get("azimuth_deg") is None:
            continue
        pred = float(peak["azimuth_deg"])
        full_gt = float(gt["gt_azimuth_full_deg"])
        folded_gt = float(gt["gt_azimuth_folded_deg"])
        row = {
            "sample_id": sample_id,
            "gt_full_deg": full_gt,
            "gt_folded_deg": folded_gt,
            "predicted_latent_azimuth_deg": pred,
            "peak_activity_norm": float(peak["activity_norm"]),
            "folded_error_deg": circular_error(pred, folded_gt),
            "full_error_deg": circular_error(pred, full_gt),
        }
        rows.append(row)
        by_angle[str(round(full_gt / 30.0) * 30)].append(row["folded_error_deg"])
        if abs(full_gt) <= 90.0:
            front.append(row["folded_error_deg"])
        else:
            rear_folded.append(row["folded_error_deg"])
            rear_full.append(row["full_error_deg"])
        if abs(folded_gt) >= 1.0:
            left_right.append(np.sign(pred) == np.sign(folded_gt))

    angles = [row["predicted_latent_azimuth_deg"] for row in rows]
    result = {
        "definition": "Subthreshold latent diagnostic only; target class x/y peak, regardless of ACCDOA activity threshold. Not SELD detection/DOA evidence.",
        "n_manifest": len(manifest),
        "n_latent_rows": len(rows),
        "folded_azimuth_mae_deg": mean([row["folded_error_deg"] for row in rows]),
        "left_right_accuracy": float(np.mean(left_right)) if left_right else None,
        "front_hemisphere_mae_deg": mean(front),
        "rear_hemisphere_folded_mae_deg": mean(rear_folded),
        "rear_hemisphere_full360_mae_deg": mean(rear_full),
        "per_angle": {
            angle: {
                "n": len(errors),
                "folded_mae_deg": mean(errors),
            }
            for angle, errors in sorted(by_angle.items(), key=lambda item: float(item[0]))
        },
        "prediction_distribution": {
            "circular_mean_deg": None if not angles else float(np.rad2deg(np.arctan2(np.mean(np.sin(np.deg2rad(angles))), np.mean(np.cos(np.deg2rad(angles)))))) ,
            "circular_std_deg": circular_std(angles),
            "min_deg": min(angles) if angles else None,
            "max_deg": max(angles) if angles else None,
            "range_deg": max(angles) - min(angles) if angles else None,
            "unique_rounded_deg": sorted(set(round(value, 1) for value in angles)),
        },
        "direction_flip_check": {
            "left_right_accuracy": float(np.mean(left_right)) if left_right else None,
            "all_signs_reversed_heuristic": bool(left_right and np.mean(left_right) <= 0.15),
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
