"""Channel/sign audit for Candidate B PSELD_Mamba on SoundSpaces.

The audit is intentionally small. It selects a controlled subset from a
SoundSpaces revalidation manifest, runs the PSELD_Mamba adapter on:

- original stereo audio
- L/R swapped audio
- original audio with the PFOA IVy channel sign inverted

It then scores both global azimuth conventions, theta_project = +/- theta_dcase,
using target-class localization only. This mirrors the Candidate A coordinate
sign audit without modifying third-party code or existing prediction files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
TRANSFER_DIR = THIS_DIR.parent
sys.path.insert(0, str(TRANSFER_DIR))

from run_pseld_mamba import (  # noqa: E402
    CLASS_NAMES,
    decode_output,
    extract_pfoa_feature,
    infer_batches,
    load_manifest,
    load_pseld_model,
    load_soundspaces_wav,
    resolve_path,
    row_sample_id,
    row_wav_path,
)


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def circular_error(pred: float, target: float) -> float:
    return abs((float(pred) - float(target) + 180.0) % 360.0 - 180.0)


def project(theta_dcase: Optional[float], sign: int) -> Optional[float]:
    if theta_dcase is None:
        return None
    return float(sign) * float(theta_dcase)


def percentile_summary(values: Sequence[float]) -> Dict[str, Any]:
    clean = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(clean.size),
        "min": float(clean.min()),
        "median": float(np.median(clean)),
        "mean": float(clean.mean()),
        "max": float(clean.max()),
    }


def metric_block(records: Sequence[Mapping[str, Any]], variant: str, sign: int) -> Dict[str, Any]:
    items = [r for r in records if r["variant"] == variant]
    active = [r for r in items if r["target_active"]]
    errors = []
    lr = []
    for row in active:
        pred = project(row["target_azimuth_dcase_deg"], sign)
        gt = row["gt_azimuth_project_deg"]
        if pred is None or gt is None:
            continue
        errors.append(circular_error(pred, gt))
        if abs(gt) >= 1.0:
            lr.append(np.sign(pred) == np.sign(gt))
    return {
        "variant": variant,
        "sign": sign,
        "n_samples": len(items),
        "n_target_active": len(active),
        "target_detection_rate": float(len(active) / len(items)) if items else None,
        "mae_deg": float(np.mean(errors)) if errors else None,
        "median_ae_deg": float(np.median(errors)) if errors else None,
        "left_right_accuracy": float(np.mean(lr)) if lr else None,
        "error_distribution_deg": percentile_summary(errors),
        "target_confidence_distribution": percentile_summary([r["target_confidence"] for r in active if r["target_confidence"] is not None]),
    }


def mirror_summary(records: Sequence[Mapping[str, Any]], sign: int) -> Dict[str, Any]:
    by_key = {(r["sample_id"], r["variant"]): r for r in records}
    pairs = []
    for row in records:
        if row["variant"] != "original" or not row["target_active"]:
            continue
        swapped = by_key.get((row["sample_id"], "lr_swap"))
        if not swapped or not swapped["target_active"]:
            continue
        orig = project(row["target_azimuth_dcase_deg"], sign)
        swap = project(swapped["target_azimuth_dcase_deg"], sign)
        if orig is None or swap is None:
            continue
        pairs.append({
            "sample_id": row["sample_id"],
            "gt_azimuth_project_deg": row["gt_azimuth_project_deg"],
            "original_project_deg": orig,
            "lr_swap_project_deg": swap,
            "mirror_error_deg": circular_error(swap, -orig),
        })
    errors = [p["mirror_error_deg"] for p in pairs]
    return {"n_pairs": len(pairs), "mirror_error_deg": percentile_summary(errors), "pairs": pairs}


def selected_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    wanted_angles = {float(v) for v in args.angles}
    class_names = set(args.class_name or [])
    selected = []
    per_angle_count = defaultdict(int)
    for row in rows:
        if args.condition and row.get("condition") != args.condition:
            continue
        if class_names and row.get("class_name") not in class_names:
            continue
        angle = as_float(row.get("gt_azimuth_project_deg"))
        if angle is None or angle not in wanted_angles:
            continue
        if args.max_per_angle > 0 and per_angle_count[angle] >= args.max_per_angle:
            continue
        per_angle_count[angle] += 1
        selected.append(dict(row))
    return selected


def make_variant_feature(audio: np.ndarray, variant: str, params: Mapping[str, Any], pseld_utils: Any) -> np.ndarray:
    if variant == "original":
        variant_audio = audio
        feature = extract_pfoa_feature(variant_audio, params, pseld_utils)
    elif variant == "lr_swap":
        variant_audio = audio[[1, 0], :]
        feature = extract_pfoa_feature(variant_audio, params, pseld_utils)
    elif variant == "pfoa_ivy_invert":
        feature = extract_pfoa_feature(audio, params, pseld_utils)
        # PSELD_Mamba PFOA feature layout is 4 log-mel channels + [IVy, IVz, IVx].
        # Inverting IVy isolates the pseudo-FOA left/right axis without swapping audio.
        feature = feature.copy()
        feature[4] *= -1.0
    else:
        raise ValueError("unknown variant: {}".format(variant))
    return feature


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "variant", "condition", "class_name", "dry_name",
        "gt_azimuth_project_deg", "target_active", "target_max_score",
        "target_active_frame_count", "target_confidence", "target_azimuth_dcase_deg",
        "project_sign_minus_deg", "project_sign_plus_deg", "error_sign_minus_deg",
        "error_sign_plus_deg", "target_distance_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            minus = project(row["target_azimuth_dcase_deg"], -1)
            plus = project(row["target_azimuth_dcase_deg"], 1)
            gt = row["gt_azimuth_project_deg"]
            output["project_sign_minus_deg"] = minus
            output["project_sign_plus_deg"] = plus
            output["error_sign_minus_deg"] = circular_error(minus, gt) if minus is not None and gt is not None else None
            output["error_sign_plus_deg"] = circular_error(plus, gt) if plus is not None and gt is not None else None
            writer.writerow({field: output.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSELD_Mamba channel/sign audit")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--exp", default="EXP_CNN14_BiMambaAC")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--condition", default="near_off")
    parser.add_argument("--class-name", action="append", default=["Male speech"])
    parser.add_argument("--angles", type=float, nargs="+", default=[-60.0, -30.0, 0.0, 30.0, 60.0])
    parser.add_argument("--max-per-angle", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    candidate_root = Path(args.candidate_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_root = Path(args.output_root)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows = selected_rows(load_manifest(Path(args.manifest)), args)
    if not rows:
        raise RuntimeError("no rows matched audit filters")

    model, params, pseld_utils, checkpoint_obj, load_result = load_pseld_model(candidate_root, args.exp, checkpoint, device)
    target_sr = int(params["sampling_rate"])
    variants = ["original", "lr_swap", "pfoa_ivy_invert"]

    features = []
    items = []
    for index, row in enumerate(rows):
        wav = resolve_path(row_wav_path(row, "wav_path"), repo_root, dataset_root)
        audio, audio_info = load_soundspaces_wav(wav, target_sr, args.clip_seconds)
        for variant in variants:
            feature = make_variant_feature(audio, variant, params, pseld_utils)
            features.append(feature)
            items.append({"index": index, "row": row, "variant": variant, "audio_info": audio_info})

    logits = infer_batches(model, features, device, max(1, args.batch_size))
    records = []
    for item, output in zip(items, logits):
        row = item["row"]
        target_class_id = int(row["class_id"])
        decoded = decode_output(output, target_class_id, args.activity_threshold, -1)
        target = decoded.get("target") or {}
        loc = target.get("localization") or None
        target_active = bool(loc and int(target.get("active_frame_count") or 0) > 0)
        records.append({
            "sample_id": row_sample_id(row, item["index"], "sample_id"),
            "variant": item["variant"],
            "condition": row.get("condition"),
            "class_id": int(row["class_id"]),
            "class_name": row.get("class_name"),
            "dry_name": row.get("dry_name"),
            "gt_azimuth_project_deg": float(row["gt_azimuth_project_deg"]),
            "gt_azimuth_dcase_deg": float(row["gt_azimuth_dcase_deg"]),
            "target_active": target_active,
            "target_max_score": target.get("max_score"),
            "target_active_frame_count": target.get("active_frame_count"),
            "target_confidence": loc.get("confidence") if loc else None,
            "target_azimuth_dcase_deg": loc.get("azimuth_dcase_deg") if loc else None,
            "target_distance_m": loc.get("distance_m") if loc else None,
        })

    summary = {
        "inputs": {
            "manifest": str(Path(args.manifest).resolve()),
            "samples_selected": len(rows),
            "conditions": sorted({r.get("condition") for r in rows}),
            "class_names": sorted({r.get("class_name") for r in rows}),
            "angles": sorted({float(r["gt_azimuth_project_deg"]) for r in rows}),
            "variants": variants,
            "device": str(device),
            "checkpoint": str(checkpoint),
            "checkpoint_missing_keys": list(getattr(load_result, "missing_keys", [])),
            "checkpoint_unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        },
        "metrics": {
            variant: {
                "sign_minus": metric_block(records, variant, -1),
                "sign_plus": metric_block(records, variant, 1),
            }
            for variant in variants
        },
        "mirror": {
            "sign_minus": mirror_summary(records, -1),
            "sign_plus": mirror_summary(records, 1),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "channel_sign_audit_samples.csv", records)
    (output_root / "channel_sign_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["mirror"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
