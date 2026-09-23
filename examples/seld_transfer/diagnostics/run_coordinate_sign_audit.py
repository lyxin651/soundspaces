"""Run the compact Candidate A coordinate/sign audit without changing old results."""

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def circular_error(pred, gt):
    return abs((float(pred) - float(gt) + 180.0) % 360.0 - 180.0)


def circular_mean(values):
    if not values:
        return None
    values = np.deg2rad(np.asarray(values, dtype=np.float64))
    return float(np.rad2deg(np.arctan2(np.sin(values).mean(), np.cos(values).mean())))


def independent_local_azimuth(source, listener, yaw_deg):
    """World-to-agent inverse yaw, written independently of upstream gt_azimuth."""
    delta = np.asarray(source, dtype=np.float64) - np.asarray(listener, dtype=np.float64)
    yaw = math.radians(float(yaw_deg))
    c, s = math.cos(yaw), math.sin(yaw)
    local_x = c * delta[0] - s * delta[2]
    local_y = delta[1]
    local_z = s * delta[0] + c * delta[2]
    angle = math.degrees(math.atan2(local_x, -local_z))
    return [float(local_x), float(local_y), float(local_z)], float(angle)


def load_rows(path, repo_root):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wav = Path(row["wav_path"])
            row["_wav_path"] = str(wav if wav.is_absolute() else repo_root / wav)
            rows.append(row)
    return rows


def d1_audit(rows):
    selected = []
    for target_angle in (-60.0, 0.0, 60.0):
        candidates = [row for row in rows if float(row["gt_azimuth_full_deg"]) == target_angle]
        row = next((item for item in candidates if item["class_name"] == "Male speech"), candidates[0])
        source = json.loads(row["source_world_xyz"])
        listener = json.loads(row["listener_world_xyz"])
        local, independent = independent_local_azimuth(source, listener, row["agent_yaw_deg"])
        old = float(row["gt_azimuth_folded_deg"])
        selected.append({
            "sample_id": row["sample_id"],
            "geometry": "left" if target_angle < 0 else "front" if target_angle == 0 else "right",
            "source_world_xyz": source,
            "listener_world_xyz": listener,
            "agent_yaw_deg": float(row["agent_yaw_deg"]),
            "d_world": (np.asarray(source) - np.asarray(listener)).tolist(),
            "d_local_independent": local,
            "old_gt_azimuth_deg": old,
            "independent_gt_azimuth_deg": independent,
            "difference_deg": circular_error(independent, old),
        })
    passed = all(item["difference_deg"] < 1e-5 for item in selected)
    return {
        "status": "PASS" if passed else "FAIL",
        "axes": {"forward": "-Z", "right": "+X", "up": "+Y"},
        "positive_azimuth": "right (positive local +X)",
        "azimuth_definition": "atan2(local_right, -local_forward)",
        "samples": selected,
    }


def normalized_lag(left, right, sample_rate, max_ms=5.0):
    """Return lag for score(lag)=sum(L[n] * R[n+lag]). Positive means R is delayed."""
    max_lag = int(round(sample_rate * max_ms / 1000.0))
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = left[: len(left) - lag], right[lag:]
        else:
            x, y = left[-lag:], right[: len(right) + lag]
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        scores.append(float(np.dot(x, y) / denominator) if denominator else 0.0)
    best = int(np.argmax(scores)) - max_lag
    return best, float(best / sample_rate), float(max(scores))


def d2_audit(rows):
    records = []
    for angle in (-60.0, -30.0, 0.0, 30.0, 60.0):
        row = next(item for item in rows if item["class_name"] == "Male speech" and float(item["gt_azimuth_full_deg"]) == angle)
        audio, sample_rate = sf.read(row["_wav_path"], always_2d=True, dtype="float32")
        left, right = audio[:, 0], audio[:, 1]
        rms_l = float(np.sqrt(np.mean(left * left)))
        rms_r = float(np.sqrt(np.mean(right * right)))
        ild = float(20.0 * math.log10(rms_r / rms_l)) if rms_l and rms_r else None
        lag, lag_s, corr = normalized_lag(left, right, sample_rate)
        records.append({
            "sample_id": row["sample_id"],
            "gt_azimuth_deg": angle,
            "sample_rate": int(sample_rate),
            "rms_l": rms_l,
            "rms_r": rms_r,
            "ild_db_r_over_l": ild,
            "itd_lag_samples": lag,
            "itd_lag_seconds": lag_s,
            "itd_lag_definition": "positive means R is delayed and L leads",
            "normalized_correlation": corr,
        })
    negative = [item for item in records if item["gt_azimuth_deg"] < 0]
    positive = [item for item in records if item["gt_azimuth_deg"] > 0]
    center = next(item for item in records if item["gt_azimuth_deg"] == 0)
    ild_flip = all(item["ild_db_r_over_l"] < 0 for item in negative) and all(item["ild_db_r_over_l"] > 0 for item in positive)
    itd_flip = all(item["itd_lag_samples"] > 0 for item in negative) and all(item["itd_lag_samples"] < 0 for item in positive)
    center_near = abs(center["ild_db_r_over_l"]) < 3.0
    return {
        "status": "PASS" if ild_flip and center_near else "FAIL",
        "ild_sign_rule": "negative azimuth -> R/L ILD < 0; positive azimuth -> R/L ILD > 0",
        "itd_sign_rule": "diagnostic only; positive lag means L leads R",
        "ild_sign_flip": ild_flip,
        "itd_sign_flip": itd_flip,
        "center_ild_near_zero": center_near,
        "samples": records,
    }


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


def decode_target(raw, class_id):
    shaped = raw.reshape(50, 3, 3, 13)
    x, y = shaped[:, :, 0, class_id], shaped[:, :, 1, class_id]
    norm = np.sqrt(x * x + y * y)
    active = norm > 0.5
    frame_active = active.any(axis=1)
    predictions = []
    for frame in np.flatnonzero(frame_active):
        track = int(np.argmax(norm[frame]))
        predictions.append(float(np.degrees(np.arctan2(y[frame, track], x[frame, track]))))
    return {
        "max_norm": float(norm.max()),
        "active_frame_count": int(frame_active.sum()),
        "active_frame_ratio": float(frame_active.mean()),
        "predicted_azimuth_deg": circular_mean(predictions),
        "predicted_azimuth_samples": predictions,
    }


def d3_audit(rows, evaluation_path, model, params, feature_utils, load_audio_features, repo_root):
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    active_ids = {item["sample_id"] for item in evaluation["per_sample"] if item["active"]}
    selected = [
        row for row in rows
        if row["sample_id"] in active_ids
        and row["class_name"] in ("Male speech", "Music")
        and abs(float(row["gt_azimuth_full_deg"])) in (30.0, 60.0)
    ]
    selected = sorted(selected, key=lambda row: (row["class_name"], float(row["gt_azimuth_full_deg"])))
    if len(selected) < 8:
        raise RuntimeError("D3 selected fewer than 8 active samples")
    original_features = []
    swapped_features = []
    for row in selected:
        feature = load_audio_features(row["_wav_path"], feature_utils, params)
        original_features.append(feature)
        swapped_features.append(feature[[1, 0], :, :])
    with torch.inference_mode():
        original_raw = model(torch.from_numpy(np.stack(original_features))).cpu().numpy()
        swapped_raw = model(torch.from_numpy(np.stack(swapped_features))).cpu().numpy()
    records = []
    for index, row in enumerate(selected):
        original = decode_target(original_raw[index], int(row["class_id"]))
        swapped = decode_target(swapped_raw[index], int(row["class_id"]))
        records.append({
            "sample_id": row["sample_id"],
            "class_name": row["class_name"],
            "gt_azimuth_deg": float(row["gt_azimuth_full_deg"]),
            "original": original,
            "swapped_lr": swapped,
            "mirror_error_deg": circular_error(swapped["predicted_azimuth_deg"], -original["predicted_azimuth_deg"]) if swapped["predicted_azimuth_deg"] is not None and original["predicted_azimuth_deg"] is not None else None,
        })
    valid_mirror = [item["mirror_error_deg"] for item in records if item["mirror_error_deg"] is not None]
    stable = sum(error <= 20.0 for error in valid_mirror) >= max(1, int(math.ceil(0.75 * len(valid_mirror))))
    return {
        "status": "STABLE_MIRROR" if stable else "NO_STABLE_MIRROR",
        "n_samples": len(records),
        "mirror_tolerance_deg": 20.0,
        "mirror_pass_count": sum(error <= 20.0 for error in valid_mirror),
        "records": records,
    }


def metric(predictions):
    errors = [circular_error(item["pred"], item["gt"]) for item in predictions]
    valid_sign = [item for item in predictions if abs(item["gt"]) >= 1.0]
    lr = float(np.mean([math.copysign(1.0, item["pred"]) == math.copysign(1.0, item["gt"]) for item in valid_sign])) if valid_sign else None
    return {
        "n_predictions": len(predictions),
        "folded_mae_deg": float(np.mean(errors)) if errors else None,
        "median_ae_deg": float(np.median(errors)) if errors else None,
        "left_right_accuracy": lr,
    }


def d4_audit(evaluation_path):
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    t0, t1 = [], []
    for sample in evaluation["per_sample"]:
        for prediction in sample["active_predictions"]:
            base = {"pred": float(prediction["predicted_azimuth_deg"]), "gt": float(sample["gt_azimuth_folded_deg"])}
            t0.append(base)
            t1.append({"pred": -base["pred"], "gt": base["gt"]})
    return {
        "t0_original": metric(t0),
        "t1_global_prediction_sign_flip": metric(t1),
        "corrected_gt": None,
        "source_definition": "C4-B active_predictions only; no threshold or per-sample correction",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    rows = load_rows(args.manifest, repo_root)
    result = {"definition": "D1-D4 compact coordinate/sign audit; historical C4-B outputs are read-only."}
    result["d1"] = d1_audit(rows)
    result["d2"] = d2_audit(rows)
    model, params, feature_utils, load_audio_features = load_model(
        repo_root, Path(args.candidate_root).resolve(), args.config, args.checkpoint
    )
    result["d3"] = d3_audit(
        rows, args.evaluation, model, params, feature_utils, load_audio_features, repo_root
    )
    result["d4"] = d4_audit(args.evaluation)
    result["final_attribution"] = (
        "HRTF mismatch basically established"
        if result["d1"]["status"] == "PASS" and result["d2"]["status"] == "PASS" and result["d4"]["t1_global_prediction_sign_flip"]["left_right_accuracy"] < 0.8
        else "coordinate/sign issue requires combined interpretation"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coordinate_sign_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "d1": result["d1"]["status"],
        "d2": result["d2"]["status"],
        "d3": result["d3"]["status"],
        "d4": result["d4"],
        "final_attribution": result["final_attribution"],
    }, indent=2))


if __name__ == "__main__":
    main()
