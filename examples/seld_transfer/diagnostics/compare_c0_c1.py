"""C0/C1 audit for Candidate A without changing upstream artifacts."""

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def stats(array):
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def compare_arrays(left, right):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "max_abs_diff": float(np.max(np.abs(delta))),
        "mean_abs_diff": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "allclose_atol_1e-6_rtol_1e-5": bool(np.allclose(left, right, atol=1e-6, rtol=1e-5)),
    }


def load_rows(manifest, repo_root):
    rows = []
    with Path(manifest).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["wav_path"])
            if not path.is_absolute():
                path = repo_root / path
            row["_wav_path"] = str(path)
            rows.append(row)
    return rows


def official_features(path, official_utils, params):
    audio, sr = official_utils.load_audio(str(path), params["sampling_rate"])
    hop = int(sr * params["hop_length_s"])
    win = 2 * hop
    n_fft = 2 ** (win - 1).bit_length()
    feature = official_utils.extract_log_mel_spectrogram(audio, sr, n_fft, hop, win, params["nb_mels"])
    return np.asarray(feature, dtype=np.float32), {"sample_rate": sr, "audio_shape": list(np.asarray(audio).shape)}


def official_track_arrays(raw, official_utils, params):
    # Copy because the official helper clamps negative distance values in-place.
    decoded = official_utils.get_multiaccdoa_labels(
        torch.from_numpy(np.asarray(raw, dtype=np.float32).copy()), params["nb_classes"], "audio"
    )
    tracks = []
    for offset in (0, 5, 10):
        sed = decoded[offset].cpu().numpy()
        doa = decoded[offset + 2].cpu().numpy()
        dist = decoded[offset + 3].cpu().numpy()
        x = doa[:, :, : params["nb_classes"]]
        y = doa[:, :, params["nb_classes"] :]
        norm = np.sqrt(x * x + y * y)
        tracks.append({"x": x, "y": y, "distance": dist, "norm": norm, "active": sed.astype(bool)})
    return tracks


def wrapper_track_arrays(raw, params):
    reshaped = np.asarray(raw, dtype=np.float32).reshape(raw.shape[0], raw.shape[1], 3, 3, params["nb_classes"])
    tracks = []
    for track in range(3):
        x = reshaped[:, :, track, 0, :]
        y = reshaped[:, :, track, 1, :]
        distance = reshaped[:, :, track, 2, :]
        norm = np.sqrt(x * x + y * y)
        tracks.append({"x": x, "y": y, "distance": distance, "norm": norm, "active": norm > 0.5})
    return tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    candidate_root = Path(args.candidate_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_root / "examples" / "seld_transfer"))
    sys.path.insert(0, str(candidate_root))
    from run_dcase25_baseline import load_audio_features
    from model import SELDModel
    import utils as official_utils

    with Path(args.config).open("rb") as handle:
        params = dict(pickle.load(handle))
    params["modality"] = "audio"
    params["multiACCDOA"] = True
    rows = load_rows(args.manifest, repo_root)

    selected = []
    for target in (-60.0, 0.0, 60.0):
        selected.append(min(rows, key=lambda row: abs(float(row["gt_azimuth_folded_deg"]) - target)))

    c0_rows = []
    wrapper_features = []
    for row in selected:
        official, loading = official_features(row["_wav_path"], official_utils, params)
        wrapper = load_audio_features(row["_wav_path"], official_utils, params)
        channel_diffs = compare_arrays(official, wrapper)
        channel_diffs["L_vs_L"] = compare_arrays(official[0], wrapper[0])
        channel_diffs["R_vs_R"] = compare_arrays(official[1], wrapper[1])
        channel_diffs["official_L_vs_wrapper_R"] = compare_arrays(official[0], wrapper[1])
        channel_diffs["official_R_vs_wrapper_L"] = compare_arrays(official[1], wrapper[0])
        c0_rows.append({
            "sample_id": row["sample_id"],
            "gt_folded_azimuth_deg": float(row["gt_azimuth_folded_deg"]),
            "wav": row["_wav_path"],
            "official": {**stats(official), **loading},
            "wrapper": stats(wrapper),
            "difference": channel_diffs,
        })
    for row in rows:
        wrapper_features.append(load_audio_features(row["_wav_path"], official_utils, params))
    feature_array = np.stack(wrapper_features, axis=0)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SELDModel(params)
    state = checkpoint["seld_model"]
    load_result = model.load_state_dict(state, strict=False)
    model.eval()
    with torch.inference_mode():
        raw = model(torch.from_numpy(feature_array)).cpu().numpy()
    np.save(output_dir / "raw_output.npy", raw)

    official_tracks = official_track_arrays(raw, official_utils, params)
    wrapper_tracks = wrapper_track_arrays(raw, params)
    field_diffs = []
    all_official_active = []
    all_wrapper_active = []
    telephone_norms = []
    per_sample = []
    for sample_index, row in enumerate(rows):
        sample_diffs = []
        for track_index in range(3):
            for field in ("x", "y", "distance", "norm", "active"):
                left = official_tracks[track_index][field][sample_index]
                right = wrapper_tracks[track_index][field][sample_index]
                if field == "active":
                    equal = bool(np.array_equal(left, right))
                    all_official_active.extend(np.asarray(left).ravel().tolist())
                    all_wrapper_active.extend(np.asarray(right).ravel().tolist())
                    diff = {"equal": equal, "mismatch_count": int(np.sum(left != right))}
                else:
                    diff = compare_arrays(left, right)
                sample_diffs.append({"track": track_index, "field": field, **diff})
        telephone_norms.extend(official_tracks[0]["norm"][sample_index, :, 3].tolist())
        telephone_norms.extend(official_tracks[1]["norm"][sample_index, :, 3].tolist())
        telephone_norms.extend(official_tracks[2]["norm"][sample_index, :, 3].tolist())
        per_sample.append({"sample_id": row["sample_id"], "differences": sample_diffs})
        field_diffs.extend(sample_diffs)

    telephone_norms = np.asarray(telephone_norms, dtype=np.float64)
    thresholds = {str(threshold): int(np.sum(telephone_norms > threshold)) for threshold in (0.1, 0.2, 0.3, 0.4, 0.5)}
    norm_stats = {
        "count": int(telephone_norms.size),
        "min": float(np.min(telephone_norms)),
        "max": float(np.max(telephone_norms)),
        "mean": float(np.mean(telephone_norms)),
        "median": float(np.median(telephone_norms)),
        "p90": float(np.percentile(telephone_norms, 90)),
        "p95": float(np.percentile(telephone_norms, 95)),
        "p99": float(np.percentile(telephone_norms, 99)),
        "counts_greater_than": thresholds,
    }
    c1 = {
        "raw_output_shape": list(raw.shape),
        "checkpoint_load": {"missing_keys": list(load_result.missing_keys), "unexpected_keys": list(load_result.unexpected_keys)},
        "class_mapping": {"Telephone": 3, "source": "upstream manifest label convention; official model has 13 zero-based output classes"},
        "telephone_activity_norm": norm_stats,
        "official_wrapper_active_equal": bool(np.array_equal(all_official_active, all_wrapper_active)),
        "official_wrapper_active_mismatch_count": int(np.sum(np.asarray(all_official_active) != np.asarray(all_wrapper_active))),
        "field_comparison_max_abs": max((item.get("max_abs_diff", 0.0) for item in field_diffs), default=0.0),
        "per_sample_field_comparison": per_sample,
    }
    result = {"c0": {"selected_samples": c0_rows}, "c1": c1}
    (output_dir / "c0_c1_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
