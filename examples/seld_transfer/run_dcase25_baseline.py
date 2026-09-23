"""Candidate A: DCASE 2025 official audio-only baseline on an existing manifest.

This adapter intentionally lives outside the third-party repository.  It loads
the official config/checkpoint, extracts the repository's log-Mel features for
arbitrary 24 kHz stereo WAVs, and writes both sanity metadata and predictions.
No training or dataset download is performed.
"""

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


NB_CLASSES = 13
TRACKS = 3
FEATURE_FRAMES = 251
FEATURE_MELS = 64
MODEL_FRAMES = 50


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_stats(array):
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "min": float(np.nanmin(array)) if array.size else None,
        "max": float(np.nanmax(array)) if array.size else None,
    }


def load_manifest(manifest_path, repo_root):
    rows = []
    with Path(manifest_path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wav = Path(row["wav_path"])
            if not wav.is_absolute():
                wav = repo_root / wav
            row["_wav_path"] = str(wav)
            rows.append(row)
    return rows


def load_audio_features(wav_path, feature_utils, params):
    audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    # soundfile is samples x channels; the official helper expects channels x samples.
    audio = audio.T
    if sr != params["sampling_rate"]:
        raise ValueError("expected 24 kHz model input, got {} Hz: {}".format(sr, wav_path))
    hop_length = int(sr * params["hop_length_s"])
    win_length = 2 * hop_length
    n_fft = 2 ** (win_length - 1).bit_length()
    features = feature_utils.extract_log_mel_spectrogram(
        audio, sr, n_fft, hop_length, win_length, params["nb_mels"]
    )
    features = np.asarray(features, dtype=np.float32)
    # The upstream helper has changed its multi-channel transpose behavior across
    # librosa releases; normalize equivalent layouts without changing values.
    if features.shape == (audio.shape[0], FEATURE_FRAMES, FEATURE_MELS):
        normalized = features
    elif features.shape == (FEATURE_FRAMES, FEATURE_MELS, audio.shape[0]):
        normalized = np.transpose(features, (2, 0, 1))
    elif features.shape == (audio.shape[0], FEATURE_MELS, FEATURE_FRAMES):
        normalized = np.transpose(features, (0, 2, 1))
    else:
        per_channel = []
        for channel in audio:
            channel_feature = feature_utils.extract_log_mel_spectrogram(
                channel, sr, n_fft, hop_length, win_length, params["nb_mels"]
            )
            channel_feature = np.asarray(channel_feature, dtype=np.float32)
            if channel_feature.shape != (FEATURE_FRAMES, FEATURE_MELS):
                raise ValueError("unexpected single-channel feature shape: {}".format(channel_feature.shape))
            per_channel.append(channel_feature)
        normalized = np.stack(per_channel, axis=0)
    if normalized.shape != (2, FEATURE_FRAMES, FEATURE_MELS):
        raise ValueError("unexpected feature shape {} for {}".format(normalized.shape, wav_path))
    return normalized


def circular_mean(degrees):
    if not degrees:
        return None
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


def decode_sample(output, params, target_class_id):
    """Decode one (50,117) audio Multi-ACCDOA output without postprocessing guesses."""
    output = np.asarray(output, dtype=np.float32).reshape(MODEL_FRAMES, TRACKS, 3, NB_CLASSES)
    tracks = []
    target_candidates = []
    strongest_candidates = []
    class_records = [[] for _ in range(NB_CLASSES)]
    for frame_idx in range(MODEL_FRAMES):
        for track_idx in range(TRACKS):
            x = output[frame_idx, track_idx, 0]
            y = output[frame_idx, track_idx, 1]
            distance = output[frame_idx, track_idx, 2]
            norm = np.sqrt(x * x + y * y)
            active = norm > 0.5
            for class_idx in range(NB_CLASSES):
                if np.isfinite(x[class_idx]) and np.isfinite(y[class_idx]):
                    angle = math.degrees(math.atan2(float(y[class_idx]), float(x[class_idx])))
                    record = {
                        "frame": frame_idx,
                        "track": track_idx,
                        "class_id": class_idx,
                        "activity_norm": float(norm[class_idx]),
                        "azimuth_deg": angle,
                        "distance": float(distance[class_idx]),
                    }
                    if active[class_idx]:
                        class_records[class_idx].append(record)
                        tracks.append(record)
                    if class_idx == target_class_id:
                        target_candidates.append(record)
                    strongest_candidates.append(record)
    counts = [len(records) for records in class_records]
    predicted_class = int(np.argmax(counts)) if max(counts, default=0) else None
    selected = class_records[predicted_class] if predicted_class is not None else []
    predicted_azimuth = circular_mean([record["azimuth_deg"] for record in selected])
    target_peak = max(target_candidates, key=lambda item: item["activity_norm"], default=None)
    strongest_peak = max(strongest_candidates, key=lambda item: item["activity_norm"], default=None)
    return {
        "active": bool(selected),
        "predicted_class_id": predicted_class,
        "predicted_azimuth_deg": predicted_azimuth,
        "active_frame_count": len({record["frame"] for record in selected}),
        "class_active_frame_counts": counts,
        "target_class_id": target_class_id,
        "target_latent_peak": target_peak,
        "strongest_latent_peak": strongest_peak,
        "raw_active_predictions": tracks,
        "selected_predictions": selected,
    }


def infer(model, features, device):
    tensor = torch.from_numpy(np.stack(features, axis=0)).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(tensor)
    return output.detach().cpu().numpy(), tensor, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--sanity-output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    candidate_root = Path(args.candidate_root).resolve()
    sys.path.insert(0, str(candidate_root))
    from model import SELDModel
    import utils as feature_utils

    with Path(args.config).open("rb") as handle:
        params = pickle.load(handle)
    params = dict(params)
    params["modality"] = "audio"
    params["multiACCDOA"] = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["seld_model"] if isinstance(checkpoint, dict) and "seld_model" in checkpoint else checkpoint
    model = SELDModel(params)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError("checkpoint mismatch: missing={}, unexpected={}".format(missing_keys, unexpected_keys))
    model.eval()

    rows = load_manifest(args.manifest, repo_root)
    features = [load_audio_features(row["_wav_path"], feature_utils, params) for row in rows]
    feature_array = np.stack(features, axis=0)
    if not np.isfinite(feature_array).all():
        raise RuntimeError("input feature contains NaN/Inf")

    sanity = {
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "size_bytes": Path(args.checkpoint).stat().st_size,
            "sha256": sha256_file(args.checkpoint),
            "top_level_keys": sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else None,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "config": {"sampling_rate": params["sampling_rate"], "nb_mels": params["nb_mels"], "label_sequence_length": params["label_sequence_length"], "nb_classes": params["nb_classes"]},
        "input": finite_stats(feature_array),
        "torch": {"version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())},
    }

    cpu_start = time.perf_counter()
    cpu_output, _, _ = infer(model, features, torch.device("cpu"))
    cpu_elapsed = time.perf_counter() - cpu_start
    sanity["cpu"] = {"elapsed_seconds": cpu_elapsed, "samples": len(rows), "output": finite_stats(cpu_output)}
    if not np.isfinite(cpu_output).all():
        raise RuntimeError("CPU output contains NaN/Inf")

    selected_output = cpu_output
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("requested GPU sanity but CUDA is unavailable")
        gpu_device = torch.device(args.device)
        model = model.to(gpu_device)
        torch.cuda.reset_peak_memory_stats(gpu_device)
        torch.cuda.synchronize(gpu_device)
        gpu_start = time.perf_counter()
        gpu_output, _, _ = infer(model, features, gpu_device)
        torch.cuda.synchronize(gpu_device)
        gpu_elapsed = time.perf_counter() - gpu_start
        sanity["gpu"] = {
            "device": str(gpu_device),
            "elapsed_seconds": gpu_elapsed,
            "samples": len(rows),
            "output": finite_stats(gpu_output),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(gpu_device)),
        }
        if not np.isfinite(gpu_output).all():
            raise RuntimeError("GPU output contains NaN/Inf")
        sanity["cpu_gpu_max_abs_diff"] = float(np.max(np.abs(cpu_output - gpu_output)))
        selected_output = gpu_output

    prediction_path = Path(args.predictions)
    raw_path = Path(args.raw_predictions)
    sanity_path = Path(args.sanity_output)
    for path in (prediction_path, raw_path, sanity_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as predictions, raw_path.open("w", encoding="utf-8") as raw_predictions:
        for row, output in zip(rows, selected_output):
            target_class_id = int(row["class_id"])
            decoded = decode_sample(output, params, target_class_id)
            prediction = {key: value for key, value in decoded.items() if key not in ("raw_active_predictions", "selected_predictions", "target_latent_peak", "strongest_latent_peak")}
            prediction["sample_id"] = row["sample_id"]
            prediction["target_latent_peak"] = decoded["target_latent_peak"]
            prediction["strongest_latent_peak"] = decoded["strongest_latent_peak"]
            predictions.write(json.dumps(prediction, sort_keys=True) + "\n")
            raw_predictions.write(json.dumps({"sample_id": row["sample_id"], "output_shape": list(output.shape), **decoded}, sort_keys=True) + "\n")
    sanity_path.write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    print(json.dumps(sanity, indent=2))


if __name__ == "__main__":
    main()
