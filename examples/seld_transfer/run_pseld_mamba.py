"""Run Candidate B PSELD_Mamba on SoundSpaces WAV manifests.

This adapter stays outside the third-party PSELD_Mamba repository.  It reads a
SoundSpaces-style CSV/JSONL manifest, converts each WAV to the model's 24 kHz
stereo 5-second PFOA input, runs a pretrained checkpoint, and writes project
coordinate predictions.  It does not require DCASE labels or the upstream
DataGenerator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


NB_CLASSES = 13
TRACKS = 3
TRACK_VALUES = 3
DEFAULT_ACTIVITY_THRESHOLD = 0.5


CLASS_NAMES = {
    0: "Female speech",
    1: "Male speech",
    2: "Clapping",
    3: "Telephone",
    4: "Laughter",
    5: "Domestic sounds",
    6: "Walk/footsteps",
    7: "Door",
    8: "Music",
    9: "Musical instrument",
    10: "Water tap",
    11: "Bell",
    12: "Knock",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_stats(array: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "min": float(np.nanmin(array)) if array.size else None,
        "max": float(np.nanmax(array)) if array.size else None,
        "mean": float(np.nanmean(array)) if array.size else None,
    }


def circular_error(pred: float, target: float) -> float:
    return abs((float(pred) - float(target) + 180.0) % 360.0 - 180.0)


def circular_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    items = [float(value) for value in values]
    return min(items, key=lambda candidate: sum(circular_error(candidate, other) for other in items))


def dcase_to_project_azimuth(theta_dcase: float, sign: int) -> float:
    """Convert DCASE azimuth to the project convention.

    Prior Candidate A audits found DCASE stereo output uses the opposite positive
    direction to the project convention, so the default is sign=-1:
    theta_project = -theta_dcase.
    """
    return float(sign) * float(theta_dcase)


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


def resolve_path(value: str, repo_root: Path, dataset_root: Optional[Path]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = []
    if dataset_root is not None:
        candidates.append(dataset_root / path)
    candidates.append(repo_root / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def row_sample_id(row: Mapping[str, Any], index: int, sample_id_field: str) -> str:
    if sample_id_field in row and str(row[sample_id_field]):
        return str(row[sample_id_field])
    if row.get("episode_id") is not None and row.get("viewpoint_id") is not None:
        return "{}__{}".format(row["episode_id"], row["viewpoint_id"])
    if row.get("viewpoint_id") is not None:
        return str(row["viewpoint_id"])
    return "sample_{:06d}".format(index)


def row_wav_path(row: Mapping[str, Any], wav_path_field: str) -> str:
    for field in (wav_path_field, "wav_path", "audio_path"):
        value = row.get(field)
        if value:
            return str(value)
    raise KeyError("manifest row has no WAV/audio path field")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def load_soundspaces_wav(path: Path, target_sr: int, clip_seconds: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    waveform, source_sr = sf.read(str(path), always_2d=True, dtype="float32")
    if waveform.shape[1] == 1:
        waveform = np.repeat(waveform, 2, axis=1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    if not np.isfinite(waveform).all():
        raise ValueError("WAV contains NaN/Inf: {}".format(path))
    original_shape = list(waveform.shape)
    if int(source_sr) != int(target_sr):
        gcd = math.gcd(int(target_sr), int(source_sr))
        waveform = resample_poly(waveform, int(target_sr) // gcd, int(source_sr) // gcd, axis=0)
        waveform = waveform.astype(np.float32, copy=False)
    target_samples = int(round(float(clip_seconds) * int(target_sr)))
    if waveform.shape[0] >= target_samples:
        waveform = waveform[:target_samples]
        pad_samples = 0
    else:
        pad_samples = target_samples - waveform.shape[0]
        waveform = np.pad(waveform, ((0, pad_samples), (0, 0)), mode="constant")
    before_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    waveform = normalize_audio(waveform.T)
    return waveform, {
        "path": str(path),
        "source_sample_rate": int(source_sr),
        "source_shape_samples_channels": original_shape,
        "model_sample_rate": int(target_sr),
        "model_shape_channels_samples": list(waveform.shape),
        "clip_seconds": float(clip_seconds),
        "pad_samples": int(pad_samples),
        "pre_normalization_peak": before_peak,
    }


def extract_pfoa_feature(audio_channels_first: np.ndarray, params: Mapping[str, Any], pseld_utils: Any) -> np.ndarray:
    sr = int(params["sampling_rate"])
    hop_length = int(sr * float(params["hop_length_s"]))
    win_length = 2 * hop_length
    n_fft = 2 ** (win_length - 1).bit_length()
    feature = pseld_utils.extract_PFOA(
        audio_channels_first,
        sr,
        n_fft,
        hop_length,
        win_length,
        int(params["nb_mels"]),
    )
    feature = feature.detach().cpu().numpy().astype(np.float32, copy=False)
    if feature.ndim != 3 or feature.shape[0] != 7 or feature.shape[2] != int(params["nb_mels"]):
        raise ValueError("unexpected PFOA feature shape: {}".format(feature.shape))
    if not np.isfinite(feature).all():
        raise ValueError("PFOA feature contains NaN/Inf")
    return feature


def load_pseld_model(candidate_root: Path, exp_name: str, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(candidate_root))
    exp_module = importlib.import_module("experiments.{}".format(exp_name))
    params = dict(exp_module.get_params())
    params["modality"] = "audio"
    params["multiACCDOA"] = True
    params["checkpoints_dir"] = str(checkpoint_path.parent)

    from model import ConvConformer_Multi, HTSAT_multi  # type: ignore
    import utils as pseld_utils  # type: ignore

    if params.get("model_type") == "HTSAT":
        model = HTSAT_multi(params=params).to(device)
    elif params.get("model_type") == "CNN14_Conformer":
        model = ConvConformer_Multi(params=params).to(device)
    else:
        raise ValueError("unsupported PSELD model_type: {}".format(params.get("model_type")))

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint["seld_model"] if isinstance(checkpoint, dict) and "seld_model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict)
    model.eval()
    return model, params, pseld_utils, checkpoint, load_result


def infer_batches(model: torch.nn.Module, features: Sequence[np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = np.stack(features[start:start + batch_size], axis=0)
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            try:
                logits = model(tensor, None)
            except TypeError:
                logits = model(tensor)
        outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 50, 117), dtype=np.float32)


def active_record(frame: int, track: int, class_id: int, x: float, y: float, distance_m: float, score: float, sign: int) -> Dict[str, Any]:
    theta_dcase = math.degrees(math.atan2(float(y), float(x)))
    return {
        "frame": int(frame),
        "track": int(track),
        "class_id": int(class_id),
        "class_name": CLASS_NAMES.get(int(class_id)),
        "activity_score": float(score),
        "azimuth_dcase_deg": float(theta_dcase),
        "azimuth_project_deg": dcase_to_project_azimuth(theta_dcase, sign),
        "distance_m": max(float(distance_m), 0.0),
    }


def decode_output(output: np.ndarray, target_class_id: Optional[int], activity_threshold: float, project_sign: int) -> Dict[str, Any]:
    shaped = np.asarray(output, dtype=np.float32).reshape(-1, TRACKS, TRACK_VALUES, NB_CLASSES)
    x = shaped[:, :, 0, :]
    y = shaped[:, :, 1, :]
    distance = shaped[:, :, 2, :]
    score = np.sqrt(x * x + y * y)
    active = score > float(activity_threshold)

    class_scores = {str(class_id): float(np.max(score[:, :, class_id])) for class_id in range(NB_CLASSES)}
    class_active_frame_counts = []
    localizations = []
    all_active_records = []
    for class_id in range(NB_CLASSES):
        frame_has_class = active[:, :, class_id].any(axis=1)
        frame_count = int(frame_has_class.sum())
        class_active_frame_counts.append(frame_count)
        records = []
        for frame, track in np.argwhere(active[:, :, class_id]):
            record = active_record(
                int(frame),
                int(track),
                class_id,
                float(x[frame, track, class_id]),
                float(y[frame, track, class_id]),
                float(distance[frame, track, class_id]),
                float(score[frame, track, class_id]),
                project_sign,
            )
            records.append(record)
            all_active_records.append(record)
        if records:
            localizations.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id),
                "active_frame_count": frame_count,
                "confidence": max(record["activity_score"] for record in records),
                "azimuth_dcase_deg": circular_median([record["azimuth_dcase_deg"] for record in records]),
                "azimuth_project_deg": circular_median([record["azimuth_project_deg"] for record in records]),
                "distance_m": float(np.median([record["distance_m"] for record in records])),
            })

    predicted_class_id = None
    if max(class_active_frame_counts, default=0) > 0:
        predicted_class_id = int(np.argmax(class_active_frame_counts))
    selected = next((item for item in localizations if item["class_id"] == predicted_class_id), None)

    target_summary = None
    if target_class_id is not None:
        target_summary = {
            "class_id": int(target_class_id),
            "class_name": CLASS_NAMES.get(int(target_class_id)),
            "max_score": class_scores.get(str(int(target_class_id))),
            "active_frame_count": class_active_frame_counts[int(target_class_id)],
            "localization": next((item for item in localizations if item["class_id"] == int(target_class_id)), None),
        }

    return {
        "active": selected is not None,
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": CLASS_NAMES.get(predicted_class_id) if predicted_class_id is not None else None,
        "confidence": selected["confidence"] if selected else None,
        "predicted_azimuth_dcase_deg": selected["azimuth_dcase_deg"] if selected else None,
        "predicted_azimuth_deg": selected["azimuth_project_deg"] if selected else None,
        "distance_m": selected["distance_m"] if selected else None,
        "active_frame_count": selected["active_frame_count"] if selected else 0,
        "class_scores": class_scores,
        "class_active_frame_counts": class_active_frame_counts,
        "localizations": localizations,
        "target": target_summary,
        "raw_active_predictions": all_active_records,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_dcase_csvs(output_dir: Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for prediction in predictions:
        path = output_dir / (str(prediction["sample_id"]) + ".csv")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "class", "source", "azimuth", "distance", "onscreen"])
            for record in prediction["raw_active_predictions"]:
                writer.writerow([
                    int(record["frame"]),
                    int(record["class_id"]),
                    int(record["track"]),
                    int(round(float(record["azimuth_dcase_deg"]))),
                    int(round(float(record["distance_m"]) * 100.0)),
                    0,
                ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSELD_Mamba on SoundSpaces WAV manifests")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset-root", default="", help="Resolve relative active_audition audio_path values")
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--exp", default="EXP_CNN14_BiMambaAC")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True, help="Project-coordinate JSONL output")
    parser.add_argument("--raw-predictions", default="", help="Optional verbose JSONL with active frame records")
    parser.add_argument("--raw-logits", default="", help="Optional .npy raw model output")
    parser.add_argument("--sanity-output", default="", help="Optional JSON sanity report")
    parser.add_argument("--dcase-output-dir", default="", help="Optional DCASE-convention CSV directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--activity-threshold", type=float, default=DEFAULT_ACTIVITY_THRESHOLD)
    parser.add_argument("--project-azimuth-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--sample-id-field", default="sample_id")
    parser.add_argument("--wav-path-field", default="wav_path")
    parser.add_argument("--target-class-field", default="class_id")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    candidate_root = Path(args.candidate_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    device = torch.device(args.device)

    rows = load_manifest(manifest_path)
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError("manifest has no rows: {}".format(manifest_path))

    model, params, pseld_utils, checkpoint, load_result = load_pseld_model(candidate_root, args.exp, checkpoint_path, device)
    target_sr = int(params["sampling_rate"])

    features = []
    prepared = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        wav = resolve_path(row_wav_path(row, args.wav_path_field), repo_root, dataset_root)
        audio, audio_info = load_soundspaces_wav(wav, target_sr, args.clip_seconds)
        feature = extract_pfoa_feature(audio, params, pseld_utils)
        sample_id = row_sample_id(row, index, args.sample_id_field)
        target_class_id = None
        if args.target_class_field in row and str(row[args.target_class_field]) != "":
            target_class_id = int(row[args.target_class_field])
        prepared.append({
            "sample_id": sample_id,
            "manifest_index": index,
            "manifest_row": row,
            "wav": audio_info,
            "target_class_id": target_class_id,
            "feature_shape": list(feature.shape),
        })
        features.append(feature)

    logits = infer_batches(model, features, device, max(1, int(args.batch_size)))
    elapsed = time.perf_counter() - started
    if not np.isfinite(logits).all():
        raise RuntimeError("model output contains NaN/Inf")

    predictions = []
    raw_predictions = []
    for item, output in zip(prepared, logits):
        decoded = decode_output(output, item["target_class_id"], args.activity_threshold, args.project_azimuth_sign)
        prediction = {
            "sample_id": item["sample_id"],
            "active": decoded["active"],
            "predicted_class_id": decoded["predicted_class_id"],
            "predicted_class_name": decoded["predicted_class_name"],
            "confidence": decoded["confidence"],
            "predicted_azimuth_dcase_deg": decoded["predicted_azimuth_dcase_deg"],
            "predicted_azimuth_deg": decoded["predicted_azimuth_deg"],
            "distance_m": decoded["distance_m"],
            "active_frame_count": decoded["active_frame_count"],
            "class_scores": decoded["class_scores"],
            "class_active_frame_counts": decoded["class_active_frame_counts"],
            "localizations": decoded["localizations"],
            "target": decoded["target"],
            "metadata": {
                "model": "DCASE2025_TASK3_Stereo_PSELD_Mamba",
                "exp": args.exp,
                "checkpoint": str(checkpoint_path),
                "activity_threshold": args.activity_threshold,
                "project_azimuth_sign": args.project_azimuth_sign,
                "wav_path": item["wav"]["path"],
            },
        }
        predictions.append(prediction)
        raw_predictions.append({**prediction, "raw_active_predictions": decoded["raw_active_predictions"]})

    write_jsonl(Path(args.predictions), predictions)
    if args.raw_predictions:
        write_jsonl(Path(args.raw_predictions), raw_predictions)
    if args.raw_logits:
        raw_logits_path = Path(args.raw_logits)
        raw_logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_logits_path, logits)
    if args.dcase_output_dir:
        write_dcase_csvs(Path(args.dcase_output_dir), raw_predictions)

    sanity = {
        "manifest": str(manifest_path),
        "samples": len(rows),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "top_level_keys": sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else None,
            "missing_keys": list(getattr(load_result, "missing_keys", [])),
            "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        },
        "params": {
            "exp": args.exp,
            "model_type": params.get("model_type"),
            "decoder_type": params.get("decoder_type"),
            "feature_type": params.get("feature_type"),
            "sampling_rate": params.get("sampling_rate"),
            "nb_mels": params.get("nb_mels"),
            "label_sequence_length": params.get("label_sequence_length"),
            "nb_classes": params.get("nb_classes"),
        },
        "preprocessing": {
            "clip_seconds": args.clip_seconds,
            "activity_threshold": args.activity_threshold,
            "project_azimuth_rule": "theta_project = {} * theta_dcase".format(args.project_azimuth_sign),
            "audio_normalization": "global peak normalization after resample/trim/pad, matching PSELD_Mamba feature extraction",
        },
        "input_feature": finite_stats(np.stack(features, axis=0)),
        "output_logits": finite_stats(logits),
        "first_samples": prepared[:3],
    }
    if args.sanity_output:
        sanity_path = Path(args.sanity_output)
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        sanity_path.write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(sanity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
