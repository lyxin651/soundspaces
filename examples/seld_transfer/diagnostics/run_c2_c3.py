"""Run C2 original-domain controls and C3 semantic content controls."""

import argparse
import csv
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def feature(path, feature_utils, params):
    from run_dcase25_baseline import load_audio_features
    return load_audio_features(str(path), feature_utils, params)


def load_model(candidate_root, config_path, checkpoint_path):
    sys.path.insert(0, str(candidate_root))
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
    return model, feature_utils, params


def decode(raw, nb_classes):
    values = raw.reshape(raw.shape[0], raw.shape[1], 3, 3, nb_classes)
    output = []
    for index in range(raw.shape[0]):
        x = values[index, :, :, 0, :]
        y = values[index, :, :, 1, :]
        norm = np.sqrt(x * x + y * y)
        active = norm > 0.5
        class_max = norm.max(axis=(0, 1))
        class_active_ratio = active.mean(axis=(0, 1))
        output.append({
            "class_max_norm": class_max.tolist(),
            "class_active_frame_ratio": class_active_ratio.tolist(),
            "top_max_class": int(np.argmax(class_max)),
            "top_max_norm": float(np.max(class_max)),
            "top_active_class": int(np.argmax(class_active_ratio)),
            "top_active_ratio": float(np.max(class_active_ratio)),
        })
    return output


def forward_paths(paths, model, feature_utils, params):
    features = np.stack([feature(path, feature_utils, params) for path in paths], axis=0)
    with torch.inference_mode():
        raw = model(torch.from_numpy(features)).cpu().numpy()
    return raw


def read_metadata(metadata_zip, wav_path):
    stem = Path(wav_path).stem + ".csv"
    with zipfile.ZipFile(metadata_zip) as archive:
        matches = [name for name in archive.namelist() if name.endswith("/" + stem)]
        if len(matches) != 1:
            raise RuntimeError("metadata match count for {}: {}".format(stem, matches))
        rows = list(csv.DictReader(io.StringIO(archive.read(matches[0]).decode("utf-8"))))
    classes = sorted(set(int(row["class"]) for row in rows))
    azimuths = [float(row["azimuth"]) for row in rows]
    return {"metadata_member": matches[0], "classes": classes, "azimuth_min": min(azimuths), "azimuth_max": max(azimuths), "azimuth_median": float(np.median(azimuths)), "frames": len(set(row["frame"] for row in rows))}


def make_simple_content(input_path, output_path, target_sr=24000, seconds=5):
    audio, _ = librosa.load(str(input_path), sr=target_sr, mono=True)
    target_samples = target_sr * seconds
    audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    stereo = np.stack([audio, audio], axis=1).astype(np.float32)
    sf.write(str(output_path), stereo, target_sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-zip", required=True)
    parser.add_argument("--c2-wav-dir", required=True)
    parser.add_argument("--c3-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root / "examples" / "seld_transfer"))
    model, feature_utils, params = load_model(Path(args.candidate_root), args.config, args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c2_paths = sorted(Path(args.c2_wav_dir).glob("*.wav"))
    c2_raw = forward_paths(c2_paths, model, feature_utils, params)
    np.save(output_dir / "c2_raw_output.npy", c2_raw)
    c2_decoded = decode(c2_raw, params["nb_classes"])
    c2_rows = []
    for path, prediction in zip(c2_paths, c2_decoded):
        metadata = read_metadata(args.metadata_zip, path)
        target_class = metadata["classes"][0] if len(metadata["classes"]) == 1 else None
        c2_rows.append({"wav": str(path), "gt_class": target_class, "gt_classes": metadata["classes"], "gt_azimuth_deg": metadata["azimuth_median"], "metadata": metadata, "prediction": prediction, "target_max_norm": None if target_class is None else prediction["class_max_norm"][target_class], "target_active_ratio": None if target_class is None else prediction["class_active_frame_ratio"][target_class]})

    c3_dir = Path(args.c3_output_dir)
    c3_dir.mkdir(parents=True, exist_ok=True)
    content_sources = [
        ("telephone", repo_root / "data" / "sounds" / "telephone.wav", 3),
        ("music_singing", repo_root / "res" / "singing.wav", 8),
        ("speech_proxy_person_8", c3_dir / "person_8.wav", None),
        ("speech_proxy_person_9", c3_dir / "person_9.wav", None),
    ]
    simple_paths = []
    content_rows = []
    for name, source, target_class in content_sources:
        output = c3_dir / (name + "_lr24k_5s.wav")
        make_simple_content(source, output)
        simple_paths.append(output)
        content_rows.append({"name": name, "source": str(source), "target_class": target_class, "simple_stereo_wav": str(output)})
    c3_raw = forward_paths(simple_paths, model, feature_utils, params)
    np.save(output_dir / "c3_raw_output.npy", c3_raw)
    c3_decoded = decode(c3_raw, params["nb_classes"])
    for row, prediction in zip(content_rows, c3_decoded):
        row["prediction"] = prediction
        row["target_max_norm"] = None if row["target_class"] is None else prediction["class_max_norm"][row["target_class"]]
        row["target_active_ratio"] = None if row["target_class"] is None else prediction["class_active_frame_ratio"][row["target_class"]]
    result = {
        "pipeline": "same Candidate A wrapper feature extraction and model forward as C0/C1",
        "c2": {"samples": c2_rows, "raw_shape": list(c2_raw.shape)},
        "c3": {"samples": content_rows, "raw_shape": list(c3_raw.shape), "class_names": {"0": "Female speech", "1": "Male speech", "2": "Clapping", "3": "Telephone", "4": "Laughter", "5": "Domestic sounds", "6": "Walk, footsteps", "7": "Door", "8": "Music", "9": "Musical instrument", "10": "Water tap", "11": "Bell", "12": "Knock"}},
    }
    (output_dir / "c2_c3_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
