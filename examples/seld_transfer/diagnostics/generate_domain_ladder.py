"""Generate C4 SoundSpaces domain-ladder audio from known active events."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_binaural_transfer_set import (
    fold_azimuth,
    make_sim,
    pad_or_crop,
    read_mono,
    relative_azimuth,
    render_ir,
    set_pose,
)


CLASS_MAP = {
    0: "Female speech", 1: "Male speech", 3: "Telephone", 8: "Music"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="data/scene_datasets/replica_compat/office_0/habitat/mesh_semantic.ply")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--agent", type=float, nargs=3, default=[0.8, -0.968869, 2.443387])
    parser.add_argument("--source", type=float, nargs=3, default=[0.1810176075, -0.9688690305, 2.44338727])
    parser.add_argument("--angle", type=float, action="append", required=True)
    parser.add_argument("--dry-json", required=True, help="JSON list: [{path,class_id,class_name,name}, ...]")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    wav_dir = output_root / "wav_24k_5s"
    wav_dir.mkdir(parents=True, exist_ok=True)
    source = np.asarray(args.source, dtype=np.float32)
    agent = np.asarray(args.agent, dtype=np.float32)
    dry_items = json.loads(Path(args.dry_json).read_text(encoding="utf-8"))
    render_rate, model_rate, duration_s = 16000, 24000, 5.0
    source_audio_position = source + np.array([0.0, 1.5, 0.0])
    listener_audio_position = agent + np.array([0.0, 1.5, 0.0])
    distance = float(np.linalg.norm(source_audio_position - listener_audio_position))
    base_azimuth = relative_azimuth(source_audio_position, listener_audio_position, 0.0)
    rows = []

    sim = make_sim(args.scene, source_audio_position, agent, 0.0, False, "", render_rate)
    try:
        dry_signals = []
        for item in dry_items:
            dry, original_rate = read_mono(Path(item["path"]), render_rate)
            dry = pad_or_crop(dry, int(duration_s * render_rate))
            dry_signals.append((item, dry, original_rate))
        for angle in args.angle:
            yaw = angle - base_azimuth
            set_pose(sim, source_audio_position, agent, yaw)
            ir = render_ir(sim)
            for item, dry, original_rate in dry_signals:
                convolved = np.stack([
                    fftconvolve(dry, ir[:, channel], mode="full") for channel in range(2)
                ], axis=1).astype(np.float32)
                model_audio = resample_poly(convolved, model_rate, render_rate).astype(np.float32)
                samples = int(duration_s * model_rate)
                model_audio = model_audio[:samples]
                if model_audio.shape[0] < samples:
                    model_audio = np.pad(model_audio, ((0, samples - model_audio.shape[0]), (0, 0)))
                peak = float(np.max(np.abs(model_audio)))
                gain = min(1.0, 0.98 / peak) if peak > 0.98 else 1.0
                model_audio *= gain
                name = item["name"]
                sample_id = "{}_az_{:+03d}".format(name, int(round(angle))).replace("+", "p").replace("-", "m")
                wav_path = wav_dir / (sample_id + ".wav")
                wavfile.write(str(wav_path), model_rate, model_audio.astype(np.float32))
                rows.append({
                    "sample_id": sample_id, "dry_file": item["path"], "class_id": item["class_id"],
                    "class_name": item["class_name"], "gt_azimuth_full_deg": float(angle),
                    "gt_azimuth_folded_deg": float(fold_azimuth(float(angle))), "agent_yaw_deg": float(yaw),
                    "source_world_xyz": json.dumps(source_audio_position.tolist()),
                    "listener_world_xyz": json.dumps(listener_audio_position.tolist()),
                    "distance_m": distance, "materials_on": False, "ir_shape": json.dumps(list(ir.shape)),
                    "render_sample_rate": render_rate, "output_sample_rate": model_rate,
                    "clip_seconds": duration_s, "input_rms": float(np.sqrt(np.mean(model_audio ** 2))),
                    "input_peak": float(np.max(np.abs(model_audio))), "input_clipping_count": int(np.sum(np.abs(model_audio) > 1.0)),
                    "peak_guard_gain": gain, "wav_path": str(wav_path), "original_dry_rate": original_rate,
                })
                print("rendered {} angle={} class={} distance={:.3f}".format(sample_id, angle, item["class_name"], distance))
    finally:
        sim.close()
    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "manifest.json").write_text(json.dumps({"class_map": CLASS_MAP, "rows": rows}, indent=2), encoding="utf-8")
    print("manifest rows: {} -> {}".format(len(rows), manifest))


if __name__ == "__main__":
    main()
