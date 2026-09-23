"""Generate the Candidate A revalidation transfer set in new output folders only."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_binaural_transfer_set import (  # noqa: E402
    make_sim,
    pad_or_crop,
    read_mono,
    relative_azimuth,
    render_ir,
    set_pose,
)


ANGLES = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
SOURCE = np.asarray([0.1810176075, -0.9688690305, 2.44338727], dtype=np.float32)
CONDITIONS = {
    "near_off": {"agent": [0.8, -0.9688690305, 2.44338727], "materials_on": False},
    "dist_1p5_off": {"agent": [1.6239607337, -0.9688690305, 2.0340662003], "materials_on": False},
    "dist_3p0_off": {"agent": [1.2185535431, -0.9688690305, -0.3722417355], "materials_on": False},
    "near_on": {"agent": [0.8, -0.9688690305, 2.44338727], "materials_on": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dry-json", required=True)
    parser.add_argument("--materials-json", default="data/replica_material_config.json")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    dry_items = json.loads(Path(args.dry_json).read_text(encoding="utf-8"))
    render_rate, model_rate, duration_s = 16000, 24000, 5.0
    source_audio = SOURCE + np.asarray([0.0, 1.5, 0.0], dtype=np.float32)
    scene_name = Path(args.scene).parts[-3]
    rows = []

    for condition_name, condition in CONDITIONS.items():
        agent = np.asarray(condition["agent"], dtype=np.float32)
        listener_audio = agent + np.asarray([0.0, 1.5, 0.0], dtype=np.float32)
        distance = float(np.linalg.norm(source_audio - listener_audio))
        base_azimuth = relative_azimuth(source_audio, listener_audio, 0.0)
        condition_root = output_root / "conditions" / condition_name
        wav_dir = condition_root / "wav_24k_5s"
        wav_dir.mkdir(parents=True, exist_ok=True)
        sim = make_sim(
            args.scene,
            source_audio,
            agent,
            0.0,
            condition["materials_on"],
            str((repo_root / args.materials_json).resolve()),
            render_rate,
        )
        try:
            dry_signals = []
            for item in dry_items:
                path = Path(item["path"])
                if not path.is_absolute():
                    path = repo_root / path
                dry, original_rate = read_mono(path, render_rate)
                dry = pad_or_crop(dry, int(duration_s * render_rate))
                dry_signals.append((item, path, dry, original_rate))
            for angle in ANGLES:
                yaw = float(angle - base_azimuth)
                set_pose(sim, source_audio, agent, yaw)
                ir = render_ir(sim)
                for item, dry_path, dry, original_rate in dry_signals:
                    convolved = np.stack(
                        [fftconvolve(dry, ir[:, channel], mode="full") for channel in range(2)],
                        axis=1,
                    ).astype(np.float32)
                    model_audio = resample_poly(convolved, model_rate, render_rate).astype(np.float32)
                    samples = int(duration_s * model_rate)
                    model_audio = model_audio[:samples]
                    if model_audio.shape[0] < samples:
                        model_audio = np.pad(model_audio, ((0, samples - model_audio.shape[0]), (0, 0)))
                    peak = float(np.max(np.abs(model_audio)))
                    gain = min(1.0, 0.98 / peak) if peak > 0.98 else 1.0
                    model_audio *= gain
                    sample_id = f"{condition_name}__{item['name']}__az_{int(angle):+03d}".replace("+", "p").replace("-", "m")
                    wav_path = wav_dir / f"{sample_id}.wav"
                    wavfile.write(str(wav_path), model_rate, model_audio.astype(np.float32))
                    rows.append({
                        "sample_id": sample_id,
                        "condition": condition_name,
                        "scene": scene_name,
                        "class_id": int(item["class_id"]),
                        "class_name": item["class_name"],
                        "dry_name": item["name"],
                        "dry_file": str(dry_path),
                        "dry_semantic_source": item.get("semantic_source", "unspecified"),
                        "gt_azimuth_project_deg": float(angle),
                        "gt_azimuth_dcase_deg": float(-angle),
                        "agent_yaw_deg": yaw,
                        "source_world_xyz": json.dumps(source_audio.tolist()),
                        "agent_world_xyz": json.dumps(agent.tolist()),
                        "listener_world_xyz": json.dumps(listener_audio.tolist()),
                        "distance_m": distance,
                        "materials_on": bool(condition["materials_on"]),
                        "ir_shape": json.dumps(list(ir.shape)),
                        "render_sample_rate": render_rate,
                        "output_sample_rate": model_rate,
                        "clip_seconds": duration_s,
                        "dry_original_rate": original_rate,
                        "input_rms": float(np.sqrt(np.mean(model_audio * model_audio))),
                        "input_peak": float(np.max(np.abs(model_audio))),
                        "input_clipping_count": int(np.sum(np.abs(model_audio) > 1.0)),
                        "peak_guard_gain": gain,
                        "wav_path": str(wav_path),
                    })
                    print(f"rendered {sample_id} class={item['class_name']} distance={distance:.3f} materials={condition['materials_on']}")
        finally:
            sim.close()
    manifest = output_root / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "manifest.json").write_text(
        json.dumps({"angles_project_deg": ANGLES, "conditions": CONDITIONS, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"manifest rows: {len(rows)} -> {manifest}")


if __name__ == "__main__":
    main()
