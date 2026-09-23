"""审计本地 dry event audio，不下载或合成额外事件源。"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile


CLASS_MAP = {
    3: "Telephone",
    8: "Music",
}


def infer_class(path: Path):
    name = path.stem.lower()
    if "telephone" in name:
        return 3, CLASS_MAP[3]
    if "singing" in name or "music" in name:
        return 8, CLASS_MAP[8]
    return "", ""


def audio_stats(path: Path):
    sample_rate, signal = wavfile.read(str(path))
    signal = np.asarray(signal)
    duration = signal.shape[0] / float(sample_rate)
    channels = 1 if signal.ndim == 1 else signal.shape[1]
    return sample_rate, signal.shape, duration, channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="data/experiments/seld_transfer_gate_20260824/dry_inventory.csv")
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
        and "seld_transfer_gate_20260824" not in path.parts
        and "pose_experiments" not in path.parts
        and "orientation_check" not in path.parts
    )
    rows = []
    for path in paths:
        class_id, class_name = infer_class(path)
        if not class_name:
            continue
        try:
            sample_rate, shape, duration, channels = audio_stats(path)
            error = ""
        except Exception as exc:  # pragma: no cover - inventory should keep going
            sample_rate, shape, duration, channels = "", "", "", ""
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "file_path": str(path),
            "duration_s": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "usable_5s_segment": bool(duration >= 5.0),
            "read_error": error,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "class_id", "class_name", "file_path", "duration_s", "sample_rate",
            "channels", "usable_5s_segment", "read_error",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"inventory rows: {len(rows)}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
