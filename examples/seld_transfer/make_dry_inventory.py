"""Validate and materialize a small local mono dry-event inventory."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    rows = []
    for item in sources:
        path = Path(item["path"])
        if not path.is_absolute():
            path = repo_root / path
        info = sf.info(str(path))
        row = dict(item)
        row.update({
            "path": str(path),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "seconds": float(info.duration),
            "mono_dry": bool(info.channels == 1),
            "sha256": sha256(path),
            "status": "usable_mono" if info.channels == 1 else "reject_stereo",
        })
        rows.append(row)
        print(f"inventory {item['name']}: channels={info.channels} sr={info.samplerate} seconds={info.duration:.3f}")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"inventory rows: {len(rows)} -> {output_csv}")


if __name__ == "__main__":
    main()
