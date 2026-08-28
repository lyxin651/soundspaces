#!/usr/bin/env python3
"""Run deterministic structural QC on a Step 2A candidate inventory."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf


EXTRA_FIELDS = [
    "decode_ok", "finite", "duration_sec", "sample_rate", "channels",
    "peak_abs", "rms_dbfs", "dc_offset", "clip_fraction",
    "silence_or_low_activity_ratio", "active_duration_estimate_sec",
    "auto_qc_status", "auto_qc_reason",
]

REVIEW_FIELDS = ["manual_qc_status", "manual_qc_reason", "manual_qc_reviewer", "manual_qc_timestamp"]


def dbfs(value):
    return -float("inf") if value <= 0 else 20.0 * math.log10(value)


def qc_waveform(path):
    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # deliberate hard reject for undecodable input
        return {"decode_ok": "false", "finite": "false", "auto_qc_status": "AUTO_REJECT",
                "auto_qc_reason": "decode_failed:{}".format(type(exc).__name__)}

    flat = data.reshape(-1)
    finite = bool(np.isfinite(flat).all())
    channels = data.shape[1]
    duration = len(data) / float(sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(flat))) if len(flat) else 0.0
    rms = float(np.sqrt(np.mean(np.square(flat)))) if len(flat) else 0.0
    dc = float(np.mean(flat)) if len(flat) else 0.0
    clip_fraction = float(np.mean(np.abs(flat) >= 0.999)) if len(flat) else 1.0
    activity_threshold = max(1e-4, peak * 0.10)
    active_mask = np.abs(data).max(axis=1) >= activity_threshold
    active_duration = float(np.count_nonzero(active_mask)) / float(sample_rate) if sample_rate else 0.0
    low_activity = 1.0 - (float(np.count_nonzero(active_mask)) / len(active_mask)) if len(active_mask) else 1.0

    reason = ""
    status = "AUTO_PASS"
    if not finite:
        status, reason = "AUTO_REJECT", "empty_or_nonfinite"
    elif duration <= 0 or peak <= 1e-8:
        status, reason = "AUTO_REJECT", "empty_or_nearly_zero"
    elif clip_fraction >= 0.01:
        status, reason = "AUTO_FLAG", "severe_clipping_candidate"
    elif active_duration < 0.10 or low_activity > 0.98:
        status, reason = "AUTO_FLAG", "mostly_silence_or_low_activity"

    return {
        "decode_ok": "true", "finite": str(finite).lower(), "duration_sec": "{:.9f}".format(duration),
        "sample_rate": str(sample_rate), "channels": str(channels), "peak_abs": "{:.9f}".format(peak),
        "rms_dbfs": "{:.6f}".format(dbfs(rms)), "dc_offset": "{:.9f}".format(dc),
        "clip_fraction": "{:.9f}".format(clip_fraction),
        "silence_or_low_activity_ratio": "{:.9f}".format(low_activity),
        "active_duration_estimate_sec": "{:.9f}".format(active_duration),
        "auto_qc_status": status, "auto_qc_reason": reason,
    }


def run_qc(candidate_path, asset_root, output_path):
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        result = qc_waveform(asset_root / row["raw_relpath"])
        output.append(dict(row, **result))
    fields = list(rows[0].keys()) + EXTRA_FIELDS if rows else EXTRA_FIELDS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return output


def write_review_queue(qc_path, output_path):
    with qc_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0].keys()) + REVIEW_FIELDS if rows else REVIEW_FIELDS
    for row in rows:
        for field in REVIEW_FIELDS:
            row[field] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path)
    args = parser.parse_args()
    rows = run_qc(
        candidate_path=args.candidate_inventory,
        asset_root=args.asset_root,
        output_path=args.output,
    )
    counts = {}
    for row in rows:
        counts[row["auto_qc_status"]] = counts.get(row["auto_qc_status"], 0) + 1
    print("rows={}".format(len(rows)))
    print("counts={}".format(counts))
    if args.review_queue:
        print("review_queue_rows={}".format(write_review_queue(args.output, args.review_queue)))


if __name__ == "__main__":
    main()
