"""Minimal 16 kHz vs 24 kHz SoundSpaces binaural gate.

Run from the sound-spaces repository root:
    conda run -n ss --no-capture-output python examples/seld_transfer/check_binaural_24k_gate.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import quaternion  # must be imported before habitat_sim
import habitat_sim  # noqa: F401
import numpy as np
from scipy.signal import fftconvolve

from generate_binaural_transfer_set import ensure_samples_channels, make_sim, set_pose, write_audio


SCENE = "data/scene_datasets/replica_compat/office_0/habitat/mesh_semantic.ply"
OUTPUT_ROOT = Path("data/experiments/v1_24k_gate")
WAV_PATH = OUTPUT_ROOT / "binaural_24k_probe.wav"
METRICS_PATH = OUTPUT_ROOT / "metrics.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
LOG_PATH = OUTPUT_ROOT / "gate.log"

MATERIALS_ENABLED = False
MATERIALS_JSON = "data/replica_material_config.json"
LISTENER_AGENT = np.asarray([0.800000011920929, -0.9688689708709717, 2.443387031555176], dtype=np.float32)
LISTENER_WORLD = LISTENER_AGENT + np.asarray([0.0, 1.5, 0.0], dtype=np.float32)
# Yaw 0 faces -Z in the established SoundSpaces pose convention.  With the
# same listener position, the two source positions below are lateral mirrors.
LISTENER_YAW_DEG = 0.0
LEFT_SOURCE_WORLD = np.asarray([0.18101760745048523, 0.5311309695243835, 2.443387269973755], dtype=np.float32)
RIGHT_SOURCE_WORLD = np.asarray([1.4189824163913727, 0.5311309695243835, 2.443387269973755], dtype=np.float32)


def log(lines: List[str], message: str) -> None:
    lines.append(message)
    print(message)


def channel_stats(values: np.ndarray) -> Dict[str, float]:
    left = values[:, 0]
    right = values[:, 1]
    return {
        "energy_left": float(np.sum(left.astype(np.float64) ** 2)),
        "energy_right": float(np.sum(right.astype(np.float64) ** 2)),
        "rms_left": float(np.sqrt(np.mean(left.astype(np.float64) ** 2))),
        "rms_right": float(np.sqrt(np.mean(right.astype(np.float64) ** 2))),
        "peak_left": float(np.max(np.abs(left))) if left.size else 0.0,
        "peak_right": float(np.max(np.abs(right))) if right.size else 0.0,
    }


def render_condition(sample_rate_hz: int, condition: str, source_world: np.ndarray) -> Dict[str, object]:
    sim = make_sim(
        SCENE,
        source_world,
        LISTENER_AGENT,
        LISTENER_YAW_DEG,
        MATERIALS_ENABLED,
        MATERIALS_JSON,
        sample_rate_hz,
    )
    try:
        set_pose(sim, source_world, LISTENER_AGENT, LISTENER_YAW_DEG)
        native = np.asarray(sim.get_sensor_observations()["audio_sensor"])
        ir = ensure_samples_channels(native).astype(np.float32)
    finally:
        sim.close()

    if ir.ndim != 2 or ir.shape[1] != 2:
        raise RuntimeError("expected binaural IR with shape (samples, 2), got {}".format(ir.shape))

    stats = channel_stats(ir)
    finite = bool(np.all(np.isfinite(ir)))
    all_zero = bool(np.all(ir == 0.0))
    if condition == "left":
        channel_order_pass = (
            stats["energy_left"] > stats["energy_right"]
            and stats["rms_left"] > stats["rms_right"]
            and stats["peak_left"] > stats["peak_right"]
        )
    else:
        channel_order_pass = (
            stats["energy_right"] > stats["energy_left"]
            and stats["rms_right"] > stats["rms_left"]
            and stats["peak_right"] > stats["peak_left"]
        )

    row: Dict[str, object] = {
        "sample_rate_hz": sample_rate_hz,
        "condition": condition,
        "native_shape": json.dumps(list(native.shape)),
        "native_dtype": str(native.dtype),
        "num_samples": int(ir.shape[0]),
        "rir_duration_sec": float(ir.shape[0] / float(sample_rate_hz)),
        "finite": finite,
        "all_zero": all_zero,
        "channel_order_pass": bool(channel_order_pass),
        "shape_pass": bool(native.ndim == 2 and native.shape[0] == 2 and ir.shape[1] == 2),
        "basic_pass": bool(finite and not all_zero and native.ndim == 2 and native.shape[0] == 2 and ir.shape[1] == 2),
        "ir": ir,
    }
    row.update(stats)
    return row


def make_probe_wav(rir: np.ndarray) -> Dict[str, object]:
    sample_rate = 24000
    rng = np.random.default_rng(20260827)
    dry = rng.standard_normal(int(sample_rate * 5.0)).astype(np.float32) * np.float32(0.05)
    convolved = np.stack(
        [
            fftconvolve(dry, rir[:, channel], mode="full")
            for channel in range(2)
        ],
        axis=1,
    ).astype(np.float32)
    write_audio(WAV_PATH, sample_rate, convolved)
    return {
        "path": str(WAV_PATH),
        "sample_rate_hz": sample_rate,
        "num_channels": int(convolved.shape[1]),
        "dtype": str(convolved.dtype),
        "num_samples": int(convolved.shape[0]),
        "expected_num_samples": int(dry.shape[0] + rir.shape[0] - 1),
        "finite": bool(np.all(np.isfinite(convolved))),
        "nonzero": bool(np.any(convolved != 0.0)),
        "peak": float(np.max(np.abs(convolved))) if convolved.size else 0.0,
    }


def duration_reasonable(rows: Dict[str, Dict[int, Dict[str, object]]]) -> bool:
    for condition in ("left", "right"):
        d16 = float(rows[condition][16000]["rir_duration_sec"])
        d24 = float(rows[condition][24000]["rir_duration_sec"])
        if d16 <= 0.0 or d24 <= 0.0:
            return False
        ratio = d24 / d16
        if ratio < 0.5 or ratio > 2.0:
            return False
    return True


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []
    log(log_lines, "SoundSpaces 24 kHz binaural gate started at {}".format(datetime.now().isoformat(timespec="seconds")))

    rows_by_condition: Dict[str, Dict[int, Dict[str, object]]] = {"left": {}, "right": {}}
    rows: List[Dict[str, object]] = []
    final_grade = "INCONCLUSIVE"
    problems: List[str] = []

    try:
        for sample_rate in (16000, 24000):
            for condition, source in (("left", LEFT_SOURCE_WORLD), ("right", RIGHT_SOURCE_WORLD)):
                row = render_condition(sample_rate, condition, source)
                rows_by_condition[condition][sample_rate] = row
                rows.append(row)
                log(
                    log_lines,
                    "{sr} Hz {cond}: native={native} N={n} duration={dur:.6f}s "
                    "energy L/R={el:.6e}/{er:.6e} pass={passed}".format(
                        sr=sample_rate,
                        cond=condition,
                        native=row["native_shape"],
                        n=row["num_samples"],
                        dur=row["rir_duration_sec"],
                        el=row["energy_left"],
                        er=row["energy_right"],
                        passed=row["channel_order_pass"],
                    ),
                )

        wav_gate = make_probe_wav(rows_by_condition["left"][24000]["ir"])
        log(log_lines, "24 kHz probe WAV: {} samples, peak={:.6e}".format(wav_gate["num_samples"], wav_gate["peak"]))

        reference_pass = all(bool(rows_by_condition[c][16000]["basic_pass"]) and bool(rows_by_condition[c][16000]["channel_order_pass"]) for c in ("left", "right"))
        render_24k_pass = all(bool(rows_by_condition[c][24000]["basic_pass"]) and bool(rows_by_condition[c][24000]["channel_order_pass"]) for c in ("left", "right"))
        duration_pass = duration_reasonable(rows_by_condition)
        wav_pass = bool(
            wav_gate["num_channels"] == 2
            and wav_gate["dtype"] == "float32"
            and wav_gate["num_samples"] == wav_gate["expected_num_samples"]
            and wav_gate["finite"]
            and wav_gate["nonzero"]
        )
        if not reference_pass:
            final_grade = "INCONCLUSIVE"
            problems.append("16k_reference_failed")
        elif render_24k_pass and duration_pass and wav_pass:
            final_grade = "PASS"
        else:
            final_grade = "FAIL"
            if not render_24k_pass:
                problems.append("24k_render_or_channel_order_failed")
            if not duration_pass:
                problems.append("rir_duration_gate_failed")
            if not wav_pass:
                problems.append("wav_gate_failed")

    except Exception as exc:  # keep a report even on gate failure
        problems.append("{}: {}".format(type(exc).__name__, exc))
        wav_gate = {
            "path": str(WAV_PATH),
            "sample_rate_hz": 24000,
            "num_channels": 0,
            "dtype": "",
            "num_samples": 0,
            "expected_num_samples": 0,
            "finite": False,
            "nonzero": False,
            "peak": 0.0,
        }
        final_grade = "INCONCLUSIVE" if not rows else "FAIL"
        log(log_lines, "ERROR: {}".format(problems[-1]))

    metric_fields = [
        "sample_rate_hz",
        "condition",
        "native_shape",
        "native_dtype",
        "num_samples",
        "rir_duration_sec",
        "energy_left",
        "energy_right",
        "rms_left",
        "rms_right",
        "peak_left",
        "peak_right",
        "finite",
        "all_zero",
        "shape_pass",
        "basic_pass",
        "channel_order_pass",
    ]
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in metric_fields})

    comparison = {}
    if all(sample_rate in rows_by_condition[condition] for condition in ("left", "right") for sample_rate in (16000, 24000)):
        comparison = {
            "left_N24_over_N16": float(rows_by_condition["left"][24000]["num_samples"] / rows_by_condition["left"][16000]["num_samples"]),
            "right_N24_over_N16": float(rows_by_condition["right"][24000]["num_samples"] / rows_by_condition["right"][16000]["num_samples"]),
            "left_duration16_sec": rows_by_condition["left"][16000]["rir_duration_sec"],
            "left_duration24_sec": rows_by_condition["left"][24000]["rir_duration_sec"],
            "right_duration16_sec": rows_by_condition["right"][16000]["rir_duration_sec"],
            "right_duration24_sec": rows_by_condition["right"][24000]["rir_duration_sec"],
        }

    summary = {
        "fixture": {
            "scene": SCENE,
            "listener_position_world": [float(x) for x in LISTENER_WORLD],
            "listener_agent_position": [float(x) for x in LISTENER_AGENT],
            "listener_yaw_deg": float(LISTENER_YAW_DEG),
            "left_source_position_world": [float(x) for x in LEFT_SOURCE_WORLD],
            "right_source_position_world": [float(x) for x in RIGHT_SOURCE_WORLD],
            "materials_enabled": MATERIALS_ENABLED,
        },
        "comparison": comparison,
        "wav_gate": wav_gate,
        "final_grade": final_grade,
        "problems": problems,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(log_lines, "Final Grade: {}".format(final_grade))
    if problems:
        log(log_lines, "Problems: {}".format(", ".join(problems)))
    log(log_lines, "Wrote {}, {}, {}, {}".format(METRICS_PATH, SUMMARY_PATH, WAV_PATH, LOG_PATH))
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0 if final_grade == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
