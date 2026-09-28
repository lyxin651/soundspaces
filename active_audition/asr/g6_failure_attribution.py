"""Read-only A3-G6 failure attribution diagnostics.

This module consumes the frozen A3 instrument, G6 manifest, and A2 RIR lock.
It deliberately has no mechanism for changing an A3 contract, selecting a new
utterance/RIR, or changing the production frontend.  Every intervention is a
diagnostic output in a new run directory.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.asr.contract import asr_contract_sha256, load_asr_contract
from active_audition.asr.frontends import apply_frontend
from active_audition.asr.qualification import canonical_json_bytes, file_sha256
from active_audition.asr.rir_bridge import load_rir_lock
from active_audition.asr.run_qualification import (
    A3RunError,
    _decode_speech,
    _evaluate,
    _load_manifest,
    _rir_index,
    _transcribe_many,
)
from active_audition.asr.speechbrain_adapter import SpeechBrainASRAdapter
from active_audition.data.storage import DatasetStorage
from active_audition.evaluation.asr_metrics import aggregate_error_counts
from active_audition.receiver.qualification import _direct_window, compute_direct_metrics, load_metric_contract, metric_contract_sha256


SCHEMA_VERSION = "active-asr-a3-g6-failure-attribution-v1"
EXPECTED_CONTRACT_SHA = "b972d3ca2354ead8a10d2954a60602896ebbc5204adb7e8692c22f2b340497e2"
EXPECTED_SS2_MANIFEST_SHA = "7b7f9b79ebc8bed64881e5475180623333da94454f3838257dcc7b2372b190ba"
EXPECTED_CLEAN_MANIFEST_SHA = "3c3fe2a84fd88bdaa878d4f0b0094b29eebe074fc9b81c7e6d55a284fb1ff63c"
EXPECTED_RIR_LOCK_SHA = "bb71cc51123dbbb418910c8e055a326a2113c663e35d22856438575b4d7cc65e"
EXPECTED_G6_ARTIFACT_SHA = "9abc6c3e972435c26476fa65030e14cbfac85d6d1317bf196335fb10ac15da72"
SAMPLE_RATE_HZ = 16000
G6_CASES = ("front_near", "front_far", "side_left_far")
FRONTENDS = ("mean_lr", "fixed_L", "fixed_R")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha(value: Any) -> str:
    return _sha_bytes(np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes(order="C"))


def _stats(value: Any, sample_rate_hz: int = SAMPLE_RATE_HZ) -> Dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise A3RunError("diagnostic waveform must be a non-empty mono vector")
    square = np.square(array.astype(np.float64))
    return {
        "samples": int(array.size),
        "duration_sec": float(array.size / float(sample_rate_hz)),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "peak_abs": float(np.max(np.abs(array))),
        "rms": float(np.sqrt(np.mean(square))),
        "mean_square": float(np.mean(square)),
        "sum_square_energy": float(np.sum(square)),
        "sha256": _array_sha(array),
        "clipping_threshold": 1.0,
        "clipping": bool(np.any(np.abs(array) >= 1.0)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise A3RunError("cannot aggregate empty diagnostic rows")
    return aggregate_error_counts([dict(row["metric"]) for row in rows])


def _paired_deltas(reference_rows: Sequence[Mapping[str, Any]], comparison_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return explicit per-utterance deltas without selecting on WER."""
    reference = {str(row["utterance_id"]): row for row in reference_rows}
    comparison = {str(row["utterance_id"]): row for row in comparison_rows}
    result = []
    for utterance_id in sorted(set(reference) & set(comparison)):
        left = reference[utterance_id]
        right = comparison[utterance_id]
        result.append({
            "utterance_id": utterance_id,
            "reference_wer": float(left["WER"]),
            "comparison_wer": float(right["WER"]),
            "wer_delta_comparison_minus_reference": float(right["WER"] - left["WER"]),
            "reference_edit_count": int(left["S"] + left["D"] + left["I"]),
            "comparison_edit_count": int(right["S"] + right["D"] + right["I"]),
            "edit_count_delta_comparison_minus_reference": int((right["S"] + right["D"] + right["I"]) - (left["S"] + left["D"] + left["I"])),
            "hypothesis_equal": bool(left["hypothesis"] == right["hypothesis"]),
        })
    return result


def _metric_row(record: Mapping[str, Any], output: Any) -> Dict[str, Any]:
    speech = record.get("speech", record)
    metric = dict(output["metric"])
    return {
        "utterance_id": speech["utterance_id"],
        "record_id": record.get("record_id", speech["utterance_id"]),
        "reference": speech["normalized_transcript"],
        "hypothesis": output["hypothesis"],
        "S": int(metric["S"]),
        "D": int(metric["D"]),
        "I": int(metric["I"]),
        "N": int(metric["N"]),
        "WER": float(metric["WER"]),
        "CER": float(metric["CER"]),
        "metric": metric,
        "asr": output["asr"],
    }


def _run_asr(
    adapter: SpeechBrainASRAdapter,
    records: Sequence[Mapping[str, Any]],
    waveforms: Sequence[np.ndarray],
    frontend: str,
    waveform_stats: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    outputs = _transcribe_many(adapter, waveforms, frontend)
    raw_rows = []
    for record, output, waveform_stat in zip(records, outputs, waveform_stats):
        speech = record.get("speech", record)
        from active_audition.evaluation.asr_metrics import error_counts

        metric = _metric_row(record, {"metric": error_counts(speech["normalized_transcript"], output.hypothesis), "hypothesis": output.hypothesis, "asr": output.to_dict()})
        metric["frontend"] = frontend
        metric["waveform"] = dict(waveform_stat)
        metric.update(context)
        raw_rows.append(metric)
    return {"rows": raw_rows, "aggregate": _aggregate(raw_rows)}


def _edc_fit(edc: np.ndarray, start: int, sample_rate_hz: int, low_db: float, high_db: float) -> Dict[str, Any]:
    if start >= edc.size or edc[start] <= 0.0:
        return {"status": "N/A", "reason": "NO_POSITIVE_REFERENCE_ENERGY"}
    ref = float(edc[start])
    db = 10.0 * np.log10(np.maximum(edc[start:], np.finfo(np.float64).tiny) / ref)
    wanted = (db <= float(low_db)) & (db >= float(high_db))
    indices = np.flatnonzero(wanted)
    if indices.size < 3:
        return {"status": "N/A", "reason": "DECAY_RANGE_NOT_OBSERVED", "available_db": [float(np.min(db)), float(np.max(db))]}
    time = indices.astype(np.float64) / float(sample_rate_hz)
    values = db[indices]
    slope, intercept = np.polyfit(time, values, 1)
    predicted = slope * time + intercept
    residual = values - predicted
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(values - np.mean(values))))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    if not math.isfinite(float(slope)) or slope >= 0.0 or r2 < 0.8:
        return {"status": "N/A", "reason": "DECAY_FIT_INVALID", "slope_db_per_sec": float(slope), "r2": float(r2), "points": int(indices.size)}
    return {
        "status": "APPLICABLE",
        "slope_db_per_sec": float(slope),
        "intercept_db": float(intercept),
        "r2": float(r2),
        "points": int(indices.size),
        "fit_range_db": [float(low_db), float(high_db)],
        "decay_time_sec": float((high_db - low_db) / slope),
        "time_start_sec_after_direct": float(time[0]),
        "time_end_sec_after_direct": float(time[-1]),
    }


def _rir_diagnostic(rir: np.ndarray, metric_contract: Mapping[str, Any]) -> Dict[str, Any]:
    array = np.asarray(rir, dtype=np.float32)
    direct = compute_direct_metrics(array, SAMPLE_RATE_HZ, metric_contract)
    window = direct["direct_window"]
    start = int(window.get("start_sample", 0)) if window.get("applicability") == "APPLICABLE" else None
    end = int(window.get("end_sample_exclusive", 0)) if start is not None else None
    early_end = min(array.shape[0], (start or 0) + int(round(0.050 * SAMPLE_RATE_HZ)))
    channels = {}
    for index, name in enumerate(("L", "R")):
        channel = array[:, index]
        square = np.square(channel.astype(np.float64))
        total = float(np.sum(square))
        direct_energy = float(np.sum(square[start:end])) if start is not None else None
        early = float(np.sum(square[(start or 0):early_end]))
        late = float(np.sum(square[early_end:]))
        edc = np.cumsum(square[::-1])[::-1]
        step = max(1, SAMPLE_RATE_HZ // 1000)
        sampled = [
            {"sample": int(i), "time_sec": float(i / SAMPLE_RATE_HZ), "energy": float(edc[i]), "db_raw": float(10.0 * math.log10(max(edc[i], np.finfo(np.float64).tiny)))}
            for i in range(0, edc.size, step)
        ]
        if not sampled or sampled[-1]["sample"] != edc.size - 1:
            i = edc.size - 1
            sampled.append({"sample": int(i), "time_sec": float(i / SAMPLE_RATE_HZ), "energy": float(edc[i]), "db_raw": float(10.0 * math.log10(max(edc[i], np.finfo(np.float64).tiny)))})
        channels[name] = {
            "peak_abs": float(np.max(np.abs(channel))),
            "rms": float(np.sqrt(np.mean(square))),
            "mean_square": float(np.mean(square)),
            "sum_square_energy": total,
            "direct_energy": direct_energy,
            "early_energy_0_to_50ms_from_direct_start": early,
            "late_energy_after_50ms_from_direct_start": late,
            "drr_proxy_direct_over_late_db": None if direct_energy is None or late <= 0.0 else float(10.0 * math.log10(max(direct_energy, 1e-300) / late)),
            "schroeder_edc": {"definition": "raw reverse cumulative sum(x^2), no signal normalization", "sample_step": step, "sampled": sampled, "full_float64_sha256": _sha_bytes(np.ascontiguousarray(edc, dtype="<f8").tobytes())},
            "edt": _edc_fit(edc, start or 0, SAMPLE_RATE_HZ, 0.0, -10.0),
            "t20": _edc_fit(edc, start or 0, SAMPLE_RATE_HZ, -5.0, -25.0),
        }
    return {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "channel_order": ["L", "R"],
        "finite": bool(np.isfinite(array).all()),
        "length_samples": int(array.shape[0]),
        "length_sec": float(array.shape[0] / SAMPLE_RATE_HZ),
        "sha256": _array_sha(array),
        "direct_window": window,
        "direct_metrics_from_frozen_contract": direct,
        "energy_windows": {"early_definition": "direct_start through direct_start+50ms", "late_definition": "after early window through RIR end", "no_normalization": True},
        "channels": channels,
    }


def _make_rir_variant(rir: np.ndarray, start: int, end: int) -> np.ndarray:
    result = np.zeros_like(rir, dtype=np.float32)
    result[start:min(int(end), rir.shape[0]), :] = rir[start:min(int(end), rir.shape[0]), :]
    return result


def _safe_gain(waveforms: Sequence[np.ndarray], gain: float) -> bool:
    return all(float(np.max(np.abs(waveform))) * float(gain) < 1.0 for waveform in waveforms)


def _gain_summary(waveforms: Sequence[np.ndarray], gain: float, source: str) -> Dict[str, Any]:
    energies = [float(np.sum(np.square(waveform.astype(np.float64)))) for waveform in waveforms]
    return {"gain_linear": float(gain), "gain_db": float(20.0 * math.log10(gain)), "source": source, "formula": "sqrt(reference_case_sum_square / target_case_sum_square)" if source == "acoustic_energy_only" else "fixed diagnostic level", "unsafe_clipping": not _safe_gain(waveforms, gain), "target_waveform_sum_square": float(sum(energies)), "max_input_peak": float(max(np.max(np.abs(w)) for w in waveforms))}


def _write(storage: DatasetStorage, output: Path, name: str, value: Any) -> Dict[str, str]:
    path = output / name
    storage.atomic_write_bytes(path, canonical_json_bytes(value))
    return {"path": name, "sha256": file_sha256(path)}


def _input_identity(repo: Path, path: str, expected: str) -> Dict[str, str]:
    resolved = (repo / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not resolved.is_file():
        raise A3RunError("frozen input is missing: {}".format(resolved))
    actual = file_sha256(resolved)
    if actual != expected:
        raise A3RunError("frozen input SHA mismatch: {} expected={} actual={}".format(resolved, expected, actual))
    return {"path": str(resolved.relative_to(repo) if str(resolved).startswith(str(repo)) else resolved), "sha256": actual}


def run_g6_failure_attribution(contract_path: str, metric_contract_path: str, rir_lock_path: str, g6_artifact_path: str, output_dir: str) -> Dict[str, Any]:
    repo = Path.cwd().resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        marker = output / "g6_failure_attribution_summary.json"
        owned = False
        if marker.is_file():
            try:
                owned = json.loads(marker.read_text(encoding="utf-8")).get("schema_version") == SCHEMA_VERSION
            except (OSError, ValueError):
                owned = False
        if not owned:
            raise A3RunError("attribution output directory is non-empty; refusing to overwrite: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    contract = load_asr_contract(str(repo / contract_path) if not Path(contract_path).is_absolute() else contract_path, require_frozen=True)
    contract_sha = asr_contract_sha256(contract)
    if contract_sha != EXPECTED_CONTRACT_SHA:
        raise A3RunError("frozen A3 contract canonical SHA mismatch")
    metric_contract = load_metric_contract(str(repo / metric_contract_path) if not Path(metric_contract_path).is_absolute() else metric_contract_path)
    metric_sha = metric_contract_sha256(metric_contract)
    ss2_identity = contract["qualification"]["manifests"]["ss2_domain"]
    clean_identity = contract["qualification"]["manifests"]["clean"]
    ss2_input = _input_identity(repo, ss2_identity["path"], EXPECTED_SS2_MANIFEST_SHA)
    clean_input = _input_identity(repo, clean_identity["path"], EXPECTED_CLEAN_MANIFEST_SHA)
    g6_input = _input_identity(repo, g6_artifact_path, EXPECTED_G6_ARTIFACT_SHA)
    rir_input = _input_identity(repo, rir_lock_path, EXPECTED_RIR_LOCK_SHA)
    g6_artifact = json.loads((repo / g6_artifact_path).read_text(encoding="utf-8"))
    ss2_manifest = _load_manifest(repo, contract, "ss2_domain")
    clean_manifest = _load_manifest(repo, contract, "clean")
    rir_lock, rir_lock_file = load_rir_lock(str(repo / rir_lock_path), contract_sha)
    rir = _rir_index(rir_lock, rir_lock_file)
    if any((case, SAMPLE_RATE_HZ) not in rir for case in G6_CASES):
        raise A3RunError("frozen RIR lock does not contain all three G6 cases")
    sources_path = repo / contract["sources"]["config_path"]
    sources = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    source_root = repo / sources["librispeech"]["root"]
    speech_cache: Dict[str, np.ndarray] = {}
    speech_records: Dict[str, Mapping[str, Any]] = {}
    for record in clean_manifest["records"]:
        speech_records[str(record["utterance_id"])] = record
    for record in ss2_manifest["records"]:
        speech_records[str(record["speech"]["utterance_id"])] = record["speech"]
    def dry_for(record: Mapping[str, Any]) -> np.ndarray:
        speech = record.get("speech", record)
        utterance = str(speech["utterance_id"])
        if utterance not in speech_cache:
            speech_cache[utterance] = _decode_speech(repo, source_root, speech)[0]
        return speech_cache[utterance]

    adapter = SpeechBrainASRAdapter(contract)
    by_case: Dict[str, List[Mapping[str, Any]]] = {case: [] for case in G6_CASES}
    for record in ss2_manifest["records"]:
        by_case[str(record["rir_case"]["case_id"])].append(record)
    rir_diagnostics = {case: {"case_id": case, "distance_m": float(by_case[case][0]["rir_case"]["distance_m"]), "relative_azimuth_deg": float(by_case[case][0]["rir_case"]["relative_azimuth_deg"]), "line_of_sight": by_case[case][0]["rir_case"]["line_of_sight"], **_rir_diagnostic(rir[(case, SAMPLE_RATE_HZ)], metric_contract)} for case in G6_CASES}

    baseline: Dict[str, Dict[str, Any]] = {}
    baseline_waveforms: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for case in G6_CASES:
        records = by_case[case]
        baseline[case] = {}
        raw_binaural = [convolve_binaural(dry_for(record), rir[(case, SAMPLE_RATE_HZ)]) for record in records]
        for frontend in FRONTENDS:
            monos = [apply_frontend(value, frontend) for value in raw_binaural]
            baseline_waveforms[(case, frontend)] = monos
            stats = [_stats(value) for value in monos]
            context = {"case_id": case, "normalization": "none", "rir_sha256": _array_sha(rir[(case, SAMPLE_RATE_HZ)]), "convolution_time_axis": "full"}
            result = _run_asr(adapter, records, monos, frontend, stats, context)
            result["binaural_waveform_stats"] = [{"L": _stats(value[:, 0]), "R": _stats(value[:, 1]), "shape": list(value.shape), "sha256": _array_sha(value)} for value in raw_binaural]
            for row, record, binaural_stat in zip(result["rows"], records, result["binaural_waveform_stats"]):
                row["dry_waveform"] = _stats(dry_for(record))
                row["binaural_waveform"] = binaural_stat
            baseline[case][frontend] = result

    level_output: Dict[str, Any] = {"schema_version": "active-asr-a3-g6-level-intervention-v1", "normalization": "none", "cases": {}}
    reference_energy = sum(float(np.sum(np.square(value.astype(np.float64)))) for value in baseline_waveforms[("front_near", "mean_lr")])
    for case in ("front_far", "side_left_far"):
        raw = baseline_waveforms[(case, "mean_lr")]
        target_energy = sum(float(np.sum(np.square(value.astype(np.float64)))) for value in raw)
        energy_gain = math.sqrt(reference_energy / target_energy)
        conditions = [("raw", 1.0, "raw"), ("plus_6_db", 10.0 ** (6.0 / 20.0), "fixed diagnostic level"), ("plus_12_db", 10.0 ** (12.0 / 20.0), "fixed diagnostic level"), ("energy_matched_to_front_near", energy_gain, "acoustic_energy_only")]
        level_output["cases"][case] = {"reference_case": "front_near", "conditions": {}}
        for name, gain, source in conditions:
            summary = _gain_summary(raw, gain, source)
            item: Dict[str, Any] = {"gain": summary, "same_gain_for_all_utterances": True, "source_is_wer_independent": source != "wer", "rows": [], "aggregate": None}
            if not summary["unsafe_clipping"]:
                waves = [(value * np.float32(gain)).astype(np.float32) for value in raw]
                result = _run_asr(adapter, by_case[case], waves, "mean_lr", [_stats(value) for value in waves], {"case_id": case, "condition": name, "normalization": "none"})
                item["rows"] = result["rows"]
                item["aggregate"] = result["aggregate"]
                item["paired_deltas_to_raw"] = _paired_deltas(baseline[case]["mean_lr"]["rows"], result["rows"])
            else:
                item["status"] = "SKIPPED_UNSAFE_CLIPPING"
            level_output["cases"][case]["conditions"][name] = item

    reverb_output: Dict[str, Any] = {"schema_version": "active-asr-a3-g6-reverb-intervention-v1", "normalization": "none", "variants": ["direct_only", "direct_plus_50ms", "direct_plus_100ms", "direct_plus_200ms", "full_rir"], "cases": {}}
    for case in G6_CASES:
        array = rir[(case, SAMPLE_RATE_HZ)]
        window = rir_diagnostics[case]["direct_window"]
        if window.get("applicability") != "APPLICABLE":
            raise A3RunError("frozen direct window is not applicable for {}".format(case))
        start = int(window["start_sample"])
        variants = [("direct_only", start, int(window["end_sample_exclusive"])), ("direct_plus_50ms", start, start + int(.050 * SAMPLE_RATE_HZ)), ("direct_plus_100ms", start, start + int(.100 * SAMPLE_RATE_HZ)), ("direct_plus_200ms", start, start + int(.200 * SAMPLE_RATE_HZ)), ("full_rir", 0, array.shape[0])]
        reverb_output["cases"][case] = {"direct_window": window, "variants": {}}
        for name, variant_start, variant_end in variants:
            variant = array if name == "full_rir" else _make_rir_variant(array, variant_start, variant_end)
            waves = [apply_frontend(convolve_binaural(dry_for(record), variant), "mean_lr") for record in by_case[case]]
            result = _run_asr(adapter, by_case[case], waves, "mean_lr", [_stats(value) for value in waves], {"case_id": case, "variant": name, "rir_variant_start_sample": variant_start, "rir_variant_end_sample_exclusive": min(variant_end, array.shape[0]), "time_axis_preserved": True, "normalization": "none"})
            reverb_output["cases"][case]["variants"][name] = {"aggregate": result["aggregate"], "rows": result["rows"], "variant_rir_sha256": _array_sha(variant)}
        full_rows = reverb_output["cases"][case]["variants"]["full_rir"]["rows"]
        for item in reverb_output["cases"][case]["variants"].values():
            item["paired_deltas_to_full_rir"] = _paired_deltas(full_rows, item["rows"])

    tail_output: Dict[str, Any] = {"schema_version": "active-asr-a3-g6-tail-intervention-v1", "full_rir_convolution_preserved_before_crop": True, "normalization": "none", "cases": {}}
    tail_options = [("through_dry_end", 0), ("dry_end_plus_0.25s", .25), ("dry_end_plus_0.5s", .5), ("dry_end_plus_1.0s", 1.0), ("full_convolution", None)]
    for case in G6_CASES:
        tail_output["cases"][case] = {"variants": {}}
        for name, extra in tail_options:
            waves = []
            prefix_checks = []
            for record in by_case[case]:
                full = apply_frontend(convolve_binaural(dry_for(record), rir[(case, SAMPLE_RATE_HZ)]), "mean_lr")
                if extra is None:
                    cropped = full
                else:
                    cropped = full[:min(full.size, dry_for(record).size + int(round(extra * SAMPLE_RATE_HZ)))]
                waves.append(cropped)
                prefix_checks.append({"utterance_id": record["speech"]["utterance_id"], "full_speech_time_prefix_sha256": _array_sha(full[:dry_for(record).size]), "cropped_speech_time_prefix_sha256": _array_sha(cropped[:dry_for(record).size]), "speech_time_prefix_unchanged": bool(np.array_equal(full[:dry_for(record).size], cropped[:dry_for(record).size]))})
            result = _run_asr(adapter, by_case[case], waves, "mean_lr", [_stats(value) for value in waves], {"case_id": case, "variant": name, "extra_tail_sec": extra, "normalization": "none"})
            tail_output["cases"][case]["variants"][name] = {"aggregate": result["aggregate"], "rows": result["rows"], "prefix_checks": prefix_checks}
        full_rows = tail_output["cases"][case]["variants"]["full_convolution"]["rows"]
        for item in tail_output["cases"][case]["variants"].values():
            item["paired_deltas_to_full_convolution"] = _paired_deltas(full_rows, item["rows"])

    replication: Dict[str, Any] = {"schema_version": "active-asr-a3-g6-24utterance-replication-v1", "clean_manifest": clean_input, "planned_utterances": len(clean_manifest["records"]), "cases": {}}
    for case in G6_CASES:
        replication["cases"][case] = {}
        for frontend in FRONTENDS:
            waves = [apply_frontend(convolve_binaural(dry_for(record), rir[(case, SAMPLE_RATE_HZ)]), frontend) for record in clean_manifest["records"]]
            result = _run_asr(adapter, clean_manifest["records"], waves, frontend, [_stats(value) for value in waves], {"case_id": case, "frontend": frontend, "normalization": "none", "rir_sha256": _array_sha(rir[(case, SAMPLE_RATE_HZ)])})
            result["paired_deltas_to_frozen_six"] = _paired_deltas(baseline[case][frontend]["rows"], result["rows"])
            replication["cases"][case][frontend] = result

    def simple_aggregate(item: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: item["aggregate"][key] for key in ("S", "D", "I", "N", "WER", "CER")}

    frontend_breakdown = {"schema_version": "active-asr-a3-g6-case-frontend-breakdown-v1", "production_frontend": "mean_lr", "cases": {case: {frontend: {"aggregate": simple_aggregate(baseline[case][frontend]), "rows": baseline[case][frontend]["rows"], "paired_deltas_to_mean_lr": [] if frontend == "mean_lr" else _paired_deltas(baseline[case]["mean_lr"]["rows"], baseline[case][frontend]["rows"])} for frontend in FRONTENDS} for case in G6_CASES}}
    attribution = {
        "LEVEL_EFFECT": {"status": "OBSERVED_INTERVENTION_EVIDENCE", "evidence": {case: {condition: (None if item["aggregate"] is None else simple_aggregate(item)) for condition, item in level_output["cases"][case]["conditions"].items()} for case in ("front_far", "side_left_far")}, "interpretation_rule": "compare raw and fixed-gain WER/edit changes; gains are diagnostics only"},
        "REVERBERATION_EFFECT": {"status": "OBSERVED_INTERVENTION_EVIDENCE", "evidence": {case: {variant: simple_aggregate(item) for variant, item in reverb_output["cases"][case]["variants"].items()} for case in G6_CASES}, "interpretation_rule": "compare direct-window truncations with full-RIR without renormalization"},
        "FRONTEND_EFFECT": {"status": "OBSERVED_INTERVENTION_EVIDENCE", "evidence": {case: {frontend: simple_aggregate(baseline[case][frontend]) for frontend in FRONTENDS} for case in G6_CASES}, "interpretation_rule": "mean_lr is production; fixed ears are sensitivity diagnostics only"},
        "TAIL_LENGTH_EFFECT": {"status": "OBSERVED_INTERVENTION_EVIDENCE", "evidence": {case: {variant: simple_aggregate(item) for variant, item in tail_output["cases"][case]["variants"].items()} for case in G6_CASES}, "interpretation_rule": "speech-time prefix equality is required before interpreting tail-only deltas"},
        "SAMPLE_SELECTION_EFFECT": {"status": "OBSERVED_REPLICATION_EVIDENCE", "evidence": {case: {"frozen_six_mean_lr": simple_aggregate(baseline[case]["mean_lr"]), "clean_24_mean_lr": simple_aggregate(replication["cases"][case]["mean_lr"])} for case in G6_CASES}, "interpretation_rule": "compare the frozen six utterance sample with the immutable 24-utterance clean manifest; no replacement of G6"},
    }
    artifacts: Dict[str, Any] = {
        "g6_acoustic_diagnostics.json": {"schema_version": SCHEMA_VERSION, "input_identities": {"contract": {"path": contract_path, "sha256": contract_sha}, "metric_contract": {"path": metric_contract_path, "sha256": metric_sha}, "ss2_manifest": ss2_input, "clean_manifest": clean_input, "rir_lock": rir_input, "original_g6_artifact": g6_input}, "original_g6_artifact_status": {"status": g6_artifact.get("status"), "aggregate": g6_artifact.get("aggregate"), "artifact_was_read_only": True}, "cases": rir_diagnostics},
        "g6_case_frontend_breakdown.json": frontend_breakdown,
        "g6_level_intervention.json": level_output,
        "g6_reverb_intervention.json": reverb_output,
        "g6_tail_intervention.json": tail_output,
        "g6_24utterance_replication.json": replication,
    }
    identities = {name: _write(DatasetStorage(str(output)), output, name, value) for name, value in artifacts.items()}
    report_lines = [
        "# A3-G6 Failure Attribution (diagnostic only)",
        "",
        "- Status: **A3 SERVER_RUN_FAIL / OPEN_BLOCKED unchanged**",
        "- Scope: frozen G6 attribution; no contract, threshold, utterance, RIR, model, decoder, or production frontend change.",
        "- Frozen A3 contract SHA: `{}`".format(contract_sha),
        "- Frozen metric contract SHA: `{}`".format(metric_sha),
        "- Frozen G6 baseline artifact SHA: `{}` (read-only)".format(EXPECTED_G6_ARTIFACT_SHA),
        "",
        "## Frozen baseline",
        "",
        "| case | mean_lr S/D/I/N | WER | fixed_L WER | fixed_R WER |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in G6_CASES:
        m = baseline[case]["mean_lr"]["aggregate"]
        report_lines.append("| {} | {}/{}/{}/{} | {:.4f}% | {:.4f}% | {:.4f}% |".format(case, m["S"], m["D"], m["I"], m["N"], 100*m["WER"], 100*baseline[case]["fixed_L"]["aggregate"]["WER"], 100*baseline[case]["fixed_R"]["aggregate"]["WER"]))
    report_lines += ["", "## RIR acoustic evidence", "", "The JSON artifact records raw channel energies, the frozen direct window, a raw reverse-cumulative Schroeder EDC, and EDT/T20 applicability. The DRR field is explicitly a `drr_proxy_direct_over_late_db`, not a qualified DRR/RT60 claim.", ""]
    for case in G6_CASES:
        item = rir_diagnostics[case]
        report_lines.append("- **{}**: onset L/R = `{}`; direct window = `{}`; L total/direct/late = `{:.6g}/{:.6g}/{:.6g}`; R total/direct/late = `{:.6g}/{:.6g}/{:.6g}`; length = `{:.6f}s`.".format(case, item["direct_window"].get("channel_onset_samples"), item["direct_window"].get("length_samples"), item["channels"]["L"]["sum_square_energy"], item["channels"]["L"]["direct_energy"], item["channels"]["L"]["late_energy_after_50ms_from_direct_start"], item["channels"]["R"]["sum_square_energy"], item["channels"]["R"]["direct_energy"], item["channels"]["R"]["late_energy_after_50ms_from_direct_start"], item["length_sec"]))
    report_lines += ["", "## Intervention aggregates", "", "The full raw paired rows and per-utterance metrics are in the JSON artifacts. All entries below are aggregate edit counts from the same frozen references.", ""]
    for label, source in (("Level", level_output["cases"]), ("Reverb", reverb_output["cases"]), ("Tail", tail_output["cases"])):
        report_lines.append("### {}".format(label))
        for case, value in source.items():
            report_lines.append("- **{}**".format(case))
            conditions = value["conditions"] if label == "Level" else value["variants"]
            for condition, item in conditions.items():
                aggregate = item.get("aggregate")
                report_lines.append("  - `{}`: {}".format(condition, "SKIPPED_UNSAFE_CLIPPING" if aggregate is None else "S/D/I/N={}/{}/{}/{} WER={:.4f}%".format(aggregate["S"], aggregate["D"], aggregate["I"], aggregate["N"], 100*aggregate["WER"])))
    report_lines += ["", "## Mechanism attribution", ""]
    for name, value in attribution.items():
        report_lines.append("### {}".format(name))
        report_lines.append("- Status: **{}**".format(value["status"]))
        if name == "LEVEL_EFFECT":
            report_lines.append("- Observable evidence: front_far mean_lr WER raw `{:.4f}%`, +6 dB `{:.4f}%`, +12 dB `{:.4f}%`, energy-matched `{:.4f}%`; side_left_far raw `{:.4f}%`, +6 dB `{:.4f}%`, +12 dB `{:.4f}%`, energy-matched `{:.4f}%`. The same case-level gain was applied to all six utterances; paired rows are recorded in `g6_level_intervention.json`.".format(100*level_output["cases"]["front_far"]["conditions"]["raw"]["aggregate"]["WER"], 100*level_output["cases"]["front_far"]["conditions"]["plus_6_db"]["aggregate"]["WER"], 100*level_output["cases"]["front_far"]["conditions"]["plus_12_db"]["aggregate"]["WER"], 100*level_output["cases"]["front_far"]["conditions"]["energy_matched_to_front_near"]["aggregate"]["WER"], 100*level_output["cases"]["side_left_far"]["conditions"]["raw"]["aggregate"]["WER"], 100*level_output["cases"]["side_left_far"]["conditions"]["plus_6_db"]["aggregate"]["WER"], 100*level_output["cases"]["side_left_far"]["conditions"]["plus_12_db"]["aggregate"]["WER"], 100*level_output["cases"]["side_left_far"]["conditions"]["energy_matched_to_front_near"]["aggregate"]["WER"]))
        elif name == "REVERBERATION_EFFECT":
            report_lines.append("- Observable evidence: front_far WER falls from full `{:.4f}%` to direct-only `{:.4f}%` and rises through 50/100/200 ms `{:.4f}%/{:.4f}%/{:.4f}%`; side_left_far falls from `{:.4f}%` to `{:.4f}%` and rises through `{:.4f}%/{:.4f}%/{:.4f}%`. Paired utterance deltas are recorded for every variant.".format(100*reverb_output["cases"]["front_far"]["variants"]["full_rir"]["aggregate"]["WER"], 100*reverb_output["cases"]["front_far"]["variants"]["direct_only"]["aggregate"]["WER"], 100*reverb_output["cases"]["front_far"]["variants"]["direct_plus_50ms"]["aggregate"]["WER"], 100*reverb_output["cases"]["front_far"]["variants"]["direct_plus_100ms"]["aggregate"]["WER"], 100*reverb_output["cases"]["front_far"]["variants"]["direct_plus_200ms"]["aggregate"]["WER"], 100*reverb_output["cases"]["side_left_far"]["variants"]["full_rir"]["aggregate"]["WER"], 100*reverb_output["cases"]["side_left_far"]["variants"]["direct_only"]["aggregate"]["WER"], 100*reverb_output["cases"]["side_left_far"]["variants"]["direct_plus_50ms"]["aggregate"]["WER"], 100*reverb_output["cases"]["side_left_far"]["variants"]["direct_plus_100ms"]["aggregate"]["WER"], 100*reverb_output["cases"]["side_left_far"]["variants"]["direct_plus_200ms"]["aggregate"]["WER"]))
        elif name == "FRONTEND_EFFECT":
            report_lines.append("- Observable evidence: fixed-ear WER relative to production mean_lr is front_near `{:.4f}%/{:.4f}%`, front_far `{:.4f}%/{:.4f}%`, side_left_far `{:.4f}%/{:.4f}%` for fixed_L/fixed_R. This is a sensitivity result, not a production frontend change; paired rows are in `g6_case_frontend_breakdown.json`.".format(100*baseline["front_near"]["fixed_L"]["aggregate"]["WER"], 100*baseline["front_near"]["fixed_R"]["aggregate"]["WER"], 100*baseline["front_far"]["fixed_L"]["aggregate"]["WER"], 100*baseline["front_far"]["fixed_R"]["aggregate"]["WER"], 100*baseline["side_left_far"]["fixed_L"]["aggregate"]["WER"], 100*baseline["side_left_far"]["fixed_R"]["aggregate"]["WER"]))
        elif name == "TAIL_LENGTH_EFFECT":
            report_lines.append("- Observable evidence: speech-time prefixes are byte/sample identical for every crop. Full-to-through-dry-end WER changes are front_near `{:.4f}%` to `{:.4f}%`, front_far `{:.4f}%` to `{:.4f}%`, side_left_far `{:.4f}%` to `{:.4f}%`; all crop rows and paired deltas are retained.".format(100*tail_output["cases"]["front_near"]["variants"]["full_convolution"]["aggregate"]["WER"], 100*tail_output["cases"]["front_near"]["variants"]["through_dry_end"]["aggregate"]["WER"], 100*tail_output["cases"]["front_far"]["variants"]["full_convolution"]["aggregate"]["WER"], 100*tail_output["cases"]["front_far"]["variants"]["through_dry_end"]["aggregate"]["WER"], 100*tail_output["cases"]["side_left_far"]["variants"]["full_convolution"]["aggregate"]["WER"], 100*tail_output["cases"]["side_left_far"]["variants"]["through_dry_end"]["aggregate"]["WER"]))
        elif name == "SAMPLE_SELECTION_EFFECT":
            report_lines.append("- Observable evidence: immutable 24-utterance mean_lr WER is front_near `{:.4f}%`, front_far `{:.4f}%`, side_left_far `{:.4f}%`, versus frozen-six `{:.4f}%`, `{:.4f}%`, `{:.4f}%`; all six overlap deltas are explicitly retained in the replication JSON.".format(100*replication["cases"]["front_near"]["mean_lr"]["aggregate"]["WER"], 100*replication["cases"]["front_far"]["mean_lr"]["aggregate"]["WER"], 100*replication["cases"]["side_left_far"]["mean_lr"]["aggregate"]["WER"], 100*baseline["front_near"]["mean_lr"]["aggregate"]["WER"], 100*baseline["front_far"]["mean_lr"]["aggregate"]["WER"], 100*baseline["side_left_far"]["mean_lr"]["aggregate"]["WER"]))
        else:
            report_lines.append("- Observable evidence is retained in the case/condition aggregates and every paired row in the corresponding JSON artifact.")
        report_lines.append("- Interpretation rule: {}".format(value["interpretation_rule"]))
    report_lines += ["", "## 24-utterance replication", "", "The immutable 24-utterance clean manifest was evaluated for all three frozen RIR cases and all three frontends. This is diagnostic replication only; it does not replace the frozen six-utterance G6 denominator.", ""]
    for case in G6_CASES:
        aggregate = replication["cases"][case]["mean_lr"]["aggregate"]
        report_lines.append("- `{}` mean_lr: S/D/I/N={}/{}/{}/{}; WER={:.4f}%".format(case, aggregate["S"], aggregate["D"], aggregate["I"], aggregate["N"], 100*aggregate["WER"]))
    report_lines += ["", "## Artifact identities", ""]
    for name, identity in identities.items():
        report_lines.append("- `{}`: `{}`".format(name, identity["sha256"]))
    report_lines.append("")
    report = "\n".join(report_lines)
    report_path = output / "g6_failure_attribution_report.md"
    DatasetStorage(str(output)).atomic_write_text(report_path, report)
    identities["g6_failure_attribution_report.md"] = {"path": report_path.name, "sha256": file_sha256(report_path)}
    summary = {"schema_version": SCHEMA_VERSION, "status": "A3_SERVER_RUN_FAIL_OPEN_BLOCKED_UNCHANGED", "contract_sha256": contract_sha, "metric_contract_sha256": metric_sha, "historical_g6_artifact_unchanged": True, "attribution": attribution, "artifacts": identities}
    summary_path = output / "g6_failure_attribution_summary.json"
    DatasetStorage(str(output)).atomic_write_bytes(summary_path, canonical_json_bytes(summary))
    return {"status": summary["status"], "output_dir": str(output), "artifacts": identities, "summary_sha256": file_sha256(summary_path)}
