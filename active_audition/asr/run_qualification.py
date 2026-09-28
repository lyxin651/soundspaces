"""Frozen A3 G5-G8 qualification runner.

The runner consumes pre-frozen manifests and pre-rendered controlled RIRs.  It
does not create new geometry, tune SNR, or expose reference text to the ASR
adapter.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.acoustics.resampling import resample_array
from active_audition.asr.contract import asr_contract_sha256, load_asr_contract
from active_audition.asr.frontends import apply_frontend
from active_audition.asr.qualification import (
    add_noise_at_snr,
    canonical_json_bytes,
    deterministic_broadband_noise,
    file_sha256,
    validate_model_lock,
    validate_manifest,
    waveform_identity,
)
from active_audition.asr.rir_bridge import load_rir_lock
from active_audition.asr.speechbrain_adapter import ASROutput, SpeechBrainASRAdapter
from active_audition.data.speech_registry import decoded_waveform_sha256, load_sources_config
from active_audition.data.storage import DatasetStorage
from active_audition.evaluation.asr_metrics import aggregate_error_counts, error_counts


class A3RunError(ValueError):
    """Raised when frozen qualification material is missing or inconsistent."""


def _instrument_preflight(repo: Path, contract: Mapping[str, Any]) -> Dict[str, Any]:
    from active_audition.data.noise_registry import read_noise_registry
    from active_audition.data.speech_registry import read_speech_registry, speaker_split_audit

    source_config_path = repo / contract["sources"]["config_path"]
    if file_sha256(source_config_path) != contract["sources"]["config_sha256"]:
        raise A3RunError("frozen sources config hash mismatch")
    model_lock_path = repo / contract["artifacts"]["model_lock"]
    if file_sha256(model_lock_path) != contract["artifacts"]["model_lock_sha256"]:
        raise A3RunError("frozen model lock hash mismatch")
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    validate_model_lock(model_lock)
    if model_lock["environment"] != contract["environment"]:
        raise A3RunError("model lock environment does not match frozen A3 contract")
    dependency_path = repo / model_lock["dependency_lock"]["path"]
    if file_sha256(dependency_path) != contract["environment"]["dependency_lock_sha256"]:
        raise A3RunError("frozen dependency lock hash mismatch")
    speech_identity = contract["sources"]["speech_registry"]
    noise_identity = contract["sources"]["noise_registry"]
    speech_path = repo / speech_identity["path"]
    noise_path = repo / noise_identity["path"]
    if file_sha256(speech_path) != speech_identity["sha256"]:
        raise A3RunError("frozen speech registry hash mismatch")
    if file_sha256(noise_path) != noise_identity["sha256"]:
        raise A3RunError("frozen noise registry hash mismatch")
    speech_rows = read_speech_registry(str(speech_path))
    noise_rows = read_noise_registry(str(noise_path))
    split_audit = speaker_split_audit(speech_rows)
    if split_audit["status"] != "PASS":
        raise A3RunError("frozen LibriSpeech speaker split audit failed")
    parent_ids = [str(row["parent_recording_id"]) for row in noise_rows]
    if len(set(parent_ids)) != len(parent_ids):
        raise A3RunError("MUSAN parent recording IDs are not unique")
    included_noise = [row for row in noise_rows if not row["excluded"]]
    if not included_noise or not all(row["recorded_noise_as_localized_source_approximation"] is True for row in noise_rows):
        raise A3RunError("real MUSAN noise evidence is incomplete")
    # Runtime implementation checks for the frozen G4 relations.  These are
    # not model-output checks and cannot select a frontend by WER.
    frontend_fixture = np.asarray([[1.0, 3.0], [0.25, -0.25]], dtype=np.float32)
    mean = apply_frontend(frontend_fixture, "mean_lr")
    swap_mean = apply_frontend(frontend_fixture[:, ::-1], "mean_lr")
    if not np.array_equal(mean, swap_mean) or not np.array_equal(mean, np.asarray([2.0, 0.0], dtype=np.float32)):
        raise A3RunError("frozen mean_lr frontend invariant failed")
    metric_fixture = error_counts("A B C", "A X C D")
    if tuple(metric_fixture[key] for key in ("S", "D", "I", "N")) != (1, 0, 1, 3):
        raise A3RunError("frozen ASR metric invariant failed")
    return {
        "schema_version": "active-asr-a3-instrument-provenance-v1",
        "status": "PASS",
        "model_lock": {"path": contract["artifacts"]["model_lock"], "sha256": file_sha256(model_lock_path)},
        "sources_config": {"path": contract["sources"]["config_path"], "sha256": file_sha256(source_config_path)},
        "dependency_lock": {"path": str(dependency_path.relative_to(repo)), "sha256": file_sha256(dependency_path)},
        "speech_registry": {
            "path": speech_identity["path"],
            "sha256": speech_identity["sha256"],
            "records": len(speech_rows),
            "eligible": sum(bool(row["eligible"]) for row in speech_rows),
            "speaker_split_audit": split_audit,
        },
        "noise_registry": {
            "path": noise_identity["path"],
            "sha256": noise_identity["sha256"],
            "records": len(noise_rows),
            "included": len(included_noise),
            "excluded": len(noise_rows) - len(included_noise),
            "unique_parent_recordings": len(set(parent_ids)),
            "parent_split_policy": "future_O1_O2_assignments_must_be_parent_recording_disjoint",
            "speech_leakage_not_audited": sum(row["speech_leakage_audit"]["status"] == "NOT_AUDITED" for row in noise_rows),
            "strong_reverberation_not_audited": sum(row["strong_reverberation_audit"]["status"] == "NOT_AUDITED" for row in noise_rows),
        },
        "frontend_fixture": {"mean_lr_exact": True, "swap_invariant": True, "best_ear_present": False},
        "metric_fixture": {key: metric_fixture[key] for key in ("S", "D", "I", "N", "WER", "CER")},
    }


def _load_manifest(repo: Path, contract: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    identity = contract["qualification"]["manifests"][name]
    path = repo / identity["path"]
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        raise A3RunError("{} manifest hash mismatch".format(name))
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value, name)
    if len(value["records"]) != int(identity["records"]):
        raise A3RunError("{} manifest record count mismatch".format(name))
    return value


def _decode_speech(repo: Path, source_root: Path, record: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import soundfile as sf
    except ImportError as error:
        raise A3RunError("soundfile is required in the ASR environment") from error
    path = source_root / record["relative_source_path"]
    if not path.is_file() or file_sha256(path) != record["source_file_sha256"]:
        raise A3RunError("frozen speech source identity mismatch: {}".format(path))
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
    pcm16, pcm_rate = sf.read(str(path), dtype="int16", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    equivalent_pcm = (np.asarray(pcm16, dtype=np.float32) / np.float32(32768.0)).astype(np.float32)
    if int(rate) != 16000 or int(pcm_rate) != 16000 or waveform.ndim != 1:
        raise A3RunError("frozen speech decode format mismatch: {}".format(path))
    if waveform.size != int(record["samples"]) or decoded_waveform_sha256(waveform) != record["decoded_waveform_sha256"]:
        raise A3RunError("frozen decoded speech identity mismatch: {}".format(path))
    return waveform, equivalent_pcm


def _chunks(values: Sequence[Any], size: int = 2) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


def _transcribe_many(
    adapter: SpeechBrainASRAdapter,
    waveforms: Sequence[np.ndarray],
    frontend: str,
) -> List[ASROutput]:
    outputs: List[ASROutput] = []
    for chunk in _chunks(waveforms, 2):
        outputs.extend(adapter.transcribe_batch(chunk, 16000, frontend=frontend))
    return outputs


def _evaluate(records: Sequence[Mapping[str, Any]], outputs: Sequence[ASROutput]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(records) != len(outputs):
        raise A3RunError("ASR output count mismatch")
    rows = []
    metrics = []
    for record, output in zip(records, outputs):
        metric = error_counts(str(record["normalized_transcript"]), output.hypothesis)
        metrics.append(metric)
        rows.append({
            "utterance_id": record["utterance_id"],
            "reference": record["normalized_transcript"],
            "hypothesis": output.hypothesis,
            "metric": metric,
            "asr": output.to_dict(),
        })
    return rows, aggregate_error_counts(metrics)


def _rir_index(lock: Mapping[str, Any], lock_path: Path) -> Dict[Tuple[str, int], np.ndarray]:
    result = {}
    for record in lock["records"]:
        array = np.load(str(lock_path.parent / record["relative_path"]), allow_pickle=False)
        result[(str(record["case_id"]), int(record["sample_rate_hz"]))] = np.asarray(array, dtype=np.float32)
    return result


def _clean_gate(
    repo: Path,
    source_root: Path,
    manifest: Mapping[str, Any],
    adapter: SpeechBrainASRAdapter,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    records = [record for record in manifest["records"]]
    decoded = [_decode_speech(repo, source_root, record) for record in records]
    primary = [item[0] for item in decoded]
    pcm = [item[1] for item in decoded]
    repeat_outputs = [_transcribe_many(adapter, primary, "clean_mono") for _ in range(3)]
    pcm_outputs = _transcribe_many(adapter, pcm, "equivalent_pcm_float")
    rows, aggregate = _evaluate(records, repeat_outputs[0])
    repeat_consistency = []
    format_equivalence = []
    for index, record in enumerate(records):
        hypotheses = [outputs[index].hypothesis for outputs in repeat_outputs]
        repeat_consistency.append({
            "utterance_id": record["utterance_id"],
            "hypotheses": hypotheses,
            "consistent": len(set(hypotheses)) == 1,
        })
        pcm_metric = error_counts(record["normalized_transcript"], pcm_outputs[index].hypothesis)
        primary_metric = rows[index]["metric"]
        difference = np.asarray(primary[index], dtype=np.float64) - np.asarray(pcm[index], dtype=np.float64)
        format_equivalence.append({
            "utterance_id": record["utterance_id"],
            "flac_float_sha256": waveform_identity(primary[index]),
            "pcm_float_sha256": waveform_identity(pcm[index]),
            "waveform_byte_identical": bool(np.array_equal(primary[index], pcm[index])),
            "maximum_absolute_quantization_difference": float(np.max(np.abs(difference))),
            "tolerance": 1.0 / 32768.0,
            "hypothesis_equal": repeat_outputs[0][index].hypothesis == pcm_outputs[index].hypothesis,
            "error_counts_equal": all(primary_metric[key] == pcm_metric[key] for key in ("S", "D", "I", "N")),
            "pcm_hypothesis": pcm_outputs[index].hypothesis,
            "pcm_metric": pcm_metric,
        })
    passed = (
        aggregate["WER"] <= float(contract["qualification"]["clean"]["wer_max_fraction"])
        and all(item["consistent"] for item in repeat_consistency)
        and all(item["hypothesis_equal"] and item["error_counts_equal"] for item in format_equivalence)
    )
    return {
        "schema_version": "active-asr-a3-clean-sanity-v1",
        "status": "PASS" if passed else "FAIL",
        "planned": len(records),
        "valid": len(rows),
        "full_utterance": True,
        "sample_rate_hz": 16000,
        "dtype": "float32",
        "variable_lengths": len({record["samples"] for record in records}) > 1,
        "batch_padding": contract["input"]["batch_padding"],
        "length_semantics": contract["input"]["length_semantics"],
        "aggregate": aggregate,
        "threshold": {"wer_max_fraction": contract["qualification"]["clean"]["wer_max_fraction"]},
        "repeat_consistency": repeat_consistency,
        "format_equivalence": format_equivalence,
        "rows": rows,
    }


def _domain_gate(
    repo: Path,
    source_root: Path,
    manifest: Mapping[str, Any],
    rir: Mapping[Tuple[str, int], np.ndarray],
    adapter: SpeechBrainASRAdapter,
    contract: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    cache: Dict[str, np.ndarray] = {}
    generated = []
    for record in manifest["records"]:
        speech = record["speech"]
        dry = _decode_speech(repo, source_root, speech)[0]
        case_id = str(record["rir_case"]["case_id"])
        binaural = convolve_binaural(dry, rir[(case_id, 16000)])
        cache[str(record["record_id"])] = binaural
        generated.append((record, binaural))
    rows_by_frontend = {}
    aggregate_by_frontend = {}
    for frontend in ("mean_lr", "fixed_L", "fixed_R"):
        waveforms = [apply_frontend(binaural, frontend) for _, binaural in generated]
        outputs = _transcribe_many(adapter, waveforms, frontend)
        references = [dict(record["speech"]) for record, _ in generated]
        rows, aggregate = _evaluate(references, outputs)
        for row, (record, binaural), mono in zip(rows, generated, waveforms):
            row["record_id"] = record["record_id"]
            row["rir_case_id"] = record["rir_case"]["case_id"]
            row["rir_sha256"] = hashlib.sha256(np.ascontiguousarray(rir[(record["rir_case"]["case_id"], 16000)], dtype="<f4").tobytes()).hexdigest()
            row["binaural_waveform_sha256"] = hashlib.sha256(np.ascontiguousarray(binaural, dtype="<f4").tobytes()).hexdigest()
            row["mono_waveform_sha256"] = waveform_identity(mono)
            row["normalization"] = "none"
        rows_by_frontend[frontend] = rows
        aggregate_by_frontend[frontend] = aggregate
    production_rows = rows_by_frontend["mean_lr"]
    nonempty = sum(bool(row["hypothesis"].strip()) for row in production_rows)
    fraction = float(nonempty) / float(len(production_rows))
    aggregate = aggregate_by_frontend["mean_lr"]
    passed = (
        len(production_rows) == len(manifest["records"])
        and fraction >= float(contract["qualification"]["ss2_domain"]["minimum_nonempty_fraction"])
        and aggregate["WER"] <= float(contract["qualification"]["ss2_domain"]["wer_max_fraction"])
    )
    return ({
        "schema_version": "active-asr-a3-ss2-domain-sanity-v1",
        "status": "PASS" if passed else "FAIL",
        "planned": len(manifest["records"]),
        "valid": len(production_rows),
        "runtime_errors": 0,
        "production_frontend": "mean_lr",
        "nonempty_hypotheses": nonempty,
        "nonempty_fraction": fraction,
        "aggregate": aggregate,
        "thresholds": {
            "minimum_nonempty_fraction": contract["qualification"]["ss2_domain"]["minimum_nonempty_fraction"],
            "wer_max_fraction": contract["qualification"]["ss2_domain"]["wer_max_fraction"],
        },
        "post_convolution_normalization": "none",
        "frontends": {
            name: {"role": "hard_gate" if name == "mean_lr" else "fixed_sensitivity_diagnostic", "aggregate": aggregate_by_frontend[name], "rows": rows_by_frontend[name]}
            for name in ("mean_lr", "fixed_L", "fixed_R")
        },
    }, cache)


def _snr_gate(
    manifest: Mapping[str, Any],
    domain_cache: Mapping[str, np.ndarray],
    adapter: SpeechBrainASRAdapter,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    conditions = ("reverb_only", 10, 0, -10)
    by_condition = {str(condition): [] for condition in conditions}
    references = []
    diagnostics = []
    for record in manifest["records"]:
        record_id = str(record["record_id"])
        target = apply_frontend(domain_cache[record_id], "mean_lr")
        noise = deterministic_broadband_noise(target.size, int(record["noise_seed"]))
        references.append(dict(record["speech"]))
        row_diag = {
            "record_id": record_id,
            "utterance_id": record["speech"]["utterance_id"],
            "target_sha256": waveform_identity(target),
            "noise_sha256": waveform_identity(noise),
            "conditions": {},
        }
        by_condition["reverb_only"].append(target)
        row_diag["conditions"]["reverb_only"] = {"waveform_sha256": waveform_identity(target), "normalization": "none"}
        for level in (10, 0, -10):
            mixture, scaling = add_noise_at_snr(target, noise, float(level))
            by_condition[str(level)].append(mixture)
            row_diag["conditions"][str(level)] = {
                "waveform_sha256": waveform_identity(mixture),
                "scaling": scaling,
            }
        diagnostics.append(row_diag)
    results = {}
    for condition in conditions:
        key = str(condition)
        outputs = _transcribe_many(adapter, by_condition[key], "mean_lr")
        rows, aggregate = _evaluate(references, outputs)
        results[key] = {"aggregate": aggregate, "rows": rows}
    high = results["10"]["aggregate"]
    low = results["-10"]["aggregate"]
    delta_pp = (float(low["WER"]) - float(high["WER"])) * 100.0
    high_edits = int(high["S"] + high["D"] + high["I"])
    low_edits = int(low["S"] + low["D"] + low["I"])
    passed = (
        all(len(results[str(condition)]["rows"]) == len(references) for condition in conditions)
        and low["WER"] > high["WER"]
        and delta_pp >= float(contract["qualification"]["snr_sensitivity"]["minimum_low_high_wer_delta_percentage_points"])
        and low_edits > high_edits
    )
    return {
        "schema_version": "active-asr-a3-snr-sensitivity-result-v1",
        "status": "PASS" if passed else "FAIL",
        "planned_per_condition": len(references),
        "conditions": results,
        "low_minus_high_wer_percentage_points": delta_pp,
        "high_edit_count": high_edits,
        "low_edit_count": low_edits,
        "hard_gate": {
            "low_wer_greater_than_high": low["WER"] > high["WER"],
            "minimum_delta_percentage_points": contract["qualification"]["snr_sensitivity"]["minimum_low_high_wer_delta_percentage_points"],
            "delta_pass": delta_pp >= float(contract["qualification"]["snr_sensitivity"]["minimum_low_high_wer_delta_percentage_points"]),
            "low_edit_count_greater_than_high": low_edits > high_edits,
        },
        "diagnostics": diagnostics,
        "scope": "A3_qualification_only_not_A4_SNR_calibration",
    }


def _q4_reference_plan(manifest: Mapping[str, Any]) -> Tuple[List[Mapping[str, Any]], List[Dict[str, Any]]]:
    """Expand Q4 references in the same order as the waveform plan."""

    if not manifest["records"]:
        raise A3RunError("Q4 manifest is empty")
    references = []
    decode_meta = []
    for record in manifest["records"]:
        speech = record["speech"]
        for frontend in ("mean_lr", "fixed_L", "fixed_R"):
            references.append(dict(speech))
            decode_meta.append({
                "record_id": record["record_id"],
                "utterance_id": speech["utterance_id"],
                "case_id": str(record["rir_case"]["case_id"]),
                "frontend": frontend,
            })
    return references, decode_meta


def _q4_bridge_evidence(
    repo: Path,
    source_root: Path,
    manifest: Mapping[str, Any],
    rir: Mapping[Tuple[str, int], np.ndarray],
    adapter: SpeechBrainASRAdapter,
) -> Dict[str, Any]:
    references, decode_meta = _q4_reference_plan(manifest)
    path_waveforms = {"A_native16": [], "B_native24_to16": []}
    for record in manifest["records"]:
        speech = record["speech"]
        dry16 = _decode_speech(repo, source_root, speech)[0]
        case_id = str(record["rir_case"]["case_id"])
        path_a_binaural = convolve_binaural(dry16, rir[(case_id, 16000)])
        dry24 = resample_array(dry16, 16000, 24000)
        path_b_native = convolve_binaural(dry24, rir[(case_id, 24000)])
        path_b_binaural = resample_array(path_b_native, 24000, 16000)
        for frontend in ("mean_lr", "fixed_L", "fixed_R"):
            path_a = apply_frontend(path_a_binaural, frontend)
            path_b = apply_frontend(path_b_binaural, frontend)
            path_waveforms["A_native16"].append(path_a)
            path_waveforms["B_native24_to16"].append(path_b)
    for meta, path_a, path_b in zip(
        decode_meta,
        path_waveforms["A_native16"],
        path_waveforms["B_native24_to16"],
    ):
        meta.update({
            "path_a_waveform_sha256": waveform_identity(path_a),
            "path_b_waveform_sha256": waveform_identity(path_b),
            "path_a_samples": int(path_a.size),
            "path_b_samples": int(path_b.size),
            "separate_normalization": False,
        })
    outputs_a: List[Any] = [None] * len(decode_meta)
    outputs_b: List[Any] = [None] * len(decode_meta)
    for frontend in ("mean_lr", "fixed_L", "fixed_R"):
        indexes = [index for index, meta in enumerate(decode_meta) if meta["frontend"] == frontend]
        decoded_a = _transcribe_many(adapter, [path_waveforms["A_native16"][index] for index in indexes], frontend)
        decoded_b = _transcribe_many(adapter, [path_waveforms["B_native24_to16"][index] for index in indexes], frontend)
        for index, output_a, output_b in zip(indexes, decoded_a, decoded_b):
            outputs_a[index] = output_a
            outputs_b[index] = output_b
    rows_a, aggregate_a = _evaluate(references, outputs_a)
    rows_b, aggregate_b = _evaluate(references, outputs_b)
    paired = []
    pose_scores: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for meta, row_a, row_b in zip(decode_meta, rows_a, rows_b):
        paired.append({
            **meta,
            "path_a": row_a,
            "path_b": row_b,
            "hypothesis_equal": row_a["hypothesis"] == row_b["hypothesis"],
            "error_count_delta_B_minus_A": {
                key: int(row_b["metric"][key]) - int(row_a["metric"][key]) for key in ("S", "D", "I")
            },
        })
        key = (meta["utterance_id"], meta["frontend"])
        pose_scores.setdefault(key, {})[meta["case_id"]] = {
            "A": float(row_a["metric"]["WER"]),
            "B": float(row_b["metric"]["WER"]),
        }
    ordering = []
    for key, scores in sorted(pose_scores.items()):
        order_a = sorted(scores, key=lambda case: (scores[case]["A"], case))
        order_b = sorted(scores, key=lambda case: (scores[case]["B"], case))
        ordering.append({
            "utterance_id": key[0],
            "frontend": key[1],
            "path_a_pose_order": order_a,
            "path_b_pose_order": order_b,
            "path_a_best_pose": order_a[0],
            "path_b_best_pose": order_b[0],
            "best_pose_identity_equal": order_a[0] == order_b[0],
        })
    return {
        "schema_version": "active-asr-a3-q4-bridge-result-v1",
        "status": "EVIDENCE_COMPLETE_REVIEW_REQUIRED",
        "decision_authority": "reviewer",
        "separate_normalization": False,
        "path_a": {"description": "native_SS2_16khz", "aggregate": aggregate_a},
        "path_b": {"description": "native_SS2_24khz_resample_poly_to_16khz", "aggregate": aggregate_b},
        "paired": paired,
        "pose_ordering": ordering,
    }


def _q4_bridge(
    repo: Path,
    source_root: Path,
    manifest: Mapping[str, Any],
    rir: Mapping[Tuple[str, int], np.ndarray],
    adapter: SpeechBrainASRAdapter,
) -> Dict[str, Any]:
    provenance = manifest["records"][0]["a2_q2_q3_provenance"]
    for name in ("q2_sample_rate_ab", "q3_rir_sample_rate_convention"):
        identity = provenance[name]
        path = repo / identity["path"]
        if not path.is_file() or file_sha256(path) != identity["sha256"]:
            raise A3RunError("Q4 {} provenance integrity failure".format(name))
    return {
        "schema_version": "active-asr-a3-q4-bridge-result-v1",
        "status": "EVIDENCE_COMPLETE_REVIEW_REQUIRED",
        "decision_authority": "reviewer",
        "separate_normalization": False,
        **_q4_bridge_evidence(repo, source_root, manifest, rir, adapter),
        "a2_provenance": provenance,
    }


def run_a3_qualification(
    contract_path: str,
    rir_lock_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    repo = Path(".").resolve()
    contract = load_asr_contract(contract_path, require_frozen=True)
    contract_sha = asr_contract_sha256(contract)
    sources = load_sources_config(str(repo / contract["sources"]["config_path"]))
    source_root = repo / sources["librispeech"]["root"]
    manifests = {name: _load_manifest(repo, contract, name) for name in ("clean", "ss2_domain", "snr_sensitivity", "q4")}
    rir_lock, rir_lock_file = load_rir_lock(rir_lock_path, contract_sha)
    rir = _rir_index(rir_lock, rir_lock_file)
    instrument = _instrument_preflight(repo, contract)
    adapter = SpeechBrainASRAdapter(contract)

    clean = _clean_gate(repo, source_root, manifests["clean"], adapter, contract)
    domain, domain_cache = _domain_gate(repo, source_root, manifests["ss2_domain"], rir, adapter, contract)
    snr = _snr_gate(manifests["snr_sensitivity"], domain_cache, adapter, contract)
    q4 = _q4_bridge(repo, source_root, manifests["q4"], rir, adapter)

    output = Path(output_dir).resolve()
    storage = DatasetStorage(str(output))
    artifacts = {
        "instrument_provenance.json": instrument,
        "clean_complete_utterance_sanity.json": clean,
        "ss2_domain_sanity.json": domain,
        "snr_sensitivity.json": snr,
        "q4_bridge.json": q4,
    }
    identities = {}
    for name, value in artifacts.items():
        path = output / name
        storage.atomic_write_bytes(path, canonical_json_bytes(value))
        identities[name] = {"path": name, "sha256": file_sha256(path)}
    gates = {
        "A3-G1_model_decoder_provenance": "PASS",
        "A3-G2_librispeech_registry_split_audit": "PASS",
        "A3-G3_musan_registry_parent_provenance": "PASS",
        "A3-G4_frontends_text_wer_contract": "PASS",
        "A3-G5_clean_complete_utterance_sanity": clean["status"],
        "A3-G6_ss2_domain_sanity": domain["status"],
        "A3-G7_snr_sensitivity": snr["status"],
        "A3-G8_q4_bridge": q4["status"],
    }
    g1_g7_pass = all(value == "PASS" for key, value in gates.items() if not key.startswith("A3-G8"))
    report = "\n".join([
        "# Active-ASR V1.1 A3 ASR Instrument & Source Qualification",
        "",
        "- Status: **{}**".format("READY_FOR_Q4_REVIEW" if g1_g7_pass else "OPEN_BLOCKED"),
        "- A3 contract SHA256: `{}`".format(contract_sha),
        "- G5 clean WER: `{:.6f}`".format(clean["aggregate"]["WER"]),
        "- G6 SS2 mean_lr WER: `{:.6f}`; non-empty fraction: `{:.6f}`".format(domain["aggregate"]["WER"], domain["nonempty_fraction"]),
        "- G7 +10/-10 WER: `{:.6f}` / `{:.6f}`; delta pp: `{:.3f}`".format(snr["conditions"]["10"]["aggregate"]["WER"], snr["conditions"]["-10"]["aggregate"]["WER"], snr["low_minus_high_wer_percentage_points"]),
        "- G8: evidence complete; reviewer decision required.",
        "- A3 is not CLOSED and A4 remains CLOSED.",
        "",
    ])
    report_path = output / "a3_qualification_report.md"
    storage.atomic_write_text(report_path, report)
    identities["a3_qualification_report.md"] = {"path": "a3_qualification_report.md", "sha256": file_sha256(report_path)}
    summary = {
        "schema_version": "active-asr-a3-qualification-summary-v1",
        "gate": "A3",
        "status": "READY_FOR_Q4_REVIEW" if g1_g7_pass else "OPEN_BLOCKED",
        "a3_closed": False,
        "a4_opened": False,
        "asr_contract_sha256": contract_sha,
        "rir_lock_sha256": file_sha256(rir_lock_file),
        "acceptance_matrix": gates,
        "artifacts": identities,
        "blocker": None if g1_g7_pass else "ONE_OR_MORE_A3_G1_TO_G7_HARD_GATES_FAILED",
    }
    summary_path = output / "a3_qualification_summary.json"
    storage.atomic_write_bytes(summary_path, canonical_json_bytes(summary))
    returned = dict(summary)
    returned["summary_artifact"] = {"path": str(summary_path), "sha256": file_sha256(summary_path)}
    return returned
