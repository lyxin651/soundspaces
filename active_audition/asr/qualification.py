"""Pure A3 qualification primitives shared by frozen runners and tests."""

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


A3_MANIFEST_SCHEMA_VERSIONS = {
    "clean": "active-asr-a3-clean-qualification-v1",
    "ss2_domain": "active-asr-a3-ss2-domain-qualification-v1",
    "snr_sensitivity": "active-asr-a3-snr-sensitivity-v1",
    "q4": "active-asr-a3-q4-bridge-v1",
}


class A3QualificationError(ValueError):
    """Raised when a frozen A3 fixture or qualification input is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_manifest(value: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    if kind not in A3_MANIFEST_SCHEMA_VERSIONS:
        raise A3QualificationError("unknown A3 manifest kind: {}".format(kind))
    if not isinstance(value, Mapping):
        raise A3QualificationError("manifest must be a mapping")
    keys = ("schema_version", "kind", "selection_policy", "records")
    unknown = sorted(set(value) - set(keys))
    missing = [key for key in keys if key not in value]
    if unknown or missing:
        raise A3QualificationError("invalid {} manifest keys: unknown={} missing={}".format(kind, unknown, missing))
    if value["schema_version"] != A3_MANIFEST_SCHEMA_VERSIONS[kind] or value["kind"] != kind:
        raise A3QualificationError("invalid {} manifest identity".format(kind))
    if not isinstance(value["selection_policy"], Mapping) or not isinstance(value["records"], list):
        raise A3QualificationError("manifest selection_policy/records have invalid types")
    speech_keys = {
        "utterance_id", "split", "speaker_id", "chapter_id", "relative_source_path",
        "source_file_sha256", "decoded_waveform_sha256", "samples", "duration_sec",
        "normalized_transcript", "reference_word_count", "complete_utterance",
    }
    rir_keys = {
        "case_id", "geometry_id", "geometry_registry_sha256", "mesh_sha256",
        "receiver_sensor_position_world", "listener_yaw_deg", "source_position_world",
        "relative_azimuth_deg", "distance_m", "line_of_sight", "runtime_config_path",
        "runtime_config_sha256", "a2_runtime_lock_path", "a2_runtime_lock_sha256",
        "expected_effective_acoustics",
    }
    record_keys = {
        "clean": speech_keys,
        "ss2_domain": {"record_id", "speech", "rir_case", "sample_rate_hz", "frontends", "noise", "post_convolution_normalization", "convolution_time_axis"},
        "snr_sensitivity": {"record_id", "speech", "rir_case", "sample_rate_hz", "frontend", "conditions_db", "noise_algorithm", "noise_seed", "same_noise_realization_across_levels", "postmix_normalization", "a4_initial_pose_snr_calibration"},
        "q4": {"record_id", "speech", "rir_case", "frontends", "path_a", "path_b", "separate_normalization", "a2_q2_q3_provenance"},
    }
    for index, record in enumerate(value["records"]):
        path = "records[{}]".format(index)
        if not isinstance(record, Mapping) or set(record) != record_keys[kind]:
            raise A3QualificationError("{} {} fields are invalid".format(kind, path))
        if kind != "clean":
            if not isinstance(record["speech"], Mapping) or set(record["speech"]) != speech_keys:
                raise A3QualificationError("{}.speech fields are invalid".format(path))
            if not isinstance(record["rir_case"], Mapping) or set(record["rir_case"]) != rir_keys:
                raise A3QualificationError("{}.rir_case fields are invalid".format(path))
    return value


def manifest_sha256(value: Mapping[str, Any], kind: str) -> str:
    validate_manifest(value, kind)
    return sha256_bytes(canonical_json_bytes(value))


def write_manifest(path: str, value: Mapping[str, Any], kind: str) -> str:
    validate_manifest(value, kind)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    target.write_bytes(payload)
    return sha256_bytes(payload)


def deterministic_broadband_noise(length: int, seed: int = 20260927) -> np.ndarray:
    if isinstance(length, bool) or int(length) <= 0:
        raise A3QualificationError("noise length must be positive")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return generator.standard_normal(int(length), dtype=np.float32)


def add_noise_at_snr(
    target: Any,
    unscaled_noise: Any,
    snr_db: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Scale one fixed realization by full-waveform mean-square; no postmix gain."""

    speech = np.asarray(target, dtype=np.float32)
    noise = np.asarray(unscaled_noise, dtype=np.float32)
    if speech.ndim != 1 or noise.shape != speech.shape or speech.size == 0:
        raise A3QualificationError("target and noise must be same-length non-empty mono vectors")
    if not np.isfinite(speech).all() or not np.isfinite(noise).all():
        raise A3QualificationError("target/noise contains NaN/Inf")
    target_power = float(np.mean(np.square(speech, dtype=np.float64)))
    raw_noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    if target_power <= 0.0 or raw_noise_power <= 0.0:
        raise A3QualificationError("target and noise power must be positive")
    required_noise_power = target_power / (10.0 ** (float(snr_db) / 10.0))
    scale = math.sqrt(required_noise_power / raw_noise_power)
    scaled_noise = (noise * np.float32(scale)).astype(np.float32)
    mixture = (speech + scaled_noise).astype(np.float32)
    achieved_noise_power = float(np.mean(np.square(scaled_noise, dtype=np.float64)))
    achieved_snr = 10.0 * math.log10(target_power / achieved_noise_power)
    return mixture, {
        "target_power": target_power,
        "unscaled_noise_power": raw_noise_power,
        "noise_scale": scale,
        "scaled_noise_power": achieved_noise_power,
        "requested_snr_db": float(snr_db),
        "achieved_snr_db": achieved_snr,
        "postmix_normalization": "none",
    }


def waveform_identity(waveform: Any) -> str:
    value = np.ascontiguousarray(np.asarray(waveform, dtype="<f4"))
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise A3QualificationError("waveform identity requires finite non-empty mono float32")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_lock(value: Mapping[str, Any]) -> Mapping[str, Any]:
    keys = {
        "schema_version",
        "repo_id",
        "resolved_revision",
        "local_root",
        "loaded_files",
        "environment",
        "dependency_lock",
        "discovery",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise A3QualificationError("model lock has invalid top-level fields")
    if value["schema_version"] != "active-asr-a3-model-lock-v1":
        raise A3QualificationError("model lock schema_version is invalid")
    if not isinstance(value["loaded_files"], list) or not value["loaded_files"]:
        raise A3QualificationError("model lock loaded_files must be non-empty")
    for item in value["loaded_files"]:
        if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256", "role"}:
            raise A3QualificationError("model lock loaded file has invalid fields")
        if len(str(item["sha256"])) != 64 or int(item["bytes"]) <= 0:
            raise A3QualificationError("model lock loaded file identity is invalid")
    dependency = value["dependency_lock"]
    if not isinstance(dependency, Mapping) or set(dependency) != {"path", "sha256"} or len(str(dependency["sha256"])) != 64:
        raise A3QualificationError("model lock dependency identity is invalid")
    return value


def _qualification_speech_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "utterance_id": row["utterance_id"],
        "split": row["split"],
        "speaker_id": row["speaker_id"],
        "chapter_id": row["chapter_id"],
        "relative_source_path": row["relative_source_path"],
        "source_file_sha256": row["source_file_sha256"],
        "decoded_waveform_sha256": row["decoded_waveform_sha256"],
        "samples": row["samples"],
        "duration_sec": row["duration_sec"],
        "normalized_transcript": row["normalized_transcript"],
        "reference_word_count": row["reference_word_count"],
        "complete_utterance": True,
    }


def prepare_a3_freeze_material(
    contract_path: str,
    sources_path: str,
    registry_dir: str,
    dependency_lock_path: str,
    repo_root: str = ".",
) -> Dict[str, Any]:
    """Build real registries and pre-WER manifests from metadata only.

    This function never imports or calls SpeechBrain and cannot inspect WER.
    It is safe to run during DISCOVER/FREEZE but not a qualification runner.
    """

    from active_audition.asr.contract import load_asr_contract
    from active_audition.data.noise_registry import build_noise_registry, write_noise_registry
    from active_audition.data.speech_registry import (
        build_speech_registry,
        load_sources_config,
        select_clean_qualification,
        speaker_split_audit,
        write_jsonl,
    )
    from active_audition.receiver.geometry import load_geometry_registry, source_position_world

    repo = Path(repo_root).resolve()
    contract = load_asr_contract(str(repo / contract_path) if not Path(contract_path).is_absolute() else contract_path)
    if contract["contract"]["state"] != "DRAFT":
        raise A3QualificationError("freeze material may only be regenerated while contract.state=DRAFT")
    resolved_sources_path = repo / sources_path if not Path(sources_path).is_absolute() else Path(sources_path)
    if file_sha256(resolved_sources_path) != contract["sources"]["config_sha256"]:
        raise A3QualificationError("sources config does not match the A3 contract")
    source_config = load_sources_config(str(resolved_sources_path))
    target = Path(registry_dir)
    if not target.is_absolute():
        target = repo / target
    target.mkdir(parents=True, exist_ok=True)

    speech_rows = build_speech_registry(str(repo / source_config["librispeech"]["root"]))
    split_audit = speaker_split_audit(speech_rows)
    if split_audit["status"] != "PASS":
        raise A3QualificationError("LibriSpeech O1/O2 speaker split audit failed")
    speech_path = target / "librispeech.jsonl"
    speech_sha = write_jsonl(str(speech_path), speech_rows)

    noise_audit = contract["sources"]["noise_technical_audit"]
    noise_rows = build_noise_registry(
        str(repo / source_config["musan"]["root"]),
        minimum_duration_sec=float(noise_audit["minimum_duration_sec"]),
        extreme_silence_rms=float(noise_audit["extreme_silence_rms_max"]),
    )
    noise_path = target / "musan_noise.jsonl"
    noise_sha = write_noise_registry(str(noise_path), noise_rows)

    clean_selected = select_clean_qualification(speech_rows, per_split=12)
    clean_records = [_qualification_speech_record(row) for row in clean_selected]
    clean_manifest = {
        "schema_version": A3_MANIFEST_SCHEMA_VERSIONS["clean"],
        "kind": "clean",
        "selection_policy": {
            "selection_time": "before_any_formal_WER",
            "method": "metadata_only_lexicographic_round_robin_by_speaker",
            "splits": ["dev-clean", "dev-other"],
            "per_split": 12,
            "duration_sec": [4.0, 15.0],
            "minimum_reference_words": 10,
            "complete_utterance": True,
            "wer_or_model_output_used": False,
        },
        "records": clean_records,
    }
    clean_path = target / "clean_qualification_manifest.json"
    clean_sha = write_manifest(str(clean_path), clean_manifest, "clean")

    geometry_path = repo / "registries/active_asr_a2/qualification_geometry.yaml"
    geometry_registry = load_geometry_registry(str(geometry_path), str(repo))
    geometry = next(item for item in geometry_registry["geometries"] if item["id"] == "a2_symmetric_shoebox_v1")
    a2_runtime_lock_path = repo / "runs/active_asr_v1/a2_failure_attribution_run3/runtime.lock.json"
    if not a2_runtime_lock_path.is_file():
        raise A3QualificationError("frozen A2 runtime lock is missing")
    a2_runtime_lock = json.loads(a2_runtime_lock_path.read_text(encoding="utf-8"))
    expected_acoustics = dict(a2_runtime_lock["receiver_effective"])
    receiver = list(geometry["receiver"]["sensor_position_world"])
    yaw = float(geometry["receiver"]["yaw_deg"])
    geometry_cases = []
    for case_id, distance, angle in (
        ("front_near", 1.0, 0.0),
        ("front_far", 4.0, 0.0),
        ("side_left_far", 4.0, 60.0),
    ):
        geometry_cases.append({
            "case_id": case_id,
            "geometry_id": geometry["id"],
            "geometry_registry_sha256": geometry_registry["registry_sha256"],
            "mesh_sha256": geometry["mesh_sha256"],
            "receiver_sensor_position_world": receiver,
            "listener_yaw_deg": yaw,
            "source_position_world": list(source_position_world(receiver, yaw, angle, distance)),
            "relative_azimuth_deg": angle,
            "distance_m": distance,
            "line_of_sight": "LOS",
            "runtime_config_path": "configs/active_audition/v0_replica_debug.yaml",
            "runtime_config_sha256": file_sha256(repo / "configs/active_audition/v0_replica_debug.yaml"),
            "a2_runtime_lock_path": "runs/active_asr_v1/a2_failure_attribution_run3/runtime.lock.json",
            "a2_runtime_lock_sha256": file_sha256(a2_runtime_lock_path),
            "expected_effective_acoustics": expected_acoustics,
        })
    # Six utterances are frozen for domain/SNR qualification: the first three
    # already-selected records per source split.  This remains metadata-only.
    domain_speech = []
    for split in ("dev-clean", "dev-other"):
        domain_speech.extend([row for row in clean_selected if row["split"] == split][:3])
    ss2_records = []
    for speech in domain_speech:
        for geometry_case in geometry_cases:
            ss2_records.append({
                "record_id": "{}__{}".format(speech["utterance_id"], geometry_case["case_id"]),
                "speech": _qualification_speech_record(speech),
                "rir_case": geometry_case,
                "sample_rate_hz": 16000,
                "frontends": ["mean_lr", "fixed_L", "fixed_R"],
                "noise": "none",
                "post_convolution_normalization": "none",
                "convolution_time_axis": "full",
            })
    ss2_manifest = {
        "schema_version": A3_MANIFEST_SCHEMA_VERSIONS["ss2_domain"],
        "kind": "ss2_domain",
        "selection_policy": {
            "selection_time": "before_any_formal_WER",
            "speech_selection": "first_three_frozen_clean_records_per_split",
            "geometry_selection": "A2_authoritative_preexisting_cases_only",
            "result_friendly_selection": False,
        },
        "records": ss2_records,
    }
    ss2_path = target / "ss2_domain_qualification_manifest.json"
    ss2_sha = write_manifest(str(ss2_path), ss2_manifest, "ss2_domain")

    front_near = geometry_cases[0]
    snr_records = [{
        "record_id": speech["utterance_id"] + "__front_near",
        "speech": _qualification_speech_record(speech),
        "rir_case": front_near,
        "sample_rate_hz": 16000,
        "frontend": "mean_lr",
        "conditions_db": ["reverb_only", 10, 0, -10],
        "noise_algorithm": "numpy_pcg64_standard_normal_float32",
        "noise_seed": 20260927,
        "same_noise_realization_across_levels": True,
        "postmix_normalization": "none",
        "a4_initial_pose_snr_calibration": False,
    } for speech in domain_speech]
    snr_manifest = {
        "schema_version": A3_MANIFEST_SCHEMA_VERSIONS["snr_sensitivity"],
        "kind": "snr_sensitivity",
        "selection_policy": {
            "selection_time": "before_any_formal_WER",
            "purpose": "qualification_only_instrument_response",
            "localized_noise": False,
            "o1_snr_selection": False,
        },
        "records": snr_records,
    }
    snr_path = target / "snr_sensitivity_manifest.json"
    snr_sha = write_manifest(str(snr_path), snr_manifest, "snr_sensitivity")

    q4_speech = clean_selected[:2] + [row for row in clean_selected if row["split"] == "dev-other"][:2]
    q4_records = []
    for speech in q4_speech:
        for geometry_case in geometry_cases:
            q4_records.append({
                "record_id": "{}__{}".format(speech["utterance_id"], geometry_case["case_id"]),
                "speech": _qualification_speech_record(speech),
                "rir_case": geometry_case,
                "frontends": ["mean_lr", "fixed_L", "fixed_R"],
                "path_a": {"renderer": "native_SS2", "sample_rate_hz": 16000},
                "path_b": {"renderer": "native_SS2", "sample_rate_hz": 24000, "resample_to_hz": 16000, "resampler": "resample_poly"},
                "separate_normalization": False,
                "a2_q2_q3_provenance": {
                    "metric_v2_sha256": "f1185d2c5fe81091199d5525f224822a3227c700c60bba9ab168e6d9ddc38f83",
                    "oracle_v3_sha256": "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c",
                    "q2_sample_rate_ab": {
                        "path": "runs/active_asr_v1/a2_failure_attribution_run3/sample_rate_ab.json",
                        "sha256": "83da855c380622d7f59a785c007948992013c084a68058ddbeb0b696b002d217",
                    },
                    "q3_rir_sample_rate_convention": {
                        "path": "runs/active_asr_v1/a2_sample_rate_blocker_calibration_run5/rir_sample_rate_convention_calibration.json",
                        "sha256": "418a12465a7262bde1a1351cca2be6ad1605442b022f578f866f3a8f28091f9a",
                    },
                },
            })
    q4_manifest = {
        "schema_version": A3_MANIFEST_SCHEMA_VERSIONS["q4"],
        "kind": "q4",
        "selection_policy": {
            "selection_time": "before_any_formal_WER",
            "speech_selection": "first_two_frozen_clean_records_per_split",
            "geometry_selection": "same_three_A2_controlled_cases_as_G6",
            "reviewer_decides_sensitivity": True,
        },
        "records": q4_records,
    }
    q4_path = target / "q4_manifest.json"
    q4_sha = write_manifest(str(q4_path), q4_manifest, "q4")

    dependency_path = Path(dependency_lock_path)
    if not dependency_path.is_absolute():
        dependency_path = repo / dependency_path
    if not dependency_path.is_file():
        raise A3QualificationError("dependency lock is missing: {}".format(dependency_path))
    model_root = repo / contract["model"]["local_root"]
    roles = {
        "hyperparams.yaml": "hyperparams",
        "asr.ckpt": "asr_weights",
        "lm.ckpt": "language_model",
        "tokenizer.ckpt": "tokenizer",
        "normalizer.ckpt": "normalizer",
    }
    loaded_files = []
    for name, role in roles.items():
        path = model_root / name
        loaded_files.append({"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path), "role": role})
    model_lock = {
        "schema_version": "active-asr-a3-model-lock-v1",
        "repo_id": contract["model"]["repo_id"],
        "resolved_revision": contract["model"]["resolved_revision"],
        "local_root": contract["model"]["local_root"],
        "loaded_files": loaded_files,
        "environment": {
            **dict(contract["environment"]),
            "dependency_lock_sha256": file_sha256(dependency_path),
        },
        "dependency_lock": {"path": str(dependency_path.relative_to(repo)), "sha256": file_sha256(dependency_path)},
        "discovery": {
            "formal_wer_run_before_freeze": False,
            "speechbrain_installed_in_renderer_ss_env": False,
            "asr_conda_env": contract["environment"]["conda_env"],
        },
    }
    validate_model_lock(model_lock)
    model_lock_path = target / "model_lock.json"
    model_lock_path.write_bytes(canonical_json_bytes(model_lock))

    return {
        "status": "FREEZE_MATERIAL_PREPARED_NO_WER_RUN",
        "speech_registry": {"path": str(speech_path.relative_to(repo)), "sha256": speech_sha, "records": len(speech_rows), "eligible": sum(bool(row["eligible"]) for row in speech_rows)},
        "speaker_split_audit": split_audit,
        "noise_registry": {"path": str(noise_path.relative_to(repo)), "sha256": noise_sha, "records": len(noise_rows), "included": sum(not bool(row["excluded"]) for row in noise_rows)},
        "manifests": {
            "clean": {"path": str(clean_path.relative_to(repo)), "sha256": clean_sha, "records": len(clean_records)},
            "ss2_domain": {"path": str(ss2_path.relative_to(repo)), "sha256": ss2_sha, "records": len(ss2_records)},
            "snr_sensitivity": {"path": str(snr_path.relative_to(repo)), "sha256": snr_sha, "records": len(snr_records)},
            "q4": {"path": str(q4_path.relative_to(repo)), "sha256": q4_sha, "records": len(q4_records)},
        },
        "model_lock": {"path": str(model_lock_path.relative_to(repo)), "sha256": file_sha256(model_lock_path)},
        "dependency_lock": {"path": str(dependency_path.relative_to(repo)), "sha256": file_sha256(dependency_path)},
    }
