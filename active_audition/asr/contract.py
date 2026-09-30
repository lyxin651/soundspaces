"""Strict, canonical Active-ASR A3 instrument contract.

The contract is deliberately independent of SpeechBrain imports so that it can
be validated in the legacy renderer environment.  A DRAFT may contain explicit
``PENDING`` identities while sources are being discovered; a FROZEN contract
may not.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


A3_CONTRACT_VERSION = "active-asr-a3-instrument-v1"
A3_GATE = "A3"
A0_PARENT_SHA256 = "d731393cda3ddb29f0bdf58249f104da59f29d012b976eeb2de1f160e1df8107"
A2_V3_PARENT_SHA256 = "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ASRContractError(ValueError):
    """Raised when the A3 contract is incomplete or inconsistent."""


def _path(path: str, key: Any) -> str:
    return "{}.{}".format(path, key) if path else str(key)


def _mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ASRContractError("{} must be a mapping".format(path))
    return value


def _only(value: Mapping, keys: Iterable[str], path: str) -> None:
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise ASRContractError(
            "unknown A3 contract field(s) at {}: {}".format(path, ", ".join(unknown))
        )


def _require(value: Mapping, keys: Iterable[str], path: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ASRContractError(
            "missing A3 contract field(s) at {}: {}".format(path, ", ".join(missing))
        )


def _section(root: Mapping, name: str, keys: Iterable[str]) -> Mapping:
    value = _mapping(root[name], name)
    _only(value, keys, name)
    _require(value, keys, name)
    return value


def _string(value: Any, path: str, expected: Optional[str] = None) -> str:
    if not isinstance(value, str) or not value:
        raise ASRContractError("{} must be a non-empty string".format(path))
    if expected is not None and value != expected:
        raise ASRContractError("{} must be {!r}".format(path, expected))
    return value


def _bool(value: Any, path: str, expected: Optional[bool] = None) -> bool:
    if not isinstance(value, bool):
        raise ASRContractError("{} must be boolean".format(path))
    if expected is not None and value is not expected:
        raise ASRContractError("{} must be {}".format(path, str(expected).lower()))
    return value


def _int(value: Any, path: str, expected: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ASRContractError("{} must be an integer".format(path))
    if expected is not None and value != expected:
        raise ASRContractError("{} must be {}".format(path, expected))
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ASRContractError("{} must be numeric".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise ASRContractError("{} must be finite".format(path))
    return result


def _list(value: Any, path: str) -> List[Any]:
    if not isinstance(value, list):
        raise ASRContractError("{} must be a list".format(path))
    return value


def _exact_list(value: Any, expected: List[Any], path: str) -> None:
    if _list(value, path) != expected:
        raise ASRContractError("{} must be {!r}".format(path, expected))


def _sha(value: Any, path: str, frozen: bool) -> str:
    text = _string(value, path)
    if text == "PENDING" and not frozen:
        return text
    if not SHA256_RE.match(text):
        raise ASRContractError("{} must be a lowercase SHA256".format(path))
    return text


def _artifact_identity(value: Any, path: str, frozen: bool) -> None:
    item = _mapping(value, path)
    _only(item, ("path", "sha256"), path)
    _require(item, ("path", "sha256"), path)
    _string(item["path"], _path(path, "path"))
    _sha(item["sha256"], _path(path, "sha256"), frozen)


def _manifest_identity(value: Any, path: str, frozen: bool) -> None:
    item = _mapping(value, path)
    _only(item, ("path", "sha256", "schema_version", "records"), path)
    _require(item, ("path", "sha256", "schema_version", "records"), path)
    _string(item["path"], _path(path, "path"))
    _sha(item["sha256"], _path(path, "sha256"), frozen)
    _string(item["schema_version"], _path(path, "schema_version"))
    if _int(item["records"], _path(path, "records")) < 0:
        raise ASRContractError("{}.records must be non-negative".format(path))


def validate_asr_contract(contract: Mapping[str, Any], require_frozen: bool = False) -> Mapping[str, Any]:
    """Strictly validate an A3 DRAFT or FROZEN contract."""

    root = _mapping(contract, "root")
    top = (
        "contract",
        "serialization",
        "parents",
        "model",
        "environment",
        "decoder",
        "input",
        "frontends",
        "text",
        "sources",
        "qualification",
        "artifacts",
    )
    _only(root, top, "root")
    _require(root, top, "root")

    metadata = _section(root, "contract", ("namespace", "version", "gate", "state"))
    _string(metadata["namespace"], "contract.namespace", "active-asr")
    _string(metadata["version"], "contract.version", A3_CONTRACT_VERSION)
    _string(metadata["gate"], "contract.gate", A3_GATE)
    state = _string(metadata["state"], "contract.state")
    if state not in ("DRAFT", "FROZEN"):
        raise ASRContractError("contract.state must be DRAFT or FROZEN")
    if require_frozen and state != "FROZEN":
        raise ASRContractError("formal A3 qualification requires contract.state=FROZEN")
    frozen = state == "FROZEN"

    serialization = _section(root, "serialization", ("version", "hash_algorithm"))
    _string(serialization["version"], "serialization.version", "canonical-json-v1")
    _string(serialization["hash_algorithm"], "serialization.hash_algorithm", "sha256")

    parents = _section(root, "parents", ("a0_contract_sha256", "a2_oracle_v3_sha256"))
    _sha(parents["a0_contract_sha256"], "parents.a0_contract_sha256", True)
    _sha(parents["a2_oracle_v3_sha256"], "parents.a2_oracle_v3_sha256", True)
    if parents["a0_contract_sha256"] != A0_PARENT_SHA256 or parents["a2_oracle_v3_sha256"] != A2_V3_PARENT_SHA256:
        raise ASRContractError("A3 parent contract identity is not the accepted A0/A2 authority")

    model = _section(
        root,
        "model",
        ("family", "repo_id", "resolved_revision", "local_root", "loaded_files"),
    )
    _string(model["family"], "model.family", "SpeechBrain_LibriSpeech_Transformer")
    _string(model["repo_id"], "model.repo_id")
    revision = _string(model["resolved_revision"], "model.resolved_revision")
    if frozen and not re.match(r"^[0-9a-f]{40}$", revision):
        raise ASRContractError("model.resolved_revision must be a 40-character git commit")
    _string(model["local_root"], "model.local_root")
    files = _mapping(model["loaded_files"], "model.loaded_files")
    required_files = ("hyperparams", "asr_weights", "language_model", "tokenizer", "normalizer")
    _only(files, required_files, "model.loaded_files")
    _require(files, required_files, "model.loaded_files")
    for name in required_files:
        _artifact_identity(files[name], _path("model.loaded_files", name), frozen)

    environment = _section(
        root,
        "environment",
        (
            "conda_env",
            "python",
            "speechbrain",
            "torch",
            "torchaudio",
            "torch_cuda",
            "cuda_available",
            "device",
            "precision",
            "dependency_lock_sha256",
        ),
    )
    for key in ("conda_env", "python", "speechbrain", "torch", "torchaudio", "torch_cuda", "device", "precision"):
        _string(environment[key], _path("environment", key))
    _bool(environment["cuda_available"], "environment.cuda_available")
    if environment["precision"] != "float32":
        raise ASRContractError("environment.precision must be float32")
    _sha(environment["dependency_lock_sha256"], "environment.dependency_lock_sha256", frozen)

    decoder = _section(
        root,
        "decoder",
        (
            "beam_size",
            "blank_index",
            "pad_index",
            "bos_index",
            "eos_index",
            "ctc_weight",
            "lm_weight",
            "min_decode_ratio",
            "max_decode_ratio",
            "temperature",
            "lm_scorer_temperature",
            "using_eos_threshold",
            "eos_threshold",
            "length_normalization",
            "return_topk",
            "topk",
            "using_max_attn_shift",
            "max_attn_shift",
            "minus_inf",
            "tokenizer_type",
            "score_semantics",
        ),
    )
    if _int(decoder["beam_size"], "decoder.beam_size") <= 0:
        raise ASRContractError("decoder.beam_size must be positive")
    _int(decoder["blank_index"], "decoder.blank_index", 0)
    _int(decoder["pad_index"], "decoder.pad_index", 0)
    _int(decoder["bos_index"], "decoder.bos_index", 1)
    _int(decoder["eos_index"], "decoder.eos_index", 2)
    for key in ("ctc_weight", "lm_weight", "min_decode_ratio", "max_decode_ratio", "temperature", "lm_scorer_temperature"):
        _number(decoder[key], _path("decoder", key))
    _bool(decoder["using_eos_threshold"], "decoder.using_eos_threshold")
    _number(decoder["eos_threshold"], "decoder.eos_threshold")
    _bool(decoder["length_normalization"], "decoder.length_normalization")
    _bool(decoder["return_topk"], "decoder.return_topk")
    _int(decoder["topk"], "decoder.topk")
    _bool(decoder["using_max_attn_shift"], "decoder.using_max_attn_shift")
    _int(decoder["max_attn_shift"], "decoder.max_attn_shift")
    _number(decoder["minus_inf"], "decoder.minus_inf")
    _string(decoder["tokenizer_type"], "decoder.tokenizer_type")
    _string(decoder["score_semantics"], "decoder.score_semantics", "decoder_score_not_confidence")

    input_contract = _section(
        root,
        "input",
        (
            "sample_rate_hz",
            "channels",
            "dtype",
            "waveform_normalization",
            "utterance_unit",
            "batch_padding",
            "length_semantics",
            "post_convolution_normalization",
        ),
    )
    _int(input_contract["sample_rate_hz"], "input.sample_rate_hz", 16000)
    _string(input_contract["channels"], "input.channels", "mono")
    _string(input_contract["dtype"], "input.dtype", "float32")
    _string(input_contract["waveform_normalization"], "input.waveform_normalization", "speechbrain_frozen_global_input_normalizer")
    _string(input_contract["utterance_unit"], "input.utterance_unit", "complete")
    _string(input_contract["batch_padding"], "input.batch_padding", "zero_right_pad")
    _string(input_contract["length_semantics"], "input.length_semantics", "relative_valid_length")
    _string(input_contract["post_convolution_normalization"], "input.post_convolution_normalization", "none")

    frontends = _section(root, "frontends", ("primary", "sensitivity", "best_ear_forbidden"))
    _string(frontends["primary"], "frontends.primary", "mean_lr")
    _exact_list(frontends["sensitivity"], ["fixed_L", "fixed_R"], "frontends.sensitivity")
    _bool(frontends["best_ear_forbidden"], "frontends.best_ear_forbidden", True)

    text = _section(
        root,
        "text",
        ("normalization_version", "case", "whitespace", "punctuation", "apostrophe", "numbers", "cer_units"),
    )
    _string(text["normalization_version"], "text.normalization_version")
    _string(text["case"], "text.case", "uppercase")
    _string(text["whitespace"], "text.whitespace", "collapse_and_strip")
    _string(text["punctuation"], "text.punctuation", "remove_except_apostrophe")
    _string(text["apostrophe"], "text.apostrophe", "preserve_internal")
    _string(text["numbers"], "text.numbers", "preserve_digit_tokens")
    _string(text["cer_units"], "text.cer_units", "normalized_characters_excluding_spaces")

    sources = _section(
        root,
        "sources",
        (
            "config_path",
            "config_sha256",
            "speech_registry",
            "noise_registry",
            "o1_splits",
            "o2_splits",
            "speaker_disjoint_required",
            "duration_sec",
            "minimum_reference_words",
            "complete_utterance_required",
            "noise_parent_split_required",
            "noise_technical_audit",
        ),
    )
    _string(sources["config_path"], "sources.config_path")
    _sha(sources["config_sha256"], "sources.config_sha256", frozen)
    _artifact_identity(sources["speech_registry"], "sources.speech_registry", frozen)
    _artifact_identity(sources["noise_registry"], "sources.noise_registry", frozen)
    _exact_list(sources["o1_splits"], ["dev-clean", "dev-other"], "sources.o1_splits")
    _exact_list(sources["o2_splits"], ["test-clean", "test-other"], "sources.o2_splits")
    _bool(sources["speaker_disjoint_required"], "sources.speaker_disjoint_required", True)
    duration = _mapping(sources["duration_sec"], "sources.duration_sec")
    _only(duration, ("minimum", "maximum"), "sources.duration_sec")
    _require(duration, ("minimum", "maximum"), "sources.duration_sec")
    if _number(duration["minimum"], "sources.duration_sec.minimum") <= 0:
        raise ASRContractError("sources.duration_sec.minimum must be positive")
    if _number(duration["maximum"], "sources.duration_sec.maximum") < _number(duration["minimum"], "sources.duration_sec.minimum"):
        raise ASRContractError("sources.duration_sec.maximum must be >= minimum")
    if _int(sources["minimum_reference_words"], "sources.minimum_reference_words") <= 0:
        raise ASRContractError("sources.minimum_reference_words must be positive")
    _bool(sources["complete_utterance_required"], "sources.complete_utterance_required", True)
    _bool(sources["noise_parent_split_required"], "sources.noise_parent_split_required", True)
    noise_audit = _mapping(sources["noise_technical_audit"], "sources.noise_technical_audit")
    noise_audit_keys = (
        "minimum_duration_sec",
        "extreme_silence_rms_max",
        "unreadable_policy",
        "speech_leakage_audit",
        "strong_reverberation_audit",
        "legal_slice_policy",
    )
    _only(noise_audit, noise_audit_keys, "sources.noise_technical_audit")
    _require(noise_audit, noise_audit_keys, "sources.noise_technical_audit")
    _number(noise_audit["minimum_duration_sec"], "sources.noise_technical_audit.minimum_duration_sec")
    _number(noise_audit["extreme_silence_rms_max"], "sources.noise_technical_audit.extreme_silence_rms_max")
    _string(noise_audit["unreadable_policy"], "sources.noise_technical_audit.unreadable_policy", "reject")
    _string(noise_audit["speech_leakage_audit"], "sources.noise_technical_audit.speech_leakage_audit", "explicit_manual_or_validated_audit_field")
    _string(noise_audit["strong_reverberation_audit"], "sources.noise_technical_audit.strong_reverberation_audit", "explicit_manual_or_validated_audit_field")
    _string(noise_audit["legal_slice_policy"], "sources.noise_technical_audit.legal_slice_policy", "whole_recording_if_technically_eligible")

    qualification = _section(
        root,
        "qualification",
        ("seed", "manifests", "clean", "ss2_domain", "snr_sensitivity", "q4"),
    )
    _int(qualification["seed"], "qualification.seed", 20260927)
    manifests = _mapping(qualification["manifests"], "qualification.manifests")
    manifest_names = ("clean", "ss2_domain", "snr_sensitivity", "q4")
    _only(manifests, manifest_names, "qualification.manifests")
    _require(manifests, manifest_names, "qualification.manifests")
    for name in manifest_names:
        _manifest_identity(manifests[name], _path("qualification.manifests", name), frozen)

    clean = _mapping(qualification["clean"], "qualification.clean")
    _only(clean, ("utterances_total", "per_split", "repeat_inferences", "wer_max_fraction"), "qualification.clean")
    _require(clean, ("utterances_total", "per_split", "repeat_inferences", "wer_max_fraction"), "qualification.clean")
    _int(clean["utterances_total"], "qualification.clean.utterances_total", 24)
    _int(clean["per_split"], "qualification.clean.per_split", 12)
    _int(clean["repeat_inferences"], "qualification.clean.repeat_inferences", 3)
    _number(clean["wer_max_fraction"], "qualification.clean.wer_max_fraction")

    ss2 = _mapping(qualification["ss2_domain"], "qualification.ss2_domain")
    _only(ss2, ("sample_rate_hz", "noise", "post_convolution_normalization", "minimum_nonempty_fraction", "wer_max_fraction"), "qualification.ss2_domain")
    _require(ss2, ("sample_rate_hz", "noise", "post_convolution_normalization", "minimum_nonempty_fraction", "wer_max_fraction"), "qualification.ss2_domain")
    _int(ss2["sample_rate_hz"], "qualification.ss2_domain.sample_rate_hz", 16000)
    _string(ss2["noise"], "qualification.ss2_domain.noise", "none")
    _string(ss2["post_convolution_normalization"], "qualification.ss2_domain.post_convolution_normalization", "none")
    _number(ss2["minimum_nonempty_fraction"], "qualification.ss2_domain.minimum_nonempty_fraction")
    _number(ss2["wer_max_fraction"], "qualification.ss2_domain.wer_max_fraction")

    snr = _mapping(qualification["snr_sensitivity"], "qualification.snr_sensitivity")
    _only(snr, ("conditions_db", "noise_algorithm", "noise_seed", "target_power", "noise_power", "postmix_normalization", "minimum_low_high_wer_delta_percentage_points", "edit_count_order_required"), "qualification.snr_sensitivity")
    _require(snr, ("conditions_db", "noise_algorithm", "noise_seed", "target_power", "noise_power", "postmix_normalization", "minimum_low_high_wer_delta_percentage_points", "edit_count_order_required"), "qualification.snr_sensitivity")
    _exact_list(snr["conditions_db"], ["reverb_only", 10, 0, -10], "qualification.snr_sensitivity.conditions_db")
    _string(snr["noise_algorithm"], "qualification.snr_sensitivity.noise_algorithm", "numpy_pcg64_standard_normal_float32")
    _int(snr["noise_seed"], "qualification.snr_sensitivity.noise_seed", 20260927)
    _string(snr["target_power"], "qualification.snr_sensitivity.target_power", "mean_square_full_mono_waveform")
    _string(snr["noise_power"], "qualification.snr_sensitivity.noise_power", "mean_square_same_length_noise")
    _string(snr["postmix_normalization"], "qualification.snr_sensitivity.postmix_normalization", "none")
    _number(snr["minimum_low_high_wer_delta_percentage_points"], "qualification.snr_sensitivity.minimum_low_high_wer_delta_percentage_points")
    _bool(snr["edit_count_order_required"], "qualification.snr_sensitivity.edit_count_order_required", True)

    q4 = _mapping(qualification["q4"], "qualification.q4")
    _only(q4, ("path_a", "path_b", "separate_normalization", "decision_authority"), "qualification.q4")
    _require(q4, ("path_a", "path_b", "separate_normalization", "decision_authority"), "qualification.q4")
    _string(q4["path_a"], "qualification.q4.path_a", "native_ss2_16khz")
    _string(q4["path_b"], "qualification.q4.path_b", "native_ss2_24khz_resample_poly_to_16khz")
    _bool(q4["separate_normalization"], "qualification.q4.separate_normalization", False)
    _string(q4["decision_authority"], "qualification.q4.decision_authority", "reviewer")

    artifacts = _section(root, "artifacts", ("schema_version", "model_lock", "model_lock_sha256", "summary", "report"))
    _string(artifacts["schema_version"], "artifacts.schema_version")
    for key in ("model_lock", "summary", "report"):
        _string(artifacts[key], _path("artifacts", key))
    _sha(artifacts["model_lock_sha256"], "artifacts.model_lock_sha256", frozen)

    return contract


def _canonical(value: Any, path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key], _path(path, key)) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, list):
        return [_canonical(item, "{}[{}]".format(path, index)) for index, item in enumerate(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ASRContractError("{} contains non-finite number".format(path))
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ASRContractError("{} contains unsupported value type".format(path))


def canonical_asr_json(contract: Mapping[str, Any]) -> str:
    validate_asr_contract(contract)
    return json.dumps(_canonical(contract), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def asr_contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_asr_json(contract).encode("utf-8")).hexdigest()


def load_asr_contract(path: str, require_frozen: bool = False) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ASRContractError("A3 contract root must be a mapping")
    validate_asr_contract(value, require_frozen=require_frozen)
    return dict(value)
