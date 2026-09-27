"""Immutable LibriSpeech metadata and decoded-waveform registry for A3."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml

from active_audition.evaluation.asr_metrics import normalize_text


SPEECH_REGISTRY_SCHEMA_VERSION = "active-asr-a3-speech-registry-v1"
SPEECH_QUALIFICATION_MANIFEST_VERSION = "active-asr-a3-clean-qualification-v1"
SPEECH_ROW_KEYS = (
    "schema_version",
    "corpus",
    "split",
    "speaker_id",
    "chapter_id",
    "utterance_id",
    "relative_source_path",
    "source_file_sha256",
    "decoded_waveform_sha256",
    "source_sample_rate_hz",
    "decode_sample_rate_hz",
    "dtype",
    "samples",
    "duration_sec",
    "raw_transcript",
    "normalized_transcript",
    "reference_word_count",
    "complete_utterance",
    "eligible",
    "exclusion_reason",
)


class SpeechRegistryError(ValueError):
    """Raised when LibriSpeech metadata cannot satisfy the A3 contract."""


def load_sources_config(path: str) -> Dict[str, Any]:
    """Load the strict local source-location contract without guessing paths."""

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "librispeech", "musan"}:
        raise SpeechRegistryError("sources config has invalid top-level fields")
    if value["schema_version"] != "active-asr-a3-source-locations-v1":
        raise SpeechRegistryError("sources config schema_version is invalid")
    libri = value["librispeech"]
    if not isinstance(libri, Mapping) or set(libri) != {"corpus", "root", "archive_source", "splits"}:
        raise SpeechRegistryError("sources.librispeech has invalid fields")
    if libri["corpus"] != "LibriSpeech" or set(libri["splits"]) != {"dev-clean", "dev-other", "test-clean", "test-other"}:
        raise SpeechRegistryError("LibriSpeech corpus/splits are invalid")
    for split, item in libri["splits"].items():
        if not isinstance(item, Mapping) or set(item) != {"archive", "archive_md5"}:
            raise SpeechRegistryError("LibriSpeech {} source identity is invalid".format(split))
        if not isinstance(item["archive_md5"], str) or len(item["archive_md5"]) != 32:
            raise SpeechRegistryError("LibriSpeech {} archive_md5 is invalid".format(split))
    musan = value["musan"]
    if not isinstance(musan, Mapping) or set(musan) != {"corpus", "subset", "root", "archive", "archive_source", "archive_md5"}:
        raise SpeechRegistryError("sources.musan has invalid fields")
    if musan["corpus"] != "MUSAN" or musan["subset"] != "noise" or len(str(musan["archive_md5"])) != 32:
        raise SpeechRegistryError("MUSAN source identity is invalid")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_waveform_sha256(waveform: Any) -> str:
    value = np.ascontiguousarray(np.asarray(waveform, dtype="<f4"))
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise SpeechRegistryError("decoded speech must be a finite non-empty mono vector")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def validate_speech_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise SpeechRegistryError("speech registry row must be a mapping")
    unknown = sorted(set(row) - set(SPEECH_ROW_KEYS))
    missing = [key for key in SPEECH_ROW_KEYS if key not in row]
    if unknown or missing:
        raise SpeechRegistryError("invalid speech registry keys: unknown={} missing={}".format(unknown, missing))
    if row["schema_version"] != SPEECH_REGISTRY_SCHEMA_VERSION or row["corpus"] != "LibriSpeech":
        raise SpeechRegistryError("invalid speech registry identity")
    for key in ("split", "speaker_id", "chapter_id", "utterance_id", "relative_source_path", "raw_transcript", "normalized_transcript"):
        if not isinstance(row[key], str) or not row[key]:
            raise SpeechRegistryError("{} must be a non-empty string".format(key))
    for key in ("source_file_sha256", "decoded_waveform_sha256"):
        if not isinstance(row[key], str) or len(row[key]) != 64:
            raise SpeechRegistryError("{} must be SHA256".format(key))
    if row["source_sample_rate_hz"] != 16000 or row["decode_sample_rate_hz"] != 16000:
        raise SpeechRegistryError("A3 LibriSpeech source/decode sample rate must be 16000 Hz")
    if row["dtype"] != "float32" or not isinstance(row["samples"], int) or row["samples"] <= 0:
        raise SpeechRegistryError("invalid decoded waveform dtype or sample count")
    if not isinstance(row["duration_sec"], (int, float)) or float(row["duration_sec"]) <= 0.0:
        raise SpeechRegistryError("duration_sec must be positive")
    if row["reference_word_count"] != len(str(row["normalized_transcript"]).split()):
        raise SpeechRegistryError("reference_word_count does not match normalized transcript")
    if row["complete_utterance"] is not True or not isinstance(row["eligible"], bool):
        raise SpeechRegistryError("complete_utterance/eligible flags are invalid")
    if not isinstance(row["exclusion_reason"], str):
        raise SpeechRegistryError("exclusion_reason must be a string")
    if row["eligible"] and row["exclusion_reason"]:
        raise SpeechRegistryError("eligible row may not have an exclusion reason")
    return row


def _transcripts(split_root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for path in sorted(split_root.rglob("*.trans.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            utterance_id, separator, transcript = line.partition(" ")
            if not separator or utterance_id in result:
                raise SpeechRegistryError("invalid or duplicate transcript line in {}".format(path))
            result[utterance_id] = transcript.strip()
    return result


def build_speech_registry(
    librispeech_root: str,
    splits: Sequence[str] = ("dev-clean", "dev-other", "test-clean", "test-other"),
    minimum_duration_sec: float = 4.0,
    maximum_duration_sec: float = 15.0,
    minimum_reference_words: int = 10,
) -> List[Dict[str, Any]]:
    """Decode each full utterance and bind source and decoded representations."""

    try:
        import soundfile as sf
    except ImportError as error:
        raise SpeechRegistryError("soundfile is required to build the speech registry") from error
    root = Path(librispeech_root).resolve()
    rows: List[Dict[str, Any]] = []
    for split in splits:
        split_root = root / split
        if not split_root.is_dir():
            raise SpeechRegistryError("LibriSpeech split is missing: {}".format(split_root))
        transcripts = _transcripts(split_root)
        audio_paths = sorted(split_root.rglob("*.flac"))
        if not audio_paths:
            raise SpeechRegistryError("LibriSpeech split has no FLAC files: {}".format(split_root))
        for path in audio_paths:
            utterance_id = path.stem
            if utterance_id not in transcripts:
                raise SpeechRegistryError("missing transcript for {}".format(utterance_id))
            parts = utterance_id.split("-")
            if len(parts) != 3:
                raise SpeechRegistryError("unexpected LibriSpeech utterance id: {}".format(utterance_id))
            waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
                raise SpeechRegistryError("invalid decoded waveform: {}".format(path))
            normalized = normalize_text(transcripts[utterance_id])
            duration = float(waveform.size) / float(sample_rate)
            reasons: List[str] = []
            if int(sample_rate) != 16000:
                reasons.append("SOURCE_SAMPLE_RATE_NOT_16000")
            if duration < float(minimum_duration_sec):
                reasons.append("DURATION_BELOW_MINIMUM")
            if duration > float(maximum_duration_sec):
                reasons.append("DURATION_ABOVE_MAXIMUM")
            if len(normalized.split()) < int(minimum_reference_words):
                reasons.append("REFERENCE_WORD_COUNT_BELOW_MINIMUM")
            row = {
                "schema_version": SPEECH_REGISTRY_SCHEMA_VERSION,
                "corpus": "LibriSpeech",
                "split": split,
                "speaker_id": parts[0],
                "chapter_id": parts[1],
                "utterance_id": utterance_id,
                "relative_source_path": path.relative_to(root).as_posix(),
                "source_file_sha256": sha256_file(path),
                "decoded_waveform_sha256": decoded_waveform_sha256(waveform),
                "source_sample_rate_hz": int(sample_rate),
                "decode_sample_rate_hz": int(sample_rate),
                "dtype": "float32",
                "samples": int(waveform.size),
                "duration_sec": duration,
                "raw_transcript": transcripts[utterance_id],
                "normalized_transcript": normalized,
                "reference_word_count": len(normalized.split()),
                "complete_utterance": True,
                "eligible": not reasons,
                "exclusion_reason": ";".join(reasons),
            }
            validate_speech_row(row)
            rows.append(row)
    if len({row["utterance_id"] for row in rows}) != len(rows):
        raise SpeechRegistryError("utterance IDs are not globally unique")
    return rows


def speaker_split_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    speakers = defaultdict(set)
    for row in rows:
        validate_speech_row(row)
        speakers[str(row["split"])].add(str(row["speaker_id"]))
    o1 = speakers["dev-clean"] | speakers["dev-other"]
    o2 = speakers["test-clean"] | speakers["test-other"]
    overlap = sorted(o1 & o2)
    return {
        "o1_speaker_count": len(o1),
        "o2_speaker_count": len(o2),
        "overlap_speaker_ids": overlap,
        "speaker_disjoint": not overlap,
        "status": "PASS" if not overlap else "FAIL",
    }


def select_clean_qualification(rows: Sequence[Mapping[str, Any]], per_split: int = 12) -> List[Dict[str, Any]]:
    """Metadata-only deterministic selection with speaker breadth.

    No hypothesis, WER, decoder score, or model API is accepted by this
    function, making WER-driven selection structurally impossible.
    """

    selected: List[Dict[str, Any]] = []
    for split in ("dev-clean", "dev-other"):
        eligible = [dict(row) for row in rows if row["split"] == split and row["eligible"]]
        by_speaker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in sorted(eligible, key=lambda item: str(item["utterance_id"])):
            by_speaker[str(row["speaker_id"])].append(row)
        speakers = sorted(by_speaker)
        chosen: List[Dict[str, Any]] = []
        round_index = 0
        while len(chosen) < int(per_split):
            progressed = False
            for speaker in speakers:
                if round_index < len(by_speaker[speaker]):
                    chosen.append(by_speaker[speaker][round_index])
                    progressed = True
                    if len(chosen) == int(per_split):
                        break
            if not progressed:
                break
            round_index += 1
        if len(chosen) != int(per_split):
            raise SpeechRegistryError(
                "{} has {} metadata-eligible utterances, need {}".format(split, len(chosen), per_split)
            )
        selected.extend(chosen)
    return selected


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [json.dumps(dict(row), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) for row in rows]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = jsonl_bytes(rows)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_speech_registry(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SpeechRegistryError("invalid JSON at line {}".format(line_number)) from error
        validate_speech_row(row)
        rows.append(dict(row))
    return rows
