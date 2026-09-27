"""Versioned text normalization and hand-auditable ASR error metrics."""

import re
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple


TEXT_NORMALIZATION_VERSION = "librispeech-uppercase-apostrophe-v1"


class ASRMetricError(ValueError):
    """Raised for invalid reference or hypothesis data."""


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        raise ASRMetricError("text must be a string")
    value = unicodedata.normalize("NFKC", text).upper()
    value = "".join(
        character
        if (character.isalnum() or character == "'" or character.isspace())
        else " "
        for character in value
    )
    value = re.sub(r"(?<![A-Z0-9])'|'(?![A-Z0-9])", " ", value)
    return " ".join(value.split())


def _edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> Tuple[int, int, int]:
    # Each cell stores (total edits, substitutions, deletions, insertions).
    rows: List[List[Tuple[int, int, int, int]]] = []
    rows.append([(index, 0, 0, index) for index in range(len(hypothesis) + 1)])
    priority = {"S": 0, "D": 1, "I": 2}
    for ref_index, ref_token in enumerate(reference, 1):
        row: List[Tuple[int, int, int, int]] = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_token in enumerate(hypothesis, 1):
            if ref_token == hyp_token:
                row.append(rows[ref_index - 1][hyp_index - 1])
                continue
            diagonal = rows[ref_index - 1][hyp_index - 1]
            deletion = rows[ref_index - 1][hyp_index]
            insertion = row[hyp_index - 1]
            candidates = (
                ((diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]), "S"),
                ((deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]), "D"),
                ((insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1), "I"),
            )
            row.append(min(candidates, key=lambda item: (item[0][0], priority[item[1]]))[0])
        rows.append(row)
    final = rows[-1][-1]
    return final[1], final[2], final[3]


def error_counts(reference: str, hypothesis: str) -> Dict[str, Any]:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    if not reference_words:
        raise ASRMetricError("empty normalized reference is a data error")
    substitutions, deletions, insertions = _edit_counts(reference_words, hypothesis_words)
    characters_reference = list(normalized_reference.replace(" ", ""))
    characters_hypothesis = list(normalized_hypothesis.replace(" ", ""))
    char_s, char_d, char_i = _edit_counts(characters_reference, characters_hypothesis)
    return {
        "normalization_version": TEXT_NORMALIZATION_VERSION,
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        "S": substitutions,
        "D": deletions,
        "I": insertions,
        "N": len(reference_words),
        "WER": float(substitutions + deletions + insertions) / float(len(reference_words)),
        "char_S": char_s,
        "char_D": char_d,
        "char_I": char_i,
        "char_N": len(characters_reference),
        "CER": float(char_s + char_d + char_i) / float(len(characters_reference)),
    }


def aggregate_error_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ASRMetricError("cannot aggregate an empty qualification corpus")
    result = {key: sum(int(row[key]) for row in rows) for key in ("S", "D", "I", "N", "char_S", "char_D", "char_I", "char_N")}
    if result["N"] <= 0 or result["char_N"] <= 0:
        raise ASRMetricError("aggregate reference is empty")
    result["WER"] = float(result["S"] + result["D"] + result["I"]) / float(result["N"])
    result["CER"] = float(result["char_S"] + result["char_D"] + result["char_I"]) / float(result["char_N"])
    return result
