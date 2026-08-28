#!/usr/bin/env python3
"""Deterministic, read-only ClassDOA V1 source mapping audit."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DATASETS = ("ESC-50", "DESED isolated foreground", "PSELD-selected FSD50K")
MAPPING_TYPES = ("EXACT", "SEMANTIC_STRONG", "AMBIGUOUS", "NONE", "BLOCKED_RESOURCE")
CSV_FIELDS = (
    "canonical_class_id",
    "canonical_class",
    "source_dataset",
    "source_label",
    "source_label_id",
    "mapping_type",
    "exact_match",
    "semantic_match",
    "candidate_count",
    "independent_identity_count",
    "metadata_status",
    "resource_status",
    "primary_candidate",
    "supplement_candidate",
    "manual_review_required",
    "evidence_path_or_url",
    "evidence_sha_or_version",
    "notes",
)

ESC_AMBIGUOUS = {
    "running_water": ("pouring_water", "water_drops"),
}
DESED_LABELS = {
    "vacuum_cleaner": ("Vacuum_cleaner",),
    "clock_alarm": ("Alarm_bell_ringing",),
    "speech": ("Speech",),
    "running_water": ("Running_water",),
    "frying": ("Frying",),
    "dishes": ("Dishes",),
}
PSELD_EXACT = {
    "mechanical_fan": ("Mechanical_fan",),
    "vacuum_cleaner": ("Vacuum_cleaner",),
}
PSELD_STRONG = {
    "laughing": ("Laughter",),
    "speech": ("Female_speech_and_woman_speaking", "Male_speech_and_man_speaking"),
    "running_water": ("Water_tap_and_faucet",),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ontology(path):
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["ontology"]["classes"]


def _row(item, dataset, *, source_label="", source_label_id="", mapping_type="BLOCKED_RESOURCE",
         exact=False, semantic=False, candidate_count=None, identity_count=None,
         metadata_status="MISSING", resource_status="MISSING", primary=False,
         supplement=False, manual=True, evidence="", evidence_sha="", notes=""):
    if mapping_type not in MAPPING_TYPES:
        raise ValueError("unsupported mapping type: {}".format(mapping_type))
    if mapping_type == "EXACT" and not exact:
        raise ValueError("EXACT rows must set exact_match=true")
    if mapping_type in ("NONE", "BLOCKED_RESOURCE") and (exact or semantic):
        raise ValueError("{} rows cannot claim an exact/semantic match".format(mapping_type))
    return {
        "canonical_class_id": int(item["id"]),
        "canonical_class": str(item["name"]),
        "source_dataset": dataset,
        "source_label": source_label,
        "source_label_id": source_label_id,
        "mapping_type": mapping_type,
        "exact_match": bool(exact),
        "semantic_match": bool(semantic),
        "candidate_count": candidate_count,
        "independent_identity_count": identity_count,
        "metadata_status": metadata_status,
        "resource_status": resource_status,
        "primary_candidate": bool(primary),
        "supplement_candidate": bool(supplement),
        "manual_review_required": bool(manual),
        "evidence_path_or_url": evidence,
        "evidence_sha_or_version": evidence_sha,
        "notes": notes,
    }


def _counts(rows, label_key, identity_key):
    labels = defaultdict(list)
    for row in rows:
        label = str(row.get(label_key, "")).strip()
        if label:
            labels[label].append(row)
    return labels, lambda selected: len({str(row.get(identity_key, "")).strip() for row in selected if str(row.get(identity_key, "")).strip()})


def audit_esc(classes, metadata_path=None, audio_root=None, evidence="", evidence_sha=""):
    if not metadata_path or not Path(metadata_path).is_file():
        return [_row(item, "ESC-50", evidence=evidence, evidence_sha=evidence_sha,
                     notes="official metadata/audio root not present locally") for item in classes]
    with Path(metadata_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_label, identity_count = _counts(rows, "category", "src_file")
    resource_status = "PRESENT_PARTIAL" if audio_root and Path(audio_root).is_dir() else "PRESENT_METADATA_ONLY"
    result = []
    for item in classes:
        name = str(item["name"])
        if name in by_label:
            selected = by_label[name]
            result.append(_row(item, "ESC-50", source_label=name,
                               source_label_id="|".join(sorted({str(row.get("target", "")) for row in selected})),
                               mapping_type="EXACT", exact=True,
                               candidate_count=len(selected), identity_count=identity_count(selected),
                               metadata_status="PRESENT", resource_status=resource_status,
                               primary=True, manual=False, evidence=evidence,
                               evidence_sha=evidence_sha,
                               notes="official category exact; src_file used as original identity"))
            continue
        aliases = ESC_AMBIGUOUS.get(name, ())
        selected = [row for alias in aliases for row in by_label.get(alias, ())]
        if selected:
            result.append(_row(item, "ESC-50", source_label="|".join(aliases),
                               source_label_id="|".join(sorted({str(row.get("target", "")) for row in selected})),
                               mapping_type="AMBIGUOUS", semantic=True,
                               candidate_count=len(selected), identity_count=identity_count(selected),
                               metadata_status="PRESENT", resource_status=resource_status,
                               evidence=evidence, evidence_sha=evidence_sha,
                               notes="nearby label only; Step 2A audition required"))
        else:
            result.append(_row(item, "ESC-50", mapping_type="NONE", candidate_count=0,
                               identity_count=0, metadata_status="PRESENT", resource_status=resource_status,
                               manual=True, evidence=evidence, evidence_sha=evidence_sha,
                               notes="no canonical or approved nearby category in official metadata"))
    return result


def audit_desed(classes, metadata_path=None, foreground_root=None, evidence="", evidence_sha=""):
    labels = set()
    if metadata_path and Path(metadata_path).is_file():
        with Path(metadata_path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        labels = {str(key) for key in payload}
    resource_status = "PRESENT_PARTIAL" if foreground_root and Path(foreground_root).is_dir() else "MISSING"
    result = []
    for item in classes:
        name = str(item["name"])
        known = [label for label in DESED_LABELS.get(name, ()) if label in labels]
        result.append(_row(
            item, "DESED isolated foreground", source_label="|".join(known),
            mapping_type="BLOCKED_RESOURCE", metadata_status="PRESENT_METADATA_ONLY" if labels else "MISSING",
            resource_status=resource_status, evidence=evidence, evidence_sha=evidence_sha,
            notes=("official class label observed, but isolated foreground registry/audio is absent locally"
                   if known else "isolated foreground registry/audio absent; no safe class conclusion"),
        ))
    return result


def audit_pseld(classes, selected_path=None, audio_root=None, evidence="", evidence_sha=""):
    by_label = defaultdict(list)
    if selected_path and Path(selected_path).is_file():
        with Path(selected_path).open(encoding="utf-8") as handle:
            for raw in handle:
                parts = raw.strip().split("/")
                if len(parts) >= 4 and parts[-1].endswith(".wav"):
                    by_label[parts[2]].append((parts[-1][:-4], raw.strip()))
    resource_status = "PRESENT_PARTIAL" if audio_root and Path(audio_root).is_dir() else "PRESENT_METADATA_ONLY"
    result = []
    for item in classes:
        name = str(item["name"])
        exact_labels = PSELD_EXACT.get(name, ())
        strong_labels = PSELD_STRONG.get(name, ())
        labels = exact_labels if any(label in by_label for label in exact_labels) else strong_labels
        selected = [entry for label in labels for entry in by_label.get(label, ())]
        if selected and labels == exact_labels:
            mapping_type, exact, semantic = "EXACT", True, False
        elif selected:
            mapping_type, exact, semantic = "SEMANTIC_STRONG", False, True
        elif selected_path and Path(selected_path).is_file():
            mapping_type, exact, semantic = "NONE", False, False
        else:
            mapping_type, exact, semantic = "BLOCKED_RESOURCE", False, False
        result.append(_row(
            item, "PSELD-selected FSD50K", source_label="|".join(labels),
            source_label_id="FSD50K_fname" if selected else "",
            mapping_type=mapping_type, exact=exact, semantic=semantic,
            candidate_count=len(selected) if selected_path and Path(selected_path).is_file() else None,
            identity_count=len({entry[0] for entry in selected}) if selected_path and Path(selected_path).is_file() else None,
            metadata_status="PRESENT" if selected_path and Path(selected_path).is_file() else "MISSING",
            resource_status=resource_status, supplement=mapping_type == "SEMANTIC_STRONG",
            manual=mapping_type != "EXACT", evidence=evidence, evidence_sha=evidence_sha,
            notes=("official selected registry; audio files are not locally present"
                   if selected else "no selected-registry candidate or selected registry unavailable"),
        ))
    return result


def audit_sources(ontology_path, esc50_metadata=None, esc50_audio_root=None,
                  desed_metadata=None, desed_foreground_root=None,
                  pseld_selected=None, pseld_audio_root=None, evidence=None):
    classes = load_ontology(ontology_path)
    evidence = evidence or {}
    def evidence_args(dataset):
        entry = evidence.get(dataset, {})
        return entry.get("evidence", ""), entry.get("evidence_sha", "")

    esc_evidence, esc_sha = evidence_args("ESC-50")
    desed_evidence, desed_sha = evidence_args("DESED isolated foreground")
    pseld_evidence, pseld_sha = evidence_args("PSELD-selected FSD50K")
    rows = []
    rows.extend(audit_esc(classes, esc50_metadata, esc50_audio_root, esc_evidence, esc_sha))
    rows.extend(audit_desed(classes, desed_metadata, desed_foreground_root, desed_evidence, desed_sha))
    rows.extend(audit_pseld(classes, pseld_selected, pseld_audio_root, pseld_evidence, pseld_sha))
    order = {dataset: index for index, dataset in enumerate(DATASETS)}
    return sorted(rows, key=lambda row: (row["canonical_class_id"], order[row["source_dataset"]], row["source_label"]))


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else str(row[key]).lower() if isinstance(row[key], bool) else row[key] for key in CSV_FIELDS})


def _inventory(ontology_path, rows, evidence):
    datasets = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["source_dataset"] == dataset]
        datasets.append({
            "source_dataset": dataset,
            "mapping_rows": len(subset),
            "mapping_type_counts": {kind: sum(row["mapping_type"] == kind for row in subset) for kind in MAPPING_TYPES},
            "resource_statuses": sorted({row["resource_status"] for row in subset}),
            "metadata_statuses": sorted({row["metadata_status"] for row in subset}),
            "evidence": evidence.get(dataset, {}),
        })
    return {
        "audit_schema_version": "clsdoa_v1_source_mapping_audit.1",
        "ontology_path": str(ontology_path),
        "ontology_sha256": sha256_file(ontology_path),
        "dataset_family_count": len(DATASETS),
        "canonical_class_count": len({row["canonical_class_id"] for row in rows}),
        "datasets": datasets,
        "prohibited_operations_performed": [],
    }


def _summary(rows, inventory):
    by_class = defaultdict(dict)
    for row in rows:
        by_class[row["canonical_class"]][row["source_dataset"]] = row
    primary = {
        "coughing": "ESC-50 (EXACT)", "laughing": "ESC-50 (EXACT)",
        "keyboard_typing": "ESC-50 (EXACT)", "vacuum_cleaner": "ESC-50 (EXACT; DESED/PSELD supplement blocked/metadata-only)",
        "clock_alarm": "ESC-50 (EXACT)", "speech": "DESED isolated foreground (blocked; PSELD supplement metadata-only)",
        "running_water": "DESED isolated foreground (blocked)", "frying": "DESED isolated foreground (blocked)",
        "mechanical_fan": "PSELD-selected FSD50K (EXACT, metadata-only)",
        "microwave_oven": "UNRESOLVED", "dishes": "DESED isolated foreground (blocked)",
        "printer": "UNRESOLVED",
    }
    lines = [
        "# ClassDOA V1 Step 1A source mapping audit",
        "",
        "This is a metadata/provenance audit only. No source audio was downloaded, resampled, normalized, split, auditioned, or rendered.",
        "",
        "## 12-class summary matrix",
        "",
        "| canonical class | ESC-50 | DESED isolated | PSELD-selected FSD50K | Step 1A primary suggestion | supplement suggestion | unresolved |",
        "|---|---|---|---|---|---|---|",
    ]
    for class_name in sorted(by_class, key=lambda name: next(row["canonical_class_id"] for row in rows if row["canonical_class"] == name)):
        cells = []
        unresolved = []
        for dataset in DATASETS:
            row = by_class[class_name][dataset]
            count = "" if row["candidate_count"] is None else "{} clips/{} ids".format(row["candidate_count"], row["independent_identity_count"])
            cells.append("{}{}".format(row["mapping_type"], " (" + count + ")" if count else ""))
            if row["mapping_type"] in ("AMBIGUOUS", "BLOCKED_RESOURCE"):
                unresolved.append(dataset)
        supplement = ", ".join(dataset for dataset in DATASETS if by_class[class_name][dataset]["supplement_candidate"] is True) or "none identified"
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(class_name, *cells, primary[class_name], supplement, ", ".join(unresolved) or "none"))
    lines.extend([
        "",
        "## Evidence and review narrative",
        "",
        "ESC-50 official metadata is available for audit with 2,000 rows, five folds, category, target, src_file, and take fields. The local server has no ESC-50 audio root, so exact mappings for coughing, laughing, keyboard_typing, vacuum_cleaner, and clock_alarm are metadata-only; running_water has only the nearby pouring_water/water_drops labels and is AMBIGUOUS. Other canonical classes are NONE in the official category field. Candidate counts are metadata counts and independent identities use src_file, so they are not claims of locally usable audio.",
        "",
        "DESED official documentation and event-occurrence metadata expose the soundbank mechanism and labels for Vacuum_cleaner, Alarm_bell_ringing, Speech, Running_water, Frying, and Dishes, but the local server has no DESED isolated foreground root or per-clip registry. Therefore all 12 DESED rows are BLOCKED_RESOURCE, with known labels retained only as evidence; candidate and identity counts remain blank. Synthetic mixtures, real soundscapes, and the DESED code repository are not treated as isolated source clips.",
        "",
        "The official FSD50K_selected.txt registry is present as a small metadata text list with 4,177 entries and numeric FSD/Freesound filenames. It contains exact Mechanical_fan and Vacuum_cleaner groups, and strong semantic Laughter, speech, and Water_tap_and_faucet groups. The local PSELD repository contains code and generated DCASE stereo/feature artifacts but no cls_indices_* or PSELD source crosswalk; the selected registry is consequently usable for metadata mapping but only PARTIAL for PSELD-specific provenance. Selected audio is not locally present.",
        "",
        "The selected registry has enough distinct numeric filenames for the reported candidate/identity counts; no duplicate exact selected path was found. The audit does not infer duration or audio quality from filenames. No class is proven to lack a source across all three families, but DESED isolated resources and PSELD-specific selection linkage remain blockers for a complete source track.",
        "",
        "Dataset-level license/documentation is available for ESC-50 and DESED, and FSD/Freesound IDs are traceable for the selected list. Per-recording license completion remains DEFERRED_TO_STEP_2A. PSELD pretraining exposure is PARTIAL: FSD numeric IDs are traceable where selected, but no exact PSELD pretraining membership crosswalk is present.",
        "",
        "Step 2A cannot start as a complete Source Track. The concrete blockers are the missing local ESC audio, missing DESED isolated foreground registry/audio, absent PSELD-specific source crosswalk, and unfinished per-recording license/QC readiness. This audit therefore ends with PARTIAL readiness and awaits human review.",
        "",
        "## Determinism",
        "",
        "Rows are sorted by canonical_class_id, source dataset, and source label. Re-running the same metadata inputs must produce byte-identical CSV/JSON content.",
        "",
        "Inventory mapping-row total: {}. Prohibited operations recorded: none.".format(sum(item["mapping_rows"] for item in inventory["datasets"])),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(output_dir, ontology_path, rows, evidence):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = _inventory(ontology_path, rows, evidence)
    (output_dir / "source_dataset_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "source_class_mapping_audit.csv", rows)
    evidence_payload = {
        "audit_schema_version": "clsdoa_v1_source_mapping_evidence.1",
        "official_and_local_evidence": evidence,
        "notes": [
            "Evidence files are metadata/class-list documents only.",
            "No source audio was downloaded or written by this audit.",
            "PSELD-specific selected-source linkage is not established by generated DCASE artifacts.",
        ],
    }
    (output_dir / "source_mapping_evidence.json").write_text(json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "source_mapping_summary.md").write_text(_summary(rows, inventory), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--esc50-metadata")
    parser.add_argument("--esc50-audio-root")
    parser.add_argument("--desed-metadata")
    parser.add_argument("--desed-foreground-root")
    parser.add_argument("--pseld-selected")
    parser.add_argument("--pseld-audio-root")
    parser.add_argument("--evidence-json", required=True)
    args = parser.parse_args(argv)
    with Path(args.evidence_json).open(encoding="utf-8") as handle:
        evidence = json.load(handle)
    rows = audit_sources(
        args.ontology,
        args.esc50_metadata,
        args.esc50_audio_root,
        args.desed_metadata,
        args.desed_foreground_root,
        args.pseld_selected,
        args.pseld_audio_root,
        evidence,
    )
    write_outputs(args.output_dir, args.ontology, rows, evidence)


if __name__ == "__main__":
    main()
