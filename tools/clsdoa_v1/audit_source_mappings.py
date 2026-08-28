#!/usr/bin/env python3
"""Deterministic, metadata-only ClassDOA V1 source mapping audit."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DATASETS = ("ESC-50", "DESED isolated foreground", "PSELD-selected FSD50K")
MAPPING_TYPES = ("EXACT", "SEMANTIC_STRONG", "AMBIGUOUS", "NONE", "BLOCKED_RESOURCE")
CSV_FIELDS = (
    "canonical_class_id", "canonical_class", "source_dataset", "source_label", "source_label_id",
    "mapping_status", "mapping_type", "exact_match", "semantic_match", "candidate_count",
    "independent_identity_count", "metadata_status", "resource_status", "provenance_status",
    "pretraining_exposure_traceability", "primary_candidate", "supplement_candidate",
    "manual_review_required", "evidence_path_or_url", "evidence_sha_or_version", "notes",
)

ESC_AMBIGUOUS = {"running_water": ("pouring_water", "water_drops")}
DESED_EXACT = {
    "vacuum_cleaner": "Vacuum_cleaner", "speech": "Speech", "running_water": "Running_water",
    "frying": "Frying", "dishes": "Dishes",
}
DESED_SEMANTIC = {"clock_alarm": "Alarm_bell_ringing"}

# Explicit review rules for official PSELDNets labels. MIDs are resolved from
# cls_indices_*.tsv, never guessed from filenames.
PSELD_RULES = {
    "coughing": {"labels": ("Cough",), "mapping_status": "SEMANTIC_STRONG"},
    "laughing": {"labels": ("Laughter",), "mapping_status": "SEMANTIC_STRONG"},
    "keyboard_typing": {"labels": ("Computer keyboard", "Typing"), "mapping_status": "SEMANTIC_STRONG"},
    "vacuum_cleaner": {"labels": (), "mapping_status": "NONE"},
    "clock_alarm": {"labels": ("Alarm",), "mapping_status": "SEMANTIC_STRONG"},
    "speech": {"labels": ("Speech",), "mapping_status": "EXACT"},
    "running_water": {"labels": ("Water tap, faucet",), "mapping_status": "SEMANTIC_STRONG"},
    "frying": {"labels": ("Frying (food)",), "mapping_status": "EXACT"},
    "mechanical_fan": {"labels": ("Mechanical fan",), "mapping_status": "EXACT"},
    "microwave_oven": {"labels": ("Microwave oven",), "mapping_status": "EXACT"},
    "dishes": {"labels": ("Dishes, pots, and pans",), "mapping_status": "SEMANTIC_STRONG"},
    "printer": {"labels": ("Printer",), "mapping_status": "EXACT"},
}
PSELD_SUPPLEMENT_LABELS = {
    "speech": ("Female speech, woman speaking", "Male speech, man speaking", "Child speech, kid speaking"),
    "running_water": ("Sink (filling or washing)", "Bathtub (filling or washing)"),
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
         metadata_status="MISSING", resource_status="MISSING", provenance_status="UNKNOWN",
         pretraining_traceability="NOT_APPLICABLE", primary=False, supplement=False,
         manual=True, evidence="", evidence_sha="", notes=""):
    if mapping_type not in MAPPING_TYPES:
        raise ValueError("unsupported mapping type: {}".format(mapping_type))
    if mapping_type == "EXACT" and not exact:
        raise ValueError("EXACT rows must set exact_match=true")
    if mapping_type in ("NONE", "BLOCKED_RESOURCE") and (exact or semantic):
        raise ValueError("{} rows cannot claim an exact/semantic match".format(mapping_type))
    return {
        "canonical_class_id": int(item["id"]), "canonical_class": str(item["name"]),
        "source_dataset": dataset, "source_label": source_label, "source_label_id": source_label_id,
        "mapping_status": mapping_type, "mapping_type": mapping_type, "exact_match": bool(exact),
        "semantic_match": bool(semantic), "candidate_count": candidate_count,
        "independent_identity_count": identity_count, "metadata_status": metadata_status,
        "resource_status": resource_status, "provenance_status": provenance_status,
        "pretraining_exposure_traceability": pretraining_traceability,
        "primary_candidate": bool(primary), "supplement_candidate": bool(supplement),
        "manual_review_required": bool(manual), "evidence_path_or_url": evidence,
        "evidence_sha_or_version": evidence_sha, "notes": notes,
    }


def _counts(rows, label_key, identity_key):
    labels = defaultdict(list)
    for row in rows:
        label = str(row.get(label_key, "")).strip()
        if label:
            labels[label].append(row)
    return labels, lambda selected: len({
        str(row.get(identity_key, "")).strip() for row in selected
        if str(row.get(identity_key, "")).strip()
    })


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
            result.append(_row(
                item, "ESC-50", source_label=name,
                source_label_id="|".join(sorted({str(row.get("target", "")) for row in selected})),
                mapping_type="EXACT", exact=True, candidate_count=len(selected),
                identity_count=identity_count(selected), metadata_status="PRESENT",
                resource_status=resource_status, provenance_status="TRACEABLE_METADATA",
                primary=True, manual=False, evidence=evidence, evidence_sha=evidence_sha,
                notes="official category exact; src_file used as original identity",
            ))
            continue
        aliases = ESC_AMBIGUOUS.get(name, ())
        selected = [row for alias in aliases for row in by_label.get(alias, ())]
        if selected:
            result.append(_row(
                item, "ESC-50", source_label="|".join(aliases),
                source_label_id="|".join(sorted({str(row.get("target", "")) for row in selected})),
                mapping_type="AMBIGUOUS", semantic=True, candidate_count=len(selected),
                identity_count=identity_count(selected), metadata_status="PRESENT",
                resource_status=resource_status, provenance_status="TRACEABLE_METADATA",
                evidence=evidence, evidence_sha=evidence_sha,
                notes="nearby label only; Step 2A audition required",
            ))
        else:
            result.append(_row(
                item, "ESC-50", mapping_type="NONE", candidate_count=0, identity_count=0,
                metadata_status="PRESENT", resource_status=resource_status,
                provenance_status="TRACEABLE_METADATA", evidence=evidence, evidence_sha=evidence_sha,
                notes="no canonical or approved nearby category in official metadata",
            ))
    return result


def _load_desed_labels(metadata_path):
    if not metadata_path or not Path(metadata_path).is_file():
        return set()
    with Path(metadata_path).open(encoding="utf-8") as handle:
        return {str(key) for key in json.load(handle)}


def audit_desed(classes, metadata_path=None, foreground_root=None, evidence="", evidence_sha=""):
    labels = _load_desed_labels(metadata_path)
    resource_status = "PRESENT_PARTIAL" if foreground_root and Path(foreground_root).is_dir() else "MISSING"
    result = []
    for item in classes:
        name = str(item["name"])
        if name in DESED_EXACT and DESED_EXACT[name] in labels:
            mapping_type, exact, semantic = "EXACT", True, False
            source_label, manual = DESED_EXACT[name], False
            notes = "official DESED class label exact; isolated foreground resource status is independent"
        elif name in DESED_SEMANTIC and DESED_SEMANTIC[name] in labels:
            mapping_type, exact, semantic = "SEMANTIC_STRONG", False, True
            source_label, manual = DESED_SEMANTIC[name], True
            notes = "official DESED alarm label is broader than clock_alarm; Step 2A review required"
        else:
            mapping_type, exact, semantic = "NONE", False, False
            source_label, manual = "", True
            notes = "no approved DESED label for this canonical class; isolated resource status is independent"
        result.append(_row(
            item, "DESED isolated foreground", source_label=source_label,
            mapping_type=mapping_type, exact=exact, semantic=semantic,
            metadata_status="PRESENT_METADATA_ONLY" if labels else "MISSING",
            resource_status=resource_status,
            provenance_status="LABEL_ONLY_NO_REGISTRY" if labels else "UNKNOWN",
            evidence=evidence, evidence_sha=evidence_sha, manual=manual, notes=notes,
        ))
    return result


def _load_pseld_indices(train_path=None, test_path=None):
    index = {}
    for path, split in ((train_path, "train"), (test_path, "test")):
        if not path or not Path(path).is_file():
            continue
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if not row:
                    continue
                if len(row) != 5:
                    raise ValueError("invalid PSELD class-index row: {}".format(row))
                item = index.setdefault(row[1], {"id": row[0], "mid": row[1], "label": row[2]})
                if item["label"] != row[2] or item["id"] != row[0]:
                    raise ValueError("inconsistent PSELD class-index row for {}".format(row[1]))
                item["{}_clip_count".format(split)] = int(row[3])
                item["{}_duration".format(split)] = float(row[4])
    return index


def _source_tsv_mid(path):
    name = Path(path).name
    if name.startswith("_m_") and name.endswith(".tsv"):
        return "/m/" + name[3:-4]
    if name.startswith("_t_") and name.endswith(".tsv"):
        return "/t/" + name[3:-4]
    return ""


def _load_source_inventory(source_tsv_dir):
    inventory = {}
    if not source_tsv_dir or not Path(source_tsv_dir).is_dir():
        return inventory, []
    for path in sorted(Path(source_tsv_dir).glob("*.tsv")):
        mid = _source_tsv_mid(path)
        if not mid:
            continue
        records = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if not row:
                    continue
                if len(row) != 4:
                    raise ValueError("invalid source TSV row in {}: {}".format(path, row))
                records.append({"clip_id": row[0], "duration": float(row[1]), "labels": row[2], "split": row[3]})
        inventory[mid] = records
    manifest = []
    for mid in sorted(inventory):
        path = Path(source_tsv_dir) / ("_m_" + mid[3:] + ".tsv" if mid.startswith("/m/") else "_t_" + mid[3:] + ".tsv")
        records = inventory[mid]
        manifest.append({
            "mid": mid, "path": path.name, "size_bytes": path.stat().st_size,
            "url": "https://raw.githubusercontent.com/Jinbo-Hu/SELD-Data-Generator/main/source_datasets/single_source_samples/FSD50K/{}".format(path.name),
            "sha256": sha256_file(path), "row_count": len(records),
            "independent_clip_id_count": len({row["clip_id"] for row in records}),
        })
    return inventory, manifest


def _pseld_label_index(index, labels):
    by_label = defaultdict(list)
    for item in index.values():
        by_label[item["label"]].append(item)
    return [item for label in labels for item in by_label.get(label, [])]


def audit_pseld(classes, train_index=None, test_index=None, source_tsv_dir=None,
                audio_root=None, evidence="", evidence_sha=""):
    index = _load_pseld_indices(train_index, test_index)
    inventory, _ = _load_source_inventory(source_tsv_dir)
    resource_status = "PRESENT_PARTIAL" if audio_root and Path(audio_root).is_dir() else "PRESENT_METADATA_ONLY"
    metadata_status = "PRESENT" if index else "MISSING"
    provenance_status = "TRACEABLE_170_CLASS_TO_MID_TO_FSD_CLIP" if index and inventory else "UNESTABLISHED"
    result = []
    for item in classes:
        name = str(item["name"])
        rule = PSELD_RULES[name]
        matched = _pseld_label_index(index, rule["labels"])
        selected = [row for label_item in matched for row in inventory.get(label_item["mid"], [])]
        clip_ids = {row["clip_id"] for row in selected}
        if matched:
            mapping_type = rule["mapping_status"]
            exact, semantic = mapping_type == "EXACT", mapping_type == "SEMANTIC_STRONG"
        elif index:
            mapping_type, exact, semantic = "NONE", False, False
        else:
            mapping_type, exact, semantic = "BLOCKED_RESOURCE", False, False
        if matched and selected:
            count, identity_count = len(selected), len(clip_ids)
            count_note = "source TSV first column counted as FSD50K clip ID; unique IDs used for independent identity"
        elif matched and source_tsv_dir and Path(source_tsv_dir).is_dir():
            count, identity_count = None, None
            count_note = "official label present but corresponding source TSV inventory is incomplete"
        else:
            count = identity_count = 0 if index else None
            count_note = "official 170-class metadata contains no approved mapping label"
        result.append(_row(
            item, "PSELD-selected FSD50K",
            source_label="|".join(label_item["label"] for label_item in matched),
            source_label_id="|".join(label_item["mid"] for label_item in matched),
            mapping_type=mapping_type, exact=exact, semantic=semantic,
            candidate_count=count, identity_count=identity_count, metadata_status=metadata_status,
            resource_status=resource_status, provenance_status=provenance_status,
            pretraining_traceability="PARTIAL" if matched else "NOT_ESTABLISHED",
            supplement=bool(matched), manual=mapping_type != "EXACT",
            evidence=evidence, evidence_sha=evidence_sha,
            notes=("official PSELDNets 170-class label to MID and SELD-Data-Generator source TSV; " + count_note)
            if matched else count_note,
        ))
    return result


def audit_sources(ontology_path, esc50_metadata=None, esc50_audio_root=None,
                  desed_metadata=None, desed_foreground_root=None, pseld_selected=None,
                  pseld_audio_root=None, evidence=None, pseld_train_index=None,
                  pseld_test_index=None, pseld_source_tsv_dir=None):
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
    rows.extend(audit_pseld(classes, pseld_train_index, pseld_test_index, pseld_source_tsv_dir,
                             pseld_audio_root, pseld_evidence, pseld_sha))
    order = {dataset: index for index, dataset in enumerate(DATASETS)}
    return sorted(rows, key=lambda row: (row["canonical_class_id"], order[row["source_dataset"]], row["source_label"]))


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "" if row[key] is None else str(row[key]).lower() if isinstance(row[key], bool) else row[key]
                for key in CSV_FIELDS
            })


def _inventory(ontology_path, rows, evidence):
    datasets = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["source_dataset"] == dataset]
        datasets.append({
            "source_dataset": dataset, "mapping_rows": len(subset),
            "mapping_status_counts": {kind: sum(row["mapping_status"] == kind for row in subset) for kind in MAPPING_TYPES},
            "mapping_type_counts": {kind: sum(row["mapping_type"] == kind for row in subset) for kind in MAPPING_TYPES},
            "resource_statuses": sorted({row["resource_status"] for row in subset}),
            "metadata_statuses": sorted({row["metadata_status"] for row in subset}),
            "provenance_statuses": sorted({row["provenance_status"] for row in subset}),
            "evidence": evidence.get(dataset, {}),
        })
    return {
        "audit_schema_version": "clsdoa_v1_source_mapping_audit.2",
        "ontology_path": str(ontology_path), "ontology_sha256": sha256_file(ontology_path),
        "dataset_family_count": len(DATASETS),
        "canonical_class_count": len({row["canonical_class_id"] for row in rows}),
        "datasets": datasets, "prohibited_operations_performed": [],
    }


def _summary(rows, inventory):
    by_class = defaultdict(dict)
    for row in rows:
        by_class[row["canonical_class"]][row["source_dataset"]] = row
    primary = {
        "coughing": "ESC-50 (EXACT)", "laughing": "ESC-50 (EXACT)",
        "keyboard_typing": "ESC-50 (EXACT)", "vacuum_cleaner": "ESC-50 (EXACT)",
        "clock_alarm": "ESC-50 (EXACT)", "speech": "DESED isolated foreground (EXACT; resource missing)",
        "running_water": "DESED isolated foreground (EXACT; resource missing)",
        "frying": "DESED isolated foreground (EXACT; resource missing)",
        "mechanical_fan": "PSELDNets 170-class FSD50K inventory (EXACT, metadata-only)",
        "microwave_oven": "PSELDNets 170-class FSD50K inventory (EXACT, metadata-only)",
        "dishes": "DESED isolated foreground (EXACT; resource missing)",
        "printer": "PSELDNets 170-class FSD50K inventory (EXACT, metadata-only)",
    }
    lines = [
        "# ClassDOA V1 Step 1A.1 source mapping repair", "",
        "This is a metadata/provenance audit only. No source audio was downloaded, resampled, normalized, split, auditioned, or rendered.", "",
        "## 12-class summary matrix", "",
        "| canonical class | ESC-50 | DESED isolated | PSELDNets 170-class FSD inventory | Step 1A primary suggestion | supplement suggestion | unresolved |",
        "|---|---|---|---|---|---|---|",
    ]
    for class_name in sorted(by_class, key=lambda name: next(row["canonical_class_id"] for row in rows if row["canonical_class"] == name)):
        cells, unresolved = [], []
        for dataset in DATASETS:
            row = by_class[class_name][dataset]
            count = "" if row["candidate_count"] is None else "{} clips/{} ids".format(row["candidate_count"], row["independent_identity_count"])
            cells.append("{}{}".format(row["mapping_status"], " (" + count + ")" if count else ""))
            if row["mapping_status"] in ("AMBIGUOUS", "BLOCKED_RESOURCE") or row["manual_review_required"]:
                unresolved.append(dataset)
        pseld = by_class[class_name]["PSELD-selected FSD50K"]
        supplement = "PSELDNets 170-class FSD inventory" if pseld["supplement_candidate"] and not primary[class_name].startswith("PSELDNets") else "none identified"
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(class_name, *cells, primary[class_name], supplement, ", ".join(unresolved) or "none"))
    lines.extend([
        "", "## Repair conclusions", "",
        "DESED mapping status is now independent from resource status. With official event-occurrence metadata present, Speech, Running_water, Dishes, Frying, and Vacuum_cleaner are EXACT; Alarm_bell_ringing to clock_alarm is SEMANTIC_STRONG and requires manual review. The isolated foreground registry/audio is still absent locally, so DESED resource_status remains MISSING and candidate/identity counts remain blank.", "",
        "PSELDNets class-index metadata contains 170 official classes. The canonical mapping is resolved through the official PSELD label, its AudioSet MID, and the matching SELD-Data-Generator FSD50K TSV. Printer, Microwave oven, and Mechanical fan are now supported by exact PSELD labels and source TSV inventories. Vacuum cleaner is not an official PSELDNets 170-class label and is therefore NONE in this PSELD row; the previous static Vacuum_cleaner mapping was removed.", "",
        "The Zenodo FSD50K_selected.txt file is retained only as supplementary DCASE2022 evidence. It is explicitly not treated as a PSELD pretraining registry: DCASE2022_SELECTED_FSD50K != PSELD_PRETRAIN_SELECTED_FSD50K. The repaired PSELD crosswalk is traceable to the 170-class index and generator TSVs, but direct checkpoint-training membership is not proven; pretraining_exposure_traceability remains PARTIAL.", "",
        "PSELD metadata coverage is exact for speech, frying, mechanical_fan, microwave_oven, and printer; semantic-strong for coughing, laughing, keyboard_typing, clock_alarm, running_water, and dishes; and NONE for vacuum_cleaner. Supplement labels for speech and water-related sounds are recorded in evidence but are not silently promoted to exact mappings.", "",
        "License/provenance is dataset-level or source-ID-level only. Per-recording license completion, audio availability, source QC, and audibility remain DEFERRED_TO_STEP_2A. No source audio was downloaded.", "",
        "Step 2A cannot start as a complete Source Track. Remaining blockers are missing local ESC/DESED/PSELD source audio, DESED isolated foreground registry, per-recording license/QC completion, and lack of direct PSELD checkpoint-training membership crosswalk. Readiness remains PARTIAL and awaits human review.", "",
        "## Determinism", "", "Rows are sorted by canonical_class_id, source dataset, and source label. Re-running the same metadata inputs must produce byte-identical CSV/JSON content.", "",
        "Inventory mapping-row total: {}. Prohibited operations recorded: none.".format(sum(item["mapping_rows"] for item in inventory["datasets"])),
    ])
    return "\n".join(lines) + "\n"


def _build_pseld_evidence(train_index, test_index, source_tsv_dir):
    index = _load_pseld_indices(train_index, test_index)
    inventory, manifest = _load_source_inventory(source_tsv_dir)
    crosswalk = []
    for name, rule in PSELD_RULES.items():
        matched = _pseld_label_index(index, rule["labels"])
        supplements = _pseld_label_index(index, PSELD_SUPPLEMENT_LABELS.get(name, ()))
        entries = []
        for label_item in matched + supplements:
            mid, records = label_item["mid"], inventory.get(label_item["mid"], [])
            manifest_item = next((entry for entry in manifest if entry["mid"] == mid), None)
            entries.append({
                "label": label_item["label"], "mid": mid, "pseld_class_id": int(label_item["id"]),
                "train_clip_count": label_item.get("train_clip_count"), "test_clip_count": label_item.get("test_clip_count"),
                "source_tsv": manifest_item, "source_tsv_clip_count": len(records),
                "source_tsv_independent_identity_count": len({row["clip_id"] for row in records}),
                "role": "primary_mapping" if label_item in matched else "supplement_review_candidate",
            })
        crosswalk.append({
            "canonical_class": name, "mapping_status": rule["mapping_status"] if matched else "NONE",
            "official_170_label_present": bool(matched), "labels": entries,
            "supplement_labels_not_promoted": [item["label"] for item in supplements],
        })
    return {
        "official_170_class_count": len(index),
        "train_index": {"path": str(train_index) if train_index else "MISSING", "sha256": sha256_file(train_index) if train_index else None},
        "test_index": {"path": str(test_index) if test_index else "MISSING", "sha256": sha256_file(test_index) if test_index else None},
        "source_tsv_count": len(manifest), "source_tsv_manifest": manifest,
        "canonical_to_pseld_to_mid_to_fsd_crosswalk": crosswalk,
        "pretraining_exposure_traceability": "PARTIAL",
        "statement": "DCASE2022_SELECTED_FSD50K != PSELD_PRETRAIN_SELECTED_FSD50K; the DCASE2022 registry is supplementary only.",
    }


def write_outputs(output_dir, ontology_path, rows, evidence):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = _inventory(ontology_path, rows, evidence)
    (output_dir / "source_dataset_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "source_class_mapping_audit.csv", rows)
    evidence_payload = {
        "audit_schema_version": "clsdoa_v1_source_mapping_evidence.2",
        "official_and_local_evidence": evidence,
        "notes": [
            "Evidence files are metadata/class-list documents only.",
            "No source audio was downloaded or written by this audit.",
            "DESED mapping_status is independent of DESED resource_status.",
            "DCASE2022_SELECTED_FSD50K != PSELD_PRETRAIN_SELECTED_FSD50K.",
            "PSELD pretraining exposure remains PARTIAL because direct checkpoint-training membership is not proven.",
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
    parser.add_argument("--pseld-selected")  # compatibility only; never used as PSELD registry
    parser.add_argument("--pseld-audio-root")
    parser.add_argument("--pseld-train-index")
    parser.add_argument("--pseld-test-index")
    parser.add_argument("--pseld-source-tsv-dir")
    parser.add_argument("--evidence-json", required=True)
    args = parser.parse_args(argv)
    with Path(args.evidence_json).open(encoding="utf-8") as handle:
        evidence = json.load(handle)
    # Permit deterministic re-runs using the previously generated evidence output.
    while "official_and_local_evidence" in evidence:
        evidence = evidence["official_and_local_evidence"]
    if args.pseld_train_index or args.pseld_test_index or args.pseld_source_tsv_dir:
        evidence.setdefault("PSELD-selected FSD50K", {})["repaired_crosswalk"] = _build_pseld_evidence(
            args.pseld_train_index, args.pseld_test_index, args.pseld_source_tsv_dir
        )
    rows = audit_sources(
        args.ontology, args.esc50_metadata, args.esc50_audio_root,
        args.desed_metadata, args.desed_foreground_root, args.pseld_selected,
        args.pseld_audio_root, evidence, args.pseld_train_index, args.pseld_test_index,
        args.pseld_source_tsv_dir,
    )
    write_outputs(args.output_dir, args.ontology, rows, evidence)


if __name__ == "__main__":
    main()
