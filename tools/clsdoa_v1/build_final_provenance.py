#!/usr/bin/env python
"""Build the ClassDOA V1 final resource lock from frozen local evidence."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "registries/source_audio.csv"
SCENE = ROOT / "registries/clsdoa_v1_scenes.yaml"
ONTOLOGY = ROOT / "registries/ontology.yaml"
AUTHORITY = ROOT / "docs/audits/clsdoa_v1/integration/step2b_evidence_authority.json"
POOL = ROOT / "docs/audits/clsdoa_v1/source_pool_pilot_001/pool_freeze_manifest.json"
RLR = Path("/home/leiyuxin/miniconda3/envs/ss/lib/python3.9/site-packages/habitat_sim/_ext/libRLRAudioPropagation.so")

COMMITS = {
    "production_renderer_origin_commit": "4dd94a4b06977acaecdb0ba9af7ab7a7ba2ca78d",
    "integrated_generation_code_commit": "552c299fa14d355f877eca7ff068154dcc463c21",
    "integration_evidence_commit": "825c391685ba5c56869f03b7eeda96604766d0d4",
    "integration_evidence_hygiene_commit": "5adb70a59490da4db3bf13e3e31e4fb23974230b",
    "foa_contract_commit": "3e3c21ab0151a0b0d8e8d1e0946e9fa015c7bf92",
    "source_pool_generation_commit": "d55153ffab31fd1f9841fc65bb5ae0909fa58a55",
    "source_track_final_hardening_commit": "8f2e0888fde6fbc69d92293b0204066fcd54398d",
    "scene_track_final_commit": "86d8b6f8c65220fa5801d81437009ed4382ab34d",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _source_summary():
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    splits = Counter(row["split"] for row in rows)
    classes = {row["canonical_class"] for row in rows}
    identities = [row["base_clip_id"] for row in rows]
    paths = [Path(row["canonical_path"]) for row in rows]
    split_ids = defaultdict(set)
    for row in rows:
        split_ids[row["split"]].add(row["base_clip_id"])
    overlaps = {"train_val": len(split_ids["train"] & split_ids["val"]), "train_test": len(split_ids["train"] & split_ids["test"]), "val_test": len(split_ids["val"] & split_ids["test"])}
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    return {
        "registry_path": "registries/source_audio.csv", "registry_sha256": sha256(SOURCE),
        "rows": len(rows), "classes": len(classes), "split_version": rows[0]["split_version"],
        "split_counts": dict(sorted(splits.items())), "base_clip_id_unique": len(set(identities)) == len(identities),
        "split_overlap": overlaps, "canonical_paths_existing": sum(path.is_file() for path in paths),
        "external_pool_identity": {"pool_id": pool["pool_id"], "root": pool["external_pool_root"], "manifest": "docs/audits/clsdoa_v1/source_pool_pilot_001/pool_freeze_manifest.json", "snapshot_sha256": pool["source_registry_snapshot_sha256"]},
        "pool_generation_commit": COMMITS["source_pool_generation_commit"],
        "track_final_hardening_commit": COMMITS["source_track_final_hardening_commit"],
    }


def _scene_summary():
    data = yaml.safe_load(SCENE.read_text(encoding="utf-8"))
    scene_table = data.get("scenes", {}) if isinstance(data, dict) else data
    rows = []
    if isinstance(scene_table, dict):
        rows = [dict(value, scene_id=key) for key, value in scene_table.items()]
    else:
        rows = list(scene_table or [])
    status = Counter(str(row.get("admitted", row.get("status", row.get("admission_status", "")))) for row in rows)
    family_status = Counter((row.get("scene_family", row.get("family")), row.get("admitted", row.get("status", row.get("admission_status", "")))) for row in rows)
    split_ids = defaultdict(set)
    for row in rows:
        split = row.get("split", "UNASSIGNED")
        split_ids[split].add(row.get("scene_id"))
    overlap = sum(len(split_ids[a] & split_ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    return {
        "registry_path": "registries/clsdoa_v1_scenes.yaml", "registry_sha256": sha256(SCENE),
        "total_records": len(rows), "status_counts": dict(sorted(status.items())),
        "family_status_counts": {"Replica": {key[1]: value for key, value in family_status.items() if key[0] == "Replica"}, "MP3D": {key[1]: value for key, value in family_status.items() if key[0] == "MP3D"}},
        "split_counts": dict(sorted(Counter(row.get("split", "UNASSIGNED") for row in rows if str(row.get("admitted", row.get("status", row.get("admission_status", "")))) == "PASS").items())),
        "split_overlap": overlap, "split_version": "clsdoa_v1_scene_split_v2", "materials_mode": "OFF", "unit_scale_status": sorted({row.get("unit_scale_status") for row in rows}),
        "track_final_commit": COMMITS["scene_track_final_commit"],
    }


def build_lock():
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    rlr = {"path": str(RLR), "size_bytes": RLR.stat().st_size, "sha256": sha256(RLR)}
    version = importlib.metadata.version("habitat-sim")
    return {
        "schema_version": "clsdoa_v1.resources_lock.1", "dataset_schema_version": "clsdoa_v1.0", "generated_from_commit": _git_head(),
        "code": dict(COMMITS),
        "source": _source_summary(), "scene": dict(_scene_summary()),
        "step2b_evidence_authority": {"pointer": "docs/audits/clsdoa_v1/integration/step2b_evidence_authority.json", "final_commit": authority["authoritative"]["final_commit"], "authoritative_files": authority["authoritative"]["acoustic_evidence"], "superseded_evidence_in_lock": False},
        "ontology": {"path": "registries/ontology.yaml", "sha256": sha256(ONTOLOGY)},
        "acoustics": {"sample_rate_hz": 24000, "clip_duration_sec": 5.0, "indirect_ray_count": 5000, "source_ray_count": 200, "materials_enabled": False, "receiver_contract": "listener_base + [0,1.5,0]", "normalization": {"per_render": False, "per_viewpoint": False}, "paired_lifecycle": "sequential_sensor_lifecycle"},
        "foa": {"contract_commit": COMMITS["foa_contract_commit"], "native": "N3D [W,Y_RLR,Z_RLR,X_RLR]", "canonical": "AmbiX ACN [W,Y,Z,X] SN3D"},
        "runtime": {"habitat_sim_version": version, "rlraudio_propagation": rlr, "hrtf": {"mode": "embedded_or_not_exposed", "path": None, "sha256": None, "status": "NOT_EXPOSED_BY_INSTALLED_HABITAT_SIM", "enclosing_implementation": rlr}},
        "soundspaces_commit": _git_head(), "habitat_sim_version": version, "rlraudio_propagation": rlr, "hrtf": {"mode": "embedded_or_not_exposed", "path": None, "sha256": None, "status": "NOT_EXPOSED_BY_INSTALLED_HABITAT_SIM", "enclosing_implementation": rlr}, "foa_contract_version": "P0-B@" + COMMITS["foa_contract_commit"], "ontology_sha256": sha256(ONTOLOGY), "source_registry_sha256": sha256(SOURCE), "scene_registry_sha256": sha256(SCENE), "source_split_version": "clsdoa_source_split_v2_stratified", "scene_split_version": "clsdoa_v1_scene_split_v2", "sample_rate_hz": 24000, "clip_duration_sec": 5.0, "indirectRayCount": 5000, "sourceRayCount": 200, "materials_mode": "OFF", "random_seed": 20260829,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_lock(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
