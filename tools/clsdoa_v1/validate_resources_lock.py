#!/usr/bin/env python
"""Strict, explicit validator for the ClassDOA V1 final resource lock."""

import argparse
import json
import sys
from pathlib import Path

try:
    from .build_final_provenance import AUTHORITY, COMMITS, ONTOLOGY, SCENE, SOURCE, _scene_summary, _source_summary, sha256
except ImportError:  # pragma: no cover - direct ``python tools/...py`` entrypoint
    from build_final_provenance import AUTHORITY, COMMITS, ONTOLOGY, SCENE, SOURCE, _scene_summary, _source_summary, sha256


class LockValidationError(ValueError):
    pass


def _same(actual, expected, name):
    if actual != expected:
        raise LockValidationError("{} mismatch: {!r} != {!r}".format(name, actual, expected))


def validate(lock):
    required = ("schema_version", "dataset_schema_version", "code", "source", "scene", "ontology", "runtime", "acoustics", "foa", "step2b_evidence_authority")
    for key in required:
        if key not in lock:
            raise LockValidationError("missing top-level field: {}".format(key))
    _same(lock["schema_version"], "clsdoa_v1.resources_lock.1", "schema_version")
    _same(lock["dataset_schema_version"], "clsdoa_v1.0", "dataset_schema_version")
    for key, value in COMMITS.items():
        _same(lock["code"].get(key), value, "code." + key)
    source = lock["source"]
    expected_source = _source_summary()
    for key in ("registry_path", "registry_sha256", "rows", "classes", "split_version", "split_counts", "split_overlap", "base_clip_id_unique", "canonical_paths_existing"):
        _same(source.get(key), expected_source[key], "source." + key)
    if source["registry_sha256"] != sha256(SOURCE) or source["external_pool_identity"]["snapshot_sha256"] != source["registry_sha256"]:
        raise LockValidationError("source registry or pool snapshot drift")
    scene = lock["scene"]
    expected_scene = _scene_summary()
    for key in ("registry_path", "registry_sha256", "total_records", "status_counts", "split_version", "split_overlap", "materials_mode", "unit_scale_status"):
        _same(scene.get(key), expected_scene[key], "scene." + key)
    if scene["registry_sha256"] != sha256(SCENE):
        raise LockValidationError("scene registry drift")
    _same(lock["ontology"], {"path": "registries/ontology.yaml", "sha256": sha256(ONTOLOGY)}, "ontology")
    _same(lock["step2b_evidence_authority"]["pointer"], "docs/audits/clsdoa_v1/integration/step2b_evidence_authority.json", "authority pointer")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    _same(lock["step2b_evidence_authority"]["final_commit"], authority["authoritative"]["final_commit"], "authority final commit")
    _same(lock["step2b_evidence_authority"]["authoritative_files"], authority["authoritative"]["acoustic_evidence"], "authority files")
    if lock["step2b_evidence_authority"].get("superseded_evidence_in_lock") is not False:
        raise LockValidationError("superseded evidence is marked authoritative")
    acoustic = lock["acoustics"]
    for key, expected in (("sample_rate_hz", 24000), ("clip_duration_sec", 5.0), ("indirect_ray_count", 5000), ("source_ray_count", 200), ("materials_enabled", False), ("receiver_contract", "listener_base + [0,1.5,0]"), ("paired_lifecycle", "sequential_sensor_lifecycle")):
        _same(acoustic.get(key), expected, "acoustics." + key)
    _same(acoustic["normalization"], {"per_render": False, "per_viewpoint": False}, "acoustics.normalization")
    _same(lock["foa"]["contract_commit"], COMMITS["foa_contract_commit"], "foa contract")
    _same(lock["foa"]["canonical"], "AmbiX ACN [W,Y,Z,X] SN3D", "foa canonical")
    runtime = lock["runtime"]
    _same(runtime["habitat_sim_version"], "0.2.2", "Habitat-Sim version")
    rlr = runtime["rlraudio_propagation"]
    if not Path(rlr["path"]).is_file() or rlr["sha256"] != sha256(Path(rlr["path"])) or rlr["size_bytes"] != Path(rlr["path"]).stat().st_size:
        raise LockValidationError("RLRAudioPropagation fingerprint mismatch")
    hrtf = runtime["hrtf"]
    _same(hrtf.get("status"), "NOT_EXPOSED_BY_INSTALLED_HABITAT_SIM", "HRTF status")
    if hrtf.get("path") is not None or hrtf.get("sha256") is not None:
        raise LockValidationError("HRTF standalone hash must remain null")
    if lock.get("superseded_evidence") is not None:
        raise LockValidationError("superseded evidence must not be embedded in authoritative lock")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate(json.loads(args.lock.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("FAIL: {}".format(exc), file=sys.stderr)
        return 1
    print("PASS: {}".format(args.lock))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
