"""Build a read-only Step 1B scene asset readiness inventory."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


STATUSES = {"PRESENT_COMPLETE", "PRESENT_PARTIAL", "MISSING", "BROKEN_PATH", "BLOCKED_PERMISSION"}


def small_file_fingerprint(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    if path.stat().st_size > 8 * 1024 * 1024:
        stat = path.stat()
        return "size:{}:mtime_ns:{}".format(stat.st_size, stat.st_mtime_ns)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:{}".format(digest)


def resolve_ref(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def parse_stage_refs(config_path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"parse_error": "{}: {}".format(type(exc).__name__, exc), "references": {}}
    refs = {}
    for key in ("render_asset", "semantic_asset", "nav_asset", "semantic_descriptor_filename"):
        value = data.get(key)
        if isinstance(value, str):
            refs[key] = {"declared": value, "resolved": str(resolve_ref(config_path, value)), "exists": resolve_ref(config_path, value).is_file()}
    return {"parse_error": None, "references": refs}


def readable(path: Optional[Path]) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def make_entry(scene_id: str, family: str, root: Path, habitat_root: Optional[Path]) -> Dict[str, Any]:
    habitat = habitat_root or root
    stage = next(iter(sorted(habitat.glob("*.stage_config.json"))), None)
    mesh = habitat / "mesh_semantic.ply"
    navmesh = habitat / "mesh_semantic.navmesh"
    semantic = habitat / "info_semantic.json"
    stage_info = parse_stage_refs(stage) if stage else {"parse_error": "missing stage config", "references": {}}
    core = {"mesh": mesh, "navmesh": navmesh, "semantic_info": semantic}
    missing = [key for key, path in core.items() if not path.is_file()]
    blocked = any(path.exists() and not readable(path) for path in core.values())
    core_reference_names = {
        "semantic_asset": mesh.name,
        "nav_asset": navmesh.name,
    }
    core_path_errors = [
        key for key, expected in core_reference_names.items()
        if key in stage_info["references"]
        and stage_info["references"][key]["declared"] != expected
        and Path(stage_info["references"][key]["resolved"]).name != expected
        and (core["mesh" if key == "semantic_asset" else "navmesh"].is_file())
    ]
    stage_config_warning = [
        {"key": key, **item}
        for key, item in stage_info["references"].items()
        if not item["exists"] and key not in core_path_errors
    ]
    if blocked:
        status = "BLOCKED_PERMISSION"
    elif missing:
        status = "PRESENT_PARTIAL"
    elif stage_info["parse_error"] or core_path_errors:
        status = "BROKEN_PATH"
    else:
        status = "PRESENT_COMPLETE"
    assert status in STATUSES
    required = {"stage_config": stage, **core}
    key_paths = {key: (str(path) if path else None) for key, path in required.items()}
    sizes = {key: (path.stat().st_size if path and path.is_file() else None) for key, path in required.items()}
    fingerprint = {key: small_file_fingerprint(path) for key, path in required.items()}
    return {
        "scene_id": scene_id,
        "scene_family": family,
        "root": str(root),
        "habitat_root": str(habitat),
        "stage_config_path": key_paths["stage_config"],
        "scene_asset_path": key_paths["mesh"],
        "mesh_path": key_paths["mesh"],
        "semantic_mesh_path": key_paths["mesh"],
        "navmesh_path": key_paths["navmesh"],
        "semantic_info_path": key_paths["semantic_info"],
        "textures_present": any(habitat.glob("*.jpg")) or any(habitat.glob("*.png")),
        "file_readable": not blocked and not missing,
        "key_resource_size_bytes": sizes,
        "resource_fingerprint": fingerprint,
        "stage_config_parse_error": stage_info["parse_error"],
        "stage_config_references": stage_info["references"],
        "readiness_status": status,
        "missing_fields": missing,
        "stage_config_warning": stage_config_warning,
        "broken_core_references": core_path_errors,
        "unit_scale_status": "TO_VERIFY_IN_STEP_2B",
        "materials_mode_expected": "off",
        "notes": "Step 1B structural readiness only; no Habitat load, render, repair, admission, or split.",
    }


def audit_replica(root: Path) -> List[Dict[str, Any]]:
    entries = []
    for scene_root in sorted(path for path in root.iterdir() if path.is_dir()):
        habitat = scene_root / "habitat"
        entries.append(make_entry("replica." + scene_root.name, "Replica", scene_root, habitat if habitat.is_dir() else None))
    return entries


def audit_mp3d(root: Path) -> Dict[str, Any]:
    scans = []
    for scan_root in sorted(path for path in root.iterdir() if path.is_dir()):
        if scan_root.name == "v1":
            for candidate in sorted(path for path in scan_root.rglob("*") if path.is_dir() and path.name != "tasks"):
                scans.append(make_entry("mp3d." + candidate.name, "MP3D", candidate, candidate))
        else:
            scans.append(make_entry("mp3d." + scan_root.name, "MP3D", scan_root, scan_root))
    archives = sorted(str(path) for path in root.rglob("*.zip"))
    return {"scene_family": "MP3D", "root": str(root), "scans": scans, "archives_present": archives, "scan_count": len(scans), "notes": "Archives are not treated as extracted scans."}


def candidate(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scene_id": entry["scene_id"], "scene_family": entry["scene_family"],
        "scene_asset": entry["scene_asset_path"], "stage_config": entry["stage_config_path"],
        "navmesh": entry["navmesh_path"], "semantic_info": entry["semantic_info_path"],
        "materials_mode": "off", "readiness_status": entry["readiness_status"],
        "resource_fingerprint": entry["resource_fingerprint"], "unit_scale_status": "TO_VERIFY_IN_STEP_2B",
        "scene_load": "NOT_RUN", "navmesh_admission": "NOT_RUN", "source_clearance": "NOT_RUN",
        "binaural_admission": "NOT_RUN", "foa_admission": "NOT_RUN", "acoustic_sanity": "NOT_RUN",
        "admitted": "NOT_RUN", "exclude_reason": None, "split": "UNASSIGNED",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-root", required=True)
    parser.add_argument("--mp3d-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    replica = sorted(audit_replica(Path(args.replica_root)), key=lambda item: item["scene_id"])
    mp3d = audit_mp3d(Path(args.mp3d_root))
    all_entries = replica + sorted(mp3d["scans"], key=lambda item: item["scene_id"])
    write_json(output / "replica_inventory.json", replica)
    write_json(output / "mp3d_inventory.json", mp3d)
    write_json(output / "scene_candidate_inventory.json", [candidate(entry) for entry in all_entries])
    print(json.dumps({"replica_count": len(replica), "replica_statuses": {status: sum(item["readiness_status"] == status for item in replica) for status in sorted(STATUSES)}, "mp3d_scan_count": mp3d["scan_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
