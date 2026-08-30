"""Shared immutable plan and payload integrity gates."""

import hashlib
import json
from pathlib import Path

import yaml


class IntegrityError(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_plan_integrity(root, repo_root):
    root, repo_root = Path(root), Path(repo_root)
    lock = json.loads((root / "manifests/plan.lock.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((root / "config_resolved.yaml").read_text(encoding="utf-8"))
    checks = {
        "episodes_sha256": root / "manifests/episodes.jsonl",
        "config_resolved_sha256": root / "config_resolved.yaml",
        "resources_lock_sha256": root / "resources.lock.json",
    }
    for field, path in checks.items():
        if lock.get(field) != _sha(path):
            raise IntegrityError(field + " mismatch")
    def configured_path(section, fallback):
        path = Path(resolved.get(section, {}).get("registry_path", fallback))
        return path if path.is_absolute() else repo_root / path

    for field, path in (("source_registry_sha256", configured_path("source", "registries/source_audio.csv")), ("scene_registry_sha256", configured_path("scene", "registries/clsdoa_v1_scenes.yaml"))):
        if lock.get(field) != _sha(path) or identity.get(field) != _sha(path):
            raise IntegrityError(field + " mismatch")
    if identity.get("generation_code_commit") != lock.get("plan_generation_code_commit"):
        raise IntegrityError("generation commit mismatch")
    if resolved.get("storage", {}).get("require_rir") is not True:
        raise IntegrityError("resolved config must require RIR")
    return {"status": "PASS", "plan_generation_code_commit": lock["plan_generation_code_commit"]}
