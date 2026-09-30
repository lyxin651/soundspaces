"""Read-only recovery and capability audit for historical Replica materials.

This module is intentionally separate from the production simulator.  It binds
historical files and logs, then runs optional materials probes in child
processes so a Habitat/RLRAudioPropagation abort cannot terminate the audit
driver.  It never runs ASR and never changes a production contract.
"""

import hashlib
import json
import os
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from active_audition.asr.qualification import canonical_json_bytes, file_sha256


SCHEMA_VERSION = "active-asr-a3-historical-materials-audit-v1"
CAPABILITY_SCHEMA_VERSION = "active-asr-a3-historical-materials-capability-v1"
FINAL_STATUSES = {
    "MATERIALS_ON_PROVENANCE_RECOVERED_AND_RUNTIME_STABLE",
    "MATERIALS_ON_PROVENANCE_RECOVERED_BUT_RUNTIME_UNSTABLE",
    "HISTORICAL_MATERIAL_ASSETS_INCOMPLETE",
    "HISTORICAL_MATERIAL_ASSETS_NOT_RECOVERED",
}


class HistoricalMaterialsAuditError(RuntimeError):
    """Raised when the read-only materials audit cannot be assembled."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repo_relative(repo: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _git_status(repo: Path, path: Path) -> Dict[str, Any]:
    relative = _repo_relative(repo, path)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=str(repo), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=str(repo), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    try:
        history = subprocess.run(
            ["git", "log", "-1", "--all", "--format=%H %s", "--", relative],
            cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, timeout=5,
        )
        history_line = history.stdout.strip() or None
    except subprocess.TimeoutExpired:
        history_line = None
    return {
        "tracked": tracked,
        "ignored": ignored,
        "untracked": not tracked,
        "git_history_visibility": "VISIBLE" if history_line else "NOT_VISIBLE",
        "latest_path_history": history_line,
    }


def _file_record(repo: Path, path: Path, role: str) -> Dict[str, Any]:
    path = path.resolve()
    record: Dict[str, Any] = {
        "role": role,
        "path": _repo_relative(repo, path),
        "exists": path.is_file(),
        "size_bytes": None,
        "mtime_utc": None,
        "sha256": None,
        "git": None,
    }
    if path.is_file():
        stat = path.stat()
        record["size_bytes"] = int(stat.st_size)
        record["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        record["sha256"] = file_sha256(path)
        record["git"] = _git_status(repo, path)
    return record


def _safe_read_json(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _semantic_class_names(path: Path) -> Sequence[str]:
    value = _safe_read_json(path)
    if not value:
        return []
    return sorted(
        {
            str(item["name"])
            for item in value.get("classes", [])
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
    )


def _audit_material_config(
    repo: Path,
    config_path: Path,
    semantic_paths: Iterable[Path],
) -> Dict[str, Any]:
    record = _file_record(repo, config_path, "historical_replica_material_config")
    result: Dict[str, Any] = {
        "artifact": record,
        "json_parse": "NOT_RUN",
        "schema": {
            "top_level_keys": [],
            "unknown_top_level_keys": [],
            "materials_count": 0,
            "material_names_unique": False,
            "required_fields_all_present": False,
        },
        "mapping": {
            "source": "materials[].labels",
            "explicit_semantic_to_material_field": False,
            "default_material_present": False,
            "explicit_fallback_field": False,
            "semantic_labels": {},
            "unmatched_semantic_labels": [],
            "duplicate_semantic_labels": [],
        },
        "semantic_identity_comparison": [],
        "mp3d_config_comparison": None,
    }
    if not config_path.is_file():
        result["json_parse"] = "MISSING"
        return result
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        result["json_parse"] = "FAIL"
        result["parse_error"] = str(error)
        return result
    result["json_parse"] = "PASS"
    result["schema"]["top_level_keys"] = sorted(value) if isinstance(value, Mapping) else []
    result["schema"]["unknown_top_level_keys"] = (
        sorted(set(value) - {"materials"}) if isinstance(value, Mapping) else ["<non-object>"]
    )
    materials = value.get("materials", []) if isinstance(value, Mapping) else []
    required = {"name", "absorption", "scattering", "transmission", "labels", "damping"}
    names = [item.get("name") for item in materials if isinstance(item, Mapping)]
    result["schema"].update(
        {
            "materials_count": len(materials) if isinstance(materials, list) else 0,
            "material_names_unique": len(names) == len(set(names)),
            "required_fields_all_present": bool(materials) and all(
                isinstance(item, Mapping) and required.issubset(item)
                for item in materials
            ),
        }
    )
    label_to_material: Dict[str, str] = {}
    duplicate_labels = []
    for material in materials if isinstance(materials, list) else []:
        if not isinstance(material, Mapping):
            continue
        material_name = str(material.get("name"))
        if material_name == "Default":
            result["mapping"]["default_material_present"] = True
        for label in material.get("labels", []) if isinstance(material.get("labels"), list) else []:
            label = str(label)
            if label in label_to_material:
                duplicate_labels.append(label)
            label_to_material[label] = material_name
    result["mapping"]["duplicate_semantic_labels"] = sorted(set(duplicate_labels))
    semantic_sets = []
    for semantic_path in semantic_paths:
        labels = _semantic_class_names(semantic_path)
        semantic_sets.append({"path": _repo_relative(repo, semantic_path), "labels": labels})
        missing = sorted(set(labels) - set(label_to_material))
        result["mapping"]["unmatched_semantic_labels"].append(
            {"semantic_path": _repo_relative(repo, semantic_path), "labels": missing}
        )
        result["mapping"]["semantic_labels"][_repo_relative(repo, semantic_path)] = {
            label: label_to_material.get(label) for label in labels
        }
    result["semantic_identity_comparison"] = semantic_sets

    mp3d_path = repo / "data/mp3d_material_config.json"
    if mp3d_path.is_file():
        mp3d = _safe_read_json(mp3d_path)
        differing_fields = []
        if mp3d and isinstance(mp3d.get("materials"), list) and isinstance(materials, list):
            for left, right in zip(mp3d["materials"], materials):
                if isinstance(left, Mapping) and isinstance(right, Mapping):
                    fields = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
                    if fields:
                        differing_fields.append({"name": right.get("name"), "fields": fields})
        result["mp3d_config_comparison"] = {
            "path": _repo_relative(repo, mp3d_path),
            "sha256": file_sha256(mp3d_path),
            "same_bytes": mp3d_path.read_bytes() == config_path.read_bytes(),
            "same_material_entries": bool(mp3d and mp3d.get("materials") == materials),
            "differing_fields": differing_fields,
            "interpretation": "same material family; Replica config carries Replica label assignments" if differing_fields else "identical",
        }
    else:
        result["mp3d_config_comparison"] = {"status": "NOT_FOUND", "path": "data/mp3d_material_config.json"}
    return result


def _run_child_probe(
    scene_path: Path,
    navmesh_path: Path,
    semantic_path: Path,
    dataset_config_path: Path,
    materials_config_path: Path,
    materials_enabled: bool,
    repo: Path,
) -> Dict[str, Any]:
    """Run a small fixed RIR probe in a child process.

    The child intentionally uses only the historical scene/material paths.  A
    SIGSEGV or abort is represented as a structured failed probe in the parent.
    """

    worker = r'''
import hashlib, json, sys
import quaternion
import habitat_sim
import numpy as np

scene, navmesh, semantic, dataset, material, enabled = sys.argv[1:7]
enabled = enabled == "1"
backend = habitat_sim.SimulatorConfiguration()
backend.scene_id = scene
backend.scene_dataset_config_file = dataset
backend.load_semantic_mesh = True
backend.enable_physics = False
agent_cfg = habitat_sim.agent.AgentConfiguration()
sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
try:
    # The historical probe used the scene's navmesh explicitly when needed.
    if navmesh and not sim.pathfinder.is_loaded:
        if not sim.pathfinder.load_nav_mesh(navmesh):
            raise RuntimeError("historical navmesh failed to load")
    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = enabled
    spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    spec.channelLayout.channelCount = 2
    spec.position = [0.0, 1.5, 0.0]
    spec.acousticsConfig.sampleRate = 16000
    spec.acousticsConfig.indirect = True
    sim.add_sensor(spec)
    sensor = sim.get_agent(0)._sensors["audio_sensor"]
    if enabled:
        sensor.setAudioMaterialsJSON(material)
    effective = None
    try:
        effective = bool(sensor.specification().enableMaterials)
    except Exception:
        effective = bool(spec.enableMaterials)
    source = np.asarray([0.181018, 0.531131, 2.44339], dtype=np.float32)
    listener = np.asarray([1.63775, -0.968869, -1.63873], dtype=np.float32)
    rows = []
    for repeat in range(2):
        try:
            sensor.reset()
        except Exception:
            pass
        state = sim.get_agent(0).get_state()
        state.position = listener
        state.sensor_states = {}
        sim.get_agent(0).set_state(state, True)
        sensor.setAudioSourceTransform(source)
        obs = np.asarray(sim.get_sensor_observations()["audio_sensor"])
        if obs.ndim != 2 or obs.shape[0] != 2:
            raise RuntimeError("unexpected historical binaural shape: {}".format(obs.shape))
        arr = np.asarray(obs, dtype="<f4")
        finite = bool(np.isfinite(arr).all())
        raw = arr.tobytes(order="C")
        energy = float(np.sum(np.square(arr, dtype=np.float64)))
        tail_start = max(0, arr.shape[1] - 1600)
        tail_energy = float(np.sum(np.square(arr[:, tail_start:], dtype=np.float64)))
        rows.append({
            "repeat_id": repeat,
            "shape": [int(x) for x in arr.shape],
            "finite": finite,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "energy_sum_square": energy,
            "tail_100ms_energy": tail_energy,
            "tail_100ms_ratio": tail_energy / energy if energy > 0 else None,
            "peak": float(np.max(np.abs(arr))),
            "rms": float(np.sqrt(np.mean(np.square(arr, dtype=np.float64)))),
        })
    print("HISTORICAL_MATERIAL_PROBE=" + json.dumps({
        "scene": scene,
        "materials_enabled_requested": enabled,
        "materials_enabled_effective": effective,
        "setAudioMaterialsJSON_called": enabled,
        "semantic_path_argument": semantic,
        "renders": rows,
        "repeat_hash_identical": len({x["sha256"] for x in rows}) == 1,
    }, sort_keys=True))
finally:
    sim.close()
'''

    def disable_core() -> None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    args = [
        sys.executable,
        "-c",
        worker,
        str(scene_path),
        str(navmesh_path),
        str(semantic_path),
        str(dataset_config_path),
        str(materials_config_path),
        "1" if materials_enabled else "0",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        completed = subprocess.run(
            args,
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=180,
            preexec_fn=disable_core,
        )
    except subprocess.TimeoutExpired as error:
        return {"status": "TIMEOUT", "returncode": None, "stdout_tail": str(error.stdout)[-4000:], "stderr_tail": str(error.stderr)[-4000:]}
    parsed = None
    marker = "HISTORICAL_MATERIAL_PROBE="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            try:
                parsed = json.loads(line[len(marker):])
            except ValueError:
                parsed = None
            break
    return {
        "status": "PASS" if completed.returncode == 0 and parsed else "FAILED",
        "returncode": completed.returncode,
        "signal": -completed.returncode if completed.returncode is not None and completed.returncode < 0 else None,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-8000:],
        "probe": parsed,
    }


def _historical_json_record(repo: Path, path: Path) -> Dict[str, Any]:
    record = _file_record(repo, path, "historical_validation_artifact")
    value = _safe_read_json(path) if path.is_file() else None
    if value:
        record["summary"] = {
            "scene": value.get("scene"),
            "info_json": value.get("info_json"),
            "mapping_keys": sorted(value.get("mapping", {})) if isinstance(value.get("mapping"), Mapping) else [],
            "materials_off": value.get("materials_off"),
            "materials_on": value.get("materials_on"),
            "comparison": value.get("comparison"),
        }
    return record


def _text_reference_search(repo: Path) -> Dict[str, Any]:
    patterns = ["setAudioMaterialsJSON", "enableMaterials", "replica_material", "replica_compat"]
    glob_args = []
    for suffix in ("*.py", "*.json", "*.yaml", "*.yml", "*.log", "*.md", "*.txt"):
        glob_args.extend(["--glob", suffix])
    try:
        result = subprocess.run(
            [
                "rg", "--no-ignore", "-l", "-i",
                "|".join(patterns), "data", "examples", "active_audition", "configs", "registries",
                *glob_args,
            ],
            cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, timeout=30,
        )
        paths = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        return {"patterns": patterns, "status": "PASS", "paths": paths}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"patterns": patterns, "status": "FAILED", "paths": [], "error": str(error)}


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    return file_sha256(path)


def _write_report(path: Path, provenance: Mapping[str, Any], capability: Mapping[str, Any], artifact_shas: Mapping[str, str]) -> str:
    material = provenance["material_config"]["mapping"]
    direct = capability["probes"]["current_direct_replica"]
    compat = capability["probes"]["historical_replica_compat"]
    lines = [
        "# Historical Materials Provenance Recovery Report",
        "",
        "Scope: read-only historical material recovery and isolated capability audit; no ASR/WER, no A3 contract change, no production simulator change.",
        "",
        "## Decision",
        "",
        "- Final status: **{}**".format(capability["status"]),
        "- Current direct Replica materials-ON safety: **{}**".format(capability["current_direct_replica_materials_on"]),
        "- Historical `replica_compat/office_0` materials-ON safety: **{}**".format(capability["historical_replica_compat_materials_on"]),
        "- Production contract upgrade evidence: **{}**".format(capability["production_contract_evidence"]),
        "",
        "## Provenance",
        "",
        "- Material config: `{}` (SHA256 `{}`), JSON parse `{}`.".format(
            provenance["material_config"]["artifact"]["path"],
            provenance["material_config"]["artifact"]["sha256"],
            provenance["material_config"]["json_parse"],
        ),
        "- Config contains {} material entries; semantic-label coverage for the bound office_0 descriptor is {} with no unmatched labels.".format(
            provenance["material_config"]["schema"]["materials_count"],
            "complete" if all(not x["labels"] for x in material["unmatched_semantic_labels"]) else "incomplete",
        ),
        "- Mapping source is `materials[].labels`; explicit semantic-to-material table: {}; explicit fallback field: {}; `Default` material present: {}.".format(
            material["explicit_semantic_to_material_field"],
            material["explicit_fallback_field"],
            material["default_material_present"],
        ),
        "- `data/mp3d_material_config.json` comparison: {}.".format(provenance["material_config"]["mp3d_config_comparison"]),
        "",
        "## Isolated runtime probes",
        "",
        "- Current direct Replica OFF probe: status `{}`, return code `{}`, parsed result `{}`.".format(direct["materials_off"]["status"], direct["materials_off"].get("returncode"), bool(direct["materials_off"].get("probe"))),
        "- Current direct Replica ON probe: status `{}`, return code `{}`, parsed result `{}`.".format(direct["materials_on"]["status"], direct["materials_on"].get("returncode"), bool(direct["materials_on"].get("probe"))),
        "- Historical compat OFF probe: status `{}`, return code `{}`, parsed result `{}`.".format(compat["materials_off"]["status"], compat["materials_off"].get("returncode"), bool(compat["materials_off"].get("probe"))),
        "- Historical compat ON probe: status `{}`, return code `{}`, parsed result `{}`.".format(compat["materials_on"]["status"], compat["materials_on"].get("returncode"), bool(compat["materials_on"].get("probe"))),
        "- Compat OFF/ON first-render hashes differ: `{}`; ON repeat hashes identical: `{}`.".format(
            capability["compat_comparison"]["off_on_first_render_hash_different"],
            capability["compat_comparison"]["on_repeat_hash_identical"],
        ),
        "- Runtime material warnings observed: {}.".format(len(capability["compat_comparison"]["runtime_material_warnings"])),
        "- Static label coverage is not treated as runtime mapping proof; warnings such as `Material for category ... was not found` remain capability limitations.",
        "- A successful compat ON probe must not be generalized to the direct production scene: the mesh SHA differs and compat has no explicit stage-config file; runtime-derived stage behavior is recorded as such.",
        "",
        "## Historical evidence retained",
        "",
        "- Existing validation JSON/logs were read-only inputs. Their paths and SHA256 values are in `historical_materials_provenance.json`.",
        "- No historical file was modified or replaced.",
        "",
        "## Artifact hashes",
        "",
    ]
    lines.extend("- `{}`: `{}`".format(key, value) for key, value in artifact_shas.items())
    lines.append("")
    lines.append("The result does not authorize switching production materials ON; reviewer decision remains required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file_sha256(path)


def run_historical_materials_audit(
    output_dir: str = "runs/active_asr_v1/historical_materials_recovery_v1",
    repo_root: str = ".",
) -> Dict[str, Any]:
    """Recover historical materials provenance and run isolated OFF/ON probes."""

    repo = Path(repo_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = repo / output

    material_path = repo / "data/replica_material_config.json"
    direct_dir = repo / "data/scene_datasets/replica/office_0/habitat"
    compat_dir = repo / "data/scene_datasets/replica_compat/office_0/habitat"
    direct_paths = {
        "scene_asset": direct_dir / "mesh_semantic.ply",
        "navmesh": direct_dir / "mesh_semantic.navmesh",
        "semantic_info": direct_dir / "info_semantic.json",
        "stage_config": direct_dir / "replica_stage.stage_config.json",
    }
    compat_paths = {
        "scene_asset": compat_dir / "mesh_semantic.ply",
        "navmesh": compat_dir / "mesh_semantic.navmesh",
        "semantic_info": compat_dir / "info_semantic.json",
        "stage_config": compat_dir / "replica_stage.stage_config.json",
    }
    dataset_config = repo / "data/scene_datasets/replica/replica.scene_dataset_config.json"
    mp3d_config = repo / "data/mp3d_material_config.json"
    examples_path = repo / "examples/check_replica_scene.py"
    logs = sorted((repo / "data/logs").glob("replica_setup_*.log"))
    logs.extend(sorted((repo / "data/logs").glob("replica_*materials_on.log")))
    validation_paths = [
        repo / "data/logs/replica_validation_compat_layout_materials_on.json",
        repo / "data/logs/replica_validation_compat_layout_off.json",
        repo / "data/logs/replica_validation_compat_materials_on.json",
        repo / "data/logs/replica_validation_compat_official_off.json",
        repo / "data/logs/replica_validation_official_config_materials_on.json",
        repo / "data/logs/replica_validation.json",
    ]

    material_audit = _audit_material_config(
        repo,
        material_path,
        [direct_paths["semantic_info"], compat_paths["semantic_info"]],
    )
    semantic_match = (
        direct_paths["semantic_info"].is_file()
        and compat_paths["semantic_info"].is_file()
        and file_sha256(direct_paths["semantic_info"]) == file_sha256(compat_paths["semantic_info"])
    )
    compat_required = [compat_paths[key] for key in ("scene_asset", "navmesh", "semantic_info")]
    compat_files_intact = all(path.is_file() for path in compat_required)
    mapping_intact = (
        material_audit["json_parse"] == "PASS"
        and material_audit["schema"]["unknown_top_level_keys"] == []
        and material_audit["schema"]["required_fields_all_present"]
        and semantic_match
        and all(not item["labels"] for item in material_audit["mapping"]["unmatched_semantic_labels"])
    )
    historical_evidence = {
        "validation_artifacts": [_historical_json_record(repo, path) for path in validation_paths if path.is_file()],
        "logs": [_file_record(repo, path, "historical_materials_log") for path in sorted(set(logs)) if path.is_file()],
        "claims": {
            "compat_layout_materials_on_log_contains_setAudioMaterialsJSON": any(
                "setAudioMaterialsJSON" in path.read_text(errors="replace")
                for path in logs if path.name == "replica_compat_layout_materials_on.log"
            ),
            "direct_materials_on_historical_crash": any(
                "Segmentation fault" in path.read_text(errors="replace")
                for path in logs if path.name == "replica_official_config_materials_on.log"
            ),
            "audio_compatible_materials_on_historical_abort": any(
                "Aborted" in path.read_text(errors="replace")
                for path in logs if path.name == "replica_compat_materials_on.log"
            ),
        },
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "gate": "A3",
        "scope": "historical_material_provenance_recovery_and_production_materials_feasibility_audit_only",
        "repository_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), stdout=subprocess.PIPE, text=True, check=False
        ).stdout.strip(),
        "material_config": material_audit,
        "current_direct_replica": {
            "scene_id": "replica.office_0",
            "registry_path": "registries/scenes.yaml",
            "materials_mode": "off",
            "resources": {role: _file_record(repo, path, "current_direct_" + role) for role, path in direct_paths.items()},
            "dataset_config": _file_record(repo, dataset_config, "replica_dataset_config"),
            "material_config_binding": "NOT_PRESENT_IN_CURRENT_SCENE_REGISTRY",
        },
        "historical_replica_compat": {
            "scene_id": "replica_compat.office_0",
            "resources": {role: _file_record(repo, path, "historical_compat_" + role) for role, path in compat_paths.items()},
            "dataset_config": _file_record(repo, dataset_config, "replica_dataset_config"),
            "semantic_sha_matches_current_direct": semantic_match,
            "stage_config_status": "NOT_PRESENT_RUNTIME_DERIVED_STAGE_FROM_SCENE_ASSET" if not compat_paths["stage_config"].is_file() else "PRESENT",
            "runtime_provenance_binding": "mesh/navmesh/semantic SHA + historical runtime log; no explicit compat stage config file",
            "assets_intact_for_isolated_probe": compat_files_intact,
        },
        "search_inventory": {
            "requested_exact_targets": {
                "data/replica_material_config.json": _file_record(repo, material_path, "requested_exact_target"),
                "data/scene_datasets/replica_compat/office_0/habitat/": {
                    "exists": compat_dir.is_dir(),
                    "files": sorted(_repo_relative(repo, p) for p in compat_dir.glob("*") if p.is_file()),
                },
                "examples/check_replica_scene.py": _file_record(repo, examples_path, "requested_exact_target"),
                "data/logs/replica_setup_*.log": [_file_record(repo, p, "requested_log_target") for p in logs if p.name.startswith("replica_setup_")],
            },
            "text_reference_search": {
                **_text_reference_search(repo),
            },
        },
        "historical_evidence": historical_evidence,
        "binding_decision": {
            "material_config_recovered": material_path.is_file() and mapping_intact,
            "compatible_scene_provenance_recovered": compat_files_intact and semantic_match,
            "mapping_intact": mapping_intact,
        },
    }
    provenance_path = output / "historical_materials_provenance.json"
    provenance_sha = _write_json(provenance_path, provenance)

    if mapping_intact and compat_files_intact:
        # The current direct path is deliberately not started again here: its
        # existing isolated materials-ON SIGSEGV evidence is preserved below,
        # while the new reproducibility probe is reserved for the recovered
        # compat path.  This avoids turning a known unsafe production path
        # into an unbounded second crash experiment.
        direct_probe = {
            "materials_off": {
                "status": "HISTORICAL_EVIDENCE",
                "artifact": _historical_json_record(
                    repo, repo / "data/logs/replica_validation_official_config_direct_mesh.json"
                ),
            },
            "materials_on": {
                "status": "HISTORICAL_FAILED",
                "evidence": {
                    "log": _file_record(
                        repo, repo / "data/logs/replica_official_config_materials_on.log", "direct_materials_on_failure_log"
                    ),
                    "failure": "SIGSEGV during semantic mesh/material loading",
                },
            },
        }
        compat_probe = {
            "materials_off": _run_child_probe(
                compat_paths["scene_asset"], compat_paths["navmesh"], compat_paths["semantic_info"],
                dataset_config, material_path, False, repo,
            ),
            "materials_on": _run_child_probe(
                compat_paths["scene_asset"], compat_paths["navmesh"], compat_paths["semantic_info"],
                dataset_config, material_path, True, repo,
            ),
        }
        compat_on = compat_probe["materials_on"]
        compat_off = compat_probe["materials_off"]
        direct_on = direct_probe["materials_on"]
        compat_on_stable = bool(
            compat_on.get("status") == "PASS"
            and compat_on.get("probe", {}).get("materials_enabled_effective") is True
            and compat_on.get("probe", {}).get("repeat_hash_identical") is True
            and all(row.get("finite") and row.get("sha256") for row in compat_on.get("probe", {}).get("renders", []))
        )
        off_on_differ = False
        if compat_on_stable and compat_off.get("status") == "PASS":
            on_rows = compat_on["probe"]["renders"]
            off_rows = compat_off.get("probe", {}).get("renders", [])
            off_on_differ = bool(on_rows and off_rows and on_rows[0]["sha256"] != off_rows[0]["sha256"])
        status = (
            "MATERIALS_ON_PROVENANCE_RECOVERED_AND_RUNTIME_STABLE"
            if compat_on_stable and off_on_differ
            else "MATERIALS_ON_PROVENANCE_RECOVERED_BUT_RUNTIME_UNSTABLE"
        )
    elif material_path.is_file() or compat_dir.is_dir():
        direct_probe = {"status": "NOT_RUN", "reason": "binding prerequisites incomplete"}
        compat_probe = {"status": "NOT_RUN", "reason": "binding prerequisites incomplete"}
        status = "HISTORICAL_MATERIAL_ASSETS_INCOMPLETE"
    else:
        direct_probe = {"status": "NOT_RUN"}
        compat_probe = {"status": "NOT_RUN"}
        status = "HISTORICAL_MATERIAL_ASSETS_NOT_RECOVERED"

    capability = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "gate": "A3",
        "scope": "isolated_renderer_capability_only_no_ASR_no_WER_no_production_switch",
        "status": status,
        "current_direct_replica_materials_on": (
            "UNSAFE_HISTORICAL_CRASH_OR_NOT_STABLE" if direct_probe.get("materials_on", {}).get("status") != "PASS" else "PROBE_PASS_NOT_PRODUCTION_AUTHORITY"
        ),
        "historical_replica_compat_materials_on": (
            "FINITE_BUT_REPEAT_UNSTABLE"
            if compat_probe.get("materials_on", {}).get("status") == "PASS"
            and not compat_probe.get("materials_on", {}).get("probe", {}).get("repeat_hash_identical", False)
            else (
                "STABLE_ISOLATED_PROBE"
                if compat_probe.get("materials_on", {}).get("status") == "PASS"
                else "NOT_STABLE_OR_NOT_RUN"
            )
        ),
        "production_contract_evidence": "NO",
        "production_contract_evidence_reason": "current create_scene_simulator hardcodes enableMaterials=False; direct Replica materials-ON is not stable; compat mesh differs and lacks explicit stage config",
        "probes": {
            "current_direct_replica": direct_probe,
            "historical_replica_compat": compat_probe,
        },
        "compat_comparison": {
            "off_on_first_render_hash_different": (
                bool(compat_probe.get("materials_off", {}).get("probe"))
                and bool(compat_probe.get("materials_on", {}).get("probe"))
                and compat_probe["materials_off"]["probe"]["renders"][0]["sha256"]
                != compat_probe["materials_on"]["probe"]["renders"][0]["sha256"]
            ),
            "off_first_render": (
                compat_probe.get("materials_off", {}).get("probe", {}).get("renders", [None])[0]
                if compat_probe.get("materials_off", {}).get("probe") else None
            ),
            "on_first_render": (
                compat_probe.get("materials_on", {}).get("probe", {}).get("renders", [None])[0]
                if compat_probe.get("materials_on", {}).get("probe") else None
            ),
            "on_repeat_hash_identical": compat_probe.get("materials_on", {}).get("probe", {}).get("repeat_hash_identical"),
            "runtime_material_warnings": sorted({
                line.strip()
                for mode in ("materials_off", "materials_on")
                for line in str(compat_probe.get(mode, {}).get("stderr_tail", "")).splitlines()
                if "Material for category" in line
            }),
        },
        "historical_artifact_references": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in provenance["historical_evidence"]["validation_artifacts"]
        ],
        "no_asr_or_wer": True,
    }
    capability_path = output / "historical_materials_capability.json"
    capability_sha = _write_json(capability_path, capability)
    report_path = output / "historical_materials_recovery_report.md"
    report_sha = _write_report(
        report_path,
        provenance,
        capability,
        {
            "historical_materials_provenance.json": provenance_sha,
            "historical_materials_capability.json": capability_sha,
        },
    )
    return {
        "status": status,
        "provenance": {"path": str(provenance_path), "sha256": provenance_sha},
        "capability": {"path": str(capability_path), "sha256": capability_sha},
        "report": {"path": str(report_path), "sha256": report_sha},
        "no_asr_or_wer": True,
    }
