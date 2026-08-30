#!/usr/bin/env python
"""Independent hard-gate validator for the metadata-only Step 3 pilot plan."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import yaml
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.clsdoa_v1.scheduler import distance_schedule, elevation_schedule, azimuth_schedule, gain_schedule
from tools.clsdoa_v1.integrity import verify_plan_integrity
CLASSES = list(range(12))


class PilotPlanValidationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise PilotPlanValidationError(message)


def validate_episode_membership(episodes, source_registry, scene_registry):
    sources = {row["base_clip_id"]: row for row in source_registry}
    scenes = scene_registry.get("scenes", scene_registry)
    for row in episodes:
        source = sources.get(row["source"]["base_clip_id"])
        scene = scenes.get(row["scene"]["scene_id"])
        _require(source is not None, "episode source is absent from registry")
        _require(scene is not None, "episode scene is absent from registry")
        _require(scene.get("admitted", "PASS") == "PASS", "episode scene is not PASS admitted")
        _require(source["split"] == row["split"], "source split mismatch")
        _require(scene["split"] == row["split"], "scene split mismatch")
        _require(scene["scene_family"] == row["scene"]["scene_family"], "scene family mismatch")


def validate_scene_set(used_scene_ids, scene_registry):
    scenes = scene_registry.get("scenes", scene_registry)
    pass_ids = {sid for sid, row in scenes.items() if row.get("admitted") == "PASS"}
    _require(set(used_scene_ids) == pass_ids, "scene set must equal exact PASS set")


def validate_episode_ids(episodes):
    _require(len({row["episode_id"] for row in episodes}) == len(episodes), "episode IDs must be unique")


def validate_source_reuse(episodes, class_id, split):
    counts = Counter(row["source"]["base_clip_id"] for row in episodes if row["label"]["class_id"] == class_id and row["split"] == split)
    _require(counts and max(counts.values()) - min(counts.values()) <= 1, "source reuse imbalance")


def validate_family_block(block, class_id, split, family):
    """Hard-check one class/split/family micro-pattern."""
    _require(Counter(row["diagnostics"]["distance_bin"] for row in block) == Counter(distance_schedule(split, class_id, family)), "family distance micro-pattern mismatch")
    _require(Counter(row["diagnostics"]["elevation_bin"] for row in block) == Counter(elevation_schedule(split, class_id, family)), "family elevation micro-pattern mismatch")
    _require(Counter(row["diagnostics"]["azimuth_bin"] for row in block) == Counter(azimuth_schedule(split, class_id, family)), "family azimuth micro-pattern mismatch")
    _require(Counter(row["diagnostics"]["gain_bin"] for row in block) == Counter(gain_schedule(split, class_id, family)), "family gain micro-pattern mismatch")


def validate(root):
    root = Path(root)
    resolved_config = yaml.safe_load((root / "config_resolved.yaml").read_text())
    revision_profile = str(resolved_config.get("dataset_id", "")).startswith("clsdoa_v1_ontology_v2_revision_pilot_")
    expected_episodes = 96 if revision_profile else 960
    episodes = [json.loads(line) for line in (root / "manifests/episodes.jsonl").read_text().splitlines() if line]
    review = [json.loads(line) for line in (root / "reports/plan_review_index.jsonl").read_text().splitlines() if line]
    _require(len(episodes) == len(review) == expected_episodes, "episode/review count mismatch: expected {}".format(expected_episodes))
    validate_episode_ids(episodes)
    expected_split = Counter(train=48, val=24, test=24) if revision_profile else Counter(train=672, val=144, test=144)
    expected_family = Counter(Replica=48, MP3D=48) if revision_profile else Counter(Replica=480, MP3D=480)
    expected_class = Counter({i: 8 for i in CLASSES}) if revision_profile else Counter({i: 80 for i in CLASSES})
    _require(Counter(r["split"] for r in episodes) == expected_split, "split quota mismatch")
    _require(Counter(r["scene"]["scene_family"] for r in episodes) == expected_family, "family quota mismatch")
    _require(Counter(r["label"]["class_id"] for r in episodes) == expected_class, "class quota mismatch")
    source_path = Path(resolved_config["source"]["registry_path"])
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    registry = list(__import__("active_audition.datasets.binaural_foa_clsdoa.source_registry", fromlist=["read_source_registry"]).read_source_registry(str(source_path)))
    episode_sources = {r["source"]["base_clip_id"] for r in episodes}
    registry_sources = {r["base_clip_id"] for r in registry}
    _require(episode_sources == registry_sources if not revision_profile else episode_sources <= registry_sources, "source pool mismatch")
    scene_path = Path(resolved_config["scene"]["registry_path"])
    if not scene_path.is_absolute():
        scene_path = ROOT / scene_path
    scene_registry = yaml.safe_load(scene_path.read_text())["scenes"]
    validate_episode_membership(episodes, registry, scene_registry)
    pass_ids = {sid for sid, r in scene_registry.items() if r["admitted"] == "PASS"}
    fail_ids = {sid for sid, r in scene_registry.items() if r["admitted"] != "PASS"}
    used_scene_ids = {r["scene"]["scene_id"] for r in episodes}
    if revision_profile:
        _require(used_scene_ids <= pass_ids, "revision plan contains non-PASS scene")
    else:
        validate_scene_set(used_scene_ids, scene_registry)
    _require(not (used_scene_ids & fail_ids), "scene pool contains non-PASS scene")
    for class_id in CLASSES:
        rows = [r for r in episodes if r["label"]["class_id"] == class_id]
        _require(len({r["label"]["azimuth_project_deg"] for r in rows}) > 0, "empty azimuth schedule")
        if not revision_profile:
            _require(Counter(r["diagnostics"]["distance_bin"] for r in review if r["label"]["class_id"] == class_id) == Counter(near=32, mid=32, far=16), "distance quota mismatch")
            _require(Counter(r["diagnostics"]["elevation_bin"] for r in review if r["label"]["class_id"] == class_id) == Counter(small=60, nonzero=20), "elevation quota mismatch")
        for split in ("train", "val", "test"):
            counts = Counter(r["source"]["base_clip_id"] for r in episodes if r["label"]["class_id"] == class_id and r["split"] == split)
            validate_source_reuse(episodes, class_id, split)
            split_rows = [r for r in review if r["label"]["class_id"] == class_id and r["split"] == split]
            if not revision_profile:
                _require(set(r["diagnostics"]["azimuth_bin"] for r in split_rows) == set(range(8)), "split azimuth coverage mismatch")
                _require(set(r["diagnostics"]["gain_bin"] for r in split_rows) == set(range(8)), "split gain coverage mismatch")
            for family in ("Replica", "MP3D"):
                block = [r for r in split_rows if r["scene"]["scene_family"] == family]
                _require(block, "empty family micro-pattern")
                if revision_profile:
                    expected_size = {"train": 2, "val": 1, "test": 1}[split]
                    _require(len(block) == expected_size, "revision family block size mismatch")
                    for key, schedule in (("distance_bin", distance_schedule), ("elevation_bin", elevation_schedule), ("azimuth_bin", azimuth_schedule), ("gain_bin", gain_schedule)):
                        expected = schedule(split, class_id, family)[:expected_size]
                        _require(Counter(r["diagnostics"][key] for r in block) == Counter(expected), "revision {} schedule mismatch".format(key))
                else:
                    validate_family_block(block, class_id, split, family)
    for r in episodes:
        _require(set(r["representations"]) == {"binaural", "foa"}, "representation mismatch")
        _require(r["listener"]["sensor_position_world"] == [r["listener"]["base_position_world"][0], r["listener"]["base_position_world"][1] + 1.5, r["listener"]["base_position_world"][2]], "receiver invariant mismatch")
        _require(-6.0 <= r["source"]["source_gain_db"] <= 6.0, "gain out of range")
    for r in review:
        if r["diagnostics"].get("candidate_origin") == "STEP2B_FIXED_PROBE":
            _require(r["diagnostics"].get("source_height_offset_m") is None and r["diagnostics"].get("geodesic_distance_m") is None, "fixed probe contains fabricated geometry diagnostics")
        else:
            _require(0.5 <= r["diagnostics"]["source_height_offset_m"] <= 2.2 + 1e-6, "source height out of range")
    validate_plan_payload_absence(root)
    _require((root / "manifests/plan.lock.json").is_file(), "plan lock missing")
    verify_plan_integrity(root, ROOT)
    return {"status": "PASS", "episodes": expected_episodes, "classes": 12, "unique_sources": len(episode_sources), "pass_scene_pool": len(pass_ids), "excluded_fail_scenes": len(fail_ids), "audio_files": 0, "rir_files": 0}


def validate_plan_payload_absence(root):
    root = Path(root)
    _require(not list(root.rglob("*.wav")) and not list(root.rglob("*.rir")) and not list((root / "cache/rir").rglob("*")) and not (root / "_SUCCESS").exists(), "plan contains render payload")


def validate_render_payload(root):
    """Payload gate used by finalize; plan-only roots must fail explicitly."""
    root = Path(root)
    _require(not (root / "_SUCCESS").exists(), "dataset already finalized")
    _require((root / "manifests/episodes.jsonl").is_file(), "episodes manifest missing")
    renders = root / "manifests/renders.jsonl"
    _require(renders.is_file(), "render manifest missing")
    rows = [json.loads(line) for line in renders.read_text().splitlines() if line]
    expected = _payload_expected_keys(root)
    validate_render_rows(rows, expected)
    verify_plan_integrity(root, ROOT)
    episodes = [json.loads(line) for line in (root / "manifests/episodes.jsonl").read_text().splitlines() if line]
    expected_episode_count = _expected_episode_count(root)
    _require(len(episodes) == expected_episode_count and len({row["episode_id"] for row in episodes}) == expected_episode_count, "payload requires all {} unique episodes".format(expected_episode_count))
    for row in rows:
        validate_render_record_payload(root, row)
    return True


def _payload_expected_keys(root):
    episodes = [json.loads(line) for line in (Path(root) / "manifests/episodes.jsonl").read_text().splitlines() if line]
    expected_episode_count = _expected_episode_count(root)
    _require(len(episodes) == expected_episode_count and len({row["episode_id"] for row in episodes}) == expected_episode_count, "payload requires all {} unique episodes".format(expected_episode_count))
    return {(row["episode_id"], representation) for row in episodes for representation in ("binaural", "foa")}


def _expected_episode_count(root):
    config_path = Path(root) / "config_resolved.yaml"
    if not config_path.is_file():
        return 960
    config = yaml.safe_load(config_path.read_text())
    return 96 if str(config.get("dataset_id", "")).startswith("clsdoa_v1_ontology_v2_revision_pilot_") else 960


def validate_render_rows(rows, expected):
    _require(len(rows) == len(expected), "render journal must contain exactly {} records".format(len(expected)))
    _require(all(row.get("render_status") == "complete" for row in rows), "render journal contains non-complete records")
    keys = {(row.get("episode_id"), row.get("representation")) for row in rows}
    _require(keys == expected and len(keys) == len(expected), "payload must contain exactly one complete binaural and foa record per episode")


def validate_render_record_payload(root, row):
    root = Path(root)
    _require(row["sample_rate_hz"] == 24000 and row["num_samples"] == 120000 and row["dtype"] == "float32", "payload audio contract mismatch")
    _require(row["num_channels"] == (2 if row["representation"] == "binaural" else 4), "payload channel contract mismatch")
    _require(Path(root / row["audio_path"]).is_file(), "render WAV missing")
    _require(row.get("rir_path") and Path(root / row["rir_path"]).is_file(), "render RIR missing")
    sample_rate, audio = wavfile.read(str(root / row["audio_path"]))
    audio = np.asarray(audio)
    _require(int(sample_rate) == 24000 and audio.shape == (120000, row["num_channels"]), "WAV payload shape/rate mismatch")
    _require(audio.dtype == np.dtype("float32"), "WAV dtype mismatch")
    _require(np.isfinite(audio).all() and np.any(np.abs(audio)), "WAV must be finite and non-zero")
    rir = np.asarray(np.load(str(root / row["rir_path"]), allow_pickle=False))
    _require(rir.dtype == np.dtype("float32"), "RIR dtype mismatch")
    _require(rir.ndim == 2 and ((row["representation"] == "binaural" and rir.shape[1] == 2) or (row["representation"] == "foa" and rir.shape[0] == 4)), "RIR channel payload mismatch")
    _require(np.isfinite(rir).all() and np.any(np.abs(rir)), "RIR must be finite and non-zero")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
