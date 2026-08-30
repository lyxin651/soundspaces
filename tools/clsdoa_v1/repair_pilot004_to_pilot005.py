#!/usr/bin/env python
"""Deterministic, offline Pilot004 -> Pilot005 FOA repair infrastructure."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

from active_audition.datasets.binaural_foa_clsdoa.recipe import EpisodeRecipe
from active_audition.datasets.binaural_foa_clsdoa.schema import NUM_SAMPLES, SAMPLE_RATE_HZ
from examples.foa_adapter import native_foa_to_canonical
from tools.clsdoa_v1.git_identity import current_clean_head
from tools.clsdoa_v1.integrity import verify_plan_integrity


ROOT = Path(__file__).resolve().parents[2]
PILOT004_ID = "clsdoa_v1_pilot_004"
PILOT005_ID = "clsdoa_v1_pilot_005"
REPRESENTATIONS = ("binaural", "foa")
FROZEN_PILOT004_GENERATION_COMMIT = "5388d17ef18919a7aa7cd911b6b239f1d91813f1"
FROZEN_PILOT004_EPISODES_SHA256 = "af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484"
FROZEN_PILOT004_RENDERS_SHA256 = "262a71e947ecaf0b049d31f84b3bd393faa3dbe951767fecb79e2a1ba6d7f7e2"
FROZEN_R3A_CODE_COMMIT = "114b23608854622a9bc07949028d310260de71d9"
FROZEN_R3B_EVIDENCE_COMMIT = "9804c6a3b302980698887581f42873fd607588c9"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _validate_foa(array, name):
    array = np.asarray(array)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[0] != 4:
        raise ValueError("{} must be float32 with shape (4, N)".format(name))
    if array.shape[1] == 0 or not np.isfinite(array).all() or not np.any(array):
        raise ValueError("{} must be finite and non-zero".format(name))
    return array


def legacy_native_to_canonical(values, listener_yaw_deg=0.0):
    """Pilot004's frozen world-fixed N3D -> canonical transform."""
    native = _validate_foa(np.asarray(values, dtype=np.float32), "native FOA")
    yaw = np.deg2rad(float(listener_yaw_deg))
    output = np.empty_like(native)
    output[0] = native[0]
    output[1] = -np.cos(yaw) * native[3] + np.sin(yaw) * native[2]
    output[2] = native[1]
    output[3] = -np.sin(yaw) * native[3] - np.cos(yaw) * native[2]
    output[1:4] *= np.float32(1.0 / np.sqrt(3.0))
    return output


def legacy_canonical_to_native(values, listener_yaw_deg=0.0):
    """Mathematical inverse of the Pilot004 legacy transform."""
    canonical = _validate_foa(np.asarray(values, dtype=np.float32), "legacy canonical FOA")
    yaw = np.deg2rad(float(listener_yaw_deg))
    x_local = -canonical[1] * np.float32(np.sqrt(3.0))
    z_local = -canonical[3] * np.float32(np.sqrt(3.0))
    native = np.empty_like(canonical)
    native[0] = canonical[0]
    native[1] = canonical[2] * np.float32(np.sqrt(3.0))
    native[3] = np.cos(yaw) * x_local + np.sin(yaw) * z_local
    native[2] = -np.sin(yaw) * x_local + np.cos(yaw) * z_local
    return np.asarray(native, dtype=np.float32)


def repair_canonical_foa(values, listener_yaw_deg=0.0):
    """Repair old canonical FOA by reusing the authoritative corrected converter."""
    native = legacy_canonical_to_native(values, listener_yaw_deg)
    return native_foa_to_canonical(native, listener_yaw_deg)


def repair_foa_rir(values, listener_yaw_deg=0.0):
    return repair_canonical_foa(values, listener_yaw_deg)


def repair_foa_wav(values, listener_yaw_deg=0.0):
    array = np.asarray(values)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 4 or array.shape[0] == 0:
        raise ValueError("FOA WAV must be float32 with shape (N, 4)")
    if not np.isfinite(array).all() or not np.any(array):
        raise ValueError("FOA WAV must be finite and non-zero")
    return np.asarray(repair_canonical_foa(array.T, listener_yaw_deg).T, dtype=np.float32)


def semantic_recipe_object(recipe):
    """Return the science-only identity object, excluding episode/path/provenance fields."""
    value = recipe.to_dict() if isinstance(recipe, EpisodeRecipe) else dict(recipe)
    scene = value["scene"]
    source = value["source"]
    listener = value["listener"]
    label = value["label"]
    result = {
        "split": value["split"],
        "scene": {"scene_id": scene["scene_id"], "scene_family": scene["scene_family"]},
        "source": {key: source[key] for key in ("source_clip_id", "base_clip_id", "source_dataset", "class_id", "source_position_world", "source_gain_db", "source_offset_sec")},
        "listener": {key: listener[key] for key in ("base_position_world", "sensor_position_world", "yaw_deg")},
        "label": {key: label[key] for key in ("class_id", "azimuth_project_deg", "elevation_project_deg", "doa_unit_project", "distance_m")},
        "representations": sorted(key for key, item in value["representations"].items() if item.get("required") is True),
    }
    if "diagnostics" in value and "candidate_origin" in value["diagnostics"]:
        result["candidate_origin"] = value["diagnostics"]["candidate_origin"]
    return result


def semantic_recipe_fingerprint(recipe):
    payload = json.dumps(semantic_recipe_object(recipe), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_derivation_lock(source_root, repair_code_commit):
    source_root = Path(source_root)
    identity = json.loads((source_root / "identity.json").read_text(encoding="utf-8"))
    if identity["dataset_id"] != PILOT004_ID or identity["generation_code_commit"] != FROZEN_PILOT004_GENERATION_COMMIT:
        raise ValueError("source is not the frozen Pilot004 generation")
    episodes_sha = _sha(source_root / "manifests/episodes.jsonl")
    renders_sha = _sha(source_root / "manifests/renders.jsonl")
    if episodes_sha != FROZEN_PILOT004_EPISODES_SHA256 or renders_sha != FROZEN_PILOT004_RENDERS_SHA256:
        raise ValueError("source is not the frozen Pilot004 evidence")
    return {
        "derivation_type": "foa_coordinate_repair_v1",
        "source_dataset_id": identity["dataset_id"],
        "source_generation_commit": FROZEN_PILOT004_GENERATION_COMMIT,
        "source_episodes_sha256": episodes_sha,
        "source_renders_sha256": renders_sha,
        "r3a_code_commit": FROZEN_R3A_CODE_COMMIT,
        "r3b_evidence_commit": FROZEN_R3B_EVIDENCE_COMMIT,
        "repair_code_commit": repair_code_commit,
        "binaural_policy": "byte_identical_copy",
        "foa_rir_policy": "deterministic_linear_coordinate_repair",
        "foa_wav_policy": "deterministic_linear_coordinate_repair",
    }


def validate_derivation_lock(lock, source_root, expected_repair_code_commit=None):
    source_root = Path(source_root)
    required = {"derivation_type", "source_dataset_id", "source_generation_commit", "source_episodes_sha256", "source_renders_sha256", "r3a_code_commit", "r3b_evidence_commit", "repair_code_commit", "binaural_policy", "foa_rir_policy", "foa_wav_policy"}
    if not required.issubset(lock):
        raise ValueError("derivation lock is missing required fields")
    if lock["derivation_type"] != "foa_coordinate_repair_v1" or lock["source_dataset_id"] != PILOT004_ID or lock["source_generation_commit"] != FROZEN_PILOT004_GENERATION_COMMIT:
        raise ValueError("invalid derivation identity")
    if lock["r3a_code_commit"] != FROZEN_R3A_CODE_COMMIT or lock["r3b_evidence_commit"] != FROZEN_R3B_EVIDENCE_COMMIT:
        raise ValueError("repair provenance commit mismatch")
    if lock["source_episodes_sha256"] != _sha(source_root / "manifests/episodes.jsonl") or lock["source_renders_sha256"] != _sha(source_root / "manifests/renders.jsonl"):
        raise ValueError("source provenance SHA mismatch")
    if lock["source_episodes_sha256"] != FROZEN_PILOT004_EPISODES_SHA256 or lock["source_renders_sha256"] != FROZEN_PILOT004_RENDERS_SHA256:
        raise ValueError("source is not the frozen Pilot004 evidence")
    if expected_repair_code_commit is not None and lock["repair_code_commit"] != expected_repair_code_commit:
        raise ValueError("repair code commit mismatch")
    if lock["binaural_policy"] != "byte_identical_copy" or lock["foa_rir_policy"] != "deterministic_linear_coordinate_repair" or lock["foa_wav_policy"] != "deterministic_linear_coordinate_repair":
        raise ValueError("repair policy mismatch")
    return True


def _complete_target_record(root, row, episode_id=None, representation=None):
    if row.get("render_status") != "complete" or not row.get("audio_path"):
        return False
    if episode_id is not None and row.get("episode_id") != episode_id:
        return False
    if representation is not None and row.get("representation") != representation:
        return False
    audio = root / row["audio_path"]
    if not audio.is_file():
        return False
    rate, waveform = wavfile.read(str(audio))
    expected_channels = 2 if row.get("representation") == "binaural" else 4
    expected_shape = (NUM_SAMPLES, expected_channels)
    if int(rate) != SAMPLE_RATE_HZ or waveform.dtype != np.float32 or waveform.shape != expected_shape:
        return False
    if not np.isfinite(waveform).all() or not np.any(waveform):
        return False
    if not row.get("rir_path") or not (root / row["rir_path"]).is_file():
        return False
    rir = np.load(root / row["rir_path"], allow_pickle=False)
    if rir.dtype != np.float32 or rir.ndim != 2 or not np.isfinite(rir).all() or not np.any(rir):
        return False
    if row.get("representation") == "binaural":
        if rir.shape[1] != 2:
            return False
    elif row.get("representation") == "foa":
        if rir.shape[0] != 4:
            return False
    else:
        return False
    return True


def _target_id(source_id):
    if not source_id.startswith(PILOT004_ID + "_"):
        raise ValueError("source episode id is not Pilot004")
    return PILOT005_ID + source_id[len(PILOT004_ID):]


def repair_dataset(source_root, target_root, resume=False, repo_root=ROOT):
    """Repair a Pilot004 root into a Pilot005 root without SoundSpaces calls."""
    source_root, target_root, repo_root = Path(source_root), Path(target_root), Path(repo_root)
    if source_root.resolve() == target_root.resolve():
        raise ValueError("source and target roots must differ")
    if not source_root.is_dir():
        raise ValueError("source root is missing")
    if (target_root / "_SUCCESS").exists():
        raise ValueError("finalized target is immutable")
    if not target_root.is_dir():
        raise ValueError("target root with an existing PLAN is required")
    required_plan = ("manifests/episodes.jsonl", "manifests/plan.lock.json", "identity.json", "config_resolved.yaml", "resources.lock.json")
    if any(not (target_root / item).is_file() for item in required_plan):
        raise ValueError("target PLAN metadata is incomplete")
    try:
        verify_plan_integrity(target_root, repo_root)
    except Exception as exc:
        raise ValueError("target PLAN integrity failed: {}".format(exc)) from exc
    plan_sha_before = {item: _sha(target_root / item) for item in required_plan}
    generation_commit = current_clean_head(repo_root)
    if target_root.exists() and (target_root / "identity.json").exists():
        target_identity = json.loads((target_root / "identity.json").read_text(encoding="utf-8"))
        if target_identity.get("dataset_id") != PILOT005_ID or target_identity.get("generation_code_commit") != generation_commit:
            raise ValueError("target generation identity mismatch")
    source_episodes = _load_rows(source_root / "manifests/episodes.jsonl")
    source_renders = _load_rows(source_root / "manifests/renders.jsonl")
    source_by_fp = {}
    for row in source_episodes:
        fp = semantic_recipe_fingerprint(row)
        if fp in source_by_fp:
            raise ValueError("duplicate source semantic fingerprint")
        source_by_fp[fp] = row
    render_by_key = {(row["episode_id"], row["representation"]): row for row in source_renders}
    if len(render_by_key) != len(source_renders):
        raise ValueError("duplicate source render record")
    target_episodes = _load_rows(target_root / "manifests/episodes.jsonl")
    target_by_fp = {}
    for row in target_episodes:
        fp = semantic_recipe_fingerprint(row)
        if fp in target_by_fp:
            raise ValueError("duplicate target semantic fingerprint")
        target_by_fp[fp] = row
    if set(target_by_fp) != set(source_by_fp):
        raise ValueError("source/target semantic fingerprint mismatch")
    target_by_fp = {semantic_recipe_fingerprint(row): row for row in target_episodes}
    if len(target_by_fp) != len(target_episodes) or set(target_by_fp) != set(source_by_fp):
        raise ValueError("source/target semantic mapping is not one-to-one")
    lock = build_derivation_lock(source_root, generation_commit)
    derivation_path = target_root / "manifests/derivation.lock.json"
    if derivation_path.exists():
        validate_derivation_lock(json.loads(derivation_path.read_text(encoding="utf-8")), source_root, generation_commit)
    else:
        derivation_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    journal_path = target_root / "manifests/renders.jsonl"
    journal = _load_rows(journal_path) if journal_path.exists() else []
    existing = {}
    for row in journal:
        key = (row.get("episode_id"), row.get("representation"))
        if key in existing:
            raise ValueError("duplicate target journal record")
        existing[key] = row
    for target in target_episodes:
        source = source_by_fp[semantic_recipe_fingerprint(target)]
        for representation in REPRESENTATIONS:
            source_record = render_by_key.get((source["episode_id"], representation))
            if source_record is None or source_record.get("render_status") != "complete":
                raise ValueError("source payload is incomplete")
            key = (target["episode_id"], representation)
            if resume and key in existing and _complete_target_record(target_root, existing[key], target["episode_id"], representation):
                continue
            source_audio = source_root / source_record["audio_path"]
            source_rir = source_root / source_record["rir_path"] if source_record.get("rir_path") else None
            if not source_audio.is_file() or (source_rir is not None and not source_rir.is_file()):
                raise ValueError("source payload file is missing")
            audio_rel = Path("audio") / representation / (target["episode_id"] + ".wav")
            audio_target = target_root / audio_rel
            audio_target.parent.mkdir(parents=True, exist_ok=True)
            if representation == "binaural":
                shutil.copyfile(source_audio, audio_target)
                if _sha(source_audio) != _sha(audio_target) or source_audio.stat().st_ino == audio_target.stat().st_ino:
                    raise ValueError("binaural copy is not independent and byte-identical")
            else:
                rate, wav = wavfile.read(str(source_audio))
                repaired_wav = repair_foa_wav(wav, target["listener"]["yaw_deg"])
                if int(rate) != SAMPLE_RATE_HZ:
                    raise ValueError("source FOA WAV sample rate mismatch")
                wavfile.write(str(audio_target), SAMPLE_RATE_HZ, repaired_wav)
            rir_rel = Path("cache/rir") / representation / (target["episode_id"] + ".npy")
            rir_target = target_root / rir_rel
            rir_target.parent.mkdir(parents=True, exist_ok=True)
            if source_rir is not None:
                if representation == "binaural":
                    shutil.copyfile(source_rir, rir_target)
                    if _sha(source_rir) != _sha(rir_target) or source_rir.stat().st_ino == rir_target.stat().st_ino:
                        raise ValueError("binaural RIR copy is not independent and byte-identical")
                    repaired_rir = None
                else:
                    old_rir = np.load(source_rir, allow_pickle=False)
                    repaired_rir = repair_foa_rir(np.asarray(old_rir, dtype=np.float32), target["listener"]["yaw_deg"])
                    np.save(str(rir_target), np.asarray(repaired_rir, dtype=np.float32), allow_pickle=False)
            record = dict(source_record, episode_id=target["episode_id"], audio_path=str(audio_rel), rir_path=str(rir_rel), repair_policy="byte_identical_copy" if representation == "binaural" else "deterministic_linear_coordinate_repair")
            existing[key] = record
            _atomic_jsonl(journal_path, list(existing.values()))
    plan_sha_after = {item: _sha(target_root / item) for item in required_plan}
    if plan_sha_before != plan_sha_after:
        raise ValueError("repair modified immutable PLAN metadata")
    return list(existing.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = repair_dataset(args.source_root, args.target_root, resume=args.resume)
    print(json.dumps({"render_records": len(rows), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
