import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.io import wavfile

from examples.foa_adapter import native_foa_to_canonical
from tools.clsdoa_v1.repair_pilot004_to_pilot005 import (
    build_derivation_lock,
    legacy_canonical_to_native,
    legacy_native_to_canonical,
    repair_canonical_foa,
    repair_dataset,
    repair_foa_wav,
    semantic_recipe_fingerprint,
    validate_derivation_lock,
    FROZEN_PILOT004_GENERATION_COMMIT,
    FROZEN_R3A_CODE_COMMIT,
    FROZEN_R3B_EVIDENCE_COMMIT,
    _sha,
)


def _episode(index=1, gain=0.0, yaw=45.0, scene="replica.office_0", source="ESC-50:1"):
    return {
        "episode_id": "clsdoa_v1_pilot_004_ep_{:06d}".format(index), "split": "train",
        "scene": {"scene_id": scene, "scene_family": "Replica"},
        "source": {"source_clip_id": source, "base_clip_id": source.lower(), "source_dataset": "ESC-50", "class_id": 0, "source_position_world": [1.0, 2.0, 3.0], "source_gain_db": gain, "source_offset_sec": 0.25},
        "listener": {"base_position_world": [0.0, 1.0, 0.0], "sensor_position_world": [0.0, 2.5, 0.0], "yaw_deg": yaw},
        "label": {"class_id": 0, "azimuth_project_deg": 10.0, "elevation_project_deg": 20.0, "doa_unit_project": [0.1, 0.2, 0.97], "distance_m": 3.74},
        "representations": {"binaural": {"required": True}, "foa": {"required": True}},
    }


class R3CRepairTests(unittest.TestCase):
    def test_foa_math_roundtrip_repair_and_invariants(self):
        rng = np.random.default_rng(7)
        native = rng.normal(size=(4, 19)).astype(np.float32)
        for yaw in (0.0, 45.0, 90.0, -45.0):
            old = legacy_native_to_canonical(native, yaw)
            np.testing.assert_allclose(legacy_canonical_to_native(old, yaw), native, atol=2e-6)
            np.testing.assert_allclose(repair_canonical_foa(old, yaw), native_foa_to_canonical(native, yaw), atol=2e-6)
            repaired = repair_canonical_foa(old, yaw)
            self.assertEqual(repaired.shape, (4, 19)); self.assertEqual(repaired.dtype, np.float32)
            self.assertTrue(np.isfinite(repaired).all())
            self.assertAlmostEqual(float(np.max(np.abs(old[0] - repaired[0]))), 0.0, places=6)
            self.assertAlmostEqual(float(np.max(np.abs(old[2] - repaired[2]))), 0.0, places=6)
            self.assertAlmostEqual(float(np.sum(old[[1, 3]] ** 2)), float(np.sum(repaired[[1, 3]] ** 2)), places=4)

    def test_wav_axis_is_explicit(self):
        wav = np.ones((31, 4), dtype=np.float32)
        repaired = repair_foa_wav(wav, 45.0)
        self.assertEqual(repaired.shape, (31, 4)); self.assertEqual(repaired.dtype, np.float32)

    def test_fingerprint_excludes_episode_id_but_tracks_science(self):
        first = _episode(1)
        second = _episode(2)
        self.assertEqual(semantic_recipe_fingerprint(first), semantic_recipe_fingerprint(second))
        self.assertNotEqual(semantic_recipe_fingerprint(first), semantic_recipe_fingerprint(_episode(2, gain=1.0)))
        self.assertNotEqual(semantic_recipe_fingerprint(first), semantic_recipe_fingerprint(_episode(2, yaw=90.0)))
        self.assertNotEqual(semantic_recipe_fingerprint(first), semantic_recipe_fingerprint(_episode(2, scene="mp3d.scan")))
        self.assertNotEqual(semantic_recipe_fingerprint(first), semantic_recipe_fingerprint(_episode(2, source="ESC-50:2")))
        self.assertEqual(semantic_recipe_fingerprint(first), semantic_recipe_fingerprint(json.loads(json.dumps(first))))

    def test_derivation_lock_validates_source_sha_and_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "manifests").mkdir()
            (root / "manifests/episodes.jsonl").write_text("{}\n")
            (root / "manifests/renders.jsonl").write_text("{}\n")
            (root / "identity.json").write_text(json.dumps({"dataset_id": "clsdoa_v1_pilot_004", "generation_code_commit": FROZEN_PILOT004_GENERATION_COMMIT}))
            lock = build_derivation_lock(root, "repair")
            self.assertEqual(lock["r3a_code_commit"], FROZEN_R3A_CODE_COMMIT)
            self.assertEqual(lock["r3b_evidence_commit"], FROZEN_R3B_EVIDENCE_COMMIT)
            self.assertTrue(validate_derivation_lock(lock, root, "repair"))
            with self.assertRaises(ValueError): validate_derivation_lock(lock, root, "wrong")

    def _source_fixture(self, directory):
        source = Path(directory); (source / "manifests").mkdir(parents=True)
        episodes = [_episode(1), _episode(2, gain=1.0, yaw=-45.0)]
        (source / "manifests/episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in episodes))
        (source / "identity.json").write_text(json.dumps({"dataset_id": "clsdoa_v1_pilot_004", "generation_code_commit": FROZEN_PILOT004_GENERATION_COMMIT}))
        rows = []
        for e in episodes:
            eid = e["episode_id"]
            for rep, channels in (("binaural", 2), ("foa", 4)):
                audio = source / "audio" / rep / (eid + ".wav"); audio.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(str(audio), 24000, np.ones((120000, channels), dtype=np.float32))
                rir = source / "cache/rir" / rep / (eid + ".npy"); rir.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(rir), np.ones((channels, 9), dtype=np.float32), allow_pickle=False)
                rows.append({"episode_id": eid, "representation": rep, "audio_path": str(audio.relative_to(source)), "rir_path": str(rir.relative_to(source)), "render_status": "complete", "sample_rate_hz": 24000, "num_channels": channels, "num_samples": 120000, "dtype": "float32", "format": "WAV" if rep == "binaural" else "AmbiX ACN/SN3D"})
        (source / "manifests/renders.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        return episodes

    def _target_plan_fixture(self, source, target, generation="repair-head"):
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifests").mkdir(exist_ok=True)
        episodes = [json.loads(x) for x in (source / "manifests/episodes.jsonl").read_text().splitlines()]
        for episode in episodes:
            episode["episode_id"] = episode["episode_id"].replace("pilot_004", "pilot_005")
        (target / "manifests/episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in episodes))
        (target / "identity.json").write_text(json.dumps({"dataset_id": "clsdoa_v1_pilot_005", "generation_code_commit": generation}))
        (target / "config_resolved.yaml").write_text("dataset_id: clsdoa_v1_pilot_005\nstorage:\n  require_rir: true\n")
        (target / "resources.lock.json").write_text("{}\n")
        (target / "manifests/plan.lock.json").write_text("{}\n")

    def test_repair_copies_binaural_independently_and_resume_keeps_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"; target = Path(directory) / "target"; self._source_fixture(source); self._target_plan_fixture(source, target)
            with mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.current_clean_head", return_value="repair-head"), mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.verify_plan_integrity", return_value={"status": "PASS"}):
                plan_files = ("manifests/episodes.jsonl", "manifests/plan.lock.json", "identity.json", "config_resolved.yaml", "resources.lock.json")
                plan_hashes = {item: _sha(target / item) for item in plan_files}
                first = repair_dataset(source, target, repo_root=Path(directory))
                second = repair_dataset(source, target, resume=True, repo_root=Path(directory))
                self.assertEqual(plan_hashes, {item: _sha(target / item) for item in plan_files})
            self.assertEqual(len(first), 4); self.assertEqual(len(second), 4)
            src = source / "audio/binaural/clsdoa_v1_pilot_004_ep_000001.wav"; dst = target / "audio/binaural/clsdoa_v1_pilot_005_ep_000001.wav"
            self.assertEqual(hashlib.sha256(src.read_bytes()).digest(), hashlib.sha256(dst.read_bytes()).digest())
            self.assertNotEqual(src.stat().st_ino, dst.stat().st_ino)
            rows = [json.loads(x) for x in (target / "manifests/renders.jsonl").read_text().splitlines()]
            self.assertEqual({x["representation"] for x in rows if x["episode_id"].endswith("000001")}, {"binaural", "foa"})

    def test_safety_and_mapping_negative_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"; self._source_fixture(source)
            with mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.current_clean_head", return_value="repair-head"):
                with self.assertRaises(ValueError): repair_dataset(source, source, repo_root=Path(directory))
                with self.assertRaises(ValueError): repair_dataset(Path(directory) / "missing", Path(directory) / "target", repo_root=Path(directory))
                target = Path(directory) / "nonempty"; target.mkdir(); (target / "junk").write_text("x")
                with self.assertRaises(ValueError): repair_dataset(source, target, repo_root=Path(directory))
                done = Path(directory) / "done"; done.mkdir(); (done / "_SUCCESS").write_text("done")
                with self.assertRaises(ValueError): repair_dataset(source, done, repo_root=Path(directory))

    def test_existing_plan_and_integrity_gate_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"; self._source_fixture(source)
            target = Path(directory) / "target"; target.mkdir()
            with self.assertRaises(ValueError): repair_dataset(source, target, repo_root=Path(directory))
            self._target_plan_fixture(source, target)
            with mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.current_clean_head", return_value="repair-head"), mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.verify_plan_integrity", side_effect=RuntimeError("drift")), self.assertRaises(ValueError):
                repair_dataset(source, target, repo_root=Path(directory))

    def test_duplicate_source_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"; target = Path(directory) / "target"; self._source_fixture(source); self._target_plan_fixture(source, target)
            rows = [json.loads(x) for x in (source / "manifests/episodes.jsonl").read_text().splitlines()]
            rows[1] = json.loads(json.dumps(rows[0])); rows[1]["episode_id"] = "clsdoa_v1_pilot_004_ep_000002"
            (source / "manifests/episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
            with mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.current_clean_head", return_value="repair-head"), mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.verify_plan_integrity", return_value={"status": "PASS"}), self.assertRaises(ValueError):
                repair_dataset(source, target, repo_root=Path(directory))

    def test_resume_identity_and_source_payload_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"; target = Path(directory) / "target"; self._source_fixture(source); self._target_plan_fixture(source, target)
            with mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.current_clean_head", return_value="repair-head"), mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.verify_plan_integrity", return_value={"status": "PASS"}):
                repair_dataset(source, target, repo_root=Path(directory))
                identity = json.loads((target / "identity.json").read_text()); identity["generation_code_commit"] = "wrong"; (target / "identity.json").write_text(json.dumps(identity))
                with self.assertRaises(ValueError): repair_dataset(source, target, resume=True, repo_root=Path(directory))
            source = Path(directory) / "missing-rir"; self._source_fixture(source)
            rir = source / "cache/rir/foa/clsdoa_v1_pilot_004_ep_000001.npy"; rir.unlink()
            target = Path(directory) / "target-missing-rir"; self._target_plan_fixture(source, target)
            with mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.current_clean_head", return_value="repair-head"), mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.verify_plan_integrity", return_value={"status": "PASS"}), self.assertRaises(ValueError):
                repair_dataset(source, target, repo_root=Path(directory))

    def test_malformed_foa_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"; target = Path(directory) / "target"; self._source_fixture(source); self._target_plan_fixture(source, target)
            wav = source / "audio/foa/clsdoa_v1_pilot_004_ep_000001.wav"
            wavfile.write(str(wav), 24000, np.ones((17, 2), dtype=np.float32))
            with mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.current_clean_head", return_value="repair-head"), mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.verify_plan_integrity", return_value={"status": "PASS"}), self.assertRaises(ValueError):
                repair_dataset(source, target, repo_root=Path(directory))

    def test_corrupted_resume_repairs_only_damaged_representation_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"; target = Path(directory) / "target"; self._source_fixture(source); self._target_plan_fixture(source, target)
            with mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.current_clean_head", return_value="repair-head"), mock.patch("tools.clsdoa_v1.repair_pilot004_to_pilot005.verify_plan_integrity", return_value={"status": "PASS"}):
                repair_dataset(source, target, repo_root=Path(directory))
                binary = target / "audio/binaural/clsdoa_v1_pilot_005_ep_000001.wav"
                binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
                foa = target / "audio/foa/clsdoa_v1_pilot_005_ep_000001.wav"
                wavfile.write(str(foa), 24000, np.ones((10, 4), dtype=np.float32))
                repair_dataset(source, target, resume=True, repo_root=Path(directory))
                self.assertEqual(hashlib.sha256(binary.read_bytes()).hexdigest(), binary_hash)
                self.assertEqual(wavfile.read(str(foa))[1].shape, (120000, 4))
                repaired_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in target.rglob("*") if path.is_file()}
                repair_dataset(source, target, resume=True, repo_root=Path(directory))
                self.assertEqual(repaired_hashes, {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in target.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
