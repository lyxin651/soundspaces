import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from active_audition.datasets.binaural_foa_clsdoa.geometry import GeometryError, project_geometry, validate_geometry_independently
from active_audition.datasets.binaural_foa_clsdoa.manifest import ManifestError, read_jsonl, write_manifests
from active_audition.datasets.binaural_foa_clsdoa.qc import build_distribution_report
from active_audition.datasets.binaural_foa_clsdoa.recipe import EpisodeRecipe, make_episode_recipe
from active_audition.datasets.binaural_foa_clsdoa.renderer import PairedRenderer, render_pair
from active_audition.datasets.binaural_foa_clsdoa.schema import RenderPolicy, RenderRecord, SchemaError
from active_audition.datasets.binaural_foa_clsdoa.scene_registry import SceneRegistryError, validate_scene_rows
from active_audition.datasets.binaural_foa_clsdoa.source_registry import SourceRegistryError, build_observation_timeline, validate_source_rows
from active_audition.datasets.binaural_foa_clsdoa.storage import V1DatasetStorage, V1StorageError, merge_resolved_config
from active_audition.datasets.binaural_foa_clsdoa.validation import ValidationError, payload_is_complete, validate_audio_contract, validate_pairing, validate_split_leakage


def _recipe(episode_id="ep_000001", split="train", yaw=0.0, source=(0.0, 1.5, -2.0)):
    return make_episode_recipe(
        episode_id=episode_id, split=split, scene_id="fixture.replica", scene_family="Replica",
        source_clip_id="clip_{}".format(episode_id), base_clip_id="base_{}".format(episode_id),
        source_dataset="fixture", class_id=0, source_position_world=source, source_gain_db=0.0,
        source_offset_sec=0.0, listener_base_position_world=(0.0, 0.0, 0.0),
        listener_sensor_position_world=(0.0, 1.5, 0.0), listener_yaw_deg=yaw,
    )


def _record(recipe, representation):
    suffix = "wav" if representation == "binaural" else "foa.npy"
    return RenderRecord(
        episode_id=recipe.episode_id, representation=representation,
        audio_path="audio/{}/{}.{}".format(representation, recipe.episode_id, suffix),
        rir_path="cache/rir/{}__{}.npz".format(recipe.episode_id, representation),
        sample_rate_hz=24000, num_channels=2 if representation == "binaural" else 4,
        num_samples=120000, dtype="float32", format="WAV" if representation == "binaural" else "AmbiX ACN/SN3D",
        render_status="complete",
    )


class _FakeRenderer(PairedRenderer):
    def render_episode(self, recipe, representation):
        return _record(recipe, representation)


class ClassDOAV1DatasetInfrastructureTests(unittest.TestCase):
    def test_recipe_roundtrip_and_independent_geometry_for_front_right_left_back_elevation_and_yaw(self):
        cases = [((0.0, 1.5, -2.0), 0.0), ((2.0, 1.5, 0.0), 0.0), ((-2.0, 1.5, 0.0), 0.0), ((0.0, 1.5, 2.0), 0.0), ((0.0, 3.5, -2.0), 0.0), ((2.0, 2.5, -1.0), 37.0)]
        for index, (source, yaw) in enumerate(cases):
            recipe = _recipe("ep_{:02d}".format(index), yaw=yaw, source=source)
            restored = EpisodeRecipe.from_dict(recipe.to_dict())
            self.assertEqual(restored, recipe)
            validate_geometry_independently(source, (0.0, 1.5, 0.0), yaw, recipe.to_dict()["label"])
        self.assertAlmostEqual(_recipe(source=(0.0, 1.5, -2.0)).azimuth_project_deg, 0.0)
        self.assertAlmostEqual(_recipe(source=(2.0, 1.5, 0.0)).azimuth_project_deg, 90.0)
        self.assertAlmostEqual(_recipe(source=(-2.0, 1.5, 0.0)).azimuth_project_deg, -90.0)
        self.assertAlmostEqual(_recipe(source=(0.0, 1.5, 2.0)).azimuth_project_deg, -180.0)

    def test_recipe_rejects_invalid_class_nan_and_coincident_geometry(self):
        with self.assertRaises(GeometryError):
            _recipe(source=(0.0, 1.5, 0.0))
        with self.assertRaises(SchemaError):
            replace(_recipe(), source_gain_db=float("nan"))
        with self.assertRaises(SchemaError):
            EpisodeRecipe.from_dict({"episode_id": "missing"})
        with self.assertRaises(SchemaError):
            make_episode_recipe(
                episode_id="bad", split="train", scene_id="s", scene_family="Replica", source_clip_id="c", base_clip_id="b", source_dataset="x", class_id=12,
                source_position_world=(0, 1.5, -1), source_gain_db=0, source_offset_sec=0, listener_base_position_world=(0, 0, 0), listener_sensor_position_world=(0, 1.5, 0), listener_yaw_deg=0,
            )
        with self.assertRaises(GeometryError):
            project_geometry((0, 1.5, 0), (0, 1.5, 0), 0)

    def test_v1_storage_root_config_provenance_finalize_and_mutation_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            storage = V1DatasetStorage.from_config(temp, {"dataset_id": "clsdoa_v1_fixture_001"})
            self.assertTrue(str(storage.root).endswith("datasets/binaural_foa_clsdoa_v1/clsdoa_v1_fixture_001"))
            storage.ensure_layout()
            recipe = _recipe()
            renders = [_record(recipe, "binaural"), _record(recipe, "foa")]
            write_manifests(storage, [recipe], renders)
            config_sha = storage.write_config_resolved({"dataset": {"schema_version": "clsdoa_v1.0"}, "normalization": {"per_render": False, "per_viewpoint": False}})
            digest = "0" * 64
            storage.write_identity({"dataset_id": "clsdoa_v1_fixture_001", "dataset_family": "soundspaces_binaural_foa_clsdoa_v1", "schema_version": "clsdoa_v1.0", "generation_code_commit": "fixture", "created_at": "2026-08-28T00:00:00Z", "config_sha256": config_sha, "ontology_sha256": digest, "source_registry_sha256": digest, "scene_registry_sha256": digest})
            storage.write_resources_lock({"soundspaces_commit": "fixture", "habitat_sim_version": "fixture", "rlraudio_propagation": "fixture", "hrtf": {"path": "fixture", "sha256": digest}, "foa_contract_version": "fixture", "ontology_sha256": digest, "source_registry_sha256": digest, "scene_registry_sha256": digest, "source_split_version": "UNASSIGNED", "scene_split_version": "UNASSIGNED", "sample_rate_hz": 24000, "clip_duration_sec": 5.0, "indirectRayCount": 5000, "sourceRayCount": 200, "materials_mode": "OFF", "random_seed": 1})
            storage.finalize()
            with self.assertRaises(V1StorageError):
                storage.write_json("reports/late.json", {})

    def test_scene_registry_rejects_unknown_family_and_materials_on(self):
        row = {"scene_id": "fixture", "scene_family": "Replica", "scene_asset": "asset", "stage_config": "stage", "navmesh": "nav", "semantic_info": "semantic", "materials_mode": "OFF", "unit_scale": 1.0, "resource_hash": "fixture", "admitted": "NOT_RUN", "exclude_reason": None, "split": "UNASSIGNED"}
        self.assertEqual(validate_scene_rows({"fixture": row})[0]["scene_family"], "Replica")
        with self.assertRaises(SceneRegistryError):
            validate_scene_rows({"fixture": dict(row, scene_family="Other")})
        with self.assertRaises(SceneRegistryError):
            validate_scene_rows({"fixture": dict(row, materials_mode="ON")})

    def test_dataset_id_path_traversal_and_v0_root_are_rejected_or_separate(self):
        with self.assertRaises(V1StorageError):
            V1DatasetStorage.from_config(tempfile.gettempdir(), {"dataset_id": "../escape"})
        with tempfile.TemporaryDirectory() as temp:
            storage = V1DatasetStorage.from_config(temp, {"dataset_id": "fixture"})
            self.assertNotIn("active_audition_v0", str(storage.root))

    def test_manifest_is_sorted_atomic_and_duplicate_free(self):
        with tempfile.TemporaryDirectory() as temp:
            storage = V1DatasetStorage(Path(temp) / "dataset")
            recipes = [_recipe("ep_000002"), _recipe("ep_000001")]
            renders = [_record(recipe, rep) for recipe in recipes for rep in ("foa", "binaural")]
            write_manifests(storage, recipes, renders)
            first = (storage.manifest_path("episodes.jsonl").read_bytes(), storage.manifest_path("renders.jsonl").read_bytes())
            write_manifests(storage, reversed(recipes), reversed(renders))
            second = (storage.manifest_path("episodes.jsonl").read_bytes(), storage.manifest_path("renders.jsonl").read_bytes())
            self.assertEqual(first, second)
            self.assertEqual([row["episode_id"] for row in read_jsonl(str(storage.manifest_path("episodes.jsonl")))], ["ep_000001", "ep_000002"])
            with self.assertRaises(ManifestError):
                write_manifests(storage, recipes + [recipes[0]], renders)

    def test_qc_cross_tables_base_clip_reuse_and_custom_bins(self):
        first = _recipe("ep_000001", split="train", source=(-1.0, 1.5, -1.0))
        second = _recipe("ep_000002", split="val", source=(2.0, 2.5, -2.0))
        second = replace(second, base_clip_id=first.base_clip_id, source_dataset="other", scene_family="MP3D")
        report = build_distribution_report([first, second], [], {"azimuth_bins": [{"name": "leftish", "min": -180, "max": 0}, {"name": "rightish", "min": 0, "max": 180}], "distance_bins": {"short": [0, 2], "long": [2, 10]}, "elevation_bands": [("flat", -10, 10), ("raised", 10, 90)]})
        self.assertEqual(report["class_by_split"], {"0": {"train": 1, "val": 1}})
        self.assertEqual(report["class_by_scene_family"], {"0": {"MP3D": 1, "Replica": 1}})
        self.assertEqual(report["class_by_source_dataset"], {"0": {"fixture": 1, "other": 1}})
        self.assertEqual(report["class_by_azimuth_bin"], {"0": {"leftish": 1, "rightish": 1}})
        self.assertEqual(report["class_by_distance_bin"], {"0": {"long": 1, "short": 1}})
        self.assertEqual(report["class_by_elevation_band"], {"0": {"flat": 1, "raised": 1}})
        self.assertEqual(report["unique_base_clip_id_by_class"], {"0": 1})
        self.assertEqual(report["source_reuse"]["identity_key"], "base_clip_id")
        self.assertEqual(report["source_reuse"]["mean"], 2.0)
        self.assertEqual(report["source_reuse"]["median"], 2.0)
        self.assertEqual(report["source_reuse"]["p95"], 2.0)
        self.assertEqual(report["source_reuse"]["max"], 2)
        self.assertEqual(set(report["distance_bins"]), {"short", "long"})
        self.assertEqual(report["acoustic_qc"]["status"], "NOT_RUN")

    def test_audio_representation_and_no_normalization_invariants(self):
        self.assertEqual(_record(_recipe(), "binaural").num_channels, 2)
        self.assertEqual(_record(_recipe(), "foa").format, "AmbiX ACN/SN3D")
        config = {
            "audio": {"sample_rate_hz": 24000, "clip_duration_sec": 5.0, "num_samples": 120000, "dtype": "float32"},
            "representations": {"binaural": {"channels": 2, "channel_order": ["LEFT", "RIGHT"]}, "foa": {"channels": 4, "canonical": {"format": "AmbiX ACN/SN3D"}}},
            "normalization": {"per_render": False, "per_viewpoint": False, "separate_branch_normalization": False},
        }
        validate_audio_contract(config)
        with self.assertRaises(ValidationError):
            validate_audio_contract(dict(config, normalization={"per_render": True}))
        with self.assertRaises(SchemaError):
            RenderRecord("ep", "binaural", "a", "r", 16000, 2, 80000, "float32", "WAV", "complete")

    def test_short_source_schema_and_split_leakage(self):
        row = {key: "fixture" for key in ("source_clip_id", "base_clip_id", "canonical_class", "source_dataset", "source_label", "original_id", "original_path", "source_offset_policy", "license", "qc_status", "qc_notes", "pretrain_seen_status", "sha256")}
        row.update({"original_sample_rate": 24000, "original_channels": 1, "duration_sec": 1.25, "crop_start_sec": 0.0, "crop_end_sec": 1.25, "split": "train"})
        self.assertEqual(len(validate_source_rows([row])), 1)
        timeline = build_observation_timeline([1.0] * 24000, 24000, 0.5)
        self.assertEqual(timeline.shape, (120000,))
        self.assertEqual(timeline.dtype.name, "float32")
        self.assertEqual(float(timeline[0]), 0.0)
        self.assertGreater(float(timeline[12000]), 0.0)
        with self.assertRaises(SourceRegistryError):
            build_observation_timeline([1.0] * 24000, 24000, 4.5)
        with self.assertRaises(SourceRegistryError):
            bad = dict(row, duration_sec=0.0)
            validate_source_rows([bad])
        with self.assertRaises(ValidationError):
            validate_split_leakage([{"base_clip_id": "same", "split": "train"}, {"base_clip_id": "same", "split": "val"}], [])
        with self.assertRaises(ValidationError):
            validate_split_leakage([], [{"scene_id": "same", "split": "train"}, {"scene_id": "same", "split": "test"}])

    def test_paired_renderer_and_payload_resume_hook(self):
        recipe = _recipe()
        records = render_pair(_FakeRenderer(), recipe)
        self.assertEqual([record.representation for record in records], ["binaural", "foa"])
        validate_pairing([recipe], records)
        with tempfile.TemporaryDirectory() as temp:
            storage = V1DatasetStorage(Path(temp) / "dataset")
            for record in records:
                storage.write_bytes(record.audio_path, b"fixture")
                storage.write_bytes(record.rir_path, b"fixture")
                self.assertTrue(payload_is_complete(storage, record))
            Path(storage.root / records[0].audio_path).unlink()
            self.assertFalse(payload_is_complete(storage, records[0]))

    def test_rir_policy_supports_pilot_default_and_formal_audio_only(self):
        recipe = _recipe()
        audio_only = RenderRecord(recipe.episode_id, "binaural", "audio/binaural/ep.wav", None, 24000, 2, 120000, "float32", "WAV", "complete")
        self.assertEqual(RenderPolicy(), RenderPolicy(save_rir=True, require_rir=True))
        with tempfile.TemporaryDirectory() as temp:
            pilot = V1DatasetStorage(Path(temp) / "pilot")
            pilot.write_bytes(audio_only.audio_path, b"fixture")
            self.assertFalse(payload_is_complete(pilot, audio_only))
            formal = V1DatasetStorage(Path(temp) / "formal", save_rir=False, require_rir=False)
            formal.write_bytes(audio_only.audio_path, b"fixture")
            self.assertTrue(payload_is_complete(formal, audio_only))
            self.assertTrue(payload_is_complete(formal, audio_only, require_rir=False))
        with self.assertRaises(SchemaError):
            RenderPolicy(save_rir=False, require_rir=True)

    def test_qc_report_is_schema_only_and_has_no_fake_acoustic_values(self):
        report = build_distribution_report([_recipe()], [_record(_recipe(), "binaural")])
        self.assertEqual(report["episode_count"], 1)
        self.assertEqual(report["acoustic_qc"]["status"], "NOT_RUN")
        self.assertNotIn("rms", report["acoustic_qc"])

    def test_resolved_config_is_deterministic_and_has_no_pilot_quota(self):
        result = merge_resolved_config({"dataset": {"schema_version": "clsdoa_v1.0"}}, {"build": {"episode_count": 2}}, {"runtime": {"seed": 1}})
        self.assertEqual(result["build"]["episode_count"], 2)
        self.assertNotIn("960", str(result))
        with tempfile.TemporaryDirectory() as temp:
            first = V1DatasetStorage(Path(temp) / "first")
            second = V1DatasetStorage(Path(temp) / "second")
            config = {"dataset": {"schema_version": "clsdoa_v1.0"}, "build": {"episode_count": 2}}
            self.assertEqual(first.write_config_resolved(config), second.write_config_resolved(config))


if __name__ == "__main__":
    unittest.main()
