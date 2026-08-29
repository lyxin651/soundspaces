import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from active_audition.datasets.binaural_foa_clsdoa.recipe import make_episode_recipe
from active_audition.datasets.binaural_foa_clsdoa.renderer import PairedRenderError, SoundSpacesPairedRenderer, render_pair
from active_audition.datasets.binaural_foa_clsdoa.schema import NUM_SAMPLES, RenderPolicy


def _recipe():
    return make_episode_recipe(
        episode_id="renderer_test_001", split="UNASSIGNED", scene_id="replica.office_0", scene_family="Replica",
        source_clip_id="fixture", base_clip_id="fixture", source_dataset="fixture", class_id=0,
        source_position_world=(0.0, 1.5, -1.0), source_gain_db=-3.0, source_offset_sec=0.0,
        listener_base_position_world=(0.0, 0.0, 0.0), listener_sensor_position_world=(0.0, 1.5, 0.0), listener_yaw_deg=15.0,
    )


class ProductionRendererContractTests(unittest.TestCase):
    def test_short_source_is_placed_on_episode_timeline(self):
        source = np.linspace(-1.0, 1.0, 24000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            renderer = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=source, output_dir=directory)
            recipe = replace(_recipe(), source_gain_db=0.0, source_offset_sec=2.0)
            timeline = renderer._dry(recipe)
        self.assertTrue(np.all(timeline[:48000] == 0.0))
        self.assertTrue(np.array_equal(timeline[48000:72000], source))
        self.assertTrue(np.all(timeline[72000:] == 0.0))
        self.assertEqual(timeline.shape, (NUM_SAMPLES,))
        self.assertEqual(timeline.dtype, np.float32)

    def test_offset_does_not_change_short_source_content_or_amplitude(self):
        source = np.full(24000, 0.25, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            renderer = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=source, output_dir=directory)
            zero = renderer._dry(replace(_recipe(), source_gain_db=0.0))
            shifted = renderer._dry(replace(_recipe(), source_gain_db=0.0, source_offset_sec=1.0))
        self.assertAlmostEqual(float(np.max(zero)), 0.25)
        self.assertAlmostEqual(float(np.max(shifted)), 0.25)
        self.assertEqual(float(np.sum(np.abs(zero))), float(np.sum(np.abs(shifted))))

    def test_offset_outside_timeline_is_hard_failure(self):
        source = np.ones(24000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            renderer = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=source, output_dir=directory)
            recipe = replace(_recipe(), source_offset_sec=4.1)
            with self.assertRaises(PairedRenderError):
                renderer._dry(recipe)

    def test_five_second_source_at_zero_offset_remains_valid(self):
        source = np.ones(NUM_SAMPLES, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            renderer = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=source, output_dir=directory)
            timeline = renderer._dry(replace(_recipe(), source_gain_db=0.0))
        self.assertTrue(np.array_equal(timeline, source))

    def test_source_waveform_and_source_audio_path_share_timeline_semantics(self):
        source = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        recipe = replace(_recipe(), source_gain_db=0.0, source_offset_sec=2.0)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "source.wav"
            wavfile.write(str(audio_path), 16000, source)
            from_waveform = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=source, source_sample_rate_hz=16000, output_dir=directory)._dry(recipe)
            from_path = SoundSpacesPairedRenderer(scene_path="unused", source_audio_path=str(audio_path), output_dir=directory)._dry(recipe)
        self.assertTrue(np.array_equal(from_waveform, from_path))

    def test_listener_base_sensor_invariant_is_checked_before_habitat(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=np.ones(NUM_SAMPLES, dtype=np.float32), output_dir=directory)
            renderer._validate_listener_pose(_recipe())
            close = _recipe().__class__(**dict(_recipe().__dict__, listener_sensor_position_world=(0.0, 1.500009, 0.0)))
            renderer._validate_listener_pose(close)
            bad = _recipe().__class__(**dict(_recipe().__dict__, listener_sensor_position_world=(0.0, 1.51, 0.0)))
            with self.assertRaises(PairedRenderError):
                renderer._validate_listener_pose(bad)

    def test_production_waveform_gain_ratio_is_preserved_for_binaural_and_foa(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=np.ones(NUM_SAMPLES, dtype=np.float32), output_dir=directory)
            zero = replace(_recipe(), source_gain_db=0.0)
            minus_six = replace(zero, source_gain_db=-6.0)
            binaural_rir = np.zeros((2, 8), dtype=np.float32)
            binaural_rir[:, 0] = 1.0
            foa_rir = np.zeros((4, 8), dtype=np.float32)
            foa_rir[0, 0] = 1.0
            for representation, rir in (("binaural", binaural_rir), ("foa", foa_rir)):
                first, _ = renderer._waveform_from_rir(zero, representation, rir)
                second, _ = renderer._waveform_from_rir(minus_six, representation, rir)
                ratio = np.sqrt(np.sum(second * second, dtype=np.float64) / np.sum(first * first, dtype=np.float64))
                self.assertAlmostEqual(float(ratio), 10.0 ** (-6.0 / 20.0), places=6)

    def test_pair_uses_same_immutable_recipe_and_both_payloads(self):
        recipe = _recipe()
        dry = np.ones(NUM_SAMPLES, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            renderer = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=dry, output_dir=directory, policy=RenderPolicy())
            seen = []
            native_binaural = np.zeros((2, 8), dtype=np.float32)
            native_binaural[:, 0] = 1.0
            native_foa = np.zeros((4, 8), dtype=np.float32)
            native_foa[0, 0] = 1.0
            native_foa[3, 0] = 1.0

            def fake_rir(value, representation):
                seen.append((value, representation))
                return native_binaural if representation == "binaural" else native_foa

            with patch.object(renderer, "_rir", side_effect=fake_rir):
                records = render_pair(renderer, recipe)
            self.assertEqual([item[1] for item in seen], ["binaural", "foa"])
            self.assertTrue(all(item[0] is recipe for item in seen))
            self.assertEqual([record.render_status for record in records], ["complete", "complete"])
            self.assertEqual([record.num_channels for record in records], [2, 4])
            self.assertTrue(all(Path(directory, record.audio_path).is_file() for record in records))
            self.assertTrue(all(Path(directory, record.rir_path).is_file() for record in records))

    def test_policy_can_skip_rir_without_changing_renderer_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = SoundSpacesPairedRenderer(scene_path="unused", source_waveform=np.ones(NUM_SAMPLES, dtype=np.float32), output_dir=directory, policy=RenderPolicy(save_rir=False, require_rir=False))
            with patch.object(renderer, "_rir", return_value=np.ones((2, 4), dtype=np.float32)):
                record = renderer.render_episode(_recipe(), "binaural")
            self.assertIsNone(record.rir_path)


if __name__ == "__main__":
    unittest.main()
