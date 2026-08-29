import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_audition.datasets.binaural_foa_clsdoa.recipe import make_episode_recipe
from active_audition.datasets.binaural_foa_clsdoa.renderer import SoundSpacesPairedRenderer, render_pair
from active_audition.datasets.binaural_foa_clsdoa.schema import NUM_SAMPLES, RenderPolicy


def _recipe():
    return make_episode_recipe(
        episode_id="renderer_test_001", split="UNASSIGNED", scene_id="replica.office_0", scene_family="Replica",
        source_clip_id="fixture", base_clip_id="fixture", source_dataset="fixture", class_id=0,
        source_position_world=(0.0, 1.5, -1.0), source_gain_db=-3.0, source_offset_sec=0.0,
        listener_base_position_world=(0.0, 0.0, 0.0), listener_sensor_position_world=(0.0, 1.5, 0.0), listener_yaw_deg=15.0,
    )


class ProductionRendererContractTests(unittest.TestCase):
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
