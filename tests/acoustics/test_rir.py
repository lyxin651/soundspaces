import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from active_audition.acoustics.rir import RIRRenderError, canonicalize_binaural_rir, render_native_rir
from active_audition.acoustics.rir import run_channel_order_gate


class RIRCanonicalizationTests(unittest.TestCase):
    def test_native_render_resets_audio_sensor_before_source_transform(self):
        audio_sensor = Mock()
        context = SimpleNamespace(
            audio_sensor=audio_sensor,
            agent=SimpleNamespace(
                get_state=lambda: SimpleNamespace(),
                set_state=lambda state, infer_sensor_states: None,
            ),
            simulator=SimpleNamespace(
                get_sensor_observations=lambda: {
                    "audio_sensor": np.ones((2, 4), dtype=np.float64),
                }
            ),
        )
        listener = SimpleNamespace(base_position_world=(0.0, 0.0, 0.0), yaw_deg=0.0)

        result = render_native_rir(context, (1.0, 0.0, 0.0), listener)

        self.assertEqual(result.shape, (4, 2))
        self.assertEqual(audio_sensor.method_calls[0], call.reset())
        self.assertEqual(audio_sensor.method_calls[1][0], "setAudioSourceTransform")
        np.testing.assert_array_equal(
            audio_sensor.method_calls[1][1][0],
            np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
        )

    def test_native_left_right_mapping_shape_and_dtype(self):
        native = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64)
        result = canonicalize_binaural_rir(native)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[:, 1], [10.0, 20.0, 30.0])

    def test_empty_bad_shape_and_nonfinite_native_are_rejected(self):
        for native in (np.zeros((2, 0)), np.zeros((3, 2)), np.asarray([[np.inf], [0.0]])):
            with self.assertRaises(RIRRenderError):
                canonicalize_binaural_rir(native)

    def test_channel_gate_requires_energy_rms_and_peak_mirror_dominance(self):
        class FakeContext:
            pass
        context = FakeContext()
        listener = object()
        left = np.asarray([[3.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        right = np.asarray([[0.0, 1.0], [0.0, 3.0]], dtype=np.float32)
        with patch("active_audition.acoustics.rir.render_native_rir", side_effect=[left, right]):
            result = run_channel_order_gate(context, listener, (0, 0, 0), (1, 0, 0))
        self.assertTrue(result["left_source_ch0_dominant"])
        self.assertTrue(result["right_source_ch1_dominant"])


if __name__ == "__main__":
    unittest.main()
