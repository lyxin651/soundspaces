import hashlib
import unittest

import numpy as np
from scipy.signal import fftconvolve

from active_audition.a4.cache import CacheError, MixtureCacheKey, NOISE_SOURCE_TIME_IDENTITY
from active_audition.a4.identity import stable_id
from active_audition.a4.mixer import MixtureContract, WaveformComponent
from active_audition.a4.timeline import TimelineContract, _full_convolve, build_common_receiver_timeline


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _key(target_component, noise_component, timeline_id):
    return MixtureCacheKey(
        target_rir_cache_key=stable_id("rir-cache-key", {"fixture": "target"}),
        noise_rir_cache_key=stable_id("rir-cache-key", {"fixture": "noise"}),
        target_component_identity=target_component.component_identity,
        noise_component_identity=noise_component.component_identity,
        target_dry_waveform_sha256="1" * 64,
        noise_segment_payload_sha256="2" * 64,
        noise_segment_identity=stable_id("noise-segment", {"fixture": "noise"}),
        noise_source_time_identity=NOISE_SOURCE_TIME_IDENTITY,
        calibration_artifact_identity=stable_id("calibration", {"fixture": "calibration"}),
        global_gain_identity=stable_id("global-gain", {"fixture": "gain"}),
        timeline_identity=timeline_id,
        mixer_contract_identity=MixtureContract().identity,
    )


class PropagationCollisionTests(unittest.TestCase):
    def test_production_convolution_is_byte_identical_on_repeated_calls(self):
        source = np.random.default_rng(17).normal(size=4097).astype(np.float32)
        rir = np.random.default_rng(23).normal(size=(257, 2)).astype(np.float32)
        first = _full_convolve(source, rir)
        second = _full_convolve(source, rir)
        self.assertEqual(first.dtype, np.dtype(np.float32))
        self.assertEqual(first.shape, (4353, 2))
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(_sha(first), _sha(second))

    def test_direct_fixture_and_production_backend_have_distinct_component_identities(self):
        source = np.random.default_rng(101).normal(size=4097).astype(np.float32)
        rir = np.random.default_rng(103).normal(size=(257, 2)).astype(np.float32)
        current = _full_convolve(source, rir)
        direct = np.column_stack(
            [np.convolve(source, rir[:, index], mode="full") for index in range(2)]
        ).astype(np.float32)
        self.assertEqual(current.shape, direct.shape)
        self.assertNotEqual(current.tobytes(), direct.tobytes())
        timeline = build_common_receiver_timeline(
            source,
            source,
            rir,
            rir,
            stable_id("noise-segment", {"fixture": "target-source"}),
            stable_id("noise-segment", {"fixture": "noise-source"}),
            0,
            TimelineContract(),
        )
        direct_target = WaveformComponent.from_array("target", timeline.timeline_id, direct)
        direct_noise = WaveformComponent.from_array("noise", timeline.timeline_id, direct)
        current_target = WaveformComponent.from_array("target", timeline.timeline_id, current)
        current_noise = WaveformComponent.from_array("noise", timeline.timeline_id, current)
        self.assertNotEqual(direct_target.payload_sha256, current_target.payload_sha256)
        self.assertNotEqual(direct_noise.payload_sha256, current_noise.payload_sha256)
        self.assertNotEqual(
            _key(direct_target, direct_noise, timeline.timeline_id).cache_key,
            _key(current_target, current_noise, timeline.timeline_id).cache_key,
        )

    def test_historical_v2_payload_is_not_a_valid_v3_request(self):
        source = np.ones(32, dtype=np.float32)
        rir = np.zeros((8, 2), dtype=np.float32)
        rir[1, :] = 1.0
        timeline = build_common_receiver_timeline(
            source,
            source,
            rir,
            rir,
            stable_id("noise-segment", {"fixture": "target-source"}),
            stable_id("noise-segment", {"fixture": "noise-source"}),
            0,
            TimelineContract(),
        )
        target = WaveformComponent.from_array("target", timeline.timeline_id, timeline.target_binaural)
        noise = WaveformComponent.from_array("noise", timeline.timeline_id, timeline.noise_binaural)
        current = _key(target, noise, timeline.timeline_id).to_payload()
        legacy = dict(current)
        legacy["schema_version"] = "active-asr-a4-mixture-cache-key-v2"
        legacy.pop("target_component_identity")
        legacy.pop("noise_component_identity")
        with self.assertRaises(CacheError):
            MixtureCacheKey.from_payload(legacy)


if __name__ == "__main__":
    unittest.main()
