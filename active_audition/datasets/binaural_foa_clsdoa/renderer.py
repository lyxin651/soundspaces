"""V1 paired binaural/FOA rendering contracts and live SoundSpaces backend."""

from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import quaternion  # Must be imported before habitat_sim.
from scipy.io import wavfile
from scipy.signal import fftconvolve

from active_audition.acoustics.renderer import convolve_binaural
from active_audition.data.audio import load_dry_segment
from active_audition.scene.pose import yaw_to_quaternion

from .recipe import EpisodeRecipe
from .schema import CLIP_DURATION_SEC, NUM_SAMPLES, REPRESENTATIONS, RenderPolicy, RenderRecord, SAMPLE_RATE_HZ


class PairedRenderError(ValueError):
    """Raised when a paired renderer violates the shared recipe contract."""


class PairedRenderer(ABC):
    """Backend-neutral interface for one representation from one recipe."""

    @abstractmethod
    def render_episode(self, recipe: EpisodeRecipe, representation: str) -> RenderRecord:
        raise NotImplementedError


class SoundSpacesPairedRenderer(PairedRenderer):
    """Production V1 renderer backed by the Habitat-Sim AudioSensor.

    A fresh simulator is used for each representation.  This is deliberate:
    it gives the native sensor the sequential lifecycle required by the V1
    contract while preventing state from one view or representation leaking
    into the next one.  The immutable recipe is the only per-episode input.
    """

    def __init__(
        self,
        *,
        scene_path: str,
        navmesh_path: Optional[str] = None,
        source_audio_path: Optional[str] = None,
        source_waveform: Optional[Any] = None,
        output_dir: str,
        policy: Optional[RenderPolicy] = None,
        indirect_ray_count: int = 5000,
        source_ray_count: int = 200,
        materials_enabled: bool = False,
    ) -> None:
        if source_audio_path is None and source_waveform is None:
            raise PairedRenderError("source_audio_path or source_waveform is required")
        if int(indirect_ray_count) != 5000 or int(source_ray_count) != 200:
            raise PairedRenderError("V1 acoustic ray counts are frozen at 5000/200")
        if materials_enabled:
            raise PairedRenderError("V1 production rendering requires Materials OFF")
        self.scene_path = str(scene_path)
        self.navmesh_path = str(navmesh_path) if navmesh_path else None
        self.source_audio_path = source_audio_path
        self.source_waveform = None if source_waveform is None else np.asarray(source_waveform, dtype=np.float32).copy()
        self.output_dir = Path(output_dir)
        self.policy = policy or RenderPolicy()
        self.indirect_ray_count = int(indirect_ray_count)
        self.source_ray_count = int(source_ray_count)

    @staticmethod
    def _habitat():
        import habitat_sim

        return habitat_sim

    def _new_simulator(self, representation: str, indirect: bool = True):
        habitat_sim = self._habitat()
        layout = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType
        channels = 2 if representation == "binaural" else 4
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = self.scene_path
        backend.enable_physics = False
        backend.load_semantic_mesh = True
        simulator = habitat_sim.Simulator(
            habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()])
        )
        try:
            if self.navmesh_path and not simulator.pathfinder.load_nav_mesh(self.navmesh_path):
                if not simulator.pathfinder.is_loaded:
                    raise PairedRenderError("V1 navmesh could not be loaded")
            spec = habitat_sim.AudioSensorSpec()
            spec.uuid = "audio_sensor"
            spec.enableMaterials = False
            spec.channelLayout.type = layout.Binaural if representation == "binaural" else layout.Ambisonics
            spec.channelLayout.channelCount = channels
            spec.position = [0.0, 1.5, 0.0]
            spec.acousticsConfig.sampleRate = SAMPLE_RATE_HZ
            spec.acousticsConfig.direct = True
            spec.acousticsConfig.indirect = bool(indirect)
            spec.acousticsConfig.diffraction = bool(indirect)
            spec.acousticsConfig.transmission = bool(indirect)
            spec.acousticsConfig.indirectRayCount = self.indirect_ray_count
            spec.acousticsConfig.sourceRayCount = self.source_ray_count
            simulator.add_sensor(spec)
            return simulator
        except Exception:
            simulator.close()
            raise

    @staticmethod
    def _channels(observation: Any, channels: int) -> np.ndarray:
        array = np.asarray(observation)
        if array.ndim != 2:
            raise PairedRenderError("AudioSensor observation must be 2-D: {}".format(array.shape))
        if array.shape[0] == channels:
            array = array
        elif array.shape[1] == channels:
            array = array.T
        else:
            raise PairedRenderError("AudioSensor channel count mismatch: {}".format(array.shape))
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all() or not np.any(np.abs(array)):
            raise PairedRenderError("AudioSensor RIR must be finite and non-zero")
        return array

    def _dry(self, recipe: EpisodeRecipe) -> np.ndarray:
        if self.source_waveform is not None:
            waveform = self.source_waveform
            if recipe.source_offset_sec:
                offset = int(round(recipe.source_offset_sec * SAMPLE_RATE_HZ))
                waveform = waveform[offset:]
            waveform = np.asarray(waveform[:NUM_SAMPLES], dtype=np.float32)
            if waveform.size < NUM_SAMPLES:
                waveform = np.pad(waveform, (0, NUM_SAMPLES - waveform.size))
            waveform = waveform * np.float32(10.0 ** (recipe.source_gain_db / 20.0))
        else:
            waveform = load_dry_segment(
                self.source_audio_path, recipe.source_offset_sec, CLIP_DURATION_SEC,
                SAMPLE_RATE_HZ, recipe.source_gain_db,
            )
        if waveform.shape != (NUM_SAMPLES,) or not np.isfinite(waveform).all():
            raise PairedRenderError("source waveform does not satisfy the V1 audio contract")
        return np.asarray(waveform, dtype=np.float32)

    def _rir(self, recipe: EpisodeRecipe, representation: str, *, indirect: bool = True) -> np.ndarray:
        simulator = self._new_simulator(representation, indirect=indirect)
        try:
            agent = simulator.get_agent(0)
            sensor = agent._sensors["audio_sensor"]
            state = agent.get_state()
            state.position = np.asarray(recipe.listener_base_position_world, dtype=np.float32)
            state.rotation = yaw_to_quaternion(recipe.listener_yaw_deg)
            state.sensor_states = {}
            agent.set_state(state, infer_sensor_states=True)
            sensor.setAudioSourceTransform(np.asarray(recipe.source_position_world, dtype=np.float32))
            observations = simulator.get_sensor_observations()
            return self._channels(observations["audio_sensor"], 2 if representation == "binaural" else 4)
        finally:
            simulator.close()

    def _write_payload(self, recipe: EpisodeRecipe, representation: str, waveform: np.ndarray, rir: np.ndarray):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        audio_relative = Path("audio") / representation / (recipe.episode_id + ".wav")
        audio_path = self.output_dir / audio_relative
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        if representation == "binaural":
            wavfile.write(str(audio_path), SAMPLE_RATE_HZ, np.asarray(waveform, dtype=np.float32))
        else:
            wavfile.write(str(audio_path), SAMPLE_RATE_HZ, np.asarray(waveform.T, dtype=np.float32))
        rir_path = None
        if self.policy.save_rir:
            rir_relative = Path("cache") / "rir" / representation / (recipe.episode_id + ".npy")
            rir_path = self.output_dir / rir_relative
            rir_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(rir_path), np.asarray(rir, dtype=np.float32), allow_pickle=False)
        return str(audio_relative), None if rir_path is None else str(rir_relative)

    def render_episode(self, recipe: EpisodeRecipe, representation: str) -> RenderRecord:
        if representation not in recipe.representations:
            raise PairedRenderError("recipe does not require {}".format(representation))
        dry = self._dry(recipe)
        native = self._rir(recipe, representation)
        if representation == "binaural":
            canonical_rir = native.T
            full = convolve_binaural(dry, canonical_rir)
            waveform = full[:NUM_SAMPLES, :]
        else:
            from examples.foa_adapter import native_foa_to_canonical

            canonical_rir = native_foa_to_canonical(native, recipe.listener_yaw_deg)
            waveform = np.asarray(
                [fftconvolve(dry, channel, mode="full")[:NUM_SAMPLES] for channel in canonical_rir],
                dtype=np.float32,
            )
        if waveform.shape != ((NUM_SAMPLES, 2) if representation == "binaural" else (4, NUM_SAMPLES)):
            raise PairedRenderError("rendered waveform shape is not V1-compatible")
        audio_path, rir_path = self._write_payload(recipe, representation, waveform, canonical_rir)
        return RenderRecord(
            episode_id=recipe.episode_id, representation=representation,
            audio_path=audio_path, rir_path=rir_path,
            sample_rate_hz=SAMPLE_RATE_HZ,
            num_channels=2 if representation == "binaural" else 4,
            num_samples=NUM_SAMPLES, dtype="float32",
            format="WAV" if representation == "binaural" else "AmbiX ACN/SN3D",
            render_status="complete",
        )


def render_pair(renderer: PairedRenderer, recipe: EpisodeRecipe) -> Sequence[RenderRecord]:
    """Render both representations from the same immutable recipe object."""

    records = []
    for representation in REPRESENTATIONS:
        if representation not in recipe.representations:
            raise PairedRenderError("recipe does not require {}".format(representation))
        record = renderer.render_episode(recipe, representation)
        if not isinstance(record, RenderRecord) or record.episode_id != recipe.episode_id or record.representation != representation:
            raise PairedRenderError("renderer returned a record inconsistent with the shared recipe")
        records.append(record)
    return tuple(records)
