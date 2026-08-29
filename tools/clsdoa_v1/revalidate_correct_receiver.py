"""Revalidate frozen admission probes with the frozen production receiver path."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


class _AgentProxy:
    def __init__(self, agent, capture):
        self._agent = agent
        self._capture = capture
        self._sensors = agent._sensors

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def set_state(self, state, *args, **kwargs):
        result = self._agent.set_state(state, *args, **kwargs)
        sensor = self._agent._sensors["audio_sensor"]
        self._capture["agent_world_position"] = np.asarray(self._agent.get_state().position, dtype=float).tolist()
        self._capture["effective_receiver_world"] = np.asarray(sensor.node.absolute_translation, dtype=float).tolist()
        return result


class _SimulatorProxy:
    def __init__(self, simulator, capture):
        self._simulator = simulator
        self._capture = capture

    def __getattr__(self, name):
        return getattr(self._simulator, name)

    def get_agent(self, index):
        return _AgentProxy(self._simulator.get_agent(index), self._capture)


def _resolve(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.production_root)))
    from active_audition.datasets.binaural_foa_clsdoa.recipe import make_episode_recipe
    from active_audition.datasets.binaural_foa_clsdoa.renderer import SoundSpacesPairedRenderer
    from active_audition.scene.pose import yaw_to_quaternion
    from examples.foa_adapter import native_foa_to_canonical

    rows = []
    for entry in json.loads(Path(args.input).read_text(encoding="utf-8")):
        renderer = SoundSpacesPairedRenderer(
            scene_path=str(_resolve(entry["scene_asset"])),
            navmesh_path=str(_resolve(entry["navmesh"])),
            source_waveform=np.ones(120000, dtype=np.float32),
            output_dir="/tmp/clsdoa_step2b_acoustic_revalidation_payload",
            indirect_ray_count=5000,
            source_ray_count=200,
            materials_enabled=False,
        )
        def render_on_simulator(simulator, recipe, representation):
            agent = simulator.get_agent(0)
            sensor = agent._sensors["audio_sensor"]
            sensor.reset()
            sensor.setAudioSourceTransform(np.asarray(recipe.source_position_world, dtype=np.float32))
            state = agent.get_state()
            state.position = np.asarray(recipe.listener_base_position_world, dtype=np.float32)
            state.rotation = yaw_to_quaternion(recipe.listener_yaw_deg)
            state.sensor_states = {}
            agent.set_state(state, infer_sensor_states=True)
            value = renderer._channels(simulator.get_sensor_observations()["audio_sensor"], 2 if representation == "binaural" else 4)
            return value, np.asarray(sensor.node.absolute_translation, dtype=float).tolist(), np.asarray(agent.get_state().position, dtype=float).tolist()

        binaural_simulator = renderer._new_simulator("binaural")
        foa_simulator = renderer._new_simulator("foa")
        try:
          for probe in entry["probes"]:
            recipe = make_episode_recipe(
                episode_id=entry["scene_id"] + "." + probe["probe_id"], split="UNASSIGNED",
                scene_id=entry["scene_id"], scene_family=entry["scene_family"],
                source_clip_id="revalidation", base_clip_id="revalidation", source_dataset="revalidation", class_id=0,
                source_position_world=tuple(probe["source_position_world"]), source_gain_db=0.0, source_offset_sec=0.0,
                listener_base_position_world=tuple(probe["listener_base_position_world"]),
                listener_sensor_position_world=tuple(probe["listener_sensor_position_world"]),
                listener_yaw_deg=probe["listener_yaw_deg"],
            )
            native_binaural, binaural_receiver, agent_position = render_on_simulator(binaural_simulator, recipe, "binaural")
            native_foa, foa_receiver, _ = render_on_simulator(foa_simulator, recipe, "foa")
            canonical_foa = native_foa_to_canonical(native_foa, recipe.listener_yaw_deg)
            base = np.asarray(probe["listener_base_position_world"], dtype=float)
            sensor = np.asarray(probe["listener_sensor_position_world"], dtype=float)
            effective = np.asarray(binaural_receiver, dtype=float)
            def metrics(value, channels):
                value = np.asarray(value)
                if value.ndim == 2 and value.shape[0] != channels:
                    value = value.T
                return {"channel_count": channels, "shape": list(value.shape), "finite": bool(np.isfinite(value).all()), "nonzero": bool(np.any(np.abs(value))), "rir_length": int(value.shape[-1]), "per_channel_energy": np.sum(value * value, axis=1, dtype=np.float64).tolist()}
            rows.append({"scene_id":entry["scene_id"],"scene_family":entry["scene_family"],"probe_id":probe["probe_id"],"scene_asset":entry["scene_asset"],"listener_base_position_world":base.tolist(),"listener_sensor_position_world":sensor.tolist(),"agent_world_position":agent_position,"effective_receiver_world":effective.tolist(),"foa_effective_receiver_world":foa_receiver,"receiver_error_m":float(np.linalg.norm(effective-sensor)),"foa_receiver_error_m":float(np.linalg.norm(np.asarray(foa_receiver)-sensor)),"source_position_world":probe["source_position_world"],"listener_yaw_deg":probe["listener_yaw_deg"],"distance_m":probe["distance_m"],"sample_rate_hz":24000,"materials_enabled":False,"indirect_ray_count":5000,"source_ray_count":200,"sequential_sensor_lifecycle":True,"normalization_applied":False,"binaural":{"channel_semantics":["LEFT","RIGHT"],**metrics(native_binaural,2)},"foa":{"native_channel_contract":"N3D [W,Y_RLR,Z_RLR,X_RLR]","converter_identity":"examples.foa_adapter.native_foa_to_canonical","canonical_order":"ACN [W,Y,Z,X]","normalization":"SN3D only; no amplitude normalization",**metrics(canonical_foa,4)}})
        finally:
            binaural_simulator.close()
            foa_simulator.close()
    Path(args.output).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
