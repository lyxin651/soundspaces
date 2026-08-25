"""Minimal Habitat-Sim context for M1 planning."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import quaternion  # Must be imported before habitat_sim.
import habitat_sim

from active_audition.data.catalog import load_scene_registry


class SimulatorError(RuntimeError):
    """Raised when the V0 simulator context cannot be created."""


class SimulatorContext:
    """Own the live simulator and expose only the M1 planning handles."""

    def __init__(self, simulator: habitat_sim.Simulator, scene: Dict[str, Any]):
        self._simulator = simulator
        self.scene = scene

    @property
    def simulator(self) -> habitat_sim.Simulator:
        return self._simulator

    @property
    def agent(self) -> Any:
        return self._simulator.get_agent(0)

    @property
    def audio_sensor(self) -> Any:
        return self.agent._sensors["audio_sensor"]

    @property
    def pathfinder(self) -> Any:
        return self._simulator.pathfinder

    def close(self) -> None:
        self._simulator.close()

    def __enter__(self) -> "SimulatorContext":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def create_scene_simulator(config: Dict[str, Any], scene_id: Optional[str] = None) -> SimulatorContext:
    """Create a materials-OFF scene with a binaural sensor, without observing it."""

    repo_root = Path(config["_repo_root"])
    selected_scene_id = scene_id or config["scene"]["ids"][0]
    scenes = load_scene_registry(config["registries"]["scenes_path"], str(repo_root))
    if selected_scene_id not in scenes:
        raise SimulatorError("scene is not registered: {}".format(selected_scene_id))
    scene = scenes[selected_scene_id]

    backend_config = habitat_sim.SimulatorConfiguration()
    backend_config.scene_id = scene["scene_asset"]
    backend_config.enable_physics = False
    backend_config.load_semantic_mesh = True
    agent_config = habitat_sim.agent.AgentConfiguration()
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend_config, [agent_config]))
    try:
        if not simulator.pathfinder.is_loaded:
            if not simulator.pathfinder.load_nav_mesh(scene["navmesh"]):
                raise SimulatorError("failed to load navmesh: {}".format(scene["navmesh"]))

        audio_spec = habitat_sim.AudioSensorSpec()
        audio_spec.uuid = "audio_sensor"
        audio_spec.enableMaterials = False
        audio_spec.channelLayout.type = (
            habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        )
        audio_spec.channelLayout.channelCount = 2
        audio_spec.position = list(config["listener"]["sensor_offset_m"])
        audio_spec.acousticsConfig.sampleRate = config["acoustics"]["sample_rate_hz"]
        simulator.add_sensor(audio_spec)
    except Exception:
        simulator.close()
        raise
    return SimulatorContext(simulator, scene)
