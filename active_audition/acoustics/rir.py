"""Live native SoundSpaces RIR acquisition and the frozen channel adapter."""

from typing import Any, Iterable, Mapping

import numpy as np
import quaternion

from active_audition.scene.pose import yaw_to_quaternion


class RIRRenderError(ValueError):
    """Raised when a native RIR or live acoustic observation is invalid."""


def _native_channels_to_time_major(native: Any) -> np.ndarray:
    """Validate native ``(2, N)`` data and expose raw channel 0/1 as ``(N, 2)``."""

    array = np.asarray(native)
    if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] <= 0:
        raise RIRRenderError("native binaural RIR must have shape (2, N), N > 0")
    if not np.isfinite(array).all():
        raise RIRRenderError("native RIR contains NaN/Inf")
    result = np.asarray(array.T, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RIRRenderError("canonical RIR contains NaN/Inf")
    return result


def canonicalize_binaural_rir(native: Any) -> np.ndarray:
    """Map proven native ``[R, L]`` channels to canonical ``[L, R]``.

    The native channel order is an A2 behavioral finding, not an A1 API claim.
    Direction calibration records raw channel 0/1 before this mapping.  The
    mapping is kept in this single adapter so downstream V1 code only sees the
    A0 canonical ``[L, R]`` order.
    """

    raw = _native_channels_to_time_major(native)
    result = np.asarray(raw[:, [1, 0]], dtype=np.float32)
    if not np.isfinite(result).all():
        raise RIRRenderError("canonical RIR contains NaN/Inf")
    return result


def _set_listener(context: Any, base_position: Iterable[float], yaw_deg: float) -> None:
    state = context.agent.get_state()
    state.position = np.asarray(tuple(base_position), dtype=np.float32)
    state.rotation = yaw_to_quaternion(float(yaw_deg))
    context.agent.set_state(state, infer_sensor_states=True)


def _render_native_observation(context: Any, source_position_world: Iterable[float], listener_pose: Any) -> Any:
    """Render and return the native AudioSensor observation without mapping."""

    # Native AudioSensor keeps simulator state across observations.  Reset it
    # before each viewpoint so repeated renders in one scene context are
    # deterministic without changing the frozen channel mapping.
    context.audio_sensor.reset()
    _set_listener(context, listener_pose.base_position_world, listener_pose.yaw_deg)
    context.audio_sensor.setAudioSourceTransform(np.asarray(tuple(source_position_world), dtype=np.float32))
    observations = context.simulator.get_sensor_observations()
    if "audio_sensor" not in observations:
        raise RIRRenderError("audio_sensor observation is missing")
    return observations["audio_sensor"]


def render_native_rir_raw(context: Any, source_position_world: Iterable[float], listener_pose: Any) -> np.ndarray:
    """Render raw native channel 0/1 as ``(N, 2)`` for A2 evidence only."""

    return _native_channels_to_time_major(_render_native_observation(context, source_position_world, listener_pose))


def render_native_rir(context: Any, source_position_world: Iterable[float], listener_pose: Any) -> np.ndarray:
    """Render one live RIR and return the canonical A0 ``[L, R]`` channels."""

    return canonicalize_binaural_rir(_render_native_observation(context, source_position_world, listener_pose))


def _channel_stats(rir: np.ndarray) -> Mapping[str, float]:
    return {
        "left_energy": float(np.sum(np.square(rir[:, 0], dtype=np.float64))),
        "right_energy": float(np.sum(np.square(rir[:, 1], dtype=np.float64))),
        "left_rms": float(np.sqrt(np.mean(np.square(rir[:, 0], dtype=np.float64)))),
        "right_rms": float(np.sqrt(np.mean(np.square(rir[:, 1], dtype=np.float64)))),
        "left_peak": float(np.max(np.abs(rir[:, 0]))),
        "right_peak": float(np.max(np.abs(rir[:, 1]))),
    }


def run_channel_order_gate(context: Any, listener_pose: Any, left_source: Iterable[float], right_source: Iterable[float]) -> Mapping[str, Any]:
    """Rerunnable live sanity gate for native channel ordering."""

    left_rir = render_native_rir(context, left_source, listener_pose)
    right_rir = render_native_rir(context, right_source, listener_pose)
    left_stats = _channel_stats(left_rir)
    right_stats = _channel_stats(right_rir)
    result = {
        "left_source": left_stats,
        "right_source": right_stats,
        "left_source_ch0_dominant": all(
            left_stats[left_key] > left_stats[right_key]
            for left_key, right_key in (("left_energy", "right_energy"), ("left_rms", "right_rms"), ("left_peak", "right_peak"))
        ),
        "right_source_ch1_dominant": all(
            right_stats[right_key] > right_stats[left_key]
            for left_key, right_key in (("left_energy", "right_energy"), ("left_rms", "right_rms"), ("left_peak", "right_peak"))
        ),
    }
    if not result["left_source_ch0_dominant"] or not result["right_source_ch1_dominant"]:
        raise RIRRenderError("binaural channel-order gate failed: {}".format(result))
    return result
