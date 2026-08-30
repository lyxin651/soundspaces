"""Small, explicit SoundSpaces FOA to Dataset V1 converter."""

import numpy as np


NATIVE_ORDER = ("W", "Y", "Z", "X")
CANONICAL_ORDER = ("W", "Y", "Z", "X")


def _validate_foa(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 4 or array.shape[1] == 0:
        raise ValueError("expected non-empty FOA with channel-first shape (4, N)")
    if not np.isfinite(array).all():
        raise ValueError("FOA values must be finite")
    return array


def find_shared_direct_sample(values):
    """Find one arrival sample shared by every FOA channel."""
    array = _validate_foa(values)
    return int(np.argmax(np.sum(array * array, axis=0)))


def measure_shared_direct_coefficients(values, window_radius=8):
    """Measure signed FOA coefficients around one shared direct arrival."""
    array = _validate_foa(values)
    if window_radius < 0:
        raise ValueError("window_radius must be non-negative")
    sample = find_shared_direct_sample(array)
    start = max(0, sample - window_radius)
    stop = min(array.shape[1], sample + window_radius + 1)
    return {
        "direct_sample": sample,
        "signed_coefficients": array[:, sample].copy(),
        "window_energy": np.sum(array[:, start:stop] ** 2, axis=1),
    }


def native_foa_to_ambix(values):
    """Convert N3D [W,Y,Z,X] to AmbiX ACN/SN3D [W,Y,Z,X]."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] != 4:
        raise ValueError("expected FOA with channel-first shape (4, ...)")
    output = array.copy()
    output[1:4] *= np.float32(1.0 / np.sqrt(3.0))
    return output


def native_foa_to_canonical(values, listener_yaw_deg=0.0):
    """Convert world-fixed native N3D FOA to model-facing AmbiX ACN/SN3D.

    Native channels are [W,Y_RLR,Z_RLR,X_RLR], where the Habitat/RLR
    listener-local axes are +X=right, +Y=up, +Z=back. The directional
    vector is first rotated from world axes into listener-local axes. The
    DCASE/STARSS model axes are +X=front, +Y=left, +Z=up, so the output
    channels are [W,Y_DCASE,Z_DCASE,X_DCASE] = [W,-X_RLR,+Y_RLR,-Z_RLR].
    """
    native = _validate_foa(values)
    yaw = np.deg2rad(float(listener_yaw_deg))
    x_world = native[3]
    y_world = native[1]
    z_world = native[2]
    # SoundSpaces native directions are world-fixed.  Project them into the
    # listener-local RLR frame using the same right/back convention as the
    # frozen geometry contract.
    x_local = np.cos(yaw) * x_world + np.sin(yaw) * z_world
    z_local = -np.sin(yaw) * x_world + np.cos(yaw) * z_world
    output = np.empty_like(native)
    output[0] = native[0]
    output[1] = -x_local
    output[2] = y_world
    output[3] = -z_local
    output[1:4] *= np.float32(1.0 / np.sqrt(3.0))
    return output


def project_to_dcase_azimuth(azimuth_deg):
    """Project convention is right-positive; DCASE convention is left-positive."""
    return -float(azimuth_deg)
