"""Small, explicit SoundSpaces FOA to Dataset V1 converter."""

import numpy as np


NATIVE_ORDER = ("W", "Y", "Z", "X")
CANONICAL_ORDER = ("W", "Y", "Z", "X")


def native_foa_to_ambix(values):
    """Convert N3D [W,Y,Z,X] to AmbiX ACN/SN3D [W,Y,Z,X]."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] != 4:
        raise ValueError("expected FOA with channel-first shape (4, ...)")
    output = array.copy()
    output[1:4] *= np.float32(1.0 / np.sqrt(3.0))
    return output


def project_to_dcase_azimuth(azimuth_deg):
    """Project convention is right-positive; DCASE convention is left-positive."""
    return -float(azimuth_deg)
