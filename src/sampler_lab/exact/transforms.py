"""Reusable deterministic transforms for exact sampling."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def polar_to_cartesian(radius: ArrayLike, angle: ArrayLike) -> Array:
    """Convert broadcast-compatible polar coordinates to ``(..., 2)`` Cartesian points."""

    radius_array, angle_array = np.broadcast_arrays(
        np.asarray(radius, dtype=np.float64),
        np.asarray(angle, dtype=np.float64),
    )
    return np.stack(
        (radius_array * np.cos(angle_array), radius_array * np.sin(angle_array)),
        axis=-1,
    )
