"""Gaussian sampling without delegating to NumPy's normal sampler."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.numerics import validate_size

Array = NDArray[np.float64]


def box_muller(
    rng: np.random.Generator,
    size: int,
    *,
    counter: OperationCounter | None = None,
) -> Array:
    """Generate standard normal variates with the Box-Muller transform.

    For odd ``size``, one member of the last generated pair is intentionally discarded.
    Operation counts record the uniforms actually consumed and normals returned.
    """

    size = validate_size(size)
    if size == 0:
        return np.empty(0, dtype=np.float64)

    n_pairs = (size + 1) // 2
    # Generator.random() can theoretically produce exactly zero. Replacing zero by the
    # smallest positive float preserves the endpoint convention while avoiding log(0).
    u1 = np.maximum(rng.random(n_pairs), np.nextafter(0.0, 1.0))
    u2 = rng.random(n_pairs)
    if counter is not None:
        counter.increment("uniform_draws", 2 * n_pairs)
        counter.increment("normal_draws", size)

    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    pairs = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)
    return pairs.reshape(-1)[:size]
