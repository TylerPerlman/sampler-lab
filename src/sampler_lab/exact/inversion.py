"""Inverse-transform samplers, including generalized discrete inverses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.numerics import validate_size

Array = NDArray[np.float64]


def inverse_cdf_sample(
    rng: np.random.Generator,
    inverse_cdf: Callable[[Array], ArrayLike],
    size: int,
    *,
    counter: OperationCounter | None = None,
) -> Array:
    """Sample by applying an inverse CDF to independent uniforms on ``[0, 1)``."""

    size = validate_size(size)
    uniforms = rng.random(size)
    if counter is not None:
        counter.increment("uniform_draws", size)
    samples = np.asarray(inverse_cdf(uniforms), dtype=np.float64)
    if samples.shape != uniforms.shape:
        raise ValueError("inverse_cdf must preserve the input shape")
    if np.any(np.isnan(samples)):
        raise ValueError("inverse_cdf returned nan")
    return samples


def generalized_inverse_discrete(
    rng: np.random.Generator,
    values: ArrayLike,
    probabilities: ArrayLike,
    size: int,
    *,
    counter: OperationCounter | None = None,
) -> Array:
    """Sample a finite distribution using its right-continuous generalized inverse.

    ``values`` may be any finite numeric support. Probabilities are normalized after
    validation, avoiding a false precision requirement that they sum to exactly one.
    """

    size = validate_size(size)
    support = np.asarray(values, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if support.ndim != 1 or probs.ndim != 1 or support.shape != probs.shape:
        raise ValueError("values and probabilities must be matching one-dimensional arrays")
    if support.size == 0:
        raise ValueError("the support must be nonempty")
    if not np.all(np.isfinite(support)):
        raise ValueError("support values must be finite")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("at least one probability must be positive")

    cumulative = np.cumsum(probs / total)
    cumulative[-1] = 1.0
    uniforms = rng.random(size)
    if counter is not None:
        counter.increment("uniform_draws", size)
    indices = np.searchsorted(cumulative, uniforms, side="right")
    return support[indices]
