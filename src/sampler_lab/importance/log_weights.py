"""Construction and validation of importance log weights."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter

Array = NDArray[np.float64]
LogDensityFunction = Callable[[Array], float]


def log_importance_weights(
    samples: ArrayLike,
    log_target: LogDensityFunction,
    log_proposal: LogDensityFunction,
    *,
    counter: OperationCounter | None = None,
) -> Array:
    """Evaluate ``log_target - log_proposal`` for each sampled observation.

    The first array axis indexes observations. A finite target density where the
    proposal density is zero is rejected as a support violation rather than
    represented by a misleading infinite weight.
    """

    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.ndim == 0:
        raise ValueError("samples must have a leading observation axis")
    if not np.all(np.isfinite(sample_array)):
        raise ValueError("samples must be finite")

    weights = np.empty(sample_array.shape[0], dtype=np.float64)
    for index, raw_point in enumerate(sample_array):
        point = np.asarray(raw_point, dtype=np.float64)
        target_value = float(log_target(point))
        proposal_value = float(log_proposal(point))
        if counter is not None:
            counter.increment("log_density_evaluations")
            counter.increment("proposal_density_evaluations")
        if np.isnan(target_value) or np.isnan(proposal_value):
            raise ValueError("log densities may not return nan")
        if np.isposinf(target_value) or np.isposinf(proposal_value):
            raise ValueError("log densities may not return +inf")
        if np.isfinite(target_value) and np.isneginf(proposal_value):
            raise ValueError("proposal support does not cover the target support")
        if np.isneginf(target_value) and np.isneginf(proposal_value):
            raise ValueError("both log densities are -inf at a sampled point")
        weights[index] = target_value - proposal_value

    return weights
