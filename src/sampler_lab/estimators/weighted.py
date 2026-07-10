"""Generic moments under normalized nonnegative weights."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def _values_and_weights(
    samples: ArrayLike,
    normalized_weights: ArrayLike,
    observable: Callable[[Array], ArrayLike] | None,
) -> tuple[Array, Array]:
    sample_array = np.asarray(samples, dtype=np.float64)
    values = (
        sample_array
        if observable is None
        else np.asarray(observable(sample_array), dtype=np.float64)
    )
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(normalized_weights, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("observable must produce one scalar per sample")
    if values.size == 0:
        raise ValueError("at least one sample is required")
    if weights.ndim != 1 or weights.shape != values.shape:
        raise ValueError("normalized_weights must match the scalar observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("observable values must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("normalized_weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if not np.isclose(total, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("normalized_weights must sum to one")
    return values, weights / total


def weighted_mean(
    samples: ArrayLike,
    normalized_weights: ArrayLike,
    observable: Callable[[Array], ArrayLike] | None = None,
) -> float:
    """Compute a scalar weighted mean using normalized weights."""

    values, weights = _values_and_weights(samples, normalized_weights, observable)
    return float(weights @ values)


def weighted_variance(
    samples: ArrayLike,
    normalized_weights: ArrayLike,
    observable: Callable[[Array], ArrayLike] | None = None,
    *,
    unbiased: bool = False,
) -> float:
    """Compute weighted variance about the weighted mean.

    With ``unbiased=True``, the reliability-weight correction
    ``1 / (1 - sum(weights**2))`` is applied. It is undefined when a single
    observation carries all weight.
    """

    values, weights = _values_and_weights(samples, normalized_weights, observable)
    mean = float(weights @ values)
    variance = float(weights @ np.square(values - mean))
    if not unbiased:
        return variance

    correction_denominator = 1.0 - float(weights @ weights)
    if correction_denominator <= 0.0:
        raise ValueError("unbiased weighted variance requires at least two positive weights")
    return variance / correction_denominator
