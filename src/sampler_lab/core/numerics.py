"""Small numerical primitives used throughout the package."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def logsumexp(values: ArrayLike, axis: int | tuple[int, ...] | None = None) -> Array | float:
    """Compute ``log(sum(exp(values)))`` stably.

    Empty reductions and slices containing only ``-inf`` follow NumPy-style
    floating-point semantics and return ``-inf`` rather than ``nan``.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("-inf") if axis is None else np.asarray(float("-inf"))

    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    shifted_sum = np.sum(np.exp(array - finite_maximum), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        result = np.log(shifted_sum) + finite_maximum
    result = np.where(np.isneginf(maximum), float("-inf"), result)

    if axis is None:
        return float(np.squeeze(result))
    return np.squeeze(result, axis=axis)


def normalize_log_weights(log_weights: ArrayLike) -> tuple[Array, float]:
    """Normalize one-dimensional log weights.

    Returns normalized linear weights and the log normalizing constant. At least
    one weight must be finite.
    """

    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("log_weights must be one-dimensional")
    if values.size == 0:
        raise ValueError("log_weights must be nonempty")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_weights may not contain nan or +inf")

    log_normalizer = float(logsumexp(values))
    if np.isneginf(log_normalizer):
        raise ValueError("at least one log weight must be finite")
    weights = np.exp(values - log_normalizer)
    return weights, log_normalizer


def validate_size(size: int) -> int:
    """Validate a requested sample count."""

    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 0:
        raise ValueError("size must be nonnegative")
    return size
