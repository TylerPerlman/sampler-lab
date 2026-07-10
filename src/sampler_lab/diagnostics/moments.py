"""Moment diagnostics for sample validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MomentSummary:
    """Empirical first and second central moments."""

    mean: Array
    covariance: Array
    n_samples: int


def moment_summary(samples: ArrayLike) -> MomentSummary:
    """Compute mean and unbiased covariance for scalar or vector samples."""

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("samples must contain at least two scalar or vector observations")
    if not np.all(np.isfinite(array)):
        raise ValueError("samples must be finite")
    return MomentSummary(
        mean=np.mean(array, axis=0),
        covariance=np.atleast_2d(np.cov(array, rowvar=False, ddof=1)),
        n_samples=int(array.shape[0]),
    )
