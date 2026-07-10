"""Elementary estimators for independent Monte Carlo samples."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.results import IIDEstimate

Array = NDArray[np.float64]


def iid_estimate(
    samples: ArrayLike,
    observable: Callable[[Array], ArrayLike] | None = None,
) -> IIDEstimate:
    """Estimate a scalar expectation from IID samples.

    ``observable`` is evaluated once on the full sample array and should return one
    scalar value per sample. When omitted, ``samples`` itself must be one-dimensional.
    The reported variance is the unbiased sample variance for ``n >= 2``.
    """

    sample_array = np.asarray(samples, dtype=np.float64)
    values = (
        sample_array
        if observable is None
        else np.asarray(observable(sample_array), dtype=np.float64)
    )
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("observable must produce one scalar per sample")
    if values.size == 0:
        raise ValueError("at least one sample is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("observable values must be finite")

    n_samples = int(values.size)
    value = float(np.mean(values))
    if n_samples == 1:
        sample_variance = float("nan")
        standard_error = float("nan")
    else:
        sample_variance = float(np.var(values, ddof=1))
        standard_error = float(np.sqrt(sample_variance / n_samples))

    return IIDEstimate(
        value=value,
        sample_variance=sample_variance,
        standard_error=standard_error,
        n_samples=n_samples,
    )
