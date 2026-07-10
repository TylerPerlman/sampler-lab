"""Bias, variance, MSE, and RMSE summaries across replicated estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    """Empirical error decomposition for replicated scalar estimates."""

    mean_estimate: float
    bias: float
    variance: float
    mse: float
    rmse: float
    n_replications: int


def error_metrics(estimates: ArrayLike, truth: float) -> ErrorMetrics:
    """Compute empirical bias/variance/MSE using population variance convention."""

    values = np.asarray(estimates, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("estimates must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or not np.isfinite(truth):
        raise ValueError("estimates and truth must be finite")

    mean_estimate = float(np.mean(values))
    bias = mean_estimate - truth
    variance = float(np.var(values, ddof=0))
    squared_errors = np.square(values - truth)
    mse = float(np.mean(squared_errors))
    return ErrorMetrics(
        mean_estimate=mean_estimate,
        bias=bias,
        variance=variance,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        n_replications=int(values.size),
    )
