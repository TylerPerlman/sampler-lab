"""Dependence and efficiency diagnostics for complete ensemble trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.diagnostics.time_series import empirical_integrated_autocorrelation_time

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnsembleEfficiency:
    """IAT and target-draw-equivalent ESS for an ensemble observable."""

    integrated_autocorrelation_time: float
    effective_sample_size: float
    n_iterations: int
    n_walkers: int


def _observable_matrix(values: ArrayLike) -> Array:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("values must have shape (iterations >= 2, walkers >= 2)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite")
    return matrix


def ensemble_effective_sample_size(values: ArrayLike) -> EnsembleEfficiency:
    """Compute ESS from the ensemble-average time series.

    If equilibrium walkers are independent, ``Var(mean_j f(X_j)) = Var(f)/L``;
    therefore the target-draw-equivalent ESS is ``L * N / IAT``.
    """

    matrix = _observable_matrix(values)
    ensemble_means = np.mean(matrix, axis=1)
    iat = empirical_integrated_autocorrelation_time(ensemble_means)
    effective = matrix.shape[0] * matrix.shape[1] / iat
    return EnsembleEfficiency(
        integrated_autocorrelation_time=iat,
        effective_sample_size=float(effective),
        n_iterations=int(matrix.shape[0]),
        n_walkers=int(matrix.shape[1]),
    )


def mean_cross_walker_correlation(values: ArrayLike) -> float:
    """Mean off-diagonal Pearson correlation between walker time series."""

    matrix = _observable_matrix(values)
    standard_deviations = np.std(matrix, axis=0)
    valid = standard_deviations > 0.0
    if np.count_nonzero(valid) < 2:
        return float("nan")
    correlations = np.corrcoef(matrix[:, valid], rowvar=False)
    upper = correlations[np.triu_indices(correlations.shape[0], k=1)]
    return float(np.mean(upper))
