"""Gaussian conditioning and exact conditional-distribution utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.geometry.affine import AffineMap, AffineTransformedTarget
from sampler_lab.models.gaussian import GaussianTarget

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GaussianConditional:
    """Conditional Gaussian law for unobserved coordinates."""

    target: GaussianTarget
    remaining_indices: IntArray
    observed_indices: IntArray


def matrix_condition_number(matrix: ArrayLike) -> float:
    """Spectral condition number of a symmetric positive-definite matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(values, values.T, atol=1e-12, rtol=0.0):
        raise ValueError("matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(values)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("matrix must be positive definite")
    return float(eigenvalues[-1] / eigenvalues[0])


def gaussian_whitening_map(target: GaussianTarget) -> AffineMap:
    """Return ``y = L^{-1}(x-mu)`` where ``C = L L^T``."""

    covariance = target.covariance_matrix
    cholesky = np.linalg.cholesky(covariance)
    whitening = np.asarray(np.linalg.solve(cholesky, np.eye(target.dimension)), dtype=np.float64)
    shift = -whitening @ target.mean_vector
    return AffineMap(whitening, shift)


def whiten_gaussian_target(target: GaussianTarget) -> AffineTransformedTarget:
    """Push a Gaussian target to standard-normal coordinates."""

    return AffineTransformedTarget(target, gaussian_whitening_map(target))


def gaussian_conditional(
    target: GaussianTarget,
    observed_indices: ArrayLike,
    observed_values: ArrayLike,
) -> GaussianConditional:
    """Return the exact conditional law of all coordinates not observed."""

    raw_indices = np.asarray(observed_indices)
    if raw_indices.ndim != 1 or raw_indices.size == 0:
        raise ValueError("observed_indices must be a nonempty one-dimensional array")
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise TypeError("observed_indices must contain integers")
    indices = np.asarray(raw_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= target.dimension):
        raise ValueError("observed index is out of range")
    if np.unique(indices).size != indices.size:
        raise ValueError("observed indices must be unique")
    observations = np.asarray(observed_values, dtype=np.float64)
    if observations.shape != (indices.size,) or not np.all(np.isfinite(observations)):
        raise ValueError("observed_values must match observed_indices and be finite")
    remaining = np.asarray(
        [index for index in range(target.dimension) if index not in set(indices.tolist())],
        dtype=np.int64,
    )
    if remaining.size == 0:
        raise ValueError("at least one coordinate must remain unobserved")

    mean = target.mean_vector
    covariance = target.covariance_matrix
    mean_observed = mean[indices]
    mean_remaining = mean[remaining]
    cov_oo = covariance[np.ix_(indices, indices)]
    cov_ro = covariance[np.ix_(remaining, indices)]
    cov_rr = covariance[np.ix_(remaining, remaining)]
    solved_displacement = np.linalg.solve(cov_oo, observations - mean_observed)
    conditional_mean = mean_remaining + cov_ro @ solved_displacement
    conditional_covariance = cov_rr - cov_ro @ np.linalg.solve(cov_oo, cov_ro.T)
    conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
    return GaussianConditional(
        target=GaussianTarget(conditional_mean, conditional_covariance),
        remaining_indices=remaining,
        observed_indices=indices,
    )
