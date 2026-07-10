"""Covariance shrinkage and spectral regularization for adaptive proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
ShrinkageTarget = Literal["diagonal", "isotropic", "identity"]


@dataclass(frozen=True, slots=True)
class CovarianceRegularizationResult:
    """Regularized covariance and its spectral diagnostics."""

    matrix: Array
    shrunk_matrix: Array
    raw_eigenvalues: Array
    regularized_eigenvalues: Array
    shrinkage: float
    correction_frobenius_norm: float


def regularize_covariance(
    covariance: ArrayLike,
    *,
    shrinkage: float = 0.05,
    target: ShrinkageTarget = "diagonal",
    eigenvalue_floor: float = 1e-6,
    eigenvalue_ceiling: float | None = None,
) -> CovarianceRegularizationResult:
    """Shrink a symmetric covariance and clip its eigenvalues.

    The shrinkage operation occurs before spectral clipping. ``identity`` uses a
    unit target, while ``isotropic`` preserves the average marginal variance.
    """

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
        raise ValueError("covariance must be symmetric")
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive and finite")
    if eigenvalue_ceiling is not None and (
        not np.isfinite(eigenvalue_ceiling) or eigenvalue_ceiling < eigenvalue_floor
    ):
        raise ValueError("eigenvalue_ceiling must be finite and at least the floor")

    symmetric = np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)
    if target == "diagonal":
        target_matrix = np.diag(np.diag(symmetric))
    elif target == "isotropic":
        mean_variance = float(np.trace(symmetric) / symmetric.shape[0])
        target_matrix = mean_variance * np.eye(symmetric.shape[0], dtype=np.float64)
    elif target == "identity":
        target_matrix = np.eye(symmetric.shape[0], dtype=np.float64)
    else:
        raise ValueError(f"unknown shrinkage target: {target!r}")

    shrunk = np.asarray(
        (1.0 - shrinkage) * symmetric + shrinkage * target_matrix,
        dtype=np.float64,
    )
    raw_eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    regularized_eigenvalues = np.maximum(raw_eigenvalues, eigenvalue_floor)
    if eigenvalue_ceiling is not None:
        regularized_eigenvalues = np.minimum(regularized_eigenvalues, eigenvalue_ceiling)
    regularized = np.asarray(
        (eigenvectors * regularized_eigenvalues) @ eigenvectors.T,
        dtype=np.float64,
    )
    regularized = np.asarray(0.5 * (regularized + regularized.T), dtype=np.float64)
    return CovarianceRegularizationResult(
        matrix=regularized,
        shrunk_matrix=shrunk,
        raw_eigenvalues=np.asarray(raw_eigenvalues, dtype=np.float64),
        regularized_eigenvalues=np.asarray(regularized_eigenvalues, dtype=np.float64),
        shrinkage=float(shrinkage),
        correction_frobenius_norm=float(np.linalg.norm(regularized - symmetric, ord="fro")),
    )
