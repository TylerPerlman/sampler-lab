"""Hessian approximation and positive-definite repair utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.protocols import TwiceDifferentiableLogDensity

Array = NDArray[np.float64]
RepairMethod = Literal["raise", "clip", "absolute"]


@dataclass(frozen=True, slots=True)
class PositiveDefiniteRepair:
    """A repaired positive-definite matrix and its spectral diagnostics."""

    matrix: Array
    original_eigenvalues: Array
    repaired_eigenvalues: Array
    method: RepairMethod
    correction_frobenius_norm: float


def _as_symmetric(matrix: ArrayLike, *, symmetry_tolerance: float = 1e-10) -> Array:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("matrix must be nonempty and square")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    if not np.allclose(values, values.T, atol=symmetry_tolerance, rtol=0.0):
        raise ValueError("matrix must be symmetric")
    return np.asarray(0.5 * (values + values.T), dtype=np.float64)


def repair_positive_definite(
    matrix: ArrayLike,
    *,
    method: RepairMethod = "clip",
    minimum_eigenvalue: float = 1e-6,
) -> PositiveDefiniteRepair:
    """Repair a symmetric matrix by eigenvalue rejection, clipping, or absolute value."""

    if not np.isfinite(minimum_eigenvalue) or minimum_eigenvalue <= 0.0:
        raise ValueError("minimum_eigenvalue must be positive and finite")
    if method not in {"raise", "clip", "absolute"}:
        raise ValueError("unknown Hessian repair method")
    symmetric = _as_symmetric(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if method == "raise":
        if np.any(eigenvalues < minimum_eigenvalue):
            raise np.linalg.LinAlgError("matrix is not sufficiently positive definite")
        repaired_eigenvalues = eigenvalues
    elif method == "clip":
        repaired_eigenvalues = np.maximum(eigenvalues, minimum_eigenvalue)
    else:
        repaired_eigenvalues = np.maximum(np.abs(eigenvalues), minimum_eigenvalue)
    repaired = (eigenvectors * repaired_eigenvalues) @ eigenvectors.T
    repaired = np.asarray(0.5 * (repaired + repaired.T), dtype=np.float64)
    return PositiveDefiniteRepair(
        matrix=repaired,
        original_eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        repaired_eigenvalues=np.asarray(repaired_eigenvalues, dtype=np.float64),
        method=method,
        correction_frobenius_norm=float(np.linalg.norm(repaired - symmetric, ord="fro")),
    )


def finite_difference_hessian_from_gradient(
    gradient: Callable[[Array], ArrayLike],
    state: ArrayLike,
    *,
    relative_step: float = 1e-5,
) -> Array:
    """Approximate a Hessian by centered differences of a gradient."""

    point = np.asarray(state, dtype=np.float64)
    if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("state must be a nonempty finite vector")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be positive and finite")
    dimension = point.size
    hessian = np.empty((dimension, dimension), dtype=np.float64)
    for coordinate in range(dimension):
        step = relative_step * max(1.0, abs(float(point[coordinate])))
        offset = np.zeros(dimension, dtype=np.float64)
        offset[coordinate] = step
        plus = np.asarray(gradient(point + offset), dtype=np.float64)
        minus = np.asarray(gradient(point - offset), dtype=np.float64)
        if plus.shape != point.shape or minus.shape != point.shape:
            raise ValueError("gradient output must match the state shape")
        if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
            raise ValueError("gradient output must be finite")
        hessian[:, coordinate] = (plus - minus) / (2.0 * step)
    return np.asarray(0.5 * (hessian + hessian.T), dtype=np.float64)


def negative_log_hessian(
    target: TwiceDifferentiableLogDensity,
    state: ArrayLike,
    *,
    repair_method: RepairMethod = "raise",
    minimum_eigenvalue: float = 1e-6,
) -> PositiveDefiniteRepair:
    """Evaluate and repair ``-D^2 log pi`` for use as local precision."""

    point = np.asarray(state, dtype=np.float64)
    if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("state must be a nonempty finite vector")
    hessian = np.asarray(target.hessian_log_prob(point), dtype=np.float64)
    if hessian.shape != (point.size, point.size):
        raise ValueError("target Hessian must match the state dimension")
    return repair_positive_definite(
        -hessian,
        method=repair_method,
        minimum_eigenvalue=minimum_eigenvalue,
    )
