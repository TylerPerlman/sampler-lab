"""Linear-algebra validation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def as_positive_definite(matrix: ArrayLike, *, symmetry_tolerance: float = 1e-12) -> Array:
    """Validate and return a symmetric positive-definite matrix.

    A Cholesky factorization is used as the definitive positive-definiteness test.
    """

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(array, array.T, atol=symmetry_tolerance, rtol=0.0):
        raise ValueError("matrix must be symmetric")
    np.linalg.cholesky(array)
    return array
