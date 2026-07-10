"""Affine maps, transformed targets, and trajectory coordinate changes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.protocols import TwiceDifferentiableLogDensity

Array = NDArray[np.float64]


@dataclass(slots=True)
class AffineMap:
    """Invertible affine map ``y = A x + b``."""

    matrix: ArrayLike
    shift: ArrayLike
    _matrix: Array = field(init=False, repr=False)
    _shift: Array = field(init=False, repr=False)
    _inverse: Array = field(init=False, repr=False)
    _log_abs_determinant: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        shift = np.asarray(self.shift, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("matrix must be nonempty and square")
        if shift.shape != (matrix.shape[0],):
            raise ValueError("shift dimension must match the affine matrix")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(shift)):
            raise ValueError("affine map parameters must be finite")
        sign, log_abs_determinant = np.linalg.slogdet(matrix)
        if sign == 0.0:
            raise ValueError("affine matrix must be invertible")
        self._matrix = np.array(matrix, dtype=np.float64, copy=True)
        self._shift = np.array(shift, dtype=np.float64, copy=True)
        self._inverse = np.asarray(
            np.linalg.solve(matrix, np.eye(matrix.shape[0])), dtype=np.float64
        )
        self._log_abs_determinant = float(log_abs_determinant)

    @property
    def dimension(self) -> int:
        return int(self._shift.size)

    @property
    def linear_matrix(self) -> Array:
        return self._matrix.copy()

    @property
    def inverse_matrix(self) -> Array:
        return self._inverse.copy()

    @property
    def shift_vector(self) -> Array:
        return self._shift.copy()

    @property
    def log_abs_determinant(self) -> float:
        return self._log_abs_determinant

    def forward(self, x: ArrayLike) -> Array:
        """Apply ``y = A x + b`` to one point or a batch along the last axis."""

        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != self.dimension:
            raise ValueError("input's final axis must match the affine dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("input must be finite")
        return np.asarray(values @ self._matrix.T + self._shift, dtype=np.float64)

    def inverse(self, y: ArrayLike) -> Array:
        """Apply ``x = A^{-1}(y-b)`` to one point or a batch."""

        values = np.asarray(y, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != self.dimension:
            raise ValueError("input's final axis must match the affine dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("input must be finite")
        return np.asarray((values - self._shift) @ self._inverse.T, dtype=np.float64)

    def compose(self, inner: AffineMap) -> AffineMap:
        """Return ``self(inner(x))``."""

        if inner.dimension != self.dimension:
            raise ValueError("affine-map dimensions must match")
        return AffineMap(
            self._matrix @ inner._matrix,
            self._matrix @ inner._shift + self._shift,
        )

    def inverse_map(self) -> AffineMap:
        """Return the inverse affine map."""

        return AffineMap(self._inverse, -self._inverse @ self._shift)

    def transform_covariance(self, covariance: ArrayLike) -> Array:
        """Push a covariance matrix forward as ``A C A^T``."""

        matrix = np.asarray(covariance, dtype=np.float64)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("covariance shape must match the affine dimension")
        return np.asarray(self._matrix @ matrix @ self._matrix.T, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AffineTransformedTarget:
    """Pushforward of a differentiable target through an affine map."""

    base_target: TwiceDifferentiableLogDensity
    affine_map: AffineMap

    def log_prob(self, y: Array) -> float:
        x = self.affine_map.inverse(y)
        return float(self.base_target.log_prob(x) - self.affine_map.log_abs_determinant)

    def grad_log_prob(self, y: Array) -> Array:
        x = self.affine_map.inverse(y)
        gradient = np.asarray(self.base_target.grad_log_prob(x), dtype=np.float64)
        if gradient.shape != (self.affine_map.dimension,):
            raise ValueError("base-target gradient has the wrong shape")
        return np.asarray(self.affine_map.inverse_matrix.T @ gradient, dtype=np.float64)

    def hessian_log_prob(self, y: Array) -> Array:
        x = self.affine_map.inverse(y)
        hessian = np.asarray(self.base_target.hessian_log_prob(x), dtype=np.float64)
        dimension = self.affine_map.dimension
        if hessian.shape != (dimension, dimension):
            raise ValueError("base-target Hessian has the wrong shape")
        inverse = self.affine_map.inverse_matrix
        return np.asarray(inverse.T @ hessian @ inverse, dtype=np.float64)


def affine_equivariance_error(
    reference: ArrayLike, transformed: ArrayLike, mapping: AffineMap
) -> float:
    """Maximum Euclidean discrepancy between ``mapping(reference)`` and transformed."""

    expected = mapping.forward(reference)
    observed = np.asarray(transformed, dtype=np.float64)
    if observed.shape != expected.shape:
        raise ValueError("reference and transformed arrays have incompatible shapes")
    if observed.ndim == 1:
        return float(np.linalg.norm(observed - expected))
    flattened = (observed - expected).reshape(-1, mapping.dimension)
    return float(np.max(np.linalg.norm(flattened, axis=1), initial=0.0))
