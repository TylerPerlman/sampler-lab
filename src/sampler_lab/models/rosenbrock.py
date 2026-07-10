"""Rosenbrock density with exact conditional sampling structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RosenbrockTarget:
    r"""Two-dimensional Rosenbrock probability density.

    The unnormalized log density is

    .. math::

        \log \pi(x, y) = -\frac{a(y-x^2)^2 + (b-x)^2}{s}.

    The defaults ``a=100``, ``b=1``, and ``s=20`` produce a narrow curved target with an
    exact hierarchical representation

    ``X ~ N(b, s/2)`` and ``Y | X ~ N(X**2, s/(2a))``.
    """

    curvature: float = 100.0
    location: float = 1.0
    scale: float = 20.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.curvature) or self.curvature <= 0.0:
            raise ValueError("curvature must be positive and finite")
        if not np.isfinite(self.location):
            raise ValueError("location must be finite")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be positive and finite")

    @property
    def dimension(self) -> int:
        return 2

    @property
    def mode(self) -> Array:
        return np.array([self.location, self.location**2], dtype=np.float64)

    @property
    def x_variance(self) -> float:
        return self.scale / 2.0

    @property
    def conditional_y_variance(self) -> float:
        return self.scale / (2.0 * self.curvature)

    def _point(self, value: ArrayLike) -> Array:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("Rosenbrock states must be finite vectors of length two")
        return point

    def log_prob(self, x: Array) -> float:
        point = self._point(x)
        first, second = float(point[0]), float(point[1])
        residual = second - first * first
        energy = self.curvature * residual * residual + (self.location - first) ** 2
        return float(-energy / self.scale)

    def grad_log_prob(self, x: Array) -> Array:
        point = self._point(x)
        first, second = float(point[0]), float(point[1])
        residual = second - first * first
        gradient_energy = np.array(
            [
                -4.0 * self.curvature * first * residual + 2.0 * (first - self.location),
                2.0 * self.curvature * residual,
            ],
            dtype=np.float64,
        )
        return -gradient_energy / self.scale

    def hessian_log_prob(self, x: Array) -> Array:
        point = self._point(x)
        first, second = float(point[0]), float(point[1])
        hessian_energy = np.array(
            [
                [
                    12.0 * self.curvature * first * first - 4.0 * self.curvature * second + 2.0,
                    -4.0 * self.curvature * first,
                ],
                [-4.0 * self.curvature * first, 2.0 * self.curvature],
            ],
            dtype=np.float64,
        )
        return -hessian_energy / self.scale

    def sample_exact(self, rng: np.random.Generator, size: int) -> Array:
        """Draw independent exact samples from the hierarchical representation."""

        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("size must be an integer")
        if size < 0:
            raise ValueError("size must be nonnegative")
        first = rng.normal(self.location, np.sqrt(self.x_variance), size=size)
        second = first * first + rng.normal(
            0.0,
            np.sqrt(self.conditional_y_variance),
            size=size,
        )
        return np.column_stack((first, second)).astype(np.float64)

    def exact_mean(self) -> Array:
        """Return the exact target mean."""

        return np.array(
            [self.location, self.location**2 + self.x_variance],
            dtype=np.float64,
        )

    def exact_covariance(self) -> Array:
        """Return the exact covariance obtained from Gaussian moments."""

        variance_x = self.x_variance
        covariance_xy = 2.0 * self.location * variance_x
        variance_y = (
            2.0 * variance_x * variance_x
            + 4.0 * self.location**2 * variance_x
            + self.conditional_y_variance
        )
        return np.array(
            [[variance_x, covariance_xy], [covariance_xy, variance_y]],
            dtype=np.float64,
        )
