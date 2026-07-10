"""Analytically tractable Gaussian targets used for validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.linalg import as_positive_definite

Array = NDArray[np.float64]


@dataclass(slots=True)
class GaussianTarget:
    """Multivariate Gaussian target exposing log density, gradient, and Hessian."""

    mean: ArrayLike
    covariance: ArrayLike
    _mean: Array = field(init=False, repr=False)
    _covariance: Array = field(init=False, repr=False)
    _precision: Array = field(init=False, repr=False)
    _log_normalizer: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mean = np.asarray(self.mean, dtype=np.float64)
        if self._mean.ndim != 1:
            raise ValueError("mean must be one-dimensional")
        self._covariance = as_positive_definite(self.covariance)
        if self._covariance.shape != (self._mean.size, self._mean.size):
            raise ValueError("covariance shape must match mean dimension")
        identity = np.eye(self._mean.size)
        self._precision = np.asarray(np.linalg.solve(self._covariance, identity), dtype=np.float64)
        sign, log_determinant = np.linalg.slogdet(self._covariance)
        if sign <= 0.0:
            raise ValueError("covariance must have positive determinant")
        self._log_normalizer = 0.5 * (self._mean.size * np.log(2.0 * np.pi) + log_determinant)

    @property
    def dimension(self) -> int:
        """State-space dimension."""

        return int(self._mean.size)

    @property
    def mean_vector(self) -> Array:
        """Return a detached copy of the Gaussian mean."""

        return self._mean.copy()

    @property
    def covariance_matrix(self) -> Array:
        """Return a detached copy of the covariance matrix."""

        return self._covariance.copy()

    @property
    def precision_matrix(self) -> Array:
        """Return a detached copy of the precision matrix."""

        return self._precision.copy()

    def sample(self, rng: np.random.Generator, size: int) -> Array:
        """Draw exact IID samples from the Gaussian target."""

        if size <= 0:
            raise ValueError("size must be positive")
        return np.asarray(
            rng.multivariate_normal(self._mean, self._covariance, size=size),
            dtype=np.float64,
        )

    def log_prob(self, x: Array) -> float:
        """Evaluate the normalized Gaussian log density."""

        point = np.asarray(x, dtype=np.float64)
        if point.shape != self._mean.shape:
            raise ValueError("x has the wrong shape")
        displacement = point - self._mean
        return float(-0.5 * displacement @ self._precision @ displacement - self._log_normalizer)

    def grad_log_prob(self, x: Array) -> Array:
        """Evaluate the gradient of the log density."""

        point = np.asarray(x, dtype=np.float64)
        if point.shape != self._mean.shape:
            raise ValueError("x has the wrong shape")
        return -self._precision @ (point - self._mean)

    def hessian_log_prob(self, x: Array) -> Array:
        """Return the constant Hessian of the Gaussian log density."""

        point = np.asarray(x, dtype=np.float64)
        if point.shape != self._mean.shape:
            raise ValueError("x has the wrong shape")
        return -self._precision.copy()
