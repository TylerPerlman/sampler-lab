"""Normalized Gaussian-mixture targets with analytic derivatives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.numerics import logsumexp, validate_size

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class GaussianMixtureTarget:
    """Finite full-covariance Gaussian mixture."""

    weights: ArrayLike
    means: ArrayLike
    covariances: ArrayLike
    _weights: Array = field(init=False, repr=False)
    _means: Array = field(init=False, repr=False)
    _covariances: Array = field(init=False, repr=False)
    _precisions: Array = field(init=False, repr=False)
    _cholesky: Array = field(init=False, repr=False)
    _log_component_normalizers: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        covariances = np.asarray(self.covariances, dtype=np.float64)
        if weights.ndim != 1 or weights.size < 2:
            raise ValueError("weights must contain at least two components")
        if means.ndim != 2 or means.shape[0] != weights.size or means.shape[1] == 0:
            raise ValueError("means must have shape (components, dimension)")
        if covariances.shape != (weights.size, means.shape[1], means.shape[1]):
            raise ValueError("covariances must have shape (components, dimension, dimension)")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("mixture weights must be positive and finite")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(covariances)):
            raise ValueError("mixture parameters must be finite")
        weights = weights / np.sum(weights)
        cholesky = np.empty_like(covariances)
        precisions = np.empty_like(covariances)
        log_normalizers = np.empty(weights.size, dtype=np.float64)
        identity = np.eye(means.shape[1], dtype=np.float64)
        for index, covariance in enumerate(covariances):
            if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
                raise ValueError("component covariances must be symmetric")
            factor = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
            cholesky[index] = factor
            precisions[index] = np.linalg.solve(covariance, identity)
            log_determinant = float(2.0 * np.sum(np.log(np.diag(factor))))
            log_normalizers[index] = 0.5 * (means.shape[1] * np.log(2.0 * np.pi) + log_determinant)
        self._weights = np.asarray(weights, dtype=np.float64)
        self._means = np.asarray(means, dtype=np.float64)
        self._covariances = np.asarray(covariances, dtype=np.float64)
        self._precisions = precisions
        self._cholesky = cholesky
        self._log_component_normalizers = log_normalizers

    @property
    def n_components(self) -> int:
        return int(self._weights.size)

    @property
    def dimension(self) -> int:
        return int(self._means.shape[1])

    @property
    def mixture_weights(self) -> Array:
        return self._weights.copy()

    @property
    def component_means(self) -> Array:
        return self._means.copy()

    @property
    def component_covariances(self) -> Array:
        return self._covariances.copy()

    @property
    def mean_vector(self) -> Array:
        return np.asarray(self._weights @ self._means, dtype=np.float64)

    @property
    def covariance_matrix(self) -> Array:
        mean = self.mean_vector
        second = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for weight, component_mean, covariance in zip(
            self._weights,
            self._means,
            self._covariances,
            strict=True,
        ):
            second += weight * (covariance + np.outer(component_mean, component_mean))
        return np.asarray(second - np.outer(mean, mean), dtype=np.float64)

    def component_log_probs(self, x: ArrayLike) -> Array:
        point = np.asarray(x, dtype=np.float64)
        if point.shape != (self.dimension,) or not np.all(np.isfinite(point)):
            raise ValueError("x must be a finite vector matching the target dimension")
        displacements = point - self._means
        quadratic = np.einsum(
            "ki,kij,kj->k",
            displacements,
            self._precisions,
            displacements,
        )
        return np.asarray(
            np.log(self._weights) - 0.5 * quadratic - self._log_component_normalizers,
            dtype=np.float64,
        )

    def responsibilities(self, x: ArrayLike) -> Array:
        terms = self.component_log_probs(x)
        normalizer = float(logsumexp(terms))
        return np.exp(terms - normalizer).astype(np.float64)

    def log_prob(self, x: Array) -> float:
        return float(logsumexp(self.component_log_probs(x)))

    def grad_log_prob(self, x: Array) -> Array:
        point = np.asarray(x, dtype=np.float64)
        responsibilities = self.responsibilities(point)
        component_gradients = -np.einsum(
            "kij,kj->ki",
            self._precisions,
            point - self._means,
        )
        return np.asarray(responsibilities @ component_gradients, dtype=np.float64)

    def hessian_log_prob(self, x: Array) -> Array:
        point = np.asarray(x, dtype=np.float64)
        responsibilities = self.responsibilities(point)
        gradients = -np.einsum(
            "kij,kj->ki",
            self._precisions,
            point - self._means,
        )
        total_gradient = responsibilities @ gradients
        hessian = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for responsibility, precision, gradient in zip(
            responsibilities,
            self._precisions,
            gradients,
            strict=True,
        ):
            hessian += responsibility * (-precision + np.outer(gradient, gradient))
        hessian -= np.outer(total_gradient, total_gradient)
        return np.asarray(0.5 * (hessian + hessian.T), dtype=np.float64)

    def sample_with_labels(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> tuple[Array, IntArray]:
        validate_size(size)
        labels = np.asarray(
            rng.choice(self.n_components, size=size, p=self._weights), dtype=np.int64
        )
        samples = np.empty((size, self.dimension), dtype=np.float64)
        for component in range(self.n_components):
            mask = labels == component
            count = int(np.count_nonzero(mask))
            if count:
                samples[mask] = (
                    self._means[component]
                    + rng.normal(size=(count, self.dimension)) @ self._cholesky[component].T
                )
        return samples, labels

    def sample(self, rng: np.random.Generator, size: int) -> Array:
        return self.sample_with_labels(rng, size)[0]

    def mode_labels(self, samples: ArrayLike) -> IntArray:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        return np.asarray(
            [int(np.argmax(self.component_log_probs(sample))) for sample in values],
            dtype=np.int64,
        )
