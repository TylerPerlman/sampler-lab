"""Exactly sampleable bimodal anisotropic funnel benchmark target."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.numerics import logsumexp, validate_size
from sampler_lab.models.funnel import FunnelTarget, seeded_orthogonal_matrix

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class BimodalFunnelTarget:
    """Balanced mixture of two translated, differently rotated funnels."""

    dimension: int = 10
    separation: float = 12.0
    sigma_v: float = 3.0
    anisotropy_ratio: float = 20.0
    seed: int = 2022
    _components: tuple[FunnelTarget, FunnelTarget] = field(init=False, repr=False)
    _direction: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise ValueError("dimension must be at least two")
        if not np.isfinite(self.separation) or self.separation <= 0.0:
            raise ValueError("separation must be positive and finite")
        if not np.isfinite(self.anisotropy_ratio) or self.anisotropy_ratio < 1.0:
            raise ValueError("anisotropy_ratio must be finite and at least one")
        scales = np.geomspace(1.0, self.anisotropy_ratio, self.dimension - 1)
        first_rotation = seeded_orthogonal_matrix(self.dimension, self.seed)
        second_rotation = seeded_orthogonal_matrix(self.dimension, self.seed + 1)
        direction = first_rotation[:, 0]
        first_location = -0.5 * self.separation * direction
        second_location = 0.5 * self.separation * direction
        self._direction = np.asarray(direction, dtype=np.float64)
        self._components = (
            FunnelTarget(
                self.dimension,
                sigma_v=self.sigma_v,
                scales=scales,
                rotation=first_rotation,
                location=first_location,
            ),
            FunnelTarget(
                self.dimension,
                sigma_v=self.sigma_v,
                scales=scales[::-1],
                rotation=second_rotation,
                location=second_location,
            ),
        )

    @property
    def components(self) -> tuple[FunnelTarget, FunnelTarget]:
        return self._components

    @property
    def separation_direction(self) -> Array:
        return self._direction.copy()

    @property
    def mean_vector(self) -> Array:
        return 0.5 * (self._components[0].mean_vector + self._components[1].mean_vector)

    @property
    def covariance_matrix(self) -> Array:
        means = [component.mean_vector for component in self._components]
        second = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for component, mean in zip(self._components, means, strict=True):
            second += 0.5 * (component.covariance_matrix + np.outer(mean, mean))
        overall = self.mean_vector
        return np.asarray(second - np.outer(overall, overall), dtype=np.float64)

    def component_log_probs(self, x: ArrayLike) -> Array:
        point = np.asarray(x, dtype=np.float64)
        if point.shape != (self.dimension,) or not np.all(np.isfinite(point)):
            raise ValueError("x must be a finite vector matching dimension")
        return np.asarray(
            [np.log(0.5) + component.log_prob(point) for component in self._components],
            dtype=np.float64,
        )

    def responsibilities(self, x: ArrayLike) -> Array:
        terms = self.component_log_probs(x)
        return np.exp(terms - float(logsumexp(terms))).astype(np.float64)

    def log_prob(self, x: Array) -> float:
        return float(logsumexp(self.component_log_probs(x)))

    def grad_log_prob(self, x: Array) -> Array:
        point = np.asarray(x, dtype=np.float64)
        responsibilities = self.responsibilities(point)
        gradients = np.asarray(
            [component.grad_log_prob(point) for component in self._components],
            dtype=np.float64,
        )
        return np.asarray(responsibilities @ gradients, dtype=np.float64)

    def hessian_log_prob(self, x: Array) -> Array:
        point = np.asarray(x, dtype=np.float64)
        responsibilities = self.responsibilities(point)
        gradients = np.asarray(
            [component.grad_log_prob(point) for component in self._components],
            dtype=np.float64,
        )
        total_gradient = responsibilities @ gradients
        result = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for responsibility, component, gradient in zip(
            responsibilities,
            self._components,
            gradients,
            strict=True,
        ):
            result += responsibility * (
                component.hessian_log_prob(point) + np.outer(gradient, gradient)
            )
        result -= np.outer(total_gradient, total_gradient)
        return np.asarray(0.5 * (result + result.T), dtype=np.float64)

    def sample_with_labels(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> tuple[Array, IntArray]:
        validate_size(size)
        labels = np.asarray(rng.integers(0, 2, size=size), dtype=np.int64)
        samples = np.empty((size, self.dimension), dtype=np.float64)
        for component_index, component in enumerate(self._components):
            mask = labels == component_index
            count = int(np.count_nonzero(mask))
            if count:
                samples[mask] = component.sample(rng, count)
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
