"""Rotated anisotropic Gaussian funnel targets with exact direct sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.numerics import validate_size

Array = NDArray[np.float64]


def seeded_orthogonal_matrix(dimension: int, seed: int) -> Array:
    """Return a deterministic proper orthogonal matrix from a seeded QR factorization."""

    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return np.asarray(q, dtype=np.float64)


@dataclass(slots=True)
class FunnelTarget:
    r"""Neal-style funnel with coordinate anisotropy and orthogonal rotation.

    In latent coordinates ``y=(v,z)``,

    ``v ~ N(0, sigma_v^2)`` and ``z_i | v ~ N(0, s_i^2 exp(v))``.
    Observed coordinates satisfy ``x = location + rotation @ y``.
    """

    dimension: int
    sigma_v: float = 3.0
    scales: ArrayLike | None = None
    rotation: ArrayLike | None = None
    location: ArrayLike | None = None
    _scales: Array = field(init=False, repr=False)
    _rotation: Array = field(init=False, repr=False)
    _location: Array = field(init=False, repr=False)
    _log_normalizer: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension must be an integer")
        if self.dimension < 2:
            raise ValueError("funnel dimension must be at least two")
        if not np.isfinite(self.sigma_v) or self.sigma_v <= 0.0:
            raise ValueError("sigma_v must be positive and finite")
        if self.scales is None:
            scales = np.ones(self.dimension - 1, dtype=np.float64)
        else:
            scales = np.asarray(self.scales, dtype=np.float64)
        if scales.shape != (self.dimension - 1,) or np.any(scales <= 0.0):
            raise ValueError("scales must be positive and have length dimension - 1")
        if not np.all(np.isfinite(scales)):
            raise ValueError("scales must be finite")
        if self.rotation is None:
            rotation = np.eye(self.dimension, dtype=np.float64)
        else:
            rotation = np.asarray(self.rotation, dtype=np.float64)
        if rotation.shape != (self.dimension, self.dimension) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite square matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(self.dimension), atol=1e-10, rtol=0.0):
            raise ValueError("rotation must be orthogonal")
        if self.location is None:
            location = np.zeros(self.dimension, dtype=np.float64)
        else:
            location = np.asarray(self.location, dtype=np.float64)
        if location.shape != (self.dimension,) or not np.all(np.isfinite(location)):
            raise ValueError("location must be a finite vector matching dimension")
        self._scales = np.array(scales, copy=True)
        self._rotation = np.array(rotation, copy=True)
        self._location = np.array(location, copy=True)
        self._log_normalizer = float(
            0.5 * self.dimension * np.log(2.0 * np.pi)
            + np.log(self.sigma_v)
            + np.sum(np.log(self._scales))
        )

    @property
    def scale_factors(self) -> Array:
        return self._scales.copy()

    @property
    def rotation_matrix(self) -> Array:
        return self._rotation.copy()

    @property
    def location_vector(self) -> Array:
        return self._location.copy()

    @property
    def mean_vector(self) -> Array:
        return self._location.copy()

    @property
    def covariance_matrix(self) -> Array:
        latent_variances = np.concatenate(
            (
                np.asarray([self.sigma_v * self.sigma_v]),
                self._scales * self._scales * np.exp(0.5 * self.sigma_v * self.sigma_v),
            )
        )
        return np.asarray(
            self._rotation @ np.diag(latent_variances) @ self._rotation.T,
            dtype=np.float64,
        )

    def to_latent(self, x: ArrayLike) -> Array:
        point = np.asarray(x, dtype=np.float64)
        if point.shape != (self.dimension,) or not np.all(np.isfinite(point)):
            raise ValueError("x must be a finite vector matching dimension")
        return np.asarray(self._rotation.T @ (point - self._location), dtype=np.float64)

    def from_latent(self, latent: ArrayLike) -> Array:
        value = np.asarray(latent, dtype=np.float64)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("latent must be a finite vector matching dimension")
        return np.asarray(self._location + self._rotation @ value, dtype=np.float64)

    def to_noncentered(self, x: ArrayLike) -> Array:
        latent = self.to_latent(x)
        v = latent[0]
        standardized = latent[1:] / (self._scales * np.exp(0.5 * v))
        return np.asarray(np.concatenate(([v], standardized)), dtype=np.float64)

    def from_noncentered(self, noncentered: ArrayLike) -> Array:
        value = np.asarray(noncentered, dtype=np.float64)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("noncentered coordinates must match dimension")
        v = value[0]
        latent = np.concatenate(([v], self._scales * np.exp(0.5 * v) * value[1:]))
        return self.from_latent(latent)

    def _latent_terms(self, x: ArrayLike) -> tuple[Array, float, Array, Array]:
        latent = self.to_latent(x)
        v = float(latent[0])
        z = latent[1:]
        with np.errstate(over="ignore"):
            inverse_variances = np.exp(-v) / (self._scales * self._scales)
        scaled_squares = z * z * inverse_variances
        return latent, v, inverse_variances, scaled_squares

    def log_prob(self, x: Array) -> float:
        _, v, _, scaled_squares = self._latent_terms(x)
        return float(
            -0.5 * (v / self.sigma_v) ** 2
            - 0.5 * np.sum(scaled_squares)
            - 0.5 * (self.dimension - 1) * v
            - self._log_normalizer
        )

    def grad_log_prob(self, x: Array) -> Array:
        latent, v, inverse_variances, scaled_squares = self._latent_terms(x)
        gradient_latent = np.empty(self.dimension, dtype=np.float64)
        gradient_latent[0] = -v / (self.sigma_v * self.sigma_v) + 0.5 * np.sum(scaled_squares - 1.0)
        gradient_latent[1:] = -latent[1:] * inverse_variances
        return np.asarray(self._rotation @ gradient_latent, dtype=np.float64)

    def hessian_log_prob(self, x: Array) -> Array:
        latent, _, inverse_variances, scaled_squares = self._latent_terms(x)
        hessian_latent = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        hessian_latent[0, 0] = -1.0 / (self.sigma_v * self.sigma_v) - 0.5 * np.sum(scaled_squares)
        cross = latent[1:] * inverse_variances
        hessian_latent[0, 1:] = cross
        hessian_latent[1:, 0] = cross
        hessian_latent[1:, 1:] = np.diag(-inverse_variances)
        transformed = self._rotation @ hessian_latent @ self._rotation.T
        return np.asarray(0.5 * (transformed + transformed.T), dtype=np.float64)

    def sample(self, rng: np.random.Generator, size: int) -> Array:
        validate_size(size)
        v = rng.normal(scale=self.sigma_v, size=size)
        z = rng.normal(size=(size, self.dimension - 1))
        z *= self._scales[None, :] * np.exp(0.5 * v)[:, None]
        latent = np.column_stack((v, z))
        return np.asarray(self._location + latent @ self._rotation.T, dtype=np.float64)
