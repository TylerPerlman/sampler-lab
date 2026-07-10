"""Periodic square-lattice XY model and circular observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
_TWO_PI = float(2.0 * np.pi)


def wrap_angles(angles: ArrayLike) -> Array:
    """Wrap angles componentwise to ``[-pi, pi)``."""

    values = np.asarray(angles, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("angles must be finite")
    return np.asarray((values + np.pi) % _TWO_PI - np.pi, dtype=np.float64)


def periodic_angle_difference(first: ArrayLike, second: ArrayLike) -> Array:
    """Return the shortest signed componentwise angular difference."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("angle arrays must have the same shape")
    return wrap_angles(left - right)


def modified_bessel_i0_i1(value: float, *, tolerance: float = 1e-15) -> tuple[float, float]:
    """Compute ``I_0(x)`` and ``I_1(x)`` by convergent power series.

    This is intended for validation-scale concentrations rather than extreme
    asymptotics. The recurrence avoids factorial construction and is accurate for
    the moderate fields used in the XY laboratory.
    """

    if not np.isfinite(value):
        raise ValueError("value must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    x = float(value)
    squared_quarter = 0.25 * x * x
    term_zero = 1.0
    total_zero = 1.0
    term_one = 0.5 * x
    total_one = term_one
    for index in range(1, 10_000):
        term_zero *= squared_quarter / (index * index)
        term_one *= squared_quarter / (index * (index + 1))
        total_zero += term_zero
        total_one += term_one
        scale = max(abs(total_zero), abs(total_one), 1.0)
        if max(abs(term_zero), abs(term_one)) <= tolerance * scale:
            return float(total_zero), float(total_one)
    raise ArithmeticError("Bessel series failed to converge")


def von_mises_mean_cosine(concentration: float) -> float:
    """Return ``E[cos(theta)] = I_1(kappa) / I_0(kappa)``."""

    i0, i1 = modified_bessel_i0_i1(concentration)
    return float(i1 / i0)


@dataclass(frozen=True, slots=True)
class XYModel:
    """Nearest-neighbor periodic square-lattice XY target.

    For angles ``theta`` on an ``L x L`` torus,

    ``log pi(theta) = beta * [J sum_<ij> cos(theta_i-theta_j)
                              + h sum_i cos(theta_i)]``.

    The density is unnormalized. Positions supplied to HMC should be wrapped with
    :func:`wrap_angles` after each drift substep so the state space remains the
    compact torus rather than an improper periodic lift to Euclidean space.
    """

    size: int
    inverse_temperature: float
    coupling: float = 1.0
    external_field: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise TypeError("size must be an integer")
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not np.isfinite(self.inverse_temperature) or self.inverse_temperature < 0.0:
            raise ValueError("inverse_temperature must be nonnegative and finite")
        if not np.isfinite(self.coupling) or not np.isfinite(self.external_field):
            raise ValueError("coupling and external_field must be finite")

    @property
    def dimension(self) -> int:
        return self.size * self.size

    def _grid(self, angles: ArrayLike) -> Array:
        values = np.asarray(angles, dtype=np.float64)
        if values.shape != (self.dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("angles must be a finite vector matching the lattice")
        return values.reshape(self.size, self.size)

    def interaction_sum(self, angles: ArrayLike) -> float:
        """Return the sum over rightward and downward periodic bonds."""

        grid = self._grid(angles)
        horizontal = np.cos(grid - np.roll(grid, shift=-1, axis=1))
        vertical = np.cos(grid - np.roll(grid, shift=-1, axis=0))
        return float(np.sum(horizontal) + np.sum(vertical))

    def field_alignment(self, angles: ArrayLike) -> float:
        return float(np.sum(np.cos(self._grid(angles))))

    def energy(self, angles: ArrayLike) -> float:
        return float(
            -self.coupling * self.interaction_sum(angles)
            - self.external_field * self.field_alignment(angles)
        )

    def log_prob(self, angles: Array) -> float:
        return float(-self.inverse_temperature * self.energy(angles))

    def grad_log_prob(self, angles: Array) -> Array:
        grid = self._grid(angles)
        neighbor_sine_sum = (
            np.sin(np.roll(grid, shift=1, axis=0) - grid)
            + np.sin(np.roll(grid, shift=-1, axis=0) - grid)
            + np.sin(np.roll(grid, shift=1, axis=1) - grid)
            + np.sin(np.roll(grid, shift=-1, axis=1) - grid)
        )
        gradient = self.inverse_temperature * (
            self.coupling * neighbor_sine_sum - self.external_field * np.sin(grid)
        )
        return np.asarray(gradient.reshape(self.dimension), dtype=np.float64)

    def magnetization_vector(self, angles: ArrayLike) -> Array:
        grid = self._grid(angles)
        return np.asarray([np.mean(np.cos(grid)), np.mean(np.sin(grid))], dtype=np.float64)

    def absolute_magnetization(self, angles: ArrayLike) -> float:
        return float(np.linalg.norm(self.magnetization_vector(angles)))

    def mean_cosine(self, angles: ArrayLike) -> float:
        return float(np.mean(np.cos(self._grid(angles))))

    def sample_uniform(self, rng: np.random.Generator) -> Array:
        return np.asarray(rng.uniform(-np.pi, np.pi, size=self.dimension), dtype=np.float64)

    def exact_single_site_mean_cosine(self) -> float:
        """Exact one-site field response, valid when ``size=1``.

        Self-bond terms are constant at one site, so the nontrivial law is von Mises
        with concentration ``beta * external_field``.
        """

        if self.size != 1:
            raise ValueError("exact_single_site_mean_cosine requires size=1")
        concentration = self.inverse_temperature * self.external_field
        return von_mises_mean_cosine(concentration)
