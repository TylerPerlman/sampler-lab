"""Inverse-multiquadric Stein discrepancies and SVGD particle transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.protocols import DifferentiableLogDensity

Array = NDArray[np.float64]


def _as_particles(value: ArrayLike) -> Array:
    particles = np.asarray(value, dtype=np.float64)
    if particles.ndim != 2 or min(particles.shape) <= 0 or not np.all(np.isfinite(particles)):
        raise ValueError("particles must be a nonempty finite matrix")
    return particles


@dataclass(frozen=True, slots=True)
class IMQKernel:
    r"""Inverse-multiquadric kernel ``(c^2 + ||x-y||^2)^beta``."""

    c: float = 1.0
    beta: float = -0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.c) or self.c <= 0.0:
            raise ValueError("c must be positive and finite")
        if not np.isfinite(self.beta) or not -1.0 < self.beta < 0.0:
            raise ValueError("beta must lie in (-1, 0)")

    def value(self, x: ArrayLike, y: ArrayLike) -> float:
        first = np.asarray(x, dtype=np.float64)
        second = np.asarray(y, dtype=np.float64)
        if first.shape != second.shape or first.ndim != 1:
            raise ValueError("kernel inputs must be equal-length vectors")
        difference = first - second
        return float((self.c * self.c + difference @ difference) ** self.beta)

    def gradient_first(self, x: ArrayLike, y: ArrayLike) -> Array:
        first = np.asarray(x, dtype=np.float64)
        second = np.asarray(y, dtype=np.float64)
        if first.shape != second.shape or first.ndim != 1:
            raise ValueError("kernel inputs must be equal-length vectors")
        difference = first - second
        base = self.c * self.c + difference @ difference
        return np.asarray(
            2.0 * self.beta * difference * base ** (self.beta - 1.0), dtype=np.float64
        )

    def mixed_trace(self, x: ArrayLike, y: ArrayLike) -> float:
        """Return ``trace(grad_x grad_y k(x, y))``."""

        first = np.asarray(x, dtype=np.float64)
        second = np.asarray(y, dtype=np.float64)
        if first.shape != second.shape or first.ndim != 1:
            raise ValueError("kernel inputs must be equal-length vectors")
        difference = first - second
        squared_distance = float(difference @ difference)
        base = self.c * self.c + squared_distance
        return float(
            -2.0 * self.beta * first.size * base ** (self.beta - 1.0)
            - 4.0 * self.beta * (self.beta - 1.0) * squared_distance * base ** (self.beta - 2.0)
        )


_DEFAULT_IMQ_KERNEL = IMQKernel()


def stein_kernel_value(
    x: ArrayLike,
    y: ArrayLike,
    score_x: ArrayLike,
    score_y: ArrayLike,
    *,
    kernel: IMQKernel = _DEFAULT_IMQ_KERNEL,
) -> float:
    """Evaluate the scalar Langevin-Stein kernel ``u_p(x, y)``."""

    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    first_score = np.asarray(score_x, dtype=np.float64)
    second_score = np.asarray(score_y, dtype=np.float64)
    if (
        first.shape != second.shape
        or first_score.shape != first.shape
        or second_score.shape != first.shape
    ):
        raise ValueError("states and scores must be equal-length vectors")
    gradient_x = kernel.gradient_first(first, second)
    gradient_y = -gradient_x
    return float(
        kernel.value(first, second) * (first_score @ second_score)
        + first_score @ gradient_y
        + second_score @ gradient_x
        + kernel.mixed_trace(first, second)
    )


def kernel_stein_discrepancy(
    samples: ArrayLike,
    target: DifferentiableLogDensity,
    *,
    kernel: IMQKernel = _DEFAULT_IMQ_KERNEL,
    unbiased: bool = True,
) -> float:
    """Estimate IMQ KSD using the U- or V-statistic convention."""

    particles = _as_particles(samples)
    if unbiased and particles.shape[0] < 2:
        raise ValueError("unbiased KSD requires at least two samples")
    scores = np.asarray(
        [target.grad_log_prob(np.asarray(particle, dtype=np.float64)) for particle in particles],
        dtype=np.float64,
    )
    if scores.shape != particles.shape or not np.all(np.isfinite(scores)):
        raise ValueError("target scores must match particles and be finite")
    total = 0.0
    count = 0
    for first_index in range(particles.shape[0]):
        start = 0 if not unbiased else first_index + 1
        for second_index in range(start, particles.shape[0]):
            if unbiased and first_index == second_index:
                continue
            value = stein_kernel_value(
                particles[first_index],
                particles[second_index],
                scores[first_index],
                scores[second_index],
                kernel=kernel,
            )
            if unbiased:
                total += 2.0 * value
                count += 2
            else:
                total += value
                count += 1
    estimate = total / count
    return float(np.sqrt(max(0.0, estimate)))


def svgd_direction(
    particles: ArrayLike,
    target: DifferentiableLogDensity,
    *,
    kernel: IMQKernel = _DEFAULT_IMQ_KERNEL,
) -> Array:
    """Return the finite-particle SVGD velocity field."""

    values = _as_particles(particles)
    scores = np.asarray(
        [target.grad_log_prob(np.asarray(particle, dtype=np.float64)) for particle in values],
        dtype=np.float64,
    )
    if scores.shape != values.shape or not np.all(np.isfinite(scores)):
        raise ValueError("target scores must match particles and be finite")
    direction = np.zeros_like(values)
    for destination in range(values.shape[0]):
        for source in range(values.shape[0]):
            kernel_value = kernel.value(values[source], values[destination])
            repulsion = kernel.gradient_first(values[source], values[destination])
            direction[destination] += kernel_value * scores[source] + repulsion
    direction /= values.shape[0]
    return np.asarray(direction, dtype=np.float64)


@dataclass(frozen=True, slots=True, init=False)
class SVGDResult:
    """Dependent variational particle approximation and optimization history."""

    particles: Array
    ksd_history: Array
    mean_step_norms: Array
    approximate: bool

    def __init__(
        self,
        particles: ArrayLike,
        ksd_history: ArrayLike,
        mean_step_norms: ArrayLike,
    ) -> None:
        values = _as_particles(particles)
        ksd = np.asarray(ksd_history, dtype=np.float64)
        norms = np.asarray(mean_step_norms, dtype=np.float64)
        if ksd.ndim != 1 or norms.shape != ksd.shape:
            raise ValueError("SVGD histories must be equal-length vectors")
        if not np.all(np.isfinite(ksd)) or not np.all(np.isfinite(norms)):
            raise ValueError("SVGD histories must be finite")
        particle_copy = np.array(values, copy=True)
        ksd_copy = np.array(ksd, copy=True)
        norm_copy = np.array(norms, copy=True)
        particle_copy.setflags(write=False)
        ksd_copy.setflags(write=False)
        norm_copy.setflags(write=False)
        object.__setattr__(self, "particles", particle_copy)
        object.__setattr__(self, "ksd_history", ksd_copy)
        object.__setattr__(self, "mean_step_norms", norm_copy)
        object.__setattr__(self, "approximate", True)


def run_svgd(
    initial_particles: ArrayLike,
    target: DifferentiableLogDensity,
    *,
    n_steps: int,
    step_size: float,
    kernel: IMQKernel = _DEFAULT_IMQ_KERNEL,
    record_ksd: bool = True,
) -> SVGDResult:
    """Run deterministic SVGD with a constant step size."""

    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be positive and finite")
    particles = np.array(_as_particles(initial_particles), copy=True)
    ksd_history = np.empty(n_steps, dtype=np.float64)
    mean_step_norms = np.empty(n_steps, dtype=np.float64)
    for step in range(n_steps):
        direction = svgd_direction(particles, target, kernel=kernel)
        increment = step_size * direction
        particles += increment
        mean_step_norms[step] = float(np.mean(np.linalg.norm(increment, axis=1)))
        ksd_history[step] = (
            kernel_stein_discrepancy(particles, target, kernel=kernel, unbiased=False)
            if record_ksd
            else float("nan")
        )
    if not record_ksd:
        ksd_history.fill(0.0)
    return SVGDResult(particles, ksd_history, mean_step_norms)
