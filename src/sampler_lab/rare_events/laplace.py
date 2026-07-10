"""Laplace and boundary-tail asymptotics for small-noise integrals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from sampler_lab.core.linalg import as_positive_definite
from sampler_lab.core.numerics import logsumexp
from sampler_lab.rare_events.problems import (
    GaussianTwoSidedRareEvent,
    RareGaussianProblem,
)


@dataclass(frozen=True, slots=True)
class LaplacePoint:
    """One isolated nondegenerate minimizer in a Laplace approximation."""

    minimum_value: float
    hessian: ArrayLike
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_value):
            raise ValueError("minimum_value must be finite")
        if not np.isfinite(self.amplitude) or self.amplitude <= 0.0:
            raise ValueError("amplitude must be positive and finite")
        as_positive_definite(self.hessian)

    @property
    def dimension(self) -> int:
        matrix = np.asarray(self.hessian, dtype=np.float64)
        return int(matrix.shape[0])


@dataclass(frozen=True, slots=True)
class LaplaceApproximation:
    """Log-domain and ordinary forms of a leading Laplace approximation."""

    value: float
    log_value: float
    exponential_rate: float
    log_prefactor: float
    n_minimizers: int


def laplace_log_integral(points: Sequence[LaplacePoint], epsilon: float) -> float:
    """Approximate ``integral a(x) exp(-I(x)/epsilon) dx``.

    Each supplied point must be an isolated, nondegenerate interior minimizer.  Multiple
    minimizers are summed in the log domain rather than replaced by the smallest prefactor.
    """

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not points:
        raise ValueError("at least one Laplace point is required")
    dimension = points[0].dimension
    terms: list[float] = []
    for point in points:
        if point.dimension != dimension:
            raise ValueError("all Laplace points must have the same dimension")
        hessian = as_positive_definite(point.hessian)
        sign, log_determinant = np.linalg.slogdet(hessian)
        if sign <= 0.0:  # pragma: no cover - construction already validates this
            raise RuntimeError("Hessian determinant must be positive")
        terms.append(
            math.log(point.amplitude)
            - point.minimum_value / epsilon
            + 0.5 * dimension * math.log(2.0 * math.pi * epsilon)
            - 0.5 * float(log_determinant)
        )
    return float(logsumexp(np.asarray(terms, dtype=np.float64)))


def laplace_integral(points: Sequence[LaplacePoint], epsilon: float) -> LaplaceApproximation:
    """Return a structured leading-order Laplace approximation."""

    log_value = laplace_log_integral(points, epsilon)
    minimum = min(point.minimum_value for point in points)
    log_prefactor = log_value + minimum / epsilon
    value = 0.0 if log_value < math.log(np.finfo(np.float64).tiny) else float(math.exp(log_value))
    return LaplaceApproximation(
        value=value,
        log_value=log_value,
        exponential_rate=float(minimum),
        log_prefactor=float(log_prefactor),
        n_minimizers=len(points),
    )


def gaussian_linear_event_log_asymptotic(
    problem: RareGaussianProblem,
    epsilon: float,
) -> float:
    """Leading boundary-Laplace approximation for a Gaussian linear rare event."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    variance = problem.directional_variance
    threshold = problem.threshold
    log_value = (
        -problem.rate / epsilon
        + 0.5 * math.log(epsilon * variance)
        - math.log(threshold)
        - 0.5 * math.log(2.0 * math.pi)
    )
    if isinstance(problem, GaussianTwoSidedRareEvent):
        log_value += math.log(2.0)
    return float(log_value)


def gaussian_linear_event_asymptotic(
    problem: RareGaussianProblem,
    epsilon: float,
) -> LaplaceApproximation:
    """Structured boundary-Laplace approximation for the exact Gaussian oracle."""

    log_value = gaussian_linear_event_log_asymptotic(problem, epsilon)
    value = 0.0 if log_value < math.log(np.finfo(np.float64).tiny) else float(math.exp(log_value))
    return LaplaceApproximation(
        value=value,
        log_value=log_value,
        exponential_rate=problem.rate,
        log_prefactor=float(log_value + problem.rate / epsilon),
        n_minimizers=2 if isinstance(problem, GaussianTwoSidedRareEvent) else 1,
    )


__all__ = [
    "LaplaceApproximation",
    "LaplacePoint",
    "gaussian_linear_event_asymptotic",
    "gaussian_linear_event_log_asymptotic",
    "laplace_integral",
    "laplace_log_integral",
]
