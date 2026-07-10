"""Weight-based and analytical divergence calculations."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from sampler_lab.core.numerics import normalize_log_weights


def chi_squared_divergence_estimate(log_weights: ArrayLike) -> float:
    """Estimate Pearson ``chi^2(p || q)`` up to weight normalization.

    The scale-invariant plug-in estimate is
    ``mean((w / mean(w) - 1)**2) = n * sum(omega**2) - 1``.
    It is therefore also the empirical squared coefficient of variation of the
    raw weights.
    """

    normalized, _ = normalize_log_weights(log_weights)
    return float(normalized.size * (normalized @ normalized) - 1.0)


def renyi_divergence_order_two_estimate(log_weights: ArrayLike) -> float:
    """Estimate order-two Renyi divergence as ``log(1 + chi^2)``."""

    return float(np.log1p(chi_squared_divergence_estimate(log_weights)))


def product_chi_squared(single_factor_chi_squared: float, dimension: int) -> float:
    """Lift a one-factor chi-squared divergence to an IID product target."""

    if not np.isfinite(single_factor_chi_squared) or single_factor_chi_squared < 0.0:
        if single_factor_chi_squared == float("inf"):
            return float("inf")
        raise ValueError("single_factor_chi_squared must be nonnegative")
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension < 1:
        raise ValueError("dimension must be positive")

    log_one_plus = dimension * math.log1p(single_factor_chi_squared)
    if log_one_plus > math.log(np.finfo(np.float64).max):
        return float("inf")
    return math.expm1(log_one_plus)


def gaussian_scale_chi_squared(
    dimension: int,
    *,
    target_scale: float = 1.0,
    proposal_scale: float = 1.0,
) -> float:
    """Return exact ``chi^2(p || q)`` for centered isotropic Gaussians.

    ``p = N(0, target_scale^2 I)`` and
    ``q = N(0, proposal_scale^2 I)``. The divergence is infinite when the
    proposal tails are too light: ``2 proposal_scale^2 <= target_scale^2``.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if target_scale <= 0.0 or proposal_scale <= 0.0:
        raise ValueError("Gaussian scales must be positive")

    denominator_squared = 2.0 * proposal_scale**2 - target_scale**2
    if denominator_squared <= 0.0:
        return float("inf")
    factor = proposal_scale**2 / (target_scale * math.sqrt(denominator_squared))
    log_integral = dimension * math.log(factor)
    if log_integral > math.log(np.finfo(np.float64).max):
        return float("inf")
    return math.expm1(log_integral)
