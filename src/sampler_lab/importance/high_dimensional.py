"""Controlled demonstrations of importance-weight collapse in product targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sampler_lab.core.numerics import validate_size
from sampler_lab.diagnostics.weighted import weight_diagnostics
from sampler_lab.exact.gaussian import box_muller
from sampler_lab.importance.divergences import gaussian_scale_chi_squared


@dataclass(frozen=True, slots=True)
class GaussianWeightCollapseRow:
    """Empirical and exact diagnostics for one product-space dimension."""

    dimension: int
    n_samples: int
    effective_sample_size: float
    ess_fraction: float
    max_normalized_weight: float
    empirical_chi_squared: float
    theoretical_chi_squared: float
    asymptotic_ess_fraction: float


def gaussian_product_log_weights(
    samples: np.ndarray,
    *,
    target_scale: float = 1.0,
    proposal_scale: float = 1.0,
) -> np.ndarray:
    """Compute ``log p - log q`` for centered isotropic Gaussian arrays."""

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("samples must have shape (n_samples, dimension)")
    if not np.all(np.isfinite(points)):
        raise ValueError("samples must be finite")
    if target_scale <= 0.0 or proposal_scale <= 0.0:
        raise ValueError("Gaussian scales must be positive")

    dimension = points.shape[1]
    squared_norm = np.sum(np.square(points), axis=1)
    constant = dimension * np.log(proposal_scale / target_scale)
    quadratic = 0.5 * (proposal_scale**-2 - target_scale**-2) * squared_norm
    return np.asarray(constant + quadratic, dtype=np.float64)


def gaussian_weight_collapse_experiment(
    rng: np.random.Generator,
    dimensions: Sequence[int],
    n_samples: int,
    *,
    target_scale: float = 1.0,
    proposal_scale: float = 1.25,
) -> list[GaussianWeightCollapseRow]:
    """Measure importance-weight collapse as product dimension grows."""

    n_samples = validate_size(n_samples)
    if n_samples == 0:
        raise ValueError("n_samples must be positive")
    rows: list[GaussianWeightCollapseRow] = []
    for dimension in dimensions:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimensions must contain integers")
        if dimension < 1:
            raise ValueError("dimensions must be positive")

        standard_normals = box_muller(rng, n_samples * dimension).reshape(n_samples, dimension)
        samples = proposal_scale * standard_normals
        log_weights = gaussian_product_log_weights(
            samples,
            target_scale=target_scale,
            proposal_scale=proposal_scale,
        )
        diagnostics = weight_diagnostics(log_weights)
        theoretical = gaussian_scale_chi_squared(
            dimension,
            target_scale=target_scale,
            proposal_scale=proposal_scale,
        )
        asymptotic_fraction = 0.0 if np.isinf(theoretical) else 1.0 / (1.0 + theoretical)
        rows.append(
            GaussianWeightCollapseRow(
                dimension=dimension,
                n_samples=n_samples,
                effective_sample_size=diagnostics.effective_sample_size,
                ess_fraction=diagnostics.ess_fraction,
                max_normalized_weight=diagnostics.max_normalized_weight,
                empirical_chi_squared=diagnostics.coefficient_of_variation_squared,
                theoretical_chi_squared=theoretical,
                asymptotic_ess_fraction=asymptotic_fraction,
            )
        )
    return rows
