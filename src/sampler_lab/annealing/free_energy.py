"""Jarzynski normalization ratios and dimensionless free energies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from sampler_lab.core.results import NormalizationRatioEstimate
from sampler_lab.estimators.normalization import estimate_normalization_ratio


@dataclass(frozen=True, slots=True)
class FreeEnergyEstimate:
    """Normalization-ratio estimate expressed as a free-energy difference.

    The convention is dimensionless free energy ``F = -log Z``. Therefore
    ``delta_free_energy = -log(Z_final / Z_initial)``.
    """

    ratio: NormalizationRatioEstimate
    delta_free_energy: float
    standard_error: float
    delta_method_bias: float


def jarzynski_estimate(reduced_work: ArrayLike) -> FreeEnergyEstimate:
    """Estimate a free-energy difference from independent reduced-work values.

    Reduced work follows ``E[exp(-W)] = Z_final / Z_initial``.
    """

    work = np.asarray(reduced_work, dtype=np.float64)
    if work.ndim != 1 or work.size == 0:
        raise ValueError("reduced_work must be a nonempty one-dimensional array")
    if np.any(np.isnan(work)) or np.any(np.isneginf(work)):
        raise ValueError("reduced_work may not contain nan or -inf")
    ratio = estimate_normalization_ratio(-work)
    relative_variance = ratio.relative_standard_error**2
    return FreeEnergyEstimate(
        ratio=ratio,
        delta_free_energy=-ratio.log_value,
        standard_error=ratio.relative_standard_error,
        delta_method_bias=0.5 * relative_variance,
    )


def free_energy_from_log_ratio(log_ratio: float) -> float:
    """Convert ``log(Z_final / Z_initial)`` to ``F_final - F_initial``."""

    if np.isnan(log_ratio):
        raise ValueError("log_ratio may not be nan")
    return -float(log_ratio)
