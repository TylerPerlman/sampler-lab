"""Gaussian-tail importance-sampling reference experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sampler_lab.core.numerics import validate_size
from sampler_lab.exact.gaussian import box_muller
from sampler_lab.importance.estimators import standard_importance_estimate


@dataclass(frozen=True, slots=True)
class GaussianTailRow:
    """One proposal's estimate of a standard-normal upper-tail probability."""

    proposal_mean: float
    estimate: float
    truth: float
    standard_error: float
    relative_standard_error: float
    absolute_error: float
    event_count: int
    effective_sample_size: float
    max_normalized_weight: float


def standard_normal_upper_tail(threshold: float) -> float:
    """Return ``P(Z >= threshold)`` for ``Z ~ N(0, 1)``."""

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return 0.5 * math.erfc(threshold / math.sqrt(2.0))


def gaussian_tail_experiment(
    rng: np.random.Generator,
    threshold: float,
    n_samples: int,
    proposal_means: Sequence[float],
) -> list[GaussianTailRow]:
    """Compare shifted-normal proposals for a standard-normal upper tail.

    For ``q = N(mu, 1)`` and ``p = N(0, 1)``, the exact log weight is
    ``-mu * x + mu**2 / 2``. ``mu=0`` is ordinary crude Monte Carlo.
    """

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    n_samples = validate_size(n_samples)
    if n_samples == 0:
        raise ValueError("n_samples must be positive")

    truth = standard_normal_upper_tail(threshold)
    rows: list[GaussianTailRow] = []
    for proposal_mean in proposal_means:
        if not np.isfinite(proposal_mean):
            raise ValueError("proposal means must be finite")
        samples = box_muller(rng, n_samples) + proposal_mean
        event = samples >= threshold
        log_weights = -proposal_mean * samples + 0.5 * proposal_mean**2
        estimate = standard_importance_estimate(event.astype(np.float64), log_weights)
        relative_standard_error = (
            estimate.standard_error / estimate.value if estimate.value != 0.0 else float("inf")
        )
        rows.append(
            GaussianTailRow(
                proposal_mean=float(proposal_mean),
                estimate=estimate.value,
                truth=truth,
                standard_error=estimate.standard_error,
                relative_standard_error=relative_standard_error,
                absolute_error=abs(estimate.value - truth),
                event_count=int(np.sum(event)),
                effective_sample_size=estimate.effective_sample_size,
                max_normalized_weight=estimate.max_normalized_weight,
            )
        )
    return rows
