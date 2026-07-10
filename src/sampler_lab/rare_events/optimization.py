"""Small purpose-built proposal-selection routines for Gaussian rare events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sampler_lab.rare_events.problems import RareGaussianProblem
from sampler_lab.rare_events.tempering import exact_tempered_log_second_moment
from sampler_lab.rare_events.twisting import exact_shifted_log_second_moment


@dataclass(frozen=True, slots=True)
class ProposalSelection:
    """Result of minimizing an exact or supplied second-moment criterion on a grid."""

    value: float
    log_second_moment: float
    candidate_count: int


def select_temperature(
    problem: RareGaussianProblem,
    epsilon: float,
    candidates: Sequence[float],
) -> ProposalSelection:
    """Choose a Gaussian temperature by exact second-moment minimization."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    values = np.asarray(candidates, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 1.0):
        raise ValueError("temperature candidates must be finite and at least one")
    objectives = np.asarray(
        [exact_tempered_log_second_moment(problem, float(value), epsilon) for value in values]
    )
    index = int(np.argmin(objectives))
    return ProposalSelection(
        value=float(values[index]),
        log_second_moment=float(objectives[index]),
        candidate_count=int(values.size),
    )


def select_shift_scale(
    problem: RareGaussianProblem,
    epsilon: float,
    candidates: Sequence[float],
) -> ProposalSelection:
    """Choose a multiple of one dominating point by exact second-moment minimization."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    values = np.asarray(candidates, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("shift-scale candidates must be finite")
    point = problem.dominant_point
    objectives = np.asarray(
        [
            exact_shifted_log_second_moment(problem, float(value) * point, epsilon)
            for value in values
        ]
    )
    index = int(np.argmin(objectives))
    return ProposalSelection(
        value=float(values[index]),
        log_second_moment=float(objectives[index]),
        candidate_count=int(values.size),
    )


__all__ = ["ProposalSelection", "select_shift_scale", "select_temperature"]
