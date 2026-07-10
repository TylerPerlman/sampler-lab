"""Result dataclasses returned by public algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Transition:
    """Result of one Markov-kernel step."""

    state: Array
    accepted: bool | None = None
    log_acceptance_ratio: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IIDEstimate:
    """Scalar IID Monte Carlo estimate with its elementary uncertainty summary."""

    value: float
    sample_variance: float
    standard_error: float
    n_samples: int


@dataclass(frozen=True, slots=True)
class ImportanceEstimate:
    """Scalar importance estimate with weight-quality diagnostics."""

    value: float
    standard_error: float
    n_samples: int
    effective_sample_size: float
    max_normalized_weight: float
    weight_entropy: float
    log_mean_weight: float
    delta_method_bias: float | None
    self_normalized: bool


@dataclass(frozen=True, slots=True)
class NormalizationRatioEstimate:
    """Estimate of a normalizing-constant ratio from importance weights."""

    value: float
    log_value: float
    standard_error: float
    relative_standard_error: float
    n_samples: int
    effective_sample_size: float


@dataclass(frozen=True, slots=True)
class RejectionResult:
    """Accepted samples and accounting from rejection sampling."""

    samples: Array
    n_accepted: int
    n_proposals: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        """Observed acceptance fraction."""

        if self.n_proposals == 0:
            return float("nan")
        return self.n_accepted / self.n_proposals

    @property
    def proposals_per_sample(self) -> float:
        """Observed proposal cost per accepted sample."""

        if self.n_accepted == 0:
            return float("nan")
        return self.n_proposals / self.n_accepted
