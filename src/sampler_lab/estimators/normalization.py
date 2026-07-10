"""Monte Carlo estimation of ratios of normalizing constants."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from sampler_lab.core.numerics import logsumexp, normalize_log_weights
from sampler_lab.core.results import NormalizationRatioEstimate
from sampler_lab.diagnostics.weighted import diagnostics_from_normalized_weights


def estimate_normalization_ratio(log_weights: ArrayLike) -> NormalizationRatioEstimate:
    """Estimate a normalizing-constant ratio from importance log weights.

    If ``X ~ q`` and ``log_weights = log(tilde_p(X)) - log(q(X))``, this estimates
    ``Z_p``. More generally, when samples are drawn from the normalized version
    of ``tilde_q``, the same calculation estimates ``Z_p / Z_q``.

    The logarithm is always computed stably. The linear estimate is ``inf`` only
    when the result is too large to represent in ``float64``.
    """

    values = np.asarray(log_weights, dtype=np.float64)
    normalized, _ = normalize_log_weights(values)
    n_samples = int(values.size)
    log_value = float(logsumexp(values) - np.log(n_samples))

    max_log = float(np.log(np.finfo(np.float64).max))
    value = float(np.exp(log_value)) if log_value <= max_log else float("inf")

    diagnostics = diagnostics_from_normalized_weights(normalized)
    if n_samples == 1:
        relative_standard_error = float("nan")
    else:
        finite_values = values[np.isfinite(values)]
        shift = float(np.max(finite_values))
        scaled_weights = np.exp(values - shift)
        mean_scaled_weight = float(np.mean(scaled_weights))
        relative_standard_error = float(
            np.std(scaled_weights, ddof=1) / np.sqrt(n_samples) / mean_scaled_weight
        )

    standard_error = 0.0 if relative_standard_error == 0.0 else value * relative_standard_error
    return NormalizationRatioEstimate(
        value=value,
        log_value=log_value,
        standard_error=standard_error,
        relative_standard_error=relative_standard_error,
        n_samples=n_samples,
        effective_sample_size=diagnostics.effective_sample_size,
    )
