"""Score-function and natural policy-gradient utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def discounted_returns(rewards: ArrayLike, *, discount: float = 1.0) -> Array:
    """Return reward-to-go values for one rollout."""

    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("rewards must be a finite vector")
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    result = np.empty_like(values)
    accumulator = 0.0
    for index in range(values.size - 1, -1, -1):
        accumulator = float(values[index] + discount * accumulator)
        result[index] = accumulator
    return result


@dataclass(frozen=True, slots=True)
class ReinforceEstimate:
    """REINFORCE gradient and variance diagnostics."""

    gradient: Array
    per_step_gradients: Array
    advantages: Array
    raw_variance: float
    centered_variance: float


def reinforce_gradient(
    scores: ArrayLike,
    returns: ArrayLike,
    *,
    baseline_predictions: ArrayLike | None = None,
    normalize_advantages: bool = False,
) -> ReinforceEstimate:
    """Estimate a score-function gradient from one or more transitions."""

    score_values = np.asarray(scores, dtype=np.float64)
    return_values = np.asarray(returns, dtype=np.float64)
    if score_values.ndim != 2 or score_values.shape[0] == 0:
        raise ValueError("scores must be a nonempty matrix")
    if return_values.shape != (score_values.shape[0],):
        raise ValueError("returns must have one value per score row")
    if not np.all(np.isfinite(score_values)) or not np.all(np.isfinite(return_values)):
        raise ValueError("scores and returns must be finite")
    if baseline_predictions is None:
        baseline = np.zeros_like(return_values)
    else:
        baseline = np.asarray(baseline_predictions, dtype=np.float64)
        if baseline.shape != return_values.shape or not np.all(np.isfinite(baseline)):
            raise ValueError("baseline predictions must match finite returns")
    advantages = return_values - baseline
    if normalize_advantages and advantages.size > 1:
        standard_deviation = float(np.std(advantages, ddof=1))
        if standard_deviation > 0.0:
            advantages = (advantages - np.mean(advantages)) / standard_deviation
    raw = score_values * return_values[:, None]
    centered = score_values * advantages[:, None]
    gradient = np.mean(centered, axis=0)
    raw_variance = float(np.mean(np.var(raw, axis=0, ddof=1))) if raw.shape[0] > 1 else 0.0
    centered_variance = (
        float(np.mean(np.var(centered, axis=0, ddof=1))) if centered.shape[0] > 1 else 0.0
    )
    return ReinforceEstimate(
        gradient=np.asarray(gradient, dtype=np.float64),
        per_step_gradients=np.asarray(centered, dtype=np.float64),
        advantages=np.asarray(advantages, dtype=np.float64),
        raw_variance=raw_variance,
        centered_variance=centered_variance,
    )


def categorical_kl(old_probabilities: ArrayLike, new_probabilities: ArrayLike) -> float:
    """Return ``KL(old || new)`` for two categorical laws."""

    old = np.asarray(old_probabilities, dtype=np.float64)
    new = np.asarray(new_probabilities, dtype=np.float64)
    if old.ndim != 1 or old.shape != new.shape or old.size == 0:
        raise ValueError("probability vectors must have equal nonzero shape")
    if np.any(old < 0.0) or np.any(new <= 0.0):
        raise ValueError("old probabilities must be nonnegative and new probabilities positive")
    if not np.isclose(np.sum(old), 1.0) or not np.isclose(np.sum(new), 1.0):
        raise ValueError("probability vectors must sum to one")
    positive = old > 0.0
    return float(np.sum(old[positive] * (np.log(old[positive]) - np.log(new[positive]))))


def linear_softmax_fisher(probabilities: ArrayLike, features: ArrayLike) -> Array:
    """Exact Fisher matrix for a linear softmax policy at one feature vector."""

    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    feature_vector = np.asarray(features, dtype=np.float64)
    if probabilities_array.ndim != 1 or probabilities_array.size < 2:
        raise ValueError("probabilities must contain at least two actions")
    if feature_vector.ndim != 1 or feature_vector.size == 0:
        raise ValueError("features must be a nonempty vector")
    if np.any(probabilities_array <= 0.0) or not np.isclose(np.sum(probabilities_array), 1.0):
        raise ValueError("probabilities must be positive and sum to one")
    action_covariance = np.diag(probabilities_array) - np.outer(
        probabilities_array, probabilities_array
    )
    feature_outer = np.outer(feature_vector, feature_vector)
    return np.asarray(np.kron(action_covariance, feature_outer), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NaturalGradientResult:
    """Damped natural-gradient direction and local KL prediction."""

    direction: Array
    scale: float
    predicted_kl: float
    quadratic_form: float


def natural_gradient_direction(
    gradient: ArrayLike,
    fisher: ArrayLike,
    *,
    damping: float = 1e-6,
    max_kl: float | None = None,
) -> NaturalGradientResult:
    """Solve a damped Fisher system and optionally enforce a local KL radius."""

    gradient_vector = np.asarray(gradient, dtype=np.float64)
    fisher_matrix = np.asarray(fisher, dtype=np.float64)
    if gradient_vector.ndim != 1 or fisher_matrix.shape != (
        gradient_vector.size,
        gradient_vector.size,
    ):
        raise ValueError("fisher must be square and match the gradient")
    if not np.all(np.isfinite(gradient_vector)) or not np.all(np.isfinite(fisher_matrix)):
        raise ValueError("gradient and fisher must be finite")
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be nonnegative and finite")
    if max_kl is not None and (not np.isfinite(max_kl) or max_kl <= 0.0):
        raise ValueError("max_kl must be positive and finite")
    system = 0.5 * (fisher_matrix + fisher_matrix.T) + damping * np.eye(
        gradient_vector.size, dtype=np.float64
    )
    raw_direction = np.asarray(np.linalg.solve(system, gradient_vector), dtype=np.float64)
    quadratic_form = float(raw_direction @ fisher_matrix @ raw_direction)
    scale = 1.0
    if max_kl is not None and quadratic_form > 0.0:
        scale = min(1.0, float(np.sqrt(2.0 * max_kl / quadratic_form)))
    direction = scale * raw_direction
    predicted_kl = float(0.5 * direction @ fisher_matrix @ direction)
    return NaturalGradientResult(direction, scale, predicted_kl, quadratic_form)
