"""Small stochastic policies used to adapt exact Monte Carlo kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.adaptive.warmup import FrozenPolicy

Array = NDArray[np.float64]


def _as_feature_vector(features: ArrayLike, expected: int) -> Array:
    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (expected,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"features must be a finite vector of length {expected}")
    return vector


def _softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return np.asarray(weights / np.sum(weights), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """Sampled policy action with its score-function derivative."""

    index: int | None
    value: Array
    log_prob: float
    score: Array
    probabilities: Array | None
    diagnostics: dict[str, float]

    def __post_init__(self) -> None:
        if self.index is not None and self.index < 0:
            raise ValueError("categorical action indices must be nonnegative")
        if self.value.size == 0 or not np.all(np.isfinite(self.value)):
            raise ValueError("policy action values must be nonempty and finite")
        if self.score.ndim != 1 or not np.all(np.isfinite(self.score)):
            raise ValueError("policy scores must be finite vectors")
        if not np.isfinite(self.log_prob):
            raise ValueError("policy log probability must be finite")
        if self.probabilities is not None:
            probabilities = np.asarray(self.probabilities, dtype=np.float64)
            if probabilities.ndim != 1 or np.any(probabilities < 0.0):
                raise ValueError("policy probabilities must be a nonnegative vector")
            if not np.isclose(np.sum(probabilities), 1.0):
                raise ValueError("policy probabilities must sum to one")


class LinearSoftmaxPolicy:
    """Categorical policy with logits ``weights @ features``."""

    def __init__(self, weights: ArrayLike, action_values: ArrayLike | None = None) -> None:
        values = np.asarray(weights, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) <= 0 or not np.all(np.isfinite(values)):
            raise ValueError("weights must be a nonempty finite matrix")
        self._weights = np.array(values, dtype=np.float64, copy=True)
        if action_values is None:
            actions = np.arange(values.shape[0], dtype=np.float64)[:, None]
        else:
            actions = np.asarray(action_values, dtype=np.float64)
            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.ndim != 2 or actions.shape[0] != values.shape[0]:
                raise ValueError("action_values must have one row per action")
            if not np.all(np.isfinite(actions)):
                raise ValueError("action_values must be finite")
        self._action_values = np.array(actions, dtype=np.float64, copy=True)

    @property
    def n_actions(self) -> int:
        return int(self._weights.shape[0])

    @property
    def n_features(self) -> int:
        return int(self._weights.shape[1])

    @property
    def parameters(self) -> Array:
        return self._weights.reshape(-1).copy()

    @property
    def weights(self) -> Array:
        return self._weights.copy()

    @property
    def action_values(self) -> Array:
        return self._action_values.copy()

    def set_parameters(self, parameters: ArrayLike) -> None:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (self._weights.size,) or not np.all(np.isfinite(values)):
            raise ValueError("parameters have the wrong shape or are nonfinite")
        self._weights[...] = values.reshape(self._weights.shape)

    def probabilities(self, features: ArrayLike) -> Array:
        vector = _as_feature_vector(features, self.n_features)
        return _softmax(self._weights @ vector)

    def score(self, features: ArrayLike, action_index: int) -> Array:
        """Return ``grad log pi(action | features)`` in row-major parameter order."""

        if isinstance(action_index, bool) or not isinstance(action_index, int):
            raise TypeError("action_index must be an integer")
        if not 0 <= action_index < self.n_actions:
            raise ValueError("action_index is out of range")
        vector = _as_feature_vector(features, self.n_features)
        probabilities = self.probabilities(vector)
        residual = -probabilities
        residual[action_index] += 1.0
        return np.asarray(np.outer(residual, vector).reshape(-1), dtype=np.float64)

    def act(self, rng: np.random.Generator, features: ArrayLike) -> PolicyAction:
        vector = _as_feature_vector(features, self.n_features)
        probabilities = self.probabilities(vector)
        action_index = int(rng.choice(self.n_actions, p=probabilities))
        return PolicyAction(
            index=action_index,
            value=self._action_values[action_index].copy(),
            log_prob=float(np.log(probabilities[action_index])),
            score=self.score(vector, action_index),
            probabilities=probabilities,
            diagnostics={"entropy": self.entropy(vector)},
        )

    def entropy(self, features: ArrayLike) -> float:
        probabilities = self.probabilities(features)
        positive = probabilities > 0.0
        return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))

    def freeze(self, *, name: str = "linear-softmax") -> FrozenLinearSoftmaxPolicy:
        return FrozenLinearSoftmaxPolicy(self._weights, self._action_values, name=name)


@dataclass(frozen=True, slots=True, init=False)
class FrozenLinearSoftmaxPolicy:
    """Immutable categorical policy used by frozen evaluation kernels."""

    weights: Array
    action_values: Array
    name: str

    def __init__(self, weights: ArrayLike, action_values: ArrayLike, *, name: str) -> None:
        matrix = np.asarray(weights, dtype=np.float64)
        actions = np.asarray(action_values, dtype=np.float64)
        if matrix.ndim != 2 or actions.ndim != 2 or actions.shape[0] != matrix.shape[0]:
            raise ValueError("frozen policy arrays have incompatible shapes")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(actions)):
            raise ValueError("frozen policy arrays must be finite")
        matrix_copy = np.array(matrix, copy=True)
        action_copy = np.array(actions, copy=True)
        matrix_copy.setflags(write=False)
        action_copy.setflags(write=False)
        object.__setattr__(self, "weights", matrix_copy)
        object.__setattr__(self, "action_values", action_copy)
        object.__setattr__(self, "name", str(name))

    @property
    def n_actions(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.weights.shape[1])

    @property
    def parameters(self) -> Array:
        return self.weights.reshape(-1).copy()

    def probabilities(self, features: ArrayLike) -> Array:
        vector = _as_feature_vector(features, self.n_features)
        return _softmax(self.weights @ vector)

    def act(self, rng: np.random.Generator, features: ArrayLike) -> PolicyAction:
        mutable = LinearSoftmaxPolicy(self.weights, self.action_values)
        return mutable.act(rng, features)

    def as_generic_frozen_policy(self) -> FrozenPolicy:
        return FrozenPolicy(
            self.name,
            self.parameters,
            {
                "n_actions": float(self.n_actions),
                "n_features": float(self.n_features),
            },
        )


class SquashedGaussianPolicy:
    """Diagonal Gaussian policy transformed into a finite action box."""

    def __init__(
        self,
        mean_weights: ArrayLike,
        log_scales: ArrayLike,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> None:
        weights = np.asarray(mean_weights, dtype=np.float64)
        scales = np.asarray(log_scales, dtype=np.float64)
        if weights.ndim != 2 or min(weights.shape) <= 0 or not np.all(np.isfinite(weights)):
            raise ValueError("mean_weights must be a nonempty finite matrix")
        if scales.shape != (weights.shape[0],) or not np.all(np.isfinite(scales)):
            raise ValueError("log_scales must match the action dimension")
        low = np.broadcast_to(np.asarray(lower, dtype=np.float64), scales.shape)
        high = np.broadcast_to(np.asarray(upper, dtype=np.float64), scales.shape)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(high <= low):
            raise ValueError("action bounds must be finite and strictly ordered")
        self._mean_weights = np.array(weights, copy=True)
        self._log_scales = np.array(scales, copy=True)
        self._lower = np.array(low, copy=True)
        self._upper = np.array(high, copy=True)

    @property
    def action_dimension(self) -> int:
        return int(self._mean_weights.shape[0])

    @property
    def n_features(self) -> int:
        return int(self._mean_weights.shape[1])

    @property
    def parameters(self) -> Array:
        return np.concatenate((self._mean_weights.reshape(-1), self._log_scales)).astype(np.float64)

    def set_parameters(self, parameters: ArrayLike) -> None:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (self._mean_weights.size + self.action_dimension,):
            raise ValueError("parameters have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite")
        split = self._mean_weights.size
        self._mean_weights[...] = values[:split].reshape(self._mean_weights.shape)
        self._log_scales[...] = values[split:]

    def mean(self, features: ArrayLike) -> Array:
        vector = _as_feature_vector(features, self.n_features)
        return np.asarray(self._mean_weights @ vector, dtype=np.float64)

    def log_prob_and_score(
        self,
        action: ArrayLike,
        features: ArrayLike,
    ) -> tuple[float, Array]:
        """Evaluate a bounded action density and its parameter score."""

        vector = _as_feature_vector(features, self.n_features)
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (self.action_dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("action must be a finite vector matching action dimension")
        half_span = 0.5 * (self._upper - self._lower)
        midpoint = 0.5 * (self._upper + self._lower)
        squashed = (value - midpoint) / half_span
        if np.any(np.abs(squashed) >= 1.0):
            raise ValueError("action must lie strictly inside the configured bounds")
        latent = np.arctanh(squashed)
        mean = self.mean(vector)
        scales = np.exp(self._log_scales)
        noise = (latent - mean) / scales
        base_log_prob = float(
            -0.5 * np.sum(noise * noise)
            - np.sum(self._log_scales)
            - 0.5 * self.action_dimension * np.log(2.0 * np.pi)
        )
        log_jacobian = float(
            np.sum(np.log(half_span) + np.log(np.maximum(1.0 - squashed * squashed, 1e-15)))
        )
        mean_score = np.outer(noise / scales, vector).reshape(-1)
        scale_score = noise * noise - 1.0
        score = np.concatenate((mean_score, scale_score)).astype(np.float64)
        return base_log_prob - log_jacobian, score

    def act(self, rng: np.random.Generator, features: ArrayLike) -> PolicyAction:
        vector = _as_feature_vector(features, self.n_features)
        mean = self.mean(vector)
        scales = np.exp(self._log_scales)
        noise = np.asarray(rng.normal(size=self.action_dimension), dtype=np.float64)
        latent = mean + scales * noise
        squashed = np.tanh(latent)
        half_span = 0.5 * (self._upper - self._lower)
        midpoint = 0.5 * (self._upper + self._lower)
        action = midpoint + half_span * squashed
        log_prob, score = self.log_prob_and_score(action, vector)
        entropy = float(
            0.5 * self.action_dimension * (1.0 + np.log(2.0 * np.pi)) + np.sum(self._log_scales)
        )
        return PolicyAction(
            index=None,
            value=np.asarray(action, dtype=np.float64),
            log_prob=log_prob,
            score=score,
            probabilities=None,
            diagnostics={"base_entropy": entropy, "latent_norm": float(np.linalg.norm(latent))},
        )

    def freeze(self, *, name: str = "squashed-gaussian") -> FrozenPolicy:
        return FrozenPolicy(
            name,
            self.parameters,
            {
                "action_dimension": float(self.action_dimension),
                "n_features": float(self.n_features),
            },
        )
