"""Small optimizers used by sampler adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


class ParameterOptimizer(Protocol):
    """Update a flat parameter vector from an ascent direction."""

    def step(self, parameters: ArrayLike, gradient: ArrayLike) -> Array:
        """Return updated parameters."""


def _validated_pair(parameters: ArrayLike, gradient: ArrayLike) -> tuple[Array, Array]:
    values = np.asarray(parameters, dtype=np.float64)
    direction = np.asarray(gradient, dtype=np.float64)
    if values.ndim != 1 or values.shape != direction.shape:
        raise ValueError("parameters and gradient must be equal-length vectors")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(direction)):
        raise ValueError("parameters and gradient must be finite")
    return values, direction


@dataclass(frozen=True, slots=True)
class SGD:
    """Plain stochastic-gradient ascent."""

    learning_rate: float
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be positive and finite")

    def step(self, parameters: ArrayLike, gradient: ArrayLike) -> Array:
        values, direction = _validated_pair(parameters, gradient)
        if self.gradient_clip_norm is not None:
            norm = float(np.linalg.norm(direction))
            if norm > self.gradient_clip_norm:
                direction = direction * (self.gradient_clip_norm / norm)
        return np.asarray(values + self.learning_rate * direction, dtype=np.float64)


@dataclass(slots=True)
class Adam:
    """Adam stochastic-gradient ascent on a flat parameter vector."""

    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip_norm: float | None = None
    _first: Array | None = field(default=None, init=False, repr=False)
    _second: Array | None = field(default=None, init=False, repr=False)
    _iteration: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam beta parameters must lie in [0, 1)")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be positive and finite")

    @property
    def iteration(self) -> int:
        return self._iteration

    def step(self, parameters: ArrayLike, gradient: ArrayLike) -> Array:
        values, direction = _validated_pair(parameters, gradient)
        if self.gradient_clip_norm is not None:
            norm = float(np.linalg.norm(direction))
            if norm > self.gradient_clip_norm:
                direction = direction * (self.gradient_clip_norm / norm)
        if self._first is None:
            self._first = np.zeros_like(values)
            self._second = np.zeros_like(values)
        if self._first.shape != values.shape or self._second is None:
            raise ValueError("parameter dimension changed after Adam initialization")
        self._iteration += 1
        self._first = self.beta1 * self._first + (1.0 - self.beta1) * direction
        self._second = self.beta2 * self._second + (1.0 - self.beta2) * direction * direction
        first_hat = self._first / (1.0 - self.beta1**self._iteration)
        second_hat = self._second / (1.0 - self.beta2**self._iteration)
        return np.asarray(
            values + self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon),
            dtype=np.float64,
        )
