"""Euler--Maruyama kernels for general Itô diffusions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.results import Transition

Array = NDArray[np.float64]


def _as_vector(state: ArrayLike, *, name: str = "state") -> Array:
    value = np.asarray(state, dtype=np.float64)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _as_diffusion_factor(value: ArrayLike, dimension: int) -> Array:
    factor = np.asarray(value, dtype=np.float64)
    if factor.ndim == 1:
        if factor.shape != (dimension,):
            raise ValueError("diagonal diffusion factor has the wrong dimension")
        factor = np.diag(factor)
    if factor.ndim != 2 or factor.shape[0] != dimension or factor.shape[1] == 0:
        raise ValueError("diffusion factor must have shape (dimension, noise_dimension)")
    if not np.all(np.isfinite(factor)):
        raise ValueError("diffusion factor must be finite")
    return np.asarray(factor, dtype=np.float64)


@dataclass(slots=True)
class EulerMaruyamaKernel:
    """One Euler--Maruyama step for ``dX = b(X) dt + sigma(X) dW``."""

    drift: Callable[[Array], ArrayLike]
    diffusion_factor: Callable[[Array], ArrayLike]
    step_size: float
    counter: OperationCounter | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        """Advance one explicit Euler--Maruyama step."""

        current = _as_vector(state)
        drift_value = np.asarray(self.drift(np.array(current, copy=True)), dtype=np.float64)
        if drift_value.shape != current.shape or not np.all(np.isfinite(drift_value)):
            raise ValueError("drift must return a finite vector matching the state")
        factor = _as_diffusion_factor(
            self.diffusion_factor(np.array(current, copy=True)), current.size
        )
        if self.counter is not None:
            self.counter.normal_draws += factor.shape[1]
            self.counter.increment("drift_evaluations")
            self.counter.increment("diffusion_evaluations")
        noise = rng.normal(size=factor.shape[1])
        next_state = (
            current + self.step_size * drift_value + np.sqrt(self.step_size) * (factor @ noise)
        )
        return Transition(
            state=np.asarray(next_state, dtype=np.float64),
            diagnostics={
                "step_size": float(self.step_size),
                "drift_norm": float(np.linalg.norm(drift_value)),
                "diffusion_frobenius_norm": float(np.linalg.norm(factor)),
            },
        )
