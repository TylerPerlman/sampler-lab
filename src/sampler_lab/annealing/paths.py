"""Paths of unnormalized distributions used by annealed samplers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.protocols import LogDensity

Array = NDArray[np.float64]


@runtime_checkable
class BatchAnnealingPath(Protocol):
    """Optional vectorized path-density capability."""

    def log_unnormalized_batch(self, states: Array, beta: float) -> Array:
        """Evaluate one path density value per leading-axis state."""


@runtime_checkable
class BatchLogDensity(Protocol):
    """Optional vectorized log-density capability."""

    def log_prob_batch(self, states: Array) -> Array:
        """Evaluate one log density per leading-axis state."""


class AnnealingPath(Protocol):
    """Unnormalized log density indexed by a path parameter in ``[0, 1]``."""

    def log_unnormalized(self, state: Array, beta: float) -> float:
        """Evaluate ``log gamma_beta(state)`` up to a beta-dependent constant."""


@dataclass(frozen=True, slots=True)
class GeometricAnnealingPath:
    """Geometric bridge ``gamma_beta = gamma_0^(1-beta) gamma_1^beta``."""

    initial: LogDensity
    final: LogDensity

    def log_unnormalized(self, state: Array, beta: float) -> float:
        parameter = validate_path_parameter(beta)
        initial_value = float(self.initial.log_prob(state))
        final_value = float(self.final.log_prob(state))
        if (
            np.isnan(initial_value)
            or np.isnan(final_value)
            or np.isposinf(initial_value)
            or np.isposinf(final_value)
        ):
            raise ValueError("endpoint log densities may not return nan or +inf")
        if parameter == 0.0:
            return initial_value
        if parameter == 1.0:
            return final_value
        return (1.0 - parameter) * initial_value + parameter * final_value

    def log_unnormalized_batch(self, states: Array, beta: float) -> Array:
        parameter = validate_path_parameter(beta)
        initial_values = _evaluate_log_density(self.initial, states)
        final_values = _evaluate_log_density(self.final, states)
        if parameter == 0.0:
            return initial_values
        if parameter == 1.0:
            return final_values
        return (1.0 - parameter) * initial_values + parameter * final_values


@dataclass(frozen=True, slots=True)
class FunctionalAnnealingPath:
    """Adapt a scalar function into an annealing path."""

    function: Callable[[Array, float], float]

    def log_unnormalized(self, state: Array, beta: float) -> float:
        parameter = validate_path_parameter(beta)
        value = float(self.function(state, parameter))
        if np.isnan(value) or np.isposinf(value):
            raise ValueError("path log density may not return nan or +inf")
        return value

    def log_unnormalized_batch(self, states: Array, beta: float) -> Array:
        parameter = validate_path_parameter(beta)
        return np.asarray(
            [
                self.log_unnormalized(np.asarray(state, dtype=np.float64), parameter)
                for state in states
            ],
            dtype=np.float64,
        )


def validate_path_parameter(beta: float) -> float:
    """Validate one path parameter."""

    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("path parameters must lie in [0, 1]")
    return float(beta)


def evaluate_path(path: AnnealingPath, particles: ArrayLike, beta: float) -> Array:
    """Evaluate a scalar path density for every leading-axis particle."""

    parameter = validate_path_parameter(beta)
    states = np.asarray(particles, dtype=np.float64)
    if states.ndim < 1 or states.shape[0] == 0:
        raise ValueError("particles must have a nonempty leading particle axis")
    if isinstance(path, BatchAnnealingPath):
        values = np.asarray(path.log_unnormalized_batch(states, parameter), dtype=np.float64)
        if values.shape != (states.shape[0],):
            raise ValueError("batched path evaluation must return one value per particle")
    else:
        values = np.asarray(
            [
                path.log_unnormalized(np.asarray(state, dtype=np.float64), parameter)
                for state in states
            ],
            dtype=np.float64,
        )
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("path log densities may not contain nan or +inf")
    return values


def incremental_log_weights(
    path: AnnealingPath,
    particles: ArrayLike,
    beta_from: float,
    beta_to: float,
) -> Array:
    """Return ``log gamma_beta_to - log gamma_beta_from`` for each particle."""

    start = validate_path_parameter(beta_from)
    stop = validate_path_parameter(beta_to)
    if stop <= start:
        raise ValueError("beta_to must be greater than beta_from")
    increments = evaluate_path(path, particles, stop) - evaluate_path(path, particles, start)
    if np.any(np.isnan(increments)) or np.any(np.isposinf(increments)):
        raise ValueError("path increment is undefined or has infinite positive weight")
    return increments


def _evaluate_log_density(density: LogDensity, states: Array) -> Array:
    if isinstance(density, BatchLogDensity):
        values = np.asarray(density.log_prob_batch(states), dtype=np.float64)
        if values.shape != (states.shape[0],):
            raise ValueError("batched log density must return one value per particle")
    else:
        values = np.asarray(
            [density.log_prob(np.asarray(state, dtype=np.float64)) for state in states],
            dtype=np.float64,
        )
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("endpoint log densities may not contain nan or +inf")
    return values
