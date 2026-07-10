"""Generalized Metropolis correction based on deterministic involutions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import LogDensity
from sampler_lab.core.results import Transition

Array = NDArray[np.float64]


def _as_state(value: ArrayLike, *, name: str = "state") -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


class Involution(Protocol):
    """Deterministic proposal map equal to its own inverse."""

    def apply(self, state: Array) -> Array:
        """Apply the involution."""

    def log_abs_det_jacobian(self, state: Array) -> float:
        """Return the logarithm of the absolute Jacobian determinant."""


@dataclass(frozen=True, slots=True)
class FunctionalInvolution:
    """Involution supplied by callables."""

    function: Callable[[Array], ArrayLike]
    log_abs_det_jacobian_function: Callable[[Array], float] = lambda _state: 0.0

    def apply(self, state: Array) -> Array:
        current = _as_state(state)
        proposed = np.asarray(self.function(np.array(current, copy=True)), dtype=np.float64)
        if proposed.shape != current.shape or not np.all(np.isfinite(proposed)):
            raise ValueError("involution must return a finite vector with unchanged shape")
        return proposed

    def log_abs_det_jacobian(self, state: Array) -> float:
        current = _as_state(state)
        value = float(self.log_abs_det_jacobian_function(np.array(current, copy=True)))
        if not np.isfinite(value):
            raise ValueError("involution log Jacobian must be finite")
        return value


@dataclass(frozen=True, slots=True)
class MomentumFlipInvolution:
    """Canonical momentum flip ``(q,p) -> (q,-p)``."""

    def apply(self, state: Array) -> Array:
        current = _as_state(state)
        if current.size % 2:
            raise ValueError("phase state must have even length")
        dimension = current.size // 2
        return np.concatenate((current[:dimension], -current[dimension:])).astype(
            np.float64,
            copy=False,
        )

    def log_abs_det_jacobian(self, state: Array) -> float:
        current = _as_state(state)
        if current.size % 2:
            raise ValueError("phase state must have even length")
        return 0.0


def involution_error(involution: Involution, state: ArrayLike) -> float:
    """Euclidean residual of applying an alleged involution twice."""

    current = _as_state(state)
    once = involution.apply(np.array(current, copy=True))
    twice = involution.apply(np.array(once, copy=True))
    return float(np.linalg.norm(twice - current))


def log_involutive_metropolis_ratio(
    *,
    current_log_density: float,
    proposed_log_density: float,
    log_abs_det_jacobian: float,
) -> float:
    """Log acceptance ratio for a deterministic involutive proposal."""

    if not np.isfinite(current_log_density):
        raise ValueError("current state must have finite target log density")
    if np.isnan(proposed_log_density) or proposed_log_density == float("inf"):
        raise ValueError("proposed log density must be finite or -inf")
    if not np.isfinite(log_abs_det_jacobian):
        raise ValueError("log_abs_det_jacobian must be finite")
    if proposed_log_density == float("-inf"):
        return float("-inf")
    return float(proposed_log_density - current_log_density + log_abs_det_jacobian)


@dataclass(slots=True)
class InvolutiveMetropolisKernel:
    """Metropolis correction for a deterministic involution.

    Correctness requires the supplied map to be an involution.  The Jacobian term
    permits non-volume-preserving involutions; volume-preserving reflections and
    momentum-flipped leapfrog maps simply return zero.
    """

    target: LogDensity
    involution: Involution
    counter: OperationCounter | None = None

    def _log_density(self, state: Array) -> float:
        if self.counter is not None:
            self.counter.log_density_evaluations += 1
        return float(self.target.log_prob(np.array(state, copy=True)))

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = _as_state(state)
        current_log_density = self._log_density(current)
        proposed = self.involution.apply(np.array(current, copy=True))
        if proposed.shape != current.shape or not np.all(np.isfinite(proposed)):
            raise ValueError("involution changed shape or returned nonfinite values")
        proposed_log_density = self._log_density(proposed)
        log_jacobian = float(self.involution.log_abs_det_jacobian(current))
        log_ratio = log_involutive_metropolis_ratio(
            current_log_density=current_log_density,
            proposed_log_density=proposed_log_density,
            log_abs_det_jacobian=log_jacobian,
        )
        if self.counter is not None:
            self.counter.uniform_draws += 1
        accepted = bool(np.log(float(rng.random())) < min(0.0, log_ratio))
        next_state = proposed if accepted else current
        return Transition(
            state=np.array(next_state, dtype=np.float64, copy=True),
            accepted=accepted,
            log_acceptance_ratio=log_ratio,
            diagnostics={
                "current_log_density": current_log_density,
                "proposed_log_density": proposed_log_density,
                "log_abs_det_jacobian": log_jacobian,
            },
        )
