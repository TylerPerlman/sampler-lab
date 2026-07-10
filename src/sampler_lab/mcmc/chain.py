"""Chain runners that retain repeated states after rejected proposals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.protocols import MarkovKernel

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True, init=False)
class MCMCTrajectory:
    """Immutable array-valued trajectory and transition-level diagnostics."""

    states: Array
    accepted: BoolArray
    acceptance_observed: BoolArray
    log_acceptance_ratios: Array
    transition_diagnostics: tuple[dict[str, float], ...]

    def __init__(
        self,
        states: ArrayLike,
        accepted: ArrayLike,
        acceptance_observed: ArrayLike,
        log_acceptance_ratios: ArrayLike,
        transition_diagnostics: tuple[dict[str, float], ...],
    ) -> None:
        state_array = np.asarray(states, dtype=np.float64)
        accepted_array = np.asarray(accepted, dtype=np.bool_)
        observed_array = np.asarray(acceptance_observed, dtype=np.bool_)
        ratios = np.asarray(log_acceptance_ratios, dtype=np.float64)
        if state_array.ndim < 2 or state_array.shape[0] == 0:
            raise ValueError("states must contain at least one non-scalar array state")
        n_steps = state_array.shape[0] - 1
        if accepted_array.shape != (n_steps,) or observed_array.shape != (n_steps,):
            raise ValueError("acceptance arrays must have one entry per transition")
        if ratios.shape != (n_steps,):
            raise ValueError("log_acceptance_ratios must have one entry per transition")
        if len(transition_diagnostics) != n_steps:
            raise ValueError("transition diagnostics must have one entry per transition")
        if not np.all(np.isfinite(state_array)):
            raise ValueError("trajectory states must be finite")
        copied_states = np.array(state_array, dtype=np.float64, copy=True)
        copied_states.setflags(write=False)
        copied_accepted = np.array(accepted_array, dtype=np.bool_, copy=True)
        copied_accepted.setflags(write=False)
        copied_observed = np.array(observed_array, dtype=np.bool_, copy=True)
        copied_observed.setflags(write=False)
        copied_ratios = np.array(ratios, dtype=np.float64, copy=True)
        copied_ratios.setflags(write=False)
        object.__setattr__(self, "states", copied_states)
        object.__setattr__(self, "accepted", copied_accepted)
        object.__setattr__(self, "acceptance_observed", copied_observed)
        object.__setattr__(self, "log_acceptance_ratios", copied_ratios)
        object.__setattr__(
            self,
            "transition_diagnostics",
            tuple(dict(values) for values in transition_diagnostics),
        )

    @property
    def n_steps(self) -> int:
        return int(self.states.shape[0] - 1)

    @property
    def acceptance_rate(self) -> float | None:
        """Observed acceptance fraction, or ``None`` for rejection-free kernels."""

        if not np.any(self.acceptance_observed):
            return None
        return float(np.mean(self.accepted[self.acceptance_observed]))

    @property
    def n_rejections(self) -> int:
        return int(np.count_nonzero(self.acceptance_observed & ~self.accepted))

    def samples(self, *, discard: int = 0, thin: int = 1) -> Array:
        """Return retained states after discarding and thinning state indices."""

        if isinstance(discard, bool) or not isinstance(discard, int):
            raise TypeError("discard must be an integer")
        if isinstance(thin, bool) or not isinstance(thin, int):
            raise TypeError("thin must be an integer")
        if discard < 0 or discard >= self.states.shape[0]:
            raise ValueError("discard must leave at least one state")
        if thin <= 0:
            raise ValueError("thin must be positive")
        return np.asarray(self.states[discard::thin], dtype=np.float64)

    def observable_values(
        self,
        observable: Callable[[Array], float],
        *,
        discard: int = 0,
        thin: int = 1,
    ) -> Array:
        """Evaluate a scalar observable on retained states."""

        retained = self.samples(discard=discard, thin=thin)
        values = np.asarray([observable(state) for state in retained], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("observable values must be finite")
        return values


def run_chain(
    kernel: MarkovKernel,
    initial_state: ArrayLike,
    rng: np.random.Generator,
    *,
    n_steps: int,
) -> MCMCTrajectory:
    """Run a Markov kernel and store every state, including rejection repeats."""

    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    current = np.asarray(initial_state, dtype=np.float64)
    if current.ndim == 0 or current.size == 0 or not np.all(np.isfinite(current)):
        raise ValueError("initial_state must be a nonempty finite array")

    states = np.empty((n_steps + 1, *current.shape), dtype=np.float64)
    states[0] = current
    accepted = np.zeros(n_steps, dtype=np.bool_)
    observed = np.zeros(n_steps, dtype=np.bool_)
    ratios = np.full(n_steps, np.nan, dtype=np.float64)
    diagnostics: list[dict[str, float]] = []

    for step_index in range(n_steps):
        transition = kernel.step(current, rng)
        next_state = np.asarray(transition.state, dtype=np.float64)
        if next_state.shape != current.shape:
            raise ValueError("kernel changed the state shape")
        if not np.all(np.isfinite(next_state)):
            raise ValueError("kernel returned a nonfinite state")
        if transition.accepted is not None:
            observed[step_index] = True
            accepted[step_index] = transition.accepted
        if transition.log_acceptance_ratio is not None:
            ratios[step_index] = transition.log_acceptance_ratio
        diagnostics.append(dict(transition.diagnostics))
        states[step_index + 1] = next_state
        current = next_state

    return MCMCTrajectory(
        states,
        accepted,
        observed,
        ratios,
        tuple(diagnostics),
    )
