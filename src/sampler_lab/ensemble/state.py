"""Immutable ensemble states, transitions, and trajectory runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.protocols import LogDensity

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True, init=False)
class EnsembleState:
    """A complete Markov state containing every walker and cached log density."""

    walkers: Array
    log_probabilities: Array

    def __init__(self, walkers: ArrayLike, log_probabilities: ArrayLike) -> None:
        positions = np.asarray(walkers, dtype=np.float64)
        logs = np.asarray(log_probabilities, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] == 0:
            raise ValueError("walkers must have shape (n_walkers >= 2, dimension >= 1)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("walker positions must be finite")
        if logs.shape != (positions.shape[0],):
            raise ValueError("log_probabilities must contain one value per walker")
        if not np.all(np.isfinite(logs)):
            raise ValueError("all walkers must lie in finite target support")
        copied_positions = np.array(positions, dtype=np.float64, copy=True)
        copied_logs = np.array(logs, dtype=np.float64, copy=True)
        copied_positions.setflags(write=False)
        copied_logs.setflags(write=False)
        object.__setattr__(self, "walkers", copied_positions)
        object.__setattr__(self, "log_probabilities", copied_logs)

    @classmethod
    def from_target(cls, walkers: ArrayLike, target: LogDensity) -> EnsembleState:
        """Construct a state and evaluate the product target once per walker."""

        positions = np.asarray(walkers, dtype=np.float64)
        if positions.ndim != 2:
            raise ValueError("walkers must be a two-dimensional array")
        logs = np.asarray([target.log_prob(row) for row in positions], dtype=np.float64)
        return cls(positions, logs)

    @property
    def n_walkers(self) -> int:
        return int(self.walkers.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.walkers.shape[1])

    @property
    def affine_span_rank(self) -> int:
        """Rank of centered walker differences."""

        centered = self.walkers - np.mean(self.walkers, axis=0)
        return int(np.linalg.matrix_rank(centered))

    @property
    def has_full_affine_span(self) -> bool:
        return self.affine_span_rank == self.dimension


@dataclass(frozen=True, slots=True, init=False)
class EnsembleTransition:
    """One whole-ensemble transition with per-walker acceptance information."""

    state: EnsembleState
    accepted: BoolArray
    log_acceptance_ratios: Array
    partner_indices: IntArray
    diagnostics: dict[str, float]

    def __init__(
        self,
        state: EnsembleState,
        accepted: ArrayLike,
        log_acceptance_ratios: ArrayLike,
        partner_indices: ArrayLike,
        diagnostics: dict[str, float] | None = None,
    ) -> None:
        accepted_array = np.asarray(accepted, dtype=np.bool_)
        ratios = np.asarray(log_acceptance_ratios, dtype=np.float64)
        partners = np.asarray(partner_indices, dtype=np.int64)
        expected = (state.n_walkers,)
        if (
            accepted_array.shape != expected
            or ratios.shape != expected
            or partners.shape != expected
        ):
            raise ValueError("transition arrays must contain one entry per walker")
        copied_accepted = np.array(accepted_array, dtype=np.bool_, copy=True)
        copied_ratios = np.array(ratios, dtype=np.float64, copy=True)
        copied_partners = np.array(partners, dtype=np.int64, copy=True)
        copied_accepted.setflags(write=False)
        copied_ratios.setflags(write=False)
        copied_partners.setflags(write=False)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "accepted", copied_accepted)
        object.__setattr__(self, "log_acceptance_ratios", copied_ratios)
        object.__setattr__(self, "partner_indices", copied_partners)
        object.__setattr__(self, "diagnostics", dict(diagnostics or {}))

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted))


class EnsembleKernel(Protocol):
    """One Markov transition on the full ensemble product state."""

    def step(self, state: EnsembleState, rng: np.random.Generator) -> EnsembleTransition:
        """Advance every walker according to the kernel schedule."""


@dataclass(frozen=True, slots=True, init=False)
class EnsembleTrajectory:
    """Time series of complete ensemble states."""

    walkers: Array
    log_probabilities: Array
    accepted: BoolArray
    log_acceptance_ratios: Array
    partner_indices: IntArray
    transition_diagnostics: tuple[dict[str, float], ...]

    def __init__(
        self,
        walkers: ArrayLike,
        log_probabilities: ArrayLike,
        accepted: ArrayLike,
        log_acceptance_ratios: ArrayLike,
        partner_indices: ArrayLike,
        transition_diagnostics: tuple[dict[str, float], ...],
    ) -> None:
        positions = np.asarray(walkers, dtype=np.float64)
        logs = np.asarray(log_probabilities, dtype=np.float64)
        accepts = np.asarray(accepted, dtype=np.bool_)
        ratios = np.asarray(log_acceptance_ratios, dtype=np.float64)
        partners = np.asarray(partner_indices, dtype=np.int64)
        if positions.ndim != 3 or positions.shape[0] == 0:
            raise ValueError("walkers must have shape (steps + 1, walkers, dimension)")
        steps = positions.shape[0] - 1
        n_walkers = positions.shape[1]
        if logs.shape != (steps + 1, n_walkers):
            raise ValueError("log_probabilities shape does not match walkers")
        expected = (steps, n_walkers)
        if accepts.shape != expected or ratios.shape != expected or partners.shape != expected:
            raise ValueError("transition arrays have the wrong shape")
        if len(transition_diagnostics) != steps:
            raise ValueError("transition diagnostics must contain one entry per step")
        for name, value in (
            ("walkers", positions),
            ("log_probabilities", logs),
            ("log_acceptance_ratios", ratios),
        ):
            if name != "log_acceptance_ratios" and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        copied_positions = np.array(positions, dtype=np.float64, copy=True)
        copied_logs = np.array(logs, dtype=np.float64, copy=True)
        copied_accepts = np.array(accepts, dtype=np.bool_, copy=True)
        copied_ratios = np.array(ratios, dtype=np.float64, copy=True)
        copied_partners = np.array(partners, dtype=np.int64, copy=True)
        copied_positions.setflags(write=False)
        copied_logs.setflags(write=False)
        copied_accepts.setflags(write=False)
        copied_ratios.setflags(write=False)
        copied_partners.setflags(write=False)
        object.__setattr__(self, "walkers", copied_positions)
        object.__setattr__(self, "log_probabilities", copied_logs)
        object.__setattr__(self, "accepted", copied_accepts)
        object.__setattr__(self, "log_acceptance_ratios", copied_ratios)
        object.__setattr__(self, "partner_indices", copied_partners)
        object.__setattr__(
            self,
            "transition_diagnostics",
            tuple(dict(item) for item in transition_diagnostics),
        )

    @property
    def n_steps(self) -> int:
        return int(self.walkers.shape[0] - 1)

    @property
    def n_walkers(self) -> int:
        return int(self.walkers.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.walkers.shape[2])

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted)) if self.accepted.size else float("nan")

    @property
    def per_walker_acceptance(self) -> Array:
        if self.n_steps == 0:
            return np.full(self.n_walkers, np.nan, dtype=np.float64)
        return np.asarray(np.mean(self.accepted, axis=0), dtype=np.float64)

    def samples(self, *, discard: int = 0, thin: int = 1, flatten: bool = False) -> Array:
        """Return retained ensemble states or their walker-flattened form."""

        if isinstance(discard, bool) or not isinstance(discard, int):
            raise TypeError("discard must be an integer")
        if isinstance(thin, bool) or not isinstance(thin, int):
            raise TypeError("thin must be an integer")
        if discard < 0 or discard >= self.walkers.shape[0]:
            raise ValueError("discard must leave at least one ensemble state")
        if thin <= 0:
            raise ValueError("thin must be positive")
        retained = np.asarray(self.walkers[discard::thin], dtype=np.float64)
        if flatten:
            return retained.reshape(-1, self.dimension)
        return retained


def run_ensemble_chain(
    kernel: EnsembleKernel,
    initial_state: EnsembleState,
    rng: np.random.Generator,
    *,
    n_steps: int,
) -> EnsembleTrajectory:
    """Run a kernel on the ensemble state, preserving every whole-ensemble state."""

    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    current = initial_state
    walkers = np.empty((n_steps + 1, current.n_walkers, current.dimension), dtype=np.float64)
    logs = np.empty((n_steps + 1, current.n_walkers), dtype=np.float64)
    accepted = np.zeros((n_steps, current.n_walkers), dtype=np.bool_)
    ratios = np.full((n_steps, current.n_walkers), np.nan, dtype=np.float64)
    partners = np.full((n_steps, current.n_walkers), -1, dtype=np.int64)
    diagnostics: list[dict[str, float]] = []
    walkers[0] = current.walkers
    logs[0] = current.log_probabilities
    for index in range(n_steps):
        transition = kernel.step(current, rng)
        if (
            transition.state.n_walkers != current.n_walkers
            or transition.state.dimension != current.dimension
        ):
            raise ValueError("ensemble kernel changed the product-state shape")
        current = transition.state
        walkers[index + 1] = current.walkers
        logs[index + 1] = current.log_probabilities
        accepted[index] = transition.accepted
        ratios[index] = transition.log_acceptance_ratios
        partners[index] = transition.partner_indices
        diagnostics.append(dict(transition.diagnostics))
    return EnsembleTrajectory(
        walkers,
        logs,
        accepted,
        ratios,
        partners,
        tuple(diagnostics),
    )
