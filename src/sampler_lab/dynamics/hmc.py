"""Hamiltonian Monte Carlo and persistent-momentum generalized HMC."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import DifferentiableLogDensity
from sampler_lab.core.results import Transition
from sampler_lab.dynamics.hamiltonian import HamiltonianSystem, MassMatrix, PhaseSpaceState
from sampler_lab.dynamics.integrators import (
    PositionMap,
    leapfrog_integrate,
    momentum_flip,
)

Array = NDArray[np.float64]


def _as_position(value: ArrayLike) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("state must be a nonempty finite vector")
    return array


def partially_refresh_momentum(
    momentum: ArrayLike,
    mass: MassMatrix,
    rng: np.random.Generator,
    *,
    persistence: float,
    counter: OperationCounter | None = None,
) -> Array:
    """Apply an exact Gaussian momentum-refreshment kernel.

    ``persistence=0`` is a full refresh and ``persistence=1`` leaves momentum
    unchanged. Every intermediate value preserves ``N(0, M)``.
    """

    p = np.asarray(momentum, dtype=np.float64)
    if p.shape != (mass.dimension,) or not np.all(np.isfinite(p)):
        raise ValueError("momentum must be finite and match the mass dimension")
    if not np.isfinite(persistence) or not 0.0 <= persistence <= 1.0:
        raise ValueError("persistence must lie in [0, 1]")
    if persistence == 1.0:
        return np.array(p, dtype=np.float64, copy=True)
    innovation = mass.sample_momentum(rng, counter=counter)
    scale = float(np.sqrt(max(0.0, 1.0 - persistence * persistence)))
    return np.asarray(persistence * p + scale * innovation, dtype=np.float64)


@dataclass(slots=True)
class HamiltonianPhaseDensity:
    """Phase-space density proportional to ``exp(-H(q,p))``."""

    target: DifferentiableLogDensity
    mass: MassMatrix

    def log_prob(self, state: Array) -> float:
        phase = PhaseSpaceState.from_array(state)
        if phase.dimension != self.mass.dimension:
            raise ValueError("phase-state dimension does not match the mass matrix")
        return float(
            self.target.log_prob(phase.position) - self.mass.kinetic_energy(phase.momentum)
        )


@dataclass(slots=True)
class LeapfrogMomentumFlipInvolution:
    """The involution ``R Phi`` formed from leapfrog and momentum flip."""

    system: HamiltonianSystem
    step_size: float
    n_steps: int
    position_map: PositionMap | None = None

    def apply(self, state: Array) -> Array:
        phase = PhaseSpaceState.from_array(state)
        integrated = leapfrog_integrate(
            self.system,
            phase,
            self.step_size,
            self.n_steps,
            position_map=self.position_map,
        ).final_state
        return momentum_flip(integrated).as_array()

    def log_abs_det_jacobian(self, state: Array) -> float:
        phase = PhaseSpaceState.from_array(state)
        if phase.dimension != self.system.mass.dimension:
            raise ValueError("phase-state dimension does not match the Hamiltonian system")
        return 0.0


@dataclass(slots=True)
class _HamiltonianKernelBase:
    target: DifferentiableLogDensity
    step_size: float
    n_leapfrog_steps: int
    mass_matrix: MassMatrix | ArrayLike | None = None
    position_map: PositionMap | None = None
    counter: OperationCounter | None = None
    _mass: MassMatrix | None = field(default=None, init=False, repr=False)
    _dimension: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if isinstance(self.n_leapfrog_steps, bool) or not isinstance(self.n_leapfrog_steps, int):
            raise TypeError("n_leapfrog_steps must be an integer")
        if self.n_leapfrog_steps <= 0:
            raise ValueError("n_leapfrog_steps must be positive")
        if isinstance(self.mass_matrix, MassMatrix):
            self._mass = self.mass_matrix
            self._dimension = self.mass_matrix.dimension
        elif self.mass_matrix is not None:
            self._mass = MassMatrix(self.mass_matrix)
            self._dimension = self._mass.dimension

    def _system_for_dimension(self, dimension: int) -> HamiltonianSystem:
        if self._mass is None:
            self._mass = MassMatrix.identity(dimension)
            self._dimension = dimension
        elif self._dimension != dimension:
            raise ValueError("state dimension does not match the configured mass matrix")
        return HamiltonianSystem(self.target, self._mass, counter=self.counter)


@dataclass(slots=True)
class UnadjustedHamiltonianKernel(_HamiltonianKernelBase):
    """Fresh-momentum numerical Hamiltonian trajectories without MH correction."""

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = _as_position(state)
        system = self._system_for_dimension(current.size)
        momentum = system.mass.sample_momentum(rng, counter=self.counter)
        result = leapfrog_integrate(
            system,
            PhaseSpaceState(current, momentum),
            self.step_size,
            self.n_leapfrog_steps,
            position_map=self.position_map,
        )
        return Transition(
            state=np.array(result.final_state.position, dtype=np.float64, copy=True),
            diagnostics={
                "initial_energy": result.initial_energy,
                "final_energy": result.final_energy,
                "energy_error": result.energy_error,
                "trajectory_length": self.step_size * self.n_leapfrog_steps,
            },
        )


@dataclass(slots=True)
class HamiltonianMonteCarloKernel(_HamiltonianKernelBase):
    """Position-valued HMC with fresh Gaussian momentum each transition."""

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = _as_position(state)
        system = self._system_for_dimension(current.size)
        momentum = system.mass.sample_momentum(rng, counter=self.counter)
        result = leapfrog_integrate(
            system,
            PhaseSpaceState(current, momentum),
            self.step_size,
            self.n_leapfrog_steps,
            position_map=self.position_map,
        )
        if not np.isfinite(result.initial_energy):
            raise ValueError("current state must have finite Hamiltonian energy")
        log_ratio = float(-result.energy_error)
        if np.isnan(log_ratio):
            raise ValueError("Hamiltonian energy difference is undefined")
        if self.counter is not None:
            self.counter.uniform_draws += 1
        accepted = bool(np.log(float(rng.random())) < min(0.0, log_ratio))
        next_position = result.final_state.position if accepted else current
        return Transition(
            state=np.array(next_position, dtype=np.float64, copy=True),
            accepted=accepted,
            log_acceptance_ratio=log_ratio,
            diagnostics={
                "initial_energy": result.initial_energy,
                "final_energy": result.final_energy,
                "energy_error": result.energy_error,
                "trajectory_length": self.step_size * self.n_leapfrog_steps,
                "squared_jump_distance": float(np.sum((next_position - current) ** 2)),
            },
        )


@dataclass(slots=True)
class PersistentHamiltonianKernel(_HamiltonianKernelBase):
    """Generalized HMC on phase space with partial momentum refreshment.

    The numerical flow is accepted without a terminal momentum flip. On rejection,
    refreshed momentum is flipped. This is the transformed-rejection form obtained
    by applying momentum flip after an involutive ``R Phi`` Metropolis proposal.
    """

    momentum_persistence: float = 0.0

    def __post_init__(self) -> None:
        _HamiltonianKernelBase.__post_init__(self)
        if not np.isfinite(self.momentum_persistence) or not (
            0.0 <= self.momentum_persistence <= 1.0
        ):
            raise ValueError("momentum_persistence must lie in [0, 1]")

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = PhaseSpaceState.from_array(state)
        system = self._system_for_dimension(current.dimension)
        refreshed_momentum = partially_refresh_momentum(
            current.momentum,
            system.mass,
            rng,
            persistence=self.momentum_persistence,
            counter=self.counter,
        )
        refreshed = PhaseSpaceState(current.position, refreshed_momentum)
        result = leapfrog_integrate(
            system,
            refreshed,
            self.step_size,
            self.n_leapfrog_steps,
            position_map=self.position_map,
        )
        if not np.isfinite(result.initial_energy):
            raise ValueError("current phase state must have finite Hamiltonian energy")
        log_ratio = float(-result.energy_error)
        if np.isnan(log_ratio):
            raise ValueError("Hamiltonian energy difference is undefined")
        if self.counter is not None:
            self.counter.uniform_draws += 1
        accepted = bool(np.log(float(rng.random())) < min(0.0, log_ratio))
        next_state = result.final_state if accepted else momentum_flip(refreshed)
        return Transition(
            state=next_state.as_array(),
            accepted=accepted,
            log_acceptance_ratio=log_ratio,
            diagnostics={
                "initial_energy": result.initial_energy,
                "final_energy": result.final_energy,
                "energy_error": result.energy_error,
                "trajectory_length": self.step_size * self.n_leapfrog_steps,
                "momentum_persistence": self.momentum_persistence,
                "rejection_momentum_flipped": float(not accepted),
                "squared_position_jump": float(
                    np.sum((next_state.position - current.position) ** 2)
                ),
            },
        )


def randomized_trajectory_steps(
    rng: np.random.Generator,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Draw an inclusive integer trajectory length for resonance avoidance."""

    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise TypeError("minimum must be an integer")
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise TypeError("maximum must be an integer")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("require 0 < minimum <= maximum")
    return int(rng.integers(minimum, maximum + 1))
