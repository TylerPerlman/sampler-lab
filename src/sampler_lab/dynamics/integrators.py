"""Time-reversible splitting integrators for separable Hamiltonians."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.dynamics.hamiltonian import HamiltonianSystem, PhaseSpaceState

Array = NDArray[np.float64]
PositionMap = Callable[[Array], ArrayLike]


def _apply_position_map(position: Array, mapping: PositionMap | None) -> Array:
    if mapping is None:
        return np.asarray(position, dtype=np.float64)
    mapped = np.asarray(mapping(np.array(position, copy=True)), dtype=np.float64)
    if mapped.shape != position.shape or not np.all(np.isfinite(mapped)):
        raise ValueError("position_map must return a finite vector with unchanged shape")
    return mapped


@dataclass(frozen=True, slots=True)
class LeapfrogResult:
    """Final phase state and Hamiltonian error of one numerical trajectory."""

    initial_state: PhaseSpaceState
    final_state: PhaseSpaceState
    initial_energy: float
    final_energy: float
    energy_error: float
    step_size: float
    n_steps: int


def leapfrog_integrate(
    system: HamiltonianSystem,
    state: PhaseSpaceState,
    step_size: float,
    n_steps: int,
    *,
    position_map: PositionMap | None = None,
) -> LeapfrogResult:
    """Integrate a separable Hamiltonian with velocity Verlet / leapfrog.

    The implementation uses ``n_steps + 1`` target-gradient evaluations by
    reusing the interior half steps. Negative step sizes are allowed so that
    time-reversibility can be tested directly.
    """

    if not np.isfinite(step_size) or step_size == 0.0:
        raise ValueError("step_size must be finite and nonzero")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    if state.dimension != system.mass.dimension:
        raise ValueError("phase-state dimension does not match the Hamiltonian system")

    initial_state = PhaseSpaceState(state.position, state.momentum)
    initial_energy = system.energy(initial_state)
    if n_steps == 0:
        return LeapfrogResult(
            initial_state=initial_state,
            final_state=initial_state,
            initial_energy=initial_energy,
            final_energy=initial_energy,
            energy_error=0.0,
            step_size=float(step_size),
            n_steps=0,
        )

    h = float(step_size)
    q = np.array(state.position, dtype=np.float64, copy=True)
    p = np.array(state.momentum, dtype=np.float64, copy=True)
    p += 0.5 * h * system.log_density_gradient(q)
    for step_index in range(n_steps):
        q += h * system.mass.velocity(p)
        q = _apply_position_map(q, position_map)
        gradient = system.log_density_gradient(q)
        if step_index == n_steps - 1:
            p += 0.5 * h * gradient
        else:
            p += h * gradient

    final_state = PhaseSpaceState(q, p)
    final_energy = system.energy(final_state)
    return LeapfrogResult(
        initial_state=initial_state,
        final_state=final_state,
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_error=float(final_energy - initial_energy),
        step_size=h,
        n_steps=n_steps,
    )


def leapfrog_map(
    system: HamiltonianSystem,
    state: ArrayLike,
    step_size: float,
    n_steps: int,
    *,
    position_map: PositionMap | None = None,
) -> Array:
    """Array-valued wrapper around :func:`leapfrog_integrate`."""

    phase = PhaseSpaceState.from_array(state)
    return leapfrog_integrate(
        system,
        phase,
        step_size,
        n_steps,
        position_map=position_map,
    ).final_state.as_array()


def momentum_flip(state: PhaseSpaceState) -> PhaseSpaceState:
    """Flip momentum while retaining position."""

    return PhaseSpaceState(state.position, -state.momentum)


def leapfrog_reversibility_error(
    system: HamiltonianSystem,
    state: PhaseSpaceState,
    step_size: float,
    n_steps: int,
    *,
    position_map: PositionMap | None = None,
    difference_map: Callable[[Array, Array], ArrayLike] | None = None,
) -> float:
    """Forward/backward integration error in phase-space Euclidean norm.

    ``difference_map`` can supply a periodic difference for toroidal positions.
    """

    forward = leapfrog_integrate(
        system,
        state,
        step_size,
        n_steps,
        position_map=position_map,
    ).final_state
    backward = leapfrog_integrate(
        system,
        forward,
        -step_size,
        n_steps,
        position_map=position_map,
    ).final_state
    if difference_map is None:
        position_difference = backward.position - state.position
    else:
        position_difference = np.asarray(
            difference_map(backward.position, state.position), dtype=np.float64
        )
        if position_difference.shape != state.position.shape:
            raise ValueError("difference_map must return a vector matching position")
    momentum_difference = backward.momentum - state.momentum
    return float(np.linalg.norm(np.concatenate((position_difference, momentum_difference))))
