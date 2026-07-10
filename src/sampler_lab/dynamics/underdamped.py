"""Underdamped Langevin dynamics, exact OU refreshment, and splittings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import DifferentiableLogDensity
from sampler_lab.core.results import Transition
from sampler_lab.dynamics.hamiltonian import HamiltonianSystem, MassMatrix, PhaseSpaceState
from sampler_lab.dynamics.hmc import PersistentHamiltonianKernel, partially_refresh_momentum
from sampler_lab.dynamics.integrators import PositionMap

Array = NDArray[np.float64]


def ornstein_uhlenbeck_persistence(friction: float, duration: float) -> float:
    """Return ``exp(-friction * duration)`` for the exact momentum OU flow."""

    if not np.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be nonnegative and finite")
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be nonnegative and finite")
    return float(np.exp(-friction * duration))


def ornstein_uhlenbeck_momentum_step(
    momentum: ArrayLike,
    mass: MassMatrix,
    rng: np.random.Generator,
    *,
    friction: float,
    duration: float,
    counter: OperationCounter | None = None,
) -> Array:
    """Advance the exact OU momentum subflow preserving ``N(0, M)``."""

    persistence = ornstein_uhlenbeck_persistence(friction, duration)
    return partially_refresh_momentum(
        momentum,
        mass,
        rng,
        persistence=persistence,
        counter=counter,
    )


@dataclass(frozen=True, slots=True)
class UnderdampedGeneratorComponents:
    """Hamiltonian (skew), OU (symmetric), and total generator values."""

    hamiltonian: float
    ornstein_uhlenbeck: float
    total: float


def underdamped_generator_value(
    target: DifferentiableLogDensity,
    state: PhaseSpaceState,
    gradient_position: ArrayLike,
    gradient_momentum: ArrayLike,
    hessian_momentum: ArrayLike,
    *,
    mass: MassMatrix,
    friction: float,
) -> UnderdampedGeneratorComponents:
    """Evaluate the underdamped generator on supplied derivatives of a test function.

    For ``dq = M^{-1}p dt`` and
    ``dp = grad(log pi(q)) dt - gamma p dt + sqrt(2 gamma M) dW``, the
    Hamiltonian contribution is skew-adjoint and the OU contribution is symmetric
    in the canonical phase-space law.
    """

    if state.dimension != mass.dimension:
        raise ValueError("phase-state dimension does not match the mass matrix")
    if not np.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be nonnegative and finite")
    grad_q = np.asarray(gradient_position, dtype=np.float64)
    grad_p = np.asarray(gradient_momentum, dtype=np.float64)
    hess_p = np.asarray(hessian_momentum, dtype=np.float64)
    dimension = state.dimension
    if grad_q.shape != (dimension,) or grad_p.shape != (dimension,):
        raise ValueError("test-function gradients must match phase dimension")
    if hess_p.shape != (dimension, dimension):
        raise ValueError("momentum Hessian must be square with phase dimension")
    if not np.all(np.isfinite(grad_q)) or not np.all(np.isfinite(grad_p)):
        raise ValueError("test-function gradients must be finite")
    if not np.all(np.isfinite(hess_p)):
        raise ValueError("momentum Hessian must be finite")
    target_gradient = np.asarray(target.grad_log_prob(state.position), dtype=np.float64)
    if target_gradient.shape != (dimension,) or not np.all(np.isfinite(target_gradient)):
        raise ValueError("target gradient must be finite and match position")
    hamiltonian = float(mass.velocity(state.momentum) @ grad_q + target_gradient @ grad_p)
    ou = float(-friction * state.momentum @ grad_p + friction * np.trace(mass.covariance @ hess_p))
    return UnderdampedGeneratorComponents(
        hamiltonian=hamiltonian,
        ornstein_uhlenbeck=ou,
        total=hamiltonian + ou,
    )


@dataclass(slots=True)
class UnderdampedLangevinKernel:
    """BAOAB splitting for underdamped Langevin dynamics on phase space."""

    target: DifferentiableLogDensity
    step_size: float
    friction: float
    mass_matrix: MassMatrix | ArrayLike | None = None
    position_map: PositionMap | None = None
    counter: OperationCounter | None = None
    _mass: MassMatrix | None = field(default=None, init=False, repr=False)
    _dimension: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if not np.isfinite(self.friction) or self.friction < 0.0:
            raise ValueError("friction must be nonnegative and finite")
        if isinstance(self.mass_matrix, MassMatrix):
            self._mass = self.mass_matrix
            self._dimension = self.mass_matrix.dimension
        elif self.mass_matrix is not None:
            self._mass = MassMatrix(self.mass_matrix)
            self._dimension = self._mass.dimension

    def _system_for(self, dimension: int) -> HamiltonianSystem:
        if self._mass is None:
            self._mass = MassMatrix.identity(dimension)
            self._dimension = dimension
        elif self._dimension != dimension:
            raise ValueError("phase-state dimension does not match the mass matrix")
        return HamiltonianSystem(self.target, self._mass, counter=self.counter)

    def _map_position(self, position: Array) -> Array:
        if self.position_map is None:
            return np.asarray(position, dtype=np.float64)
        mapped = np.asarray(self.position_map(np.array(position, copy=True)), dtype=np.float64)
        if mapped.shape != position.shape or not np.all(np.isfinite(mapped)):
            raise ValueError("position_map must return a finite vector with unchanged shape")
        return mapped

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = PhaseSpaceState.from_array(state)
        system = self._system_for(current.dimension)
        initial_energy = system.energy(current)
        if not np.isfinite(initial_energy):
            raise ValueError("current phase state must have finite Hamiltonian energy")

        h = self.step_size
        q = np.array(current.position, dtype=np.float64, copy=True)
        p = np.array(current.momentum, dtype=np.float64, copy=True)
        p += 0.5 * h * system.log_density_gradient(q)
        q = self._map_position(q + 0.5 * h * system.mass.velocity(p))
        p = ornstein_uhlenbeck_momentum_step(
            p,
            system.mass,
            rng,
            friction=self.friction,
            duration=h,
            counter=self.counter,
        )
        q = self._map_position(q + 0.5 * h * system.mass.velocity(p))
        p += 0.5 * h * system.log_density_gradient(q)
        final = PhaseSpaceState(q, p)
        final_energy = system.energy(final)
        return Transition(
            state=final.as_array(),
            diagnostics={
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_change": final_energy - initial_energy,
                "friction": self.friction,
                "ou_persistence": ornstein_uhlenbeck_persistence(self.friction, h),
            },
        )


@dataclass(slots=True)
class MetropolizedUnderdampedLangevinKernel:
    """OU momentum refresh followed by generalized HMC correction.

    Accepted proposals retain the forward leapfrog momentum. Rejections transform
    the refreshed state by momentum flip. This transformed rejection is essential:
    simply retaining momentum after a failed persistent trajectory does not satisfy
    the generalized detailed-balance construction.
    """

    target: DifferentiableLogDensity
    step_size: float
    friction: float
    n_leapfrog_steps: int = 1
    mass_matrix: MassMatrix | ArrayLike | None = None
    position_map: PositionMap | None = None
    counter: OperationCounter | None = None
    _kernel: PersistentHamiltonianKernel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        persistence = ornstein_uhlenbeck_persistence(self.friction, self.step_size)
        self._kernel = PersistentHamiltonianKernel(
            target=self.target,
            step_size=self.step_size,
            n_leapfrog_steps=self.n_leapfrog_steps,
            mass_matrix=self.mass_matrix,
            position_map=self.position_map,
            counter=self.counter,
            momentum_persistence=persistence,
        )

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        return self._kernel.step(state, rng)
