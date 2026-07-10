"""Population transitions used between annealing weight increments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from sampler_lab.core.protocols import MarkovKernel
from sampler_lab.mcmc.proposals import Array


class PopulationTransition(Protocol):
    """Move a particle population using a kernel invariant at one path stage."""

    def move(
        self,
        particles: Array,
        beta: float,
        rng: np.random.Generator,
    ) -> Array:
        """Return one moved state for every input state."""


@dataclass(frozen=True, slots=True)
class IdentityPopulationTransition:
    """Leave every particle unchanged."""

    def move(
        self,
        particles: Array,
        beta: float,
        rng: np.random.Generator,
    ) -> Array:
        del beta, rng
        return np.array(particles, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class FunctionalPopulationTransition:
    """Adapt a population-valued transition function."""

    function: Callable[[Array, float, np.random.Generator], ArrayLike]

    def move(
        self,
        particles: Array,
        beta: float,
        rng: np.random.Generator,
    ) -> Array:
        moved = np.asarray(self.function(particles, beta, rng), dtype=np.float64)
        _validate_moved_particles(particles, moved)
        return np.array(moved, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class KernelPopulationTransition:
    """Apply a path-dependent scalar-state Markov kernel to each particle."""

    kernel_factory: Callable[[float], MarkovKernel]
    n_steps: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int):
            raise TypeError("n_steps must be an integer")
        if self.n_steps < 0:
            raise ValueError("n_steps must be nonnegative")

    def move(
        self,
        particles: Array,
        beta: float,
        rng: np.random.Generator,
    ) -> Array:
        states = np.asarray(particles, dtype=np.float64)
        if states.ndim < 1 or states.shape[0] == 0:
            raise ValueError("particles must have a nonempty leading particle axis")
        moved = np.array(states, dtype=np.float64, copy=True)
        if self.n_steps == 0:
            return moved
        kernel = self.kernel_factory(float(beta))
        for index in range(moved.shape[0]):
            state = np.array(moved[index], dtype=np.float64, copy=True)
            for _ in range(self.n_steps):
                state = np.asarray(kernel.step(state, rng).state, dtype=np.float64)
            moved[index] = state
        _validate_moved_particles(states, moved)
        return moved


def _validate_moved_particles(previous: Array, moved: Array) -> None:
    if moved.shape != previous.shape:
        raise ValueError("population transition changed the particle-array shape")
    if not np.all(np.isfinite(moved)):
        raise ValueError("population transition returned nonfinite particles")
