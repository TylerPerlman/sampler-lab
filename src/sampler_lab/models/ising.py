"""Finite two-dimensional Ising targets and local MCMC kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.results import Transition
from sampler_lab.markov import FiniteStateMarkovChain
from sampler_lab.mcmc.gibbs import (
    DeterministicSweepGibbsKernel,
    RandomScanGibbsKernel,
)
from sampler_lab.mcmc.proposals import Array

IntArray = NDArray[np.int64]


def _logsumexp(values: Array) -> float:
    maximum = float(np.max(values))
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))


def _validate_site(site: tuple[int, int], length: int) -> tuple[int, int]:
    row, column = site
    if isinstance(row, bool) or isinstance(column, bool):
        raise TypeError("site coordinates must be integers")
    if not isinstance(row, (int, np.integer)) or not isinstance(column, (int, np.integer)):
        raise TypeError("site coordinates must be integers")
    row_index = int(row)
    column_index = int(column)
    if not 0 <= row_index < length or not 0 <= column_index < length:
        raise IndexError("site lies outside the lattice")
    return row_index, column_index


@dataclass(frozen=True, slots=True, init=False)
class IsingModel:
    """Periodic square-lattice Ising model.

    The unnormalized log density is

    ``beta * (coupling * sum_<i,j> s_i s_j + field * sum_i s_i)``.

    Each undirected nearest-neighbor bond is counted once via right and down
    neighbors. On a two-site periodic axis this retains the natural torus
    multiplicity, so every site still has four incident bond directions.
    """

    length: int
    beta: float
    coupling: float
    field: float

    def __init__(
        self,
        length: int,
        beta: float,
        *,
        coupling: float = 1.0,
        field: float = 0.0,
    ) -> None:
        if isinstance(length, bool) or not isinstance(length, int) or length < 2:
            raise ValueError("length must be an integer at least two")
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and nonnegative")
        if not np.isfinite(coupling):
            raise ValueError("coupling must be finite")
        if not np.isfinite(field):
            raise ValueError("field must be finite")
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "beta", float(beta))
        object.__setattr__(self, "coupling", float(coupling))
        object.__setattr__(self, "field", float(field))

    @property
    def n_sites(self) -> int:
        return self.length * self.length

    def validate_state(self, state: ArrayLike) -> Array:
        spins = np.asarray(state, dtype=np.float64)
        if spins.shape != (self.length, self.length):
            raise ValueError(f"state must have shape {(self.length, self.length)}")
        if not np.all((spins == -1.0) | (spins == 1.0)):
            raise ValueError("Ising spins must be exactly -1 or +1")
        return spins

    def interaction_sum(self, state: ArrayLike) -> float:
        spins = self.validate_state(state)
        horizontal = np.sum(spins * np.roll(spins, -1, axis=1))
        vertical = np.sum(spins * np.roll(spins, -1, axis=0))
        return float(horizontal + vertical)

    def energy(self, state: ArrayLike) -> float:
        spins = self.validate_state(state)
        return float(-self.coupling * self.interaction_sum(spins) - self.field * np.sum(spins))

    def log_prob(self, state: Array) -> float:
        """Return the unnormalized target log density."""

        return float(-self.beta * self.energy(state))

    def log_prob_batch(self, states: ArrayLike) -> Array:
        """Evaluate the unnormalized log density for a particle population."""

        spins = np.asarray(states, dtype=np.float64)
        if spins.ndim != 3 or spins.shape[1:] != (self.length, self.length):
            raise ValueError(f"states must have shape (n_particles, {self.length}, {self.length})")
        if spins.shape[0] == 0:
            raise ValueError("states must contain at least one particle")
        if not np.all((spins == -1.0) | (spins == 1.0)):
            raise ValueError("Ising spins must be exactly -1 or +1")
        horizontal = np.sum(spins * np.roll(spins, -1, axis=2), axis=(1, 2))
        vertical = np.sum(spins * np.roll(spins, -1, axis=1), axis=(1, 2))
        interaction = horizontal + vertical
        magnetization = np.sum(spins, axis=(1, 2))
        return np.asarray(
            self.beta * (self.coupling * interaction + self.field * magnetization),
            dtype=np.float64,
        )

    def magnetization(self, state: ArrayLike, *, normalized: bool = False) -> float:
        spins = self.validate_state(state)
        total = float(np.sum(spins))
        return total / self.n_sites if normalized else total

    def local_field(self, state: ArrayLike, site: tuple[int, int]) -> float:
        spins = self.validate_state(state)
        row, column = _validate_site(site, self.length)
        neighbors = (
            spins[(row - 1) % self.length, column]
            + spins[(row + 1) % self.length, column]
            + spins[row, (column - 1) % self.length]
            + spins[row, (column + 1) % self.length]
        )
        return float(self.coupling * neighbors + self.field)

    def conditional_plus_probability(
        self,
        state: ArrayLike,
        site: tuple[int, int],
    ) -> float:
        """Return ``P(s_site = +1 | all other spins)`` stably."""

        argument = 2.0 * self.beta * self.local_field(state, site)
        if argument >= 0.0:
            return float(1.0 / (1.0 + np.exp(-argument)))
        exponential = float(np.exp(argument))
        return exponential / (1.0 + exponential)

    def flip(self, state: ArrayLike, site: tuple[int, int]) -> Array:
        spins = np.array(self.validate_state(state), dtype=np.float64, copy=True)
        row, column = _validate_site(site, self.length)
        spins[row, column] *= -1.0
        return spins

    def log_flip_ratio(self, state: ArrayLike, site: tuple[int, int]) -> float:
        """Return ``log pi(flipped) - log pi(current)`` using only neighbors."""

        spins = self.validate_state(state)
        row, column = _validate_site(site, self.length)
        return float(-2.0 * self.beta * spins[row, column] * self.local_field(spins, site))

    def random_state(self, rng: np.random.Generator) -> Array:
        return np.asarray(
            2 * rng.integers(0, 2, size=(self.length, self.length)) - 1,
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class IsingExactDistribution:
    """Enumerated finite-lattice Ising law."""

    states: Array
    probabilities: Array
    log_partition: float
    energies: Array
    magnetizations: Array

    def expectation(self, values: ArrayLike) -> float:
        observable = np.asarray(values, dtype=np.float64)
        if observable.shape != self.probabilities.shape:
            raise ValueError("observable must have one value per enumerated state")
        return float(np.dot(self.probabilities, observable))


def enumerate_ising_states(length: int, *, max_sites: int = 20) -> Array:
    """Enumerate spin states in little-endian row-major bit order."""

    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer at least two")
    n_sites = length * length
    if n_sites > max_sites:
        raise ValueError(f"exact enumeration is limited to at most {max_sites} sites")
    indices = np.arange(1 << n_sites, dtype=np.uint64)
    bit_positions = np.arange(n_sites, dtype=np.uint64)
    bits = ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.float64)
    return np.asarray((2.0 * bits - 1.0).reshape(-1, length, length), dtype=np.float64)


def ising_state_index(state: ArrayLike) -> int:
    spins = np.asarray(state, dtype=np.float64)
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1]:
        raise ValueError("state must be a square spin lattice")
    if not np.all((spins == -1.0) | (spins == 1.0)):
        raise ValueError("Ising spins must be exactly -1 or +1")
    bits = (spins.reshape(-1) > 0.0).astype(np.uint64)
    powers = np.left_shift(np.uint64(1), np.arange(bits.size, dtype=np.uint64))
    return int(np.sum(bits * powers, dtype=np.uint64))


def exact_ising_distribution(
    model: IsingModel,
    *,
    max_sites: int = 20,
) -> IsingExactDistribution:
    states = enumerate_ising_states(model.length, max_sites=max_sites)
    energies = np.asarray([model.energy(state) for state in states], dtype=np.float64)
    log_weights = -model.beta * energies
    log_partition = _logsumexp(log_weights)
    probabilities = np.exp(log_weights - log_partition)
    magnetizations = np.asarray(np.sum(states, axis=(1, 2), dtype=np.float64), dtype=np.float64)
    states.setflags(write=False)
    energies.setflags(write=False)
    probabilities.setflags(write=False)
    magnetizations.setflags(write=False)
    return IsingExactDistribution(
        states=states,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        log_partition=log_partition,
        energies=energies,
        magnetizations=np.asarray(magnetizations, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class IsingSiteGibbsUpdate:
    """Exact conditional resampling of one Ising spin."""

    model: IsingModel
    site: tuple[int, int]

    def apply(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        spins = np.array(self.model.validate_state(state), dtype=np.float64, copy=True)
        row, column = _validate_site(self.site, self.model.length)
        probability_plus = self.model.conditional_plus_probability(spins, (row, column))
        spins[row, column] = 1.0 if float(rng.random()) < probability_plus else -1.0
        if counter is not None:
            counter.uniform_draws += 1
            counter.conditional_draws += 1
            counter.spin_updates += 1
        return spins


@dataclass(slots=True)
class RandomScanIsingMetropolisKernel:
    """Uniform single-spin flip Metropolis kernel using a local ratio."""

    model: IsingModel
    counter: OperationCounter | None = None

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        spins = self.model.validate_state(state)
        flat_site = int(rng.integers(0, self.model.n_sites))
        site = divmod(flat_site, self.model.length)
        log_ratio = self.model.log_flip_ratio(spins, site)
        accepted = bool(np.log(float(rng.random())) < min(0.0, log_ratio))
        next_state = self.model.flip(spins, site) if accepted else np.array(spins, copy=True)
        if self.counter is not None:
            self.counter.uniform_draws += 2
            self.counter.spin_updates += 1
            self.counter.extra["local_energy_differences"] = (
                self.counter.extra.get("local_energy_differences", 0) + 1
            )
        return Transition(
            state=next_state,
            accepted=accepted,
            log_acceptance_ratio=log_ratio,
            diagnostics={"site": float(flat_site)},
        )


def ising_site_updates(model: IsingModel) -> tuple[IsingSiteGibbsUpdate, ...]:
    return tuple(
        IsingSiteGibbsUpdate(model, (row, column))
        for row in range(model.length)
        for column in range(model.length)
    )


def random_scan_ising_gibbs(
    model: IsingModel,
    *,
    counter: OperationCounter | None = None,
) -> RandomScanGibbsKernel:
    return RandomScanGibbsKernel(ising_site_updates(model), counter=counter)


def deterministic_sweep_ising_gibbs(
    model: IsingModel,
    *,
    counter: OperationCounter | None = None,
) -> DeterministicSweepGibbsKernel:
    return DeterministicSweepGibbsKernel(ising_site_updates(model), counter=counter)


def ising_random_scan_gibbs_transition(
    model: IsingModel,
    *,
    max_sites: int = 10,
) -> FiniteStateMarkovChain:
    """Construct the exact random-scan single-site Gibbs transition matrix."""

    states = enumerate_ising_states(model.length, max_sites=max_sites)
    n_states = states.shape[0]
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    site_weight = 1.0 / model.n_sites
    for source, state in enumerate(states):
        for flat_site in range(model.n_sites):
            site = divmod(flat_site, model.length)
            probability_plus = model.conditional_plus_probability(state, site)
            plus_state = np.array(state, copy=True)
            minus_state = np.array(state, copy=True)
            plus_state[site] = 1.0
            minus_state[site] = -1.0
            transition[source, ising_state_index(plus_state)] += site_weight * probability_plus
            transition[source, ising_state_index(minus_state)] += site_weight * (
                1.0 - probability_plus
            )
    return FiniteStateMarkovChain(transition)


def ising_random_scan_metropolis_transition(
    model: IsingModel,
    *,
    max_sites: int = 10,
) -> FiniteStateMarkovChain:
    """Construct the exact uniform single-spin Metropolis transition matrix."""

    states = enumerate_ising_states(model.length, max_sites=max_sites)
    n_states = states.shape[0]
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    site_weight = 1.0 / model.n_sites
    for source, state in enumerate(states):
        for flat_site in range(model.n_sites):
            site = divmod(flat_site, model.length)
            acceptance = min(1.0, float(np.exp(min(0.0, model.log_flip_ratio(state, site)))))
            target = ising_state_index(model.flip(state, site))
            transition[source, target] += site_weight * acceptance
            transition[source, source] += site_weight * (1.0 - acceptance)
    return FiniteStateMarkovChain(transition)


def ising_deterministic_sweep_gibbs_transition(
    model: IsingModel,
    *,
    max_sites: int = 10,
) -> FiniteStateMarkovChain:
    """Construct one exact lexicographic Gibbs sweep without dense matrix products."""

    states = enumerate_ising_states(model.length, max_sites=max_sites)
    n_states = states.shape[0]
    composed = np.eye(n_states, dtype=np.float64)
    for flat_site in range(model.n_sites):
        site = divmod(flat_site, model.length)
        updated = np.zeros_like(composed)
        for intermediate, state in enumerate(states):
            probability_plus = model.conditional_plus_probability(state, site)
            plus_state = np.array(state, copy=True)
            minus_state = np.array(state, copy=True)
            plus_state[site] = 1.0
            minus_state[site] = -1.0
            plus_index = ising_state_index(plus_state)
            minus_index = ising_state_index(minus_state)
            updated[:, plus_index] += composed[:, intermediate] * probability_plus
            updated[:, minus_index] += composed[:, intermediate] * (1.0 - probability_plus)
        composed = updated
    return FiniteStateMarkovChain(composed)


@dataclass(frozen=True, slots=True, init=False)
class IsingGibbsPopulationTransition:
    """Vectorized deterministic Gibbs sweeps for an annealed Ising population.

    ``path_beta`` is mapped linearly from ``initial_beta`` to ``final_beta``.
    Every site update is vectorized across particles while sites themselves are
    visited in lexicographic order, matching deterministic-sweep Gibbs.
    """

    length: int
    initial_beta: float
    final_beta: float
    coupling: float
    field: float
    n_sweeps: int

    def __init__(
        self,
        length: int,
        final_beta: float,
        *,
        initial_beta: float = 0.0,
        coupling: float = 1.0,
        field: float = 0.0,
        n_sweeps: int = 1,
    ) -> None:
        # Reuse IsingModel validation for physical parameters.
        IsingModel(length, initial_beta, coupling=coupling, field=field)
        IsingModel(length, final_beta, coupling=coupling, field=field)
        if isinstance(n_sweeps, bool) or not isinstance(n_sweeps, int):
            raise TypeError("n_sweeps must be an integer")
        if n_sweeps < 0:
            raise ValueError("n_sweeps must be nonnegative")
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "initial_beta", float(initial_beta))
        object.__setattr__(self, "final_beta", float(final_beta))
        object.__setattr__(self, "coupling", float(coupling))
        object.__setattr__(self, "field", float(field))
        object.__setattr__(self, "n_sweeps", n_sweeps)

    def move(
        self,
        particles: Array,
        beta: float,
        rng: np.random.Generator,
    ) -> Array:
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("path beta must lie in [0, 1]")
        states = np.asarray(particles, dtype=np.float64)
        if (
            states.ndim != 3
            or states.shape[0] == 0
            or states.shape[1:] != (self.length, self.length)
        ):
            raise ValueError(
                f"particles must have shape (n_particles, {self.length}, {self.length})"
            )
        if not np.all((states == -1.0) | (states == 1.0)):
            raise ValueError("Ising spins must be exactly -1 or +1")
        physical_beta = self.initial_beta + float(beta) * (self.final_beta - self.initial_beta)
        moved = np.array(states, dtype=np.float64, copy=True)
        for _ in range(self.n_sweeps):
            for row in range(self.length):
                for column in range(self.length):
                    neighbors = (
                        moved[:, (row - 1) % self.length, column]
                        + moved[:, (row + 1) % self.length, column]
                        + moved[:, row, (column - 1) % self.length]
                        + moved[:, row, (column + 1) % self.length]
                    )
                    argument = 2.0 * physical_beta * (self.coupling * neighbors + self.field)
                    probabilities = np.empty_like(argument)
                    nonnegative = argument >= 0.0
                    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-argument[nonnegative]))
                    exponential = np.exp(argument[~nonnegative])
                    probabilities[~nonnegative] = exponential / (1.0 + exponential)
                    moved[:, row, column] = np.where(
                        rng.random(states.shape[0]) < probabilities,
                        1.0,
                        -1.0,
                    )
        return moved
