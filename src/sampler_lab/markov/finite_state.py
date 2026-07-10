"""Exact structural analysis for finite-state Markov chains."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.markov.operators import (
    apply_transition,
    as_observable,
    as_probability_vector,
    constant_projection,
    pushforward_measure,
)

Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]


def validate_transition_matrix(transition: ArrayLike, *, atol: float = 1e-12) -> Array:
    """Validate and copy a row-stochastic transition matrix."""

    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transition probabilities must be finite")
    if np.any(matrix < -atol):
        raise ValueError("transition probabilities must be nonnegative")

    clipped = np.maximum(matrix, 0.0)
    row_sums = np.sum(clipped, axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol, rtol=0.0):
        raise ValueError("each transition row must sum to one")
    normalized = clipped / row_sums[:, None]
    result = np.array(normalized, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _reachability(matrix: Array, tolerance: float) -> NDArray[np.bool_]:
    reachable = np.asarray(matrix > tolerance, dtype=np.bool_)
    np.fill_diagonal(reachable, True)
    for intermediate in range(matrix.shape[0]):
        reachable |= reachable[:, intermediate, None] & reachable[None, intermediate, :]
    return reachable


def _communication_classes(matrix: Array, tolerance: float) -> tuple[tuple[int, ...], ...]:
    reachable = _reachability(matrix, tolerance)
    unassigned = set(range(matrix.shape[0]))
    classes: list[tuple[int, ...]] = []
    while unassigned:
        seed = min(unassigned)
        component = tuple(
            index
            for index in sorted(unassigned)
            if reachable[seed, index] and reachable[index, seed]
        )
        classes.append(component)
        unassigned.difference_update(component)
    return tuple(classes)


def _class_period(matrix: Array, states: tuple[int, ...], tolerance: float) -> int:
    """Compute the gcd of cycle lengths in one strongly connected component."""

    if not states:
        return 0
    local_index = {state: index for index, state in enumerate(states)}
    adjacency: list[list[int]] = []
    for state in states:
        adjacency.append(
            [local_index[target] for target in states if matrix[state, target] > tolerance]
        )

    distances = np.full(len(states), -1, dtype=np.int64)
    distances[0] = 0
    queue = [0]
    cursor = 0
    while cursor < len(queue):
        source = queue[cursor]
        cursor += 1
        for target in adjacency[source]:
            if distances[target] < 0:
                distances[target] = distances[source] + 1
                queue.append(target)

    period = 0
    for source, targets in enumerate(adjacency):
        for target in targets:
            difference = int(distances[source] + 1 - distances[target])
            period = gcd(period, abs(difference))
    return period


def _stationary_on_closed_class(matrix: Array, states: tuple[int, ...]) -> Array:
    indices = np.asarray(states, dtype=np.int64)
    block = matrix[np.ix_(indices, indices)]
    system = np.asarray(block.T - np.eye(len(states)), dtype=np.float64)
    rhs = np.zeros(len(states), dtype=np.float64)
    system[-1] = 1.0
    rhs[-1] = 1.0
    try:
        local = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        local = np.linalg.lstsq(system, rhs, rcond=None)[0]
    local = np.asarray(np.real_if_close(local), dtype=np.float64)
    local[np.abs(local) < 1e-14] = 0.0
    if np.any(local < -1e-10):
        raise ArithmeticError("failed to recover a nonnegative invariant distribution")
    local = np.maximum(local, 0.0)
    local /= np.sum(local)
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    result[indices] = local
    return result


@dataclass(frozen=True, slots=True)
class CommunicationStructure:
    """Communicating classes and recurrent closed classes of a finite chain."""

    classes: tuple[tuple[int, ...], ...]
    closed_classes: tuple[tuple[int, ...], ...]
    periods: tuple[int, ...]

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_closed_classes(self) -> int:
        return len(self.closed_classes)


@dataclass(frozen=True, slots=True)
class SpectralSummary:
    """Exact finite-dimensional spectral diagnostics under an invariant law."""

    eigenvalues: ComplexArray
    absolute_spectral_gap: float
    second_singular_value: float
    singular_value_gap: float
    reversible: bool
    poincare_gap: float | None
    worst_case_iat: float | None


@dataclass(frozen=True, slots=True, init=False)
class FiniteStateMarkovChain:
    """A row-stochastic transition matrix with exact structural operations.

    Functions are column vectors and measures are row vectors. Consequently
    ``P @ f`` applies the transition operator while ``mu @ P`` evolves a
    distribution. The matrix is copied and made read-only at construction.
    """

    transition: Array
    tolerance: float

    def __init__(self, transition: ArrayLike, *, tolerance: float = 1e-12) -> None:
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        object.__setattr__(
            self, "transition", validate_transition_matrix(transition, atol=tolerance)
        )
        object.__setattr__(self, "tolerance", float(tolerance))

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def generator(self) -> Array:
        """Discrete generator ``L = P - I`` used throughout the package."""

        return np.asarray(self.transition - np.eye(self.n_states), dtype=np.float64)

    def apply(self, observable: ArrayLike, *, steps: int = 1) -> Array:
        """Apply ``P**steps`` to an observable."""

        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        values = as_observable(observable, n_states=self.n_states)
        if steps == 0:
            return values
        powered = np.linalg.matrix_power(self.transition, steps)
        return apply_transition(powered, values)

    def pushforward(self, measure: ArrayLike, *, steps: int = 1) -> Array:
        """Evolve a probability measure by ``steps`` transitions."""

        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        probabilities = as_probability_vector(measure, n_states=self.n_states)
        if steps == 0:
            return probabilities
        powered = np.linalg.matrix_power(self.transition, steps)
        return pushforward_measure(probabilities, powered)

    def communication_structure(self) -> CommunicationStructure:
        """Return all communicating classes, closed classes, and class periods."""

        classes = _communication_classes(self.transition, self.tolerance)
        closed: list[tuple[int, ...]] = []
        periods: list[int] = []
        all_states = set(range(self.n_states))
        for component in classes:
            outside = sorted(all_states.difference(component))
            outgoing = (
                0.0 if not outside else float(np.sum(self.transition[np.ix_(component, outside)]))
            )
            if outgoing <= self.tolerance:
                closed.append(component)
            periods.append(_class_period(self.transition, component, self.tolerance))
        return CommunicationStructure(classes, tuple(closed), tuple(periods))

    @property
    def is_irreducible(self) -> bool:
        return self.communication_structure().n_classes == 1

    @property
    def period(self) -> int:
        """Return the common state period of an irreducible chain."""

        structure = self.communication_structure()
        if structure.n_classes != 1:
            raise ValueError("period is only unambiguous for an irreducible chain")
        return structure.periods[0]

    @property
    def is_aperiodic(self) -> bool:
        """Whether every closed communicating class has period one."""

        structure = self.communication_structure()
        period_by_class = dict(zip(structure.classes, structure.periods, strict=True))
        return all(period_by_class[component] == 1 for component in structure.closed_classes)

    @property
    def is_ergodic(self) -> bool:
        """Use the standard finite-state meaning: irreducible and aperiodic."""

        return self.is_irreducible and self.period == 1

    def invariant_distributions(self) -> Array:
        """Return one extreme invariant distribution per closed class.

        Every invariant probability distribution is a convex combination of
        these rows. Transient states receive zero stationary mass.
        """

        structure = self.communication_structure()
        rows = [
            _stationary_on_closed_class(self.transition, component)
            for component in structure.closed_classes
        ]
        return np.asarray(rows, dtype=np.float64)

    def invariant_distribution(self) -> Array:
        """Return the unique invariant distribution, or reject ambiguity."""

        distributions = self.invariant_distributions()
        if distributions.shape[0] != 1:
            raise ValueError("the chain has multiple invariant probability distributions")
        return np.array(distributions[0], dtype=np.float64, copy=True)

    def global_balance_residual(self, probabilities: ArrayLike | None = None) -> float:
        """Return ``max_i |(pi P)_i - pi_i|``."""

        measure = (
            self.invariant_distribution()
            if probabilities is None
            else as_probability_vector(probabilities, n_states=self.n_states)
        )
        return float(np.max(np.abs(measure @ self.transition - measure)))

    def is_invariant(
        self,
        probabilities: ArrayLike,
        *,
        atol: float | None = None,
    ) -> bool:
        threshold = self.tolerance if atol is None else atol
        return self.global_balance_residual(probabilities) <= threshold

    def detailed_balance_residual(self, probabilities: ArrayLike | None = None) -> float:
        """Return the maximum pairwise stationary-flux imbalance."""

        measure = (
            self.invariant_distribution()
            if probabilities is None
            else as_probability_vector(probabilities, n_states=self.n_states)
        )
        flux = measure[:, None] * self.transition
        return float(np.max(np.abs(flux - flux.T)))

    def is_reversible(
        self,
        probabilities: ArrayLike | None = None,
        *,
        atol: float | None = None,
    ) -> bool:
        threshold = self.tolerance if atol is None else atol
        return self.detailed_balance_residual(probabilities) <= threshold

    def time_reversal(self, probabilities: ArrayLike | None = None) -> FiniteStateMarkovChain:
        """Construct the stationary time-reversed transition matrix.

        A full-support invariant distribution is required because reverse
        transitions out of a zero-mass state are not identified by stationarity.
        """

        measure = (
            self.invariant_distribution()
            if probabilities is None
            else as_probability_vector(probabilities, n_states=self.n_states)
        )
        if self.global_balance_residual(measure) > 10.0 * self.tolerance:
            raise ValueError("time reversal requires an invariant distribution")
        if np.any(measure <= self.tolerance):
            raise ValueError("time reversal requires a full-support invariant distribution")
        reversed_matrix = (self.transition.T * measure[None, :]) / measure[:, None]
        return FiniteStateMarkovChain(reversed_matrix, tolerance=self.tolerance)

    def eigenvalues(self) -> ComplexArray:
        """Return transition eigenvalues sorted by decreasing modulus."""

        values = np.asarray(np.linalg.eigvals(self.transition), dtype=np.complex128)
        order = np.argsort(-np.abs(values))
        return np.asarray(values[order], dtype=np.complex128)

    def spectral_summary(self, probabilities: ArrayLike | None = None) -> SpectralSummary:
        """Compute eigenvalue, contraction, and reversible-gap diagnostics."""

        measure = (
            self.invariant_distribution()
            if probabilities is None
            else as_probability_vector(probabilities, n_states=self.n_states)
        )
        if self.global_balance_residual(measure) > 10.0 * self.tolerance:
            raise ValueError("spectral diagnostics require an invariant distribution")
        if np.any(measure <= self.tolerance):
            raise ValueError("spectral diagnostics require full stationary support")

        eigenvalues = self.eigenvalues()
        distances_from_one = np.abs(eigenvalues - 1.0)
        stationary_index = int(np.argmin(distances_from_one))
        nonstationary = np.delete(eigenvalues, stationary_index)
        slem = 0.0 if nonstationary.size == 0 else float(np.max(np.abs(nonstationary)))
        absolute_gap = max(0.0, 1.0 - slem)

        sqrt_pi = np.sqrt(measure)
        similarity = (sqrt_pi[:, None] * self.transition) / sqrt_pi[None, :]
        singular_values = np.asarray(np.linalg.svd(similarity, compute_uv=False), dtype=np.float64)
        singular_values.sort()
        singular_values = singular_values[::-1]
        second_singular = 0.0 if singular_values.size == 1 else float(singular_values[1])
        singular_gap = max(0.0, 1.0 - second_singular)

        reversible = self.is_reversible(measure)
        poincare_gap: float | None = None
        worst_case_iat: float | None = None
        if reversible:
            symmetrized = 0.5 * (similarity + similarity.T)
            real_eigenvalues = np.asarray(np.linalg.eigvalsh(symmetrized), dtype=np.float64)
            real_eigenvalues.sort()
            real_eigenvalues = real_eigenvalues[::-1]
            lambda_second = -1.0 if real_eigenvalues.size == 1 else float(real_eigenvalues[1])
            poincare_gap = max(0.0, 1.0 - lambda_second)
            if poincare_gap > self.tolerance:
                worst_case_iat = max(0.0, 2.0 / poincare_gap - 1.0)
            else:
                worst_case_iat = float("inf")

        return SpectralSummary(
            eigenvalues=eigenvalues,
            absolute_spectral_gap=absolute_gap,
            second_singular_value=second_singular,
            singular_value_gap=singular_gap,
            reversible=reversible,
            poincare_gap=poincare_gap,
            worst_case_iat=worst_case_iat,
        )

    def constant_projection(self, probabilities: ArrayLike | None = None) -> Array:
        """Return the rank-one projection ``1 pi`` for an invariant law."""

        measure = (
            self.invariant_distribution()
            if probabilities is None
            else as_probability_vector(probabilities, n_states=self.n_states)
        )
        return constant_projection(measure)
