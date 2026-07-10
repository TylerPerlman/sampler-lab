"""Exact finite-state constructions for the principle of partial resampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.markov.finite_state import FiniteStateMarkovChain, validate_transition_matrix
from sampler_lab.markov.operators import as_probability_vector

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def normalize_joint_distribution(target: ArrayLike) -> Array:
    """Validate and normalize a finite joint probability table."""

    values = np.asarray(target, dtype=np.float64)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("target must be a nonempty probability table")
    if not np.all(np.isfinite(values)):
        raise ValueError("target probabilities must be finite")
    if np.any(values < 0.0):
        raise ValueError("target probabilities must be nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("target must have positive total mass")
    return np.asarray(values / total, dtype=np.float64)


def _normalize_axis(axis: int, ndim: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be an integer")
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError("axis is out of bounds for the target")
    return normalized


def _context_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    return shape[:axis] + shape[axis + 1 :]


def _state_from_context(
    context: tuple[int, ...],
    coordinate: int,
    axis: int,
) -> tuple[int, ...]:
    return (*context[:axis], coordinate, *context[axis:])


def conditional_component_transition(
    target: ArrayLike,
    axis: int,
    conditional_kernels: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> FiniteStateMarkovChain:
    """Lift conditional-invariant component kernels to the full state space.

    ``target`` is an arbitrary-dimensional joint probability table. For each
    fixed complement of ``axis``, ``conditional_kernels`` supplies a square
    transition matrix on that coordinate. Its shape must be
    ``target.shape without axis + (m, m)``, where ``m = target.shape[axis]``.

    Each local kernel is checked for row stochasticity and preservation of the
    corresponding conditional target. Contexts with zero target mass use an
    identity transition because their behavior cannot affect invariance.
    """

    probabilities = normalize_joint_distribution(target)
    coordinate_axis = _normalize_axis(axis, probabilities.ndim)
    coordinate_size = probabilities.shape[coordinate_axis]
    contexts = _context_shape(probabilities.shape, coordinate_axis)
    expected_shape = (*contexts, coordinate_size, coordinate_size)
    kernels = np.asarray(conditional_kernels, dtype=np.float64)
    if kernels.shape != expected_shape:
        raise ValueError(f"conditional_kernels must have shape {expected_shape}")

    n_states = probabilities.size
    full = np.zeros((n_states, n_states), dtype=np.float64)
    context_iterator = np.ndindex(contexts) if contexts else [()]
    for context in context_iterator:
        local = np.asarray(kernels[context], dtype=np.float64)
        local = validate_transition_matrix(local, atol=tolerance)
        state_tuples = [
            _state_from_context(context, coordinate, coordinate_axis)
            for coordinate in range(coordinate_size)
        ]
        flat_indices = np.asarray(
            [np.ravel_multi_index(state, probabilities.shape) for state in state_tuples],
            dtype=np.int64,
        )
        conditional_mass = np.asarray(
            probabilities[tuple(np.asarray(state_tuples).T)], dtype=np.float64
        )
        total = float(np.sum(conditional_mass))
        if total <= tolerance:
            local = np.eye(coordinate_size, dtype=np.float64)
        else:
            conditional = conditional_mass / total
            residual = float(np.max(np.abs(conditional @ local - conditional)))
            if residual > 10.0 * tolerance:
                raise ValueError("a component kernel does not preserve its conditional target")
        full[np.ix_(flat_indices, flat_indices)] = local

    return FiniteStateMarkovChain(full, tolerance=tolerance)


def conditional_resampling_transition(
    target: ArrayLike,
    axis: int,
    *,
    tolerance: float = 1e-12,
) -> FiniteStateMarkovChain:
    """Construct the exact Gibbs resampling kernel for one coordinate block."""

    probabilities = normalize_joint_distribution(target)
    coordinate_axis = _normalize_axis(axis, probabilities.ndim)
    coordinate_size = probabilities.shape[coordinate_axis]
    contexts = _context_shape(probabilities.shape, coordinate_axis)
    kernels = np.empty((*contexts, coordinate_size, coordinate_size), dtype=np.float64)
    context_iterator = np.ndindex(contexts) if contexts else [()]
    for context in context_iterator:
        state_tuples = [
            _state_from_context(context, coordinate, coordinate_axis)
            for coordinate in range(coordinate_size)
        ]
        masses = np.asarray(probabilities[tuple(np.asarray(state_tuples).T)], dtype=np.float64)
        total = float(np.sum(masses))
        conditional = (
            np.full(coordinate_size, 1.0 / coordinate_size, dtype=np.float64)
            if total <= tolerance
            else masses / total
        )
        kernels[context] = np.broadcast_to(
            conditional,
            (coordinate_size, coordinate_size),
        )
    return conditional_component_transition(
        probabilities,
        coordinate_axis,
        kernels,
        tolerance=tolerance,
    )


def mixture_transition(
    chains: Sequence[FiniteStateMarkovChain | ArrayLike],
    probabilities: ArrayLike | None = None,
    *,
    tolerance: float = 1e-12,
) -> FiniteStateMarkovChain:
    """Randomly choose one transition matrix independently of the state."""

    if not chains:
        raise ValueError("at least one transition matrix is required")
    matrices = [
        chain.transition
        if isinstance(chain, FiniteStateMarkovChain)
        else validate_transition_matrix(chain)
        for chain in chains
    ]
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("all transition matrices must have the same shape")
    weights = (
        np.full(len(matrices), 1.0 / len(matrices), dtype=np.float64)
        if probabilities is None
        else as_probability_vector(probabilities, n_states=len(matrices), atol=tolerance)
    )
    mixed = np.zeros(shape, dtype=np.float64)
    for weight, matrix in zip(weights, matrices, strict=True):
        mixed += weight * matrix
    return FiniteStateMarkovChain(mixed, tolerance=tolerance)


def compose_transitions(
    chains: Sequence[FiniteStateMarkovChain | ArrayLike],
    *,
    tolerance: float = 1e-12,
) -> FiniteStateMarkovChain:
    """Compose transitions in execution order.

    ``compose_transitions([P, Q])`` means apply ``P`` first and ``Q`` second,
    yielding the row-stochastic matrix ``P @ Q``.
    """

    if not chains:
        raise ValueError("at least one transition matrix is required")
    matrices = [
        chain.transition
        if isinstance(chain, FiniteStateMarkovChain)
        else validate_transition_matrix(chain)
        for chain in chains
    ]
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("all transition matrices must have the same shape")
    composed = np.eye(shape[0], dtype=np.float64)
    for matrix in matrices:
        composed = composed @ matrix
    return FiniteStateMarkovChain(composed, tolerance=tolerance)


def conjugate_by_permutation(
    chain: FiniteStateMarkovChain | ArrayLike,
    old_to_new: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> FiniteStateMarkovChain:
    """Relabel states by a permutation, preserving all Markov properties.

    ``old_to_new[i]`` is the new index assigned to old state ``i``. The result
    satisfies ``Q[old_to_new[i], old_to_new[j]] = P[i, j]``.
    """

    matrix = (
        chain.transition
        if isinstance(chain, FiniteStateMarkovChain)
        else validate_transition_matrix(chain)
    )
    permutation = np.asarray(old_to_new, dtype=np.int64)
    if permutation.ndim != 1 or permutation.size != matrix.shape[0]:
        raise ValueError("old_to_new must match the number of states")
    if not np.array_equal(np.sort(permutation), np.arange(matrix.shape[0])):
        raise ValueError("old_to_new must be a permutation")
    transformed = np.empty_like(matrix)
    transformed[np.ix_(permutation, permutation)] = matrix
    return FiniteStateMarkovChain(transformed, tolerance=tolerance)
