"""Ancestral relationships in sequential particle systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True, init=False)
class Ancestry:
    """Parent maps between consecutive particle populations.

    ``parent_indices[t][j]`` is the index at time ``t`` of particle ``j`` at
    time ``t + 1``. Population sizes may vary, as under Bernoulli resampling.
    """

    parent_indices: tuple[IntArray, ...]
    population_sizes: tuple[int, ...]

    def __init__(
        self,
        parent_indices: tuple[ArrayLike, ...],
        population_sizes: tuple[int, ...],
    ) -> None:
        if len(population_sizes) != len(parent_indices) + 1:
            raise ValueError("population_sizes must contain one entry per generation")
        if not population_sizes or any(size <= 0 for size in population_sizes):
            raise ValueError("all population sizes must be positive")

        validated: list[IntArray] = []
        for step, raw_indices in enumerate(parent_indices):
            indices = np.asarray(raw_indices, dtype=np.int64)
            if indices.ndim != 1 or indices.size != population_sizes[step + 1]:
                raise ValueError("each parent map must match its child population size")
            if np.any(indices < 0) or np.any(indices >= population_sizes[step]):
                raise IndexError("parent index out of range")
            copied = np.array(indices, dtype=np.int64, copy=True)
            copied.setflags(write=False)
            validated.append(copied)

        object.__setattr__(self, "parent_indices", tuple(validated))
        object.__setattr__(self, "population_sizes", tuple(population_sizes))

    @property
    def n_steps(self) -> int:
        """Number of parent-child transitions."""

        return len(self.parent_indices)

    def trace_lineage(self, final_index: int) -> IntArray:
        """Return one lineage from generation zero through the final generation."""

        if isinstance(final_index, bool) or not isinstance(final_index, int):
            raise TypeError("final_index must be an integer")
        if final_index < 0 or final_index >= self.population_sizes[-1]:
            raise IndexError("final particle index out of range")
        lineage = np.empty(self.n_steps + 1, dtype=np.int64)
        lineage[-1] = final_index
        current = final_index
        for step in range(self.n_steps - 1, -1, -1):
            current = int(self.parent_indices[step][current])
            lineage[step] = current
        return lineage

    def final_to_initial(self) -> IntArray:
        """Map every final particle to its generation-zero ancestor."""

        current = np.arange(self.population_sizes[-1], dtype=np.int64)
        for parents in reversed(self.parent_indices):
            current = parents[current]
        return current

    def unique_ancestor_counts(self) -> IntArray:
        """Count distinct ancestors of the final population at every generation."""

        counts = np.empty(self.n_steps + 1, dtype=np.int64)
        current = np.arange(self.population_sizes[-1], dtype=np.int64)
        counts[-1] = current.size
        for step in range(self.n_steps - 1, -1, -1):
            current = self.parent_indices[step][current]
            counts[step] = np.unique(current).size
        return counts
