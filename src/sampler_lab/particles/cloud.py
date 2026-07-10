"""Immutable weighted particle populations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.numerics import normalize_log_weights
from sampler_lab.diagnostics.weighted import WeightDiagnostics, diagnostics_from_normalized_weights

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True, init=False)
class ParticleCloud:
    """Particles paired with normalized log weights.

    The particle axis is always the first axis. Inputs are copied and marked
    read-only so a cloud is a stable snapshot suitable for history tracking.
    Additive shifts in the supplied log weights are discarded by normalization.
    """

    particles: Array
    log_weights: Array

    def __init__(self, particles: ArrayLike, log_weights: ArrayLike) -> None:
        particle_array = np.asarray(particles, dtype=np.float64)
        weight_array = np.asarray(log_weights, dtype=np.float64)
        if particle_array.ndim < 1 or particle_array.shape[0] == 0:
            raise ValueError("particles must have a nonempty leading particle axis")
        if weight_array.ndim != 1 or weight_array.shape[0] != particle_array.shape[0]:
            raise ValueError("log_weights must be one-dimensional and match the particle count")

        normalized_weights, _ = normalize_log_weights(weight_array)
        with np.errstate(divide="ignore"):
            normalized_log_weights = np.log(normalized_weights)

        particle_copy = np.array(particle_array, dtype=np.float64, copy=True)
        log_weight_copy = np.array(normalized_log_weights, dtype=np.float64, copy=True)
        particle_copy.setflags(write=False)
        log_weight_copy.setflags(write=False)
        object.__setattr__(self, "particles", particle_copy)
        object.__setattr__(self, "log_weights", log_weight_copy)

    @classmethod
    def uniform(cls, particles: ArrayLike) -> ParticleCloud:
        """Construct a uniformly weighted cloud."""

        particle_array = np.asarray(particles, dtype=np.float64)
        if particle_array.ndim < 1:
            raise ValueError("particles must have a leading particle axis")
        return cls(particle_array, np.zeros(particle_array.shape[0], dtype=np.float64))

    @property
    def n_particles(self) -> int:
        """Current population size."""

        return int(self.particles.shape[0])

    @property
    def weights(self) -> Array:
        """Normalized linear weights as a fresh array."""

        return np.exp(self.log_weights)

    @property
    def diagnostics(self) -> WeightDiagnostics:
        """Scale-free diagnostics for the current weights."""

        return diagnostics_from_normalized_weights(self.weights)

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size of the weights."""

        return self.diagnostics.effective_sample_size

    def expectation(self, observable: Callable[[Array], ArrayLike]) -> Array | float:
        """Evaluate a scalar- or array-valued weighted empirical expectation."""

        values = np.asarray(observable(self.particles), dtype=np.float64)
        if values.ndim == 0 or values.shape[0] != self.n_particles:
            raise ValueError("observable output must have the particle axis first")
        if not np.all(np.isfinite(values)):
            raise ValueError("observable values must be finite")
        result = np.tensordot(self.weights, values, axes=(0, 0))
        if np.ndim(result) == 0:
            return float(result)
        return np.asarray(result, dtype=np.float64)

    def select(self, parent_indices: ArrayLike) -> ParticleCloud:
        """Copy selected parents and reset their weights to uniform."""

        indices = np.asarray(parent_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("parent_indices must be a nonempty one-dimensional array")
        if np.any(indices < 0) or np.any(indices >= self.n_particles):
            raise IndexError("parent index out of range")
        return ParticleCloud.uniform(self.particles[indices])
