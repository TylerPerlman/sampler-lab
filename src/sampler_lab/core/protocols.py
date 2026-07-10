"""Structural interfaces shared by Monte Carlo algorithms.

The library deliberately avoids a universal ``Sampler`` base class. Independent
samplers, Markov kernels, weighted particles, and ensemble methods have different
semantics; protocols express only the capabilities each algorithm actually needs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sampler_lab.core.results import Transition

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


class LogDensity(Protocol):
    """Target that can evaluate a log density up to an additive constant."""

    def log_prob(self, x: Array) -> float:
        """Evaluate the log density at ``x``."""


class DifferentiableLogDensity(LogDensity, Protocol):
    """Log density with a first derivative."""

    def grad_log_prob(self, x: Array) -> Array:
        """Evaluate the gradient of the log density at ``x``."""


class TwiceDifferentiableLogDensity(DifferentiableLogDensity, Protocol):
    """Log density with first and second derivatives."""

    def hessian_log_prob(self, x: Array) -> Array:
        """Evaluate the Hessian of the log density at ``x``."""


class IndependentSampler(Protocol):
    """Sampler returning independent draws."""

    def sample(self, rng: np.random.Generator, size: int) -> Array:
        """Return ``size`` independent samples."""


class MarkovKernel(Protocol):
    """One-step Markov transition."""

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        """Advance one transition from ``state``."""


class Resampler(Protocol):
    """Particle resampling scheme."""

    def resample(
        self,
        weights: Array,
        rng: np.random.Generator,
        n_particles: int | None = None,
    ) -> IntArray:
        """Return parent indices for a resampled particle population."""
