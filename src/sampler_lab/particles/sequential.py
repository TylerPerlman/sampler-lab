"""Sequential importance sampling with optional particle resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.numerics import logsumexp, validate_size
from sampler_lab.core.protocols import Resampler
from sampler_lab.particles.ancestry import Ancestry
from sampler_lab.particles.cloud import ParticleCloud

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Proposed particles and their incremental log weights."""

    particles: Array
    log_incremental_weights: Array


class SequentialProposal(Protocol):
    """One sequential proposal-and-weighting operation."""

    def propose(
        self,
        particles: Array,
        step: int,
        rng: np.random.Generator,
    ) -> PropagationResult:
        """Extend every particle and return incremental log weights."""


class ParticleExtinctionError(RuntimeError):
    """Raised when every particle has zero weight or resampling returns none."""


@dataclass(frozen=True, slots=True)
class SequentialImportanceResult:
    """Complete history of a sequential importance-sampling run."""

    clouds: tuple[ParticleCloud, ...]
    weighted_clouds: tuple[ParticleCloud, ...]
    ancestry: Ancestry
    incremental_log_normalizers: Array
    resampled: BoolArray
    log_normalizing_constant_estimate: float

    @property
    def n_steps(self) -> int:
        """Number of sequential propagation steps."""

        return len(self.weighted_clouds)

    @property
    def final_cloud(self) -> ParticleCloud:
        """Final post-resampling cloud."""

        return self.clouds[-1]

    @property
    def final_weighted_cloud(self) -> ParticleCloud:
        """Final target-weighted cloud before any last resampling operation."""

        return self.clouds[0] if not self.weighted_clouds else self.weighted_clouds[-1]

    @property
    def normalizing_constant_estimate(self) -> float:
        """Normalizing-constant estimate in linear space."""

        if self.log_normalizing_constant_estimate > np.log(np.finfo(np.float64).max):
            return float("inf")
        return float(np.exp(self.log_normalizing_constant_estimate))

    @property
    def ess_history(self) -> Array:
        """Pre-resampling effective sample size after each propagation."""

        return np.asarray(
            [cloud.effective_sample_size for cloud in self.weighted_clouds],
            dtype=np.float64,
        )

    @property
    def population_sizes(self) -> NDArray[np.int64]:
        """Post-resampling population size at each generation."""

        return np.asarray(self.ancestry.population_sizes, dtype=np.int64)


def sequential_importance_sampling(
    initial_particles: ArrayLike,
    n_steps: int,
    proposal: SequentialProposal,
    rng: np.random.Generator,
    *,
    initial_log_weights: ArrayLike | None = None,
    resampler: Resampler | None = None,
    resample_every_step: bool = False,
    resample_ess_fraction: float | None = None,
    target_particle_count: int | None = None,
    initial_log_normalizing_constant: float = 0.0,
) -> SequentialImportanceResult:
    """Run sequential importance sampling with optional resampling.

    At each step the algorithm propagates, applies incremental weights,
    updates the normalizing-constant estimate, and then optionally resamples.
    ``resample_ess_fraction`` triggers when pre-resampling ESS is at most that
    fraction of the current population. A supplied ``target_particle_count``
    is useful for variable-population rules such as Bernoulli resampling.
    """

    steps = validate_size(n_steps)
    particle_array = np.asarray(initial_particles, dtype=np.float64)
    if particle_array.ndim < 1 or particle_array.shape[0] == 0:
        raise ValueError("initial_particles must have a nonempty particle axis")
    if initial_log_weights is None:
        initial_log_weights = np.zeros(particle_array.shape[0], dtype=np.float64)
    current = ParticleCloud(particle_array, initial_log_weights)

    if not np.isfinite(initial_log_normalizing_constant):
        raise ValueError("initial_log_normalizing_constant must be finite")
    if resample_every_step and resampler is None:
        raise ValueError("resample_every_step requires a resampler")
    if resample_ess_fraction is not None:
        if not np.isfinite(resample_ess_fraction) or not 0.0 <= resample_ess_fraction <= 1.0:
            raise ValueError("resample_ess_fraction must lie in [0, 1]")
        if resampler is None:
            raise ValueError("resample_ess_fraction requires a resampler")
    if target_particle_count is not None:
        if isinstance(target_particle_count, bool) or not isinstance(target_particle_count, int):
            raise TypeError("target_particle_count must be an integer")
        if target_particle_count <= 0:
            raise ValueError("target_particle_count must be positive")

    clouds = [current]
    weighted_clouds: list[ParticleCloud] = []
    parents: list[NDArray[np.int64]] = []
    increments: list[float] = []
    resampled_flags: list[bool] = []
    log_normalizer = float(initial_log_normalizing_constant)

    for step in range(1, steps + 1):
        propagated = proposal.propose(current.particles, step, rng)
        next_particles = np.asarray(propagated.particles, dtype=np.float64)
        log_increment = np.asarray(propagated.log_incremental_weights, dtype=np.float64)
        if next_particles.ndim < 1 or next_particles.shape[0] != current.n_particles:
            raise ValueError("proposal must return one propagated particle per input particle")
        if log_increment.ndim != 1 or log_increment.shape[0] != current.n_particles:
            raise ValueError("incremental log weights must match the particle count")
        if np.any(np.isnan(log_increment)) or np.any(np.isposinf(log_increment)):
            raise ValueError("incremental log weights may not contain nan or +inf")

        combined_log_weights = current.log_weights + log_increment
        if not np.any(np.isfinite(combined_log_weights)):
            raise ParticleExtinctionError(f"all particles have zero weight at step {step}")
        step_log_normalizer = float(logsumexp(combined_log_weights))
        log_normalizer += step_log_normalizer
        weighted = ParticleCloud(next_particles, combined_log_weights)
        weighted_clouds.append(weighted)
        increments.append(step_log_normalizer)

        trigger = resample_every_step
        if resample_ess_fraction is not None:
            trigger = trigger or (
                weighted.effective_sample_size <= resample_ess_fraction * weighted.n_particles
            )
        should_resample = resampler is not None and trigger

        if resampler is not None and trigger:
            target = target_particle_count
            parent_indices = np.asarray(
                resampler.resample(weighted.weights, rng, target),
                dtype=np.int64,
            )
            if parent_indices.ndim != 1:
                raise ValueError("resampler must return a one-dimensional parent index array")
            if parent_indices.size == 0:
                raise ParticleExtinctionError(
                    f"resampling produced an empty population at step {step}"
                )
            next_cloud = weighted.select(parent_indices)
        else:
            parent_indices = np.arange(current.n_particles, dtype=np.int64)
            next_cloud = weighted

        parents.append(parent_indices)
        resampled_flags.append(should_resample)
        clouds.append(next_cloud)
        current = next_cloud

    ancestry = Ancestry(
        tuple(parents),
        tuple(cloud.n_particles for cloud in clouds),
    )
    increment_array = np.asarray(increments, dtype=np.float64)
    increment_array.setflags(write=False)
    resampled_array = np.asarray(resampled_flags, dtype=np.bool_)
    resampled_array.setflags(write=False)

    return SequentialImportanceResult(
        clouds=tuple(clouds),
        weighted_clouds=tuple(weighted_clouds),
        ancestry=ancestry,
        incremental_log_normalizers=increment_array,
        resampled=resampled_array,
        log_normalizing_constant_estimate=log_normalizer,
    )
