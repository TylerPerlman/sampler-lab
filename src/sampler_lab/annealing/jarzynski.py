"""Annealed importance sampling and resampled path-space particle methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.annealing.free_energy import (
    FreeEnergyEstimate,
    free_energy_from_log_ratio,
    jarzynski_estimate,
)
from sampler_lab.annealing.paths import AnnealingPath, incremental_log_weights
from sampler_lab.annealing.schedules import AnnealingSchedule
from sampler_lab.annealing.transitions import PopulationTransition
from sampler_lab.core.numerics import logsumexp
from sampler_lab.core.protocols import Resampler
from sampler_lab.particles.ancestry import Ancestry
from sampler_lab.particles.cloud import ParticleCloud
from sampler_lab.particles.sequential import ParticleExtinctionError

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class AnnealedImportanceResult:
    """Complete history of annealed importance sampling or annealed SMC.

    ``weighted_clouds[k]`` is the population after reweighting from schedule
    value ``k`` to ``k + 1`` and before any resampling or mutation.
    ``clouds[k + 1]`` is the corresponding post-resampling, post-mutation cloud.
    """

    schedule: AnnealingSchedule
    clouds: tuple[ParticleCloud, ...]
    weighted_clouds: tuple[ParticleCloud, ...]
    ancestry: Ancestry
    incremental_log_weights: tuple[Array, ...]
    cumulative_trajectory_log_weights: tuple[Array, ...]
    incremental_log_normalizers: Array
    resampled: BoolArray
    log_normalizing_constant_ratio: float

    @property
    def n_steps(self) -> int:
        return self.schedule.n_steps

    @property
    def final_cloud(self) -> ParticleCloud:
        return self.clouds[-1]

    @property
    def final_weighted_cloud(self) -> ParticleCloud:
        return self.weighted_clouds[-1]

    @property
    def normalizing_constant_ratio(self) -> float:
        maximum = float(np.log(np.finfo(np.float64).max))
        if self.log_normalizing_constant_ratio > maximum:
            return float("inf")
        return float(np.exp(self.log_normalizing_constant_ratio))

    @property
    def delta_free_energy(self) -> float:
        return free_energy_from_log_ratio(self.log_normalizing_constant_ratio)

    @property
    def ess_history(self) -> Array:
        return np.asarray(
            [cloud.effective_sample_size for cloud in self.weighted_clouds],
            dtype=np.float64,
        )

    @property
    def population_sizes(self) -> NDArray[np.int64]:
        return np.asarray(self.ancestry.population_sizes, dtype=np.int64)

    @property
    def final_reduced_work(self) -> Array:
        """Full reduced work ``W = -sum(delta log gamma)`` per final trajectory."""

        work = -self.cumulative_trajectory_log_weights[-1]
        work.setflags(write=False)
        return work

    def iid_jarzynski_estimate(self) -> FreeEnergyEstimate:
        """Return the IID Jarzynski/free-energy estimate for an unresampled run.

        Resampling couples and duplicates trajectories, so the elementary IID
        standard-error formula is deliberately unavailable in that case.
        """

        if np.any(self.resampled):
            raise ValueError("IID Jarzynski uncertainty is invalid after resampling")
        return jarzynski_estimate(self.final_reduced_work)


def annealed_importance_sampling(
    initial_particles: ArrayLike,
    path: AnnealingPath,
    schedule: AnnealingSchedule,
    transition: PopulationTransition,
    rng: np.random.Generator,
    *,
    resampler: Resampler | None = None,
    resample_every_step: bool = False,
    resample_ess_fraction: float | None = None,
    target_particle_count: int | None = None,
) -> AnnealedImportanceResult:
    """Estimate ``Z_1 / Z_0`` along a path of unnormalized distributions.

    Each stage first applies the exact density-ratio increment at the current
    particle locations, optionally resamples, and then applies a transition
    intended to preserve the new path distribution. Without resampling this is
    ordinary annealed importance sampling/Jarzynski weighting. With resampling
    it is annealed sequential Monte Carlo.
    """

    particle_array = np.asarray(initial_particles, dtype=np.float64)
    if particle_array.ndim < 1 or particle_array.shape[0] == 0:
        raise ValueError("initial_particles must have a nonempty particle axis")
    if not np.all(np.isfinite(particle_array)):
        raise ValueError("initial_particles must be finite")
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

    current = ParticleCloud.uniform(particle_array)
    current_trajectory_log_weights = np.zeros(current.n_particles, dtype=np.float64)
    clouds = [current]
    weighted_clouds: list[ParticleCloud] = []
    all_increments: list[Array] = []
    trajectory_histories: list[Array] = [_readonly_copy(current_trajectory_log_weights)]
    parents: list[NDArray[np.int64]] = []
    log_normalizer_increments: list[float] = []
    resampled_flags: list[bool] = []
    total_log_ratio = 0.0

    for stage in range(schedule.n_steps):
        beta_from = float(schedule.values[stage])
        beta_to = float(schedule.values[stage + 1])
        increments = incremental_log_weights(path, current.particles, beta_from, beta_to)
        combined = current.log_weights + increments
        if not np.any(np.isfinite(combined)):
            raise ParticleExtinctionError(f"all particles have zero weight at stage {stage + 1}")
        step_log_ratio = float(logsumexp(combined))
        total_log_ratio += step_log_ratio
        weighted = ParticleCloud(current.particles, combined)
        weighted_clouds.append(weighted)
        all_increments.append(_readonly_copy(increments))
        log_normalizer_increments.append(step_log_ratio)

        trajectory_log_weights = current_trajectory_log_weights + increments
        trigger = resample_every_step
        if resample_ess_fraction is not None:
            trigger = trigger or (
                weighted.effective_sample_size <= resample_ess_fraction * weighted.n_particles
            )
        should_resample = resampler is not None and trigger
        if should_resample:
            assert resampler is not None
            parent_indices = np.asarray(
                resampler.resample(weighted.weights, rng, target_particle_count),
                dtype=np.int64,
            )
            if parent_indices.ndim != 1:
                raise ValueError("resampler must return a one-dimensional parent index array")
            if parent_indices.size == 0:
                raise ParticleExtinctionError(
                    f"resampling produced an empty population at stage {stage + 1}"
                )
            selected = weighted.select(parent_indices)
            trajectory_log_weights = trajectory_log_weights[parent_indices]
        else:
            parent_indices = np.arange(current.n_particles, dtype=np.int64)
            selected = weighted

        moved_particles = transition.move(selected.particles, beta_to, rng)
        moved = np.asarray(moved_particles, dtype=np.float64)
        if moved.shape != selected.particles.shape:
            raise ValueError("population transition changed the particle-array shape")
        if not np.all(np.isfinite(moved)):
            raise ValueError("population transition returned nonfinite particles")
        current = ParticleCloud(moved, selected.log_weights)
        current_trajectory_log_weights = np.asarray(
            trajectory_log_weights,
            dtype=np.float64,
        )
        trajectory_histories.append(_readonly_copy(current_trajectory_log_weights))
        parents.append(parent_indices)
        resampled_flags.append(should_resample)
        clouds.append(current)

    ancestry = Ancestry(tuple(parents), tuple(cloud.n_particles for cloud in clouds))
    log_increment_array = np.asarray(log_normalizer_increments, dtype=np.float64)
    log_increment_array.setflags(write=False)
    resampled_array = np.asarray(resampled_flags, dtype=np.bool_)
    resampled_array.setflags(write=False)
    return AnnealedImportanceResult(
        schedule=schedule,
        clouds=tuple(clouds),
        weighted_clouds=tuple(weighted_clouds),
        ancestry=ancestry,
        incremental_log_weights=tuple(all_increments),
        cumulative_trajectory_log_weights=tuple(trajectory_histories),
        incremental_log_normalizers=log_increment_array,
        resampled=resampled_array,
        log_normalizing_constant_ratio=total_log_ratio,
    )


def _readonly_copy(values: ArrayLike) -> Array:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied
