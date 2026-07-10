"""Scalar Robbins--Monro and dual-averaging adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sampler_lab.adaptive.schedules import RobbinsMonroSchedule


@dataclass(slots=True)
class RobbinsMonroLogScale:
    """Adapt a positive scale toward a target acceptance probability."""

    initial_scale: float
    target_acceptance: float
    schedule: RobbinsMonroSchedule = field(default_factory=RobbinsMonroSchedule)
    minimum_scale: float = 1e-8
    maximum_scale: float = 1e8
    _log_scale: float = field(init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_scale) or self.initial_scale <= 0.0:
            raise ValueError("initial_scale must be positive and finite")
        if not np.isfinite(self.target_acceptance) or not 0.0 < self.target_acceptance < 1.0:
            raise ValueError("target_acceptance must lie in (0, 1)")
        if not np.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive and finite")
        if not np.isfinite(self.maximum_scale) or self.maximum_scale < self.minimum_scale:
            raise ValueError("maximum_scale must be finite and at least minimum_scale")
        self._log_scale = float(np.log(self.initial_scale))

    @property
    def scale(self) -> float:
        return float(np.exp(self._log_scale))

    @property
    def step_index(self) -> int:
        return self._step

    def update(self, acceptance: float) -> float:
        """Update from an acceptance indicator or probability and return the new scale."""

        if not np.isfinite(acceptance) or not 0.0 <= acceptance <= 1.0:
            raise ValueError("acceptance must lie in [0, 1]")
        rate = self.schedule.rate(self._step)
        self._log_scale += rate * (acceptance - self.target_acceptance)
        self._log_scale = float(
            np.clip(self._log_scale, np.log(self.minimum_scale), np.log(self.maximum_scale))
        )
        self._step += 1
        return self.scale


@dataclass(slots=True)
class DualAveragingStepSize:
    """Nesterov dual averaging used for MCMC step-size warmup."""

    initial_step_size: float
    target_acceptance: float = 0.8
    gamma: float = 0.05
    t0: float = 10.0
    kappa: float = 0.75
    _mu: float = field(init=False, repr=False)
    _h_bar: float = field(default=0.0, init=False, repr=False)
    _log_step: float = field(init=False, repr=False)
    _log_step_average: float = field(init=False, repr=False)
    _iteration: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_step_size) or self.initial_step_size <= 0.0:
            raise ValueError("initial_step_size must be positive and finite")
        if not np.isfinite(self.target_acceptance) or not 0.0 < self.target_acceptance < 1.0:
            raise ValueError("target_acceptance must lie in (0, 1)")
        if not np.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError("gamma must be positive and finite")
        if not np.isfinite(self.t0) or self.t0 < 0.0:
            raise ValueError("t0 must be nonnegative and finite")
        if not np.isfinite(self.kappa) or not 0.5 < self.kappa <= 1.0:
            raise ValueError("kappa must lie in (0.5, 1]")
        self._mu = float(np.log(10.0 * self.initial_step_size))
        self._log_step = float(np.log(self.initial_step_size))
        self._log_step_average = self._log_step

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def current_step_size(self) -> float:
        return float(np.exp(self._log_step))

    @property
    def averaged_step_size(self) -> float:
        return float(np.exp(self._log_step_average))

    @property
    def h_bar(self) -> float:
        return self._h_bar

    def update(self, acceptance_probability: float) -> float:
        """Update the recursion and return the current, nonaveraged step size."""

        if not np.isfinite(acceptance_probability) or not 0.0 <= acceptance_probability <= 1.0:
            raise ValueError("acceptance_probability must lie in [0, 1]")
        self._iteration += 1
        iteration = float(self._iteration)
        eta = 1.0 / (iteration + self.t0)
        self._h_bar = (1.0 - eta) * self._h_bar + eta * (
            self.target_acceptance - acceptance_probability
        )
        self._log_step = self._mu - np.sqrt(iteration) * self._h_bar / self.gamma
        weight = iteration ** (-self.kappa)
        self._log_step_average = weight * self._log_step + (1.0 - weight) * self._log_step_average
        return self.current_step_size
