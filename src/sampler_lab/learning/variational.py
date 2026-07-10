"""Transparent Gaussian variational proposals and exact MH correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import DifferentiableLogDensity, LogDensity
from sampler_lab.learning.optimizers import Adam, ParameterOptimizer
from sampler_lab.mcmc.metropolis import MetropolisHastingsKernel
from sampler_lab.mcmc.proposals import GaussianIndependenceProposal

Array = NDArray[np.float64]
_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _as_vector(value: ArrayLike, *, name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return vector


@dataclass(frozen=True, slots=True)
class ReverseKLEstimate:
    """Monte Carlo reverse-KL objective up to the target normalizer."""

    value: float
    gradient: Array
    mean_log_q: float
    mean_log_target: float
    n_samples: int


class DiagonalGaussianVariational:
    """Reparameterized diagonal Gaussian variational family."""

    def __init__(self, mean: ArrayLike, log_scale: ArrayLike) -> None:
        mean_vector = _as_vector(mean, name="mean")
        log_scale_vector = _as_vector(log_scale, name="log_scale")
        if mean_vector.shape != log_scale_vector.shape:
            raise ValueError("mean and log_scale must have equal shape")
        self._mean = np.array(mean_vector, copy=True)
        self._log_scale = np.array(log_scale_vector, copy=True)

    @property
    def dimension(self) -> int:
        return int(self._mean.size)

    @property
    def mean(self) -> Array:
        return self._mean.copy()

    @property
    def log_scale(self) -> Array:
        return self._log_scale.copy()

    @property
    def scale(self) -> Array:
        return np.exp(self._log_scale).astype(np.float64)

    @property
    def parameters(self) -> Array:
        return np.concatenate((self._mean, self._log_scale)).astype(np.float64)

    def set_parameters(self, parameters: ArrayLike) -> None:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (2 * self.dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("parameters must be a finite vector of length 2 * dimension")
        self._mean[...] = values[: self.dimension]
        self._log_scale[...] = values[self.dimension :]

    def sample_from_noise(self, noise: ArrayLike) -> Array:
        values = np.asarray(noise, dtype=np.float64)
        if values.shape[-1:] != (self.dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("noise must have final dimension equal to the variational dimension")
        return np.asarray(self._mean + self.scale * values, dtype=np.float64)

    def sample(self, rng: np.random.Generator, size: int) -> Array:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("size must be an integer")
        if size <= 0:
            raise ValueError("size must be positive")
        noise = rng.normal(size=(size, self.dimension))
        return self.sample_from_noise(noise)

    def log_prob(self, value: ArrayLike) -> float:
        point = _as_vector(value, name="value")
        if point.shape != (self.dimension,):
            raise ValueError("value dimension does not match the variational family")
        standardized = (point - self._mean) / self.scale
        return float(
            -0.5 * standardized @ standardized
            - np.sum(self._log_scale)
            - 0.5 * self.dimension * _LOG_TWO_PI
        )

    @property
    def entropy(self) -> float:
        return float(0.5 * self.dimension * (1.0 + _LOG_TWO_PI) + np.sum(self._log_scale))

    def reverse_kl_estimate(
        self,
        target: DifferentiableLogDensity,
        noise: ArrayLike,
    ) -> ReverseKLEstimate:
        """Estimate ``E_q[log q - log gamma]`` and its pathwise gradient."""

        epsilon = np.asarray(noise, dtype=np.float64)
        if epsilon.ndim != 2 or epsilon.shape[1] != self.dimension:
            raise ValueError("noise must have shape (n_samples, dimension)")
        if epsilon.shape[0] == 0 or not np.all(np.isfinite(epsilon)):
            raise ValueError("noise must be nonempty and finite")
        samples = self.sample_from_noise(epsilon)
        scales = self.scale
        log_q = np.asarray(
            [self.log_prob(sample) for sample in samples],
            dtype=np.float64,
        )
        log_target = np.asarray(
            [target.log_prob(np.asarray(sample, dtype=np.float64)) for sample in samples],
            dtype=np.float64,
        )
        gradients = np.asarray(
            [target.grad_log_prob(np.asarray(sample, dtype=np.float64)) for sample in samples],
            dtype=np.float64,
        )
        if gradients.shape != samples.shape or not np.all(np.isfinite(gradients)):
            raise ValueError("target gradients must match samples and be finite")
        if not np.all(np.isfinite(log_target)):
            raise ValueError("reverse-KL samples must lie in finite target support")
        gradient_mean = -np.mean(gradients, axis=0)
        gradient_log_scale = -1.0 - np.mean(gradients * (scales * epsilon), axis=0)
        gradient = np.concatenate((gradient_mean, gradient_log_scale)).astype(np.float64)
        return ReverseKLEstimate(
            value=float(np.mean(log_q - log_target)),
            gradient=gradient,
            mean_log_q=float(np.mean(log_q)),
            mean_log_target=float(np.mean(log_target)),
            n_samples=epsilon.shape[0],
        )

    def freeze(self) -> FrozenDiagonalGaussian:
        return FrozenDiagonalGaussian(self._mean, self._log_scale)


@dataclass(frozen=True, slots=True, init=False)
class FrozenDiagonalGaussian:
    """Immutable variational approximation and independence proposal."""

    mean: Array
    log_scale: Array

    def __init__(self, mean: ArrayLike, log_scale: ArrayLike) -> None:
        mean_vector = _as_vector(mean, name="mean")
        log_scale_vector = _as_vector(log_scale, name="log_scale")
        if mean_vector.shape != log_scale_vector.shape:
            raise ValueError("mean and log_scale must have equal shape")
        mean_copy = np.array(mean_vector, copy=True)
        log_scale_copy = np.array(log_scale_vector, copy=True)
        mean_copy.setflags(write=False)
        log_scale_copy.setflags(write=False)
        object.__setattr__(self, "mean", mean_copy)
        object.__setattr__(self, "log_scale", log_scale_copy)

    @property
    def scale(self) -> Array:
        return np.exp(self.log_scale).astype(np.float64)

    @property
    def covariance(self) -> Array:
        return np.diag(self.scale * self.scale).astype(np.float64)

    def sample(self, rng: np.random.Generator, size: int) -> Array:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("size must be an integer")
        if size <= 0:
            raise ValueError("size must be positive")
        return np.asarray(
            self.mean + self.scale * rng.normal(size=(size, self.mean.size)),
            dtype=np.float64,
        )

    def as_independence_proposal(self) -> GaussianIndependenceProposal:
        return GaussianIndependenceProposal(self.mean, self.scale)

    def corrected_kernel(
        self,
        target: LogDensity,
        *,
        counter: OperationCounter | None = None,
    ) -> MetropolisHastingsKernel:
        """Return exact independence MH using the approximation only as a proposal."""

        return MetropolisHastingsKernel(
            target,
            self.as_independence_proposal(),
            counter,
        )


@dataclass(frozen=True, slots=True, init=False)
class VariationalFitResult:
    """Optimization history for an explicitly approximate variational fit."""

    objective_history: Array
    gradient_norms: Array
    parameter_history: Array
    approximation: FrozenDiagonalGaussian
    approximate: bool

    def __init__(
        self,
        objective_history: ArrayLike,
        gradient_norms: ArrayLike,
        parameter_history: ArrayLike,
        approximation: FrozenDiagonalGaussian,
    ) -> None:
        objectives = np.asarray(objective_history, dtype=np.float64)
        norms = np.asarray(gradient_norms, dtype=np.float64)
        parameters = np.asarray(parameter_history, dtype=np.float64)
        if objectives.ndim != 1 or norms.shape != objectives.shape:
            raise ValueError("objective and gradient histories must be equal-length vectors")
        if parameters.ndim != 2 or parameters.shape[0] != objectives.size + 1:
            raise ValueError("parameter history must have one more row than objectives")
        if not np.all(np.isfinite(objectives)) or not np.all(np.isfinite(parameters)):
            raise ValueError("variational histories must be finite")
        objective_copy = np.array(objectives, copy=True)
        norm_copy = np.array(norms, copy=True)
        parameter_copy = np.array(parameters, copy=True)
        objective_copy.setflags(write=False)
        norm_copy.setflags(write=False)
        parameter_copy.setflags(write=False)
        object.__setattr__(self, "objective_history", objective_copy)
        object.__setattr__(self, "gradient_norms", norm_copy)
        object.__setattr__(self, "parameter_history", parameter_copy)
        object.__setattr__(self, "approximation", approximation)
        object.__setattr__(self, "approximate", True)


def fit_reverse_kl_diagonal_gaussian(
    target: DifferentiableLogDensity,
    initial: DiagonalGaussianVariational,
    rng: np.random.Generator,
    *,
    n_steps: int,
    batch_size: int = 128,
    optimizer: ParameterOptimizer | None = None,
) -> VariationalFitResult:
    """Minimize reverse KL with analytic reparameterization gradients."""

    for name, value in (("n_steps", n_steps), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    resolved_optimizer = optimizer or Adam(learning_rate=0.03, gradient_clip_norm=20.0)
    objectives = np.empty(n_steps, dtype=np.float64)
    gradient_norms = np.empty(n_steps, dtype=np.float64)
    parameters = np.empty((n_steps + 1, initial.parameters.size), dtype=np.float64)
    parameters[0] = initial.parameters
    for step in range(n_steps):
        noise = rng.normal(size=(batch_size, initial.dimension))
        estimate = initial.reverse_kl_estimate(target, noise)
        updated = resolved_optimizer.step(initial.parameters, -estimate.gradient)
        initial.set_parameters(updated)
        objectives[step] = estimate.value
        gradient_norms[step] = float(np.linalg.norm(estimate.gradient))
        parameters[step + 1] = initial.parameters
    return VariationalFitResult(objectives, gradient_norms, parameters, initial.freeze())


def fit_forward_kl_diagonal_gaussian(
    reference_samples: ArrayLike,
    *,
    minimum_scale: float = 1e-6,
) -> FrozenDiagonalGaussian:
    """Fit the inclusive-KL Gaussian optimum from trusted reference samples."""

    samples = np.asarray(reference_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] == 0:
        raise ValueError("reference_samples must have shape (n >= 2, dimension)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("reference_samples must be finite")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive and finite")
    mean = np.mean(samples, axis=0)
    scale = np.maximum(np.std(samples, axis=0, ddof=0), minimum_scale)
    return FrozenDiagonalGaussian(mean, np.log(scale))
