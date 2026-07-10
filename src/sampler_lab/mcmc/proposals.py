"""Proposal kernels for Metropolis--Hastings algorithms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter

Array = NDArray[np.float64]

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _as_finite_state(state: ArrayLike, *, name: str = "state") -> Array:
    values = np.asarray(state, dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _broadcast_positive_scale(scale: ArrayLike, shape: tuple[int, ...]) -> Array:
    values = np.asarray(scale, dtype=np.float64)
    try:
        broadcast = np.broadcast_to(values, shape)
    except ValueError as error:
        raise ValueError("proposal scale is not broadcastable to the state shape") from error
    if not np.all(np.isfinite(broadcast)) or np.any(broadcast <= 0.0):
        raise ValueError("proposal scales must be positive and finite")
    return np.asarray(broadcast, dtype=np.float64)


def _diagonal_gaussian_log_density(value: Array, mean: Array, scale: Array) -> float:
    standardized = (value - mean) / scale
    return float(
        -0.5 * np.sum(standardized * standardized)
        - np.sum(np.log(scale))
        - 0.5 * value.size * _LOG_TWO_PI
    )


class Proposal(Protocol):
    """A proposal kernel with an explicit transition density or mass."""

    def sample(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        """Draw a proposal from ``q(. | state)``."""

    def log_transition_density(
        self,
        to_state: Array,
        from_state: Array,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        """Return ``log q(to_state | from_state)``."""


@dataclass(frozen=True, slots=True)
class GaussianRandomWalkProposal:
    """Diagonal Gaussian random walk ``Y = X + scale * Z``."""

    scale: ArrayLike

    def sample(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        current = _as_finite_state(state)
        scales = _broadcast_positive_scale(self.scale, current.shape)
        if counter is not None:
            counter.normal_draws += current.size
        return np.asarray(current + scales * rng.normal(size=current.shape), dtype=np.float64)

    def log_transition_density(
        self,
        to_state: Array,
        from_state: Array,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        destination = _as_finite_state(to_state, name="to_state")
        source = _as_finite_state(from_state, name="from_state")
        if destination.shape != source.shape:
            return float("-inf")
        scales = _broadcast_positive_scale(self.scale, source.shape)
        if counter is not None:
            counter.proposal_density_evaluations += 1
        return _diagonal_gaussian_log_density(destination, source, scales)


@dataclass(frozen=True, slots=True)
class MultivariateGaussianRandomWalkProposal:
    """Full-covariance Gaussian random walk ``Y = X + L Z``."""

    covariance: ArrayLike
    _covariance: Array = field(init=False, repr=False)
    _cholesky: Array = field(init=False, repr=False)
    _log_determinant: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("proposal covariance must be square")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("proposal covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
            raise ValueError("proposal covariance must be symmetric")
        cholesky = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
        object.__setattr__(self, "_covariance", np.array(covariance, copy=True))
        object.__setattr__(self, "_cholesky", cholesky)
        object.__setattr__(
            self,
            "_log_determinant",
            float(2.0 * np.sum(np.log(np.diag(cholesky)))),
        )

    @property
    def dimension(self) -> int:
        return int(self._covariance.shape[0])

    def sample(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        current = _as_finite_state(state)
        if current.shape != (self.dimension,):
            raise ValueError("state dimension must match the proposal covariance")
        if counter is not None:
            counter.normal_draws += current.size
        return np.asarray(
            current + self._cholesky @ rng.normal(size=current.size), dtype=np.float64
        )

    def log_transition_density(
        self,
        to_state: Array,
        from_state: Array,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        destination = _as_finite_state(to_state, name="to_state")
        source = _as_finite_state(from_state, name="from_state")
        if destination.shape != source.shape or source.shape != (self.dimension,):
            return float("-inf")
        displacement = destination - source
        standardized = np.linalg.solve(self._cholesky, displacement)
        if counter is not None:
            counter.proposal_density_evaluations += 1
        return float(
            -0.5
            * (self.dimension * _LOG_TWO_PI + self._log_determinant + standardized @ standardized)
        )


@dataclass(frozen=True, slots=True)
class GaussianIndependenceProposal:
    """State-independent diagonal Gaussian proposal."""

    mean: ArrayLike
    scale: ArrayLike

    def _parameters(self, shape: tuple[int, ...]) -> tuple[Array, Array]:
        means = np.asarray(np.broadcast_to(np.asarray(self.mean, dtype=np.float64), shape))
        if not np.all(np.isfinite(means)):
            raise ValueError("proposal means must be finite")
        scales = _broadcast_positive_scale(self.scale, shape)
        return np.asarray(means, dtype=np.float64), scales

    def sample(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        current = _as_finite_state(state)
        means, scales = self._parameters(current.shape)
        if counter is not None:
            counter.normal_draws += current.size
        return np.asarray(means + scales * rng.normal(size=current.shape), dtype=np.float64)

    def log_transition_density(
        self,
        to_state: Array,
        from_state: Array,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        destination = _as_finite_state(to_state, name="to_state")
        source = _as_finite_state(from_state, name="from_state")
        if destination.shape != source.shape:
            return float("-inf")
        means, scales = self._parameters(source.shape)
        if counter is not None:
            counter.proposal_density_evaluations += 1
        return _diagonal_gaussian_log_density(destination, means, scales)


@dataclass(frozen=True, slots=True)
class StateDependentGaussianProposal:
    """Diagonal Gaussian proposal with state-dependent mean and scale.

    This class deliberately evaluates both the forward and reverse parameters,
    making proposal asymmetry visible in the Metropolis--Hastings ratio.
    """

    mean: Callable[[Array], ArrayLike]
    scale: Callable[[Array], ArrayLike]

    def _parameters(self, state: Array) -> tuple[Array, Array]:
        means = np.asarray(self.mean(state), dtype=np.float64)
        if means.shape != state.shape:
            raise ValueError("state-dependent proposal mean must match the state shape")
        if not np.all(np.isfinite(means)):
            raise ValueError("state-dependent proposal mean must be finite")
        scales = _broadcast_positive_scale(self.scale(state), state.shape)
        return means, scales

    def sample(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        current = _as_finite_state(state)
        means, scales = self._parameters(current)
        if counter is not None:
            counter.normal_draws += current.size
        return np.asarray(means + scales * rng.normal(size=current.shape), dtype=np.float64)

    def log_transition_density(
        self,
        to_state: Array,
        from_state: Array,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        destination = _as_finite_state(to_state, name="to_state")
        source = _as_finite_state(from_state, name="from_state")
        if destination.shape != source.shape:
            return float("-inf")
        means, scales = self._parameters(source)
        if counter is not None:
            counter.proposal_density_evaluations += 1
        return _diagonal_gaussian_log_density(destination, means, scales)


@dataclass(frozen=True, slots=True)
class CoordinateGaussianRandomWalkProposal:
    """Randomly perturb one coordinate with a symmetric Gaussian increment.

    The transition density is interpreted with respect to the mixture of the
    one-dimensional coordinate reference measures. Proposals produced by this
    object differ from the source in exactly one coordinate almost surely.
    """

    scale: ArrayLike
    probabilities: ArrayLike | None = None

    def _parameters(self, state: Array) -> tuple[Array, Array]:
        flat = state.reshape(-1)
        scales = _broadcast_positive_scale(self.scale, flat.shape)
        if self.probabilities is None:
            probabilities = np.full(flat.size, 1.0 / flat.size, dtype=np.float64)
        else:
            probabilities = np.asarray(self.probabilities, dtype=np.float64)
            if probabilities.shape != flat.shape:
                raise ValueError("coordinate probabilities must match the flattened state")
            if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
                raise ValueError("coordinate probabilities must be finite and nonnegative")
            total = float(np.sum(probabilities))
            if total <= 0.0:
                raise ValueError("coordinate probabilities must have positive mass")
            probabilities = probabilities / total
        return scales, probabilities

    def sample(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        current = _as_finite_state(state)
        scales, probabilities = self._parameters(current)
        coordinate = int(rng.choice(current.size, p=probabilities))
        proposal = np.array(current, dtype=np.float64, copy=True).reshape(-1)
        proposal[coordinate] += float(scales[coordinate] * rng.normal())
        if counter is not None:
            counter.uniform_draws += 1
            counter.normal_draws += 1
        return proposal.reshape(current.shape)

    def log_transition_density(
        self,
        to_state: Array,
        from_state: Array,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        destination = _as_finite_state(to_state, name="to_state")
        source = _as_finite_state(from_state, name="from_state")
        if destination.shape != source.shape:
            return float("-inf")
        difference = (destination - source).reshape(-1)
        changed = np.flatnonzero(difference != 0.0)
        if changed.size != 1:
            return float("-inf")
        coordinate = int(changed[0])
        scales, probabilities = self._parameters(source)
        probability = float(probabilities[coordinate])
        if probability <= 0.0:
            return float("-inf")
        if counter is not None:
            counter.proposal_density_evaluations += 1
        increment = float(difference[coordinate])
        scale = float(scales[coordinate])
        return float(
            np.log(probability) - np.log(scale) - 0.5 * _LOG_TWO_PI - 0.5 * (increment / scale) ** 2
        )
