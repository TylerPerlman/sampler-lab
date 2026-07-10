"""Overdamped Langevin discretizations and Metropolis correction."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.linalg import as_positive_definite
from sampler_lab.core.protocols import DifferentiableLogDensity
from sampler_lab.core.results import Transition
from sampler_lab.mcmc.metropolis import log_metropolis_hastings_ratio

Array = NDArray[np.float64]
_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _as_vector(state: ArrayLike, *, name: str = "state") -> Array:
    value = np.asarray(state, dtype=np.float64)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class PreconditionerEvaluation:
    """One internally consistent evaluation of a matrix field and its geometry."""

    matrix: Array
    divergence: Array
    cholesky: Array
    log_determinant: float


class Preconditioner(Protocol):
    """Positive-definite matrix field and its row-wise divergence."""

    def evaluate_at(self, state: Array) -> PreconditionerEvaluation:
        """Evaluate matrix, divergence, Cholesky factor, and log determinant once."""

    def matrix_at(self, state: Array) -> Array:
        """Return the positive-definite preconditioning matrix ``M(x)``."""

    def divergence_at(self, state: Array) -> Array:
        """Return ``div M`` with component ``sum_j partial_j M_ij``."""

    def cholesky_at(self, state: Array) -> Array:
        """Return a lower Cholesky factor of ``M(x)``."""

    def log_determinant_at(self, state: Array) -> float:
        """Return ``log det M(x)``."""


@dataclass(slots=True)
class ConstantPreconditioner:
    """Constant symmetric positive-definite Langevin preconditioner."""

    matrix: ArrayLike
    _matrix: Array = field(init=False, repr=False)
    _cholesky: Array = field(init=False, repr=False)
    _log_determinant: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._matrix = np.array(as_positive_definite(self.matrix), dtype=np.float64, copy=True)
        self._cholesky = np.asarray(np.linalg.cholesky(self._matrix), dtype=np.float64)
        self._log_determinant = float(2.0 * np.sum(np.log(np.diag(self._cholesky))))

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[0])

    def _check_state(self, state: Array) -> None:
        if state.shape != (self.dimension,):
            raise ValueError("state dimension does not match the preconditioner")

    def evaluate_at(self, state: Array) -> PreconditionerEvaluation:
        self._check_state(state)
        return PreconditionerEvaluation(
            matrix=self._matrix.copy(),
            divergence=np.zeros(self.dimension, dtype=np.float64),
            cholesky=self._cholesky.copy(),
            log_determinant=self._log_determinant,
        )

    def matrix_at(self, state: Array) -> Array:
        return self.evaluate_at(state).matrix

    def divergence_at(self, state: Array) -> Array:
        return self.evaluate_at(state).divergence

    def cholesky_at(self, state: Array) -> Array:
        return self.evaluate_at(state).cholesky

    def log_determinant_at(self, state: Array) -> float:
        return self.evaluate_at(state).log_determinant


@dataclass(frozen=True, slots=True)
class FunctionalPreconditioner:
    """Position-dependent preconditioner with an explicit divergence function."""

    matrix_function: Callable[[Array], ArrayLike]
    divergence_function: Callable[[Array], ArrayLike]

    def evaluate_at(self, state: Array) -> PreconditionerEvaluation:
        current = _as_vector(state)
        raw_matrix = np.asarray(
            self.matrix_function(np.array(current, copy=True)), dtype=np.float64
        )
        if raw_matrix.shape != (current.size, current.size):
            raise ValueError("preconditioner matrix must match the state dimension")
        if not np.all(np.isfinite(raw_matrix)):
            raise ValueError("preconditioner matrix must be finite")
        if not np.allclose(raw_matrix, raw_matrix.T, atol=1e-12, rtol=0.0):
            raise ValueError("preconditioner matrix must be symmetric")
        cholesky = np.asarray(np.linalg.cholesky(raw_matrix), dtype=np.float64)
        divergence = np.asarray(
            self.divergence_function(np.array(current, copy=True)), dtype=np.float64
        )
        if divergence.shape != current.shape or not np.all(np.isfinite(divergence)):
            raise ValueError("preconditioner divergence must match the finite state vector")
        return PreconditionerEvaluation(
            matrix=np.asarray(raw_matrix, dtype=np.float64),
            divergence=divergence,
            cholesky=cholesky,
            log_determinant=float(2.0 * np.sum(np.log(np.diag(cholesky)))),
        )

    def matrix_at(self, state: Array) -> Array:
        return self.evaluate_at(state).matrix

    def divergence_at(self, state: Array) -> Array:
        return self.evaluate_at(state).divergence

    def cholesky_at(self, state: Array) -> Array:
        return self.evaluate_at(state).cholesky

    def log_determinant_at(self, state: Array) -> float:
        return self.evaluate_at(state).log_determinant


def _identity_geometry(dimension: int) -> ConstantPreconditioner:
    return ConstantPreconditioner(np.eye(dimension, dtype=np.float64))


def _resolve_preconditioner(
    preconditioner: Preconditioner | ArrayLike | None,
    dimension: int,
) -> Preconditioner:
    if preconditioner is None:
        return _identity_geometry(dimension)
    resolved: Preconditioner
    if isinstance(preconditioner, (np.ndarray, list, tuple)):
        resolved = ConstantPreconditioner(preconditioner)
    else:
        resolved = cast(Preconditioner, preconditioner)
    return resolved


def overdamped_langevin_drift(
    target: DifferentiableLogDensity,
    state: ArrayLike,
    *,
    preconditioner: Preconditioner | ArrayLike | None = None,
    include_divergence: bool = True,
) -> Array:
    """Return ``M grad(log pi) + div M`` for the preconditioned diffusion."""

    current = _as_vector(state)
    geometry = _resolve_preconditioner(preconditioner, current.size)
    gradient = np.asarray(target.grad_log_prob(np.array(current, copy=True)), dtype=np.float64)
    if gradient.shape != current.shape or not np.all(np.isfinite(gradient)):
        raise ValueError("target gradient must be a finite vector matching the state")
    evaluation = geometry.evaluate_at(current)
    drift = evaluation.matrix @ gradient
    if include_divergence:
        drift = drift + evaluation.divergence
    return np.asarray(drift, dtype=np.float64)


def gaussian_log_transition_density(
    destination: ArrayLike,
    mean: ArrayLike,
    covariance_cholesky: ArrayLike,
    log_determinant_covariance: float,
) -> float:
    """Evaluate a full-rank multivariate Gaussian transition density."""

    value = _as_vector(destination, name="destination")
    center = _as_vector(mean, name="mean")
    cholesky = np.asarray(covariance_cholesky, dtype=np.float64)
    if center.shape != value.shape:
        return float("-inf")
    if cholesky.shape != (value.size, value.size):
        raise ValueError("covariance_cholesky has the wrong shape")
    if not np.all(np.isfinite(cholesky)) or not np.isfinite(log_determinant_covariance):
        raise ValueError("Gaussian covariance information must be finite")
    displacement = value - center
    standardized = np.linalg.solve(cholesky, displacement)
    return float(
        -0.5 * (value.size * _LOG_TWO_PI + log_determinant_covariance + standardized @ standardized)
    )


@dataclass(slots=True)
class UnadjustedLangevinKernel:
    """Euler discretization of preconditioned overdamped Langevin dynamics."""

    target: DifferentiableLogDensity
    step_size: float
    preconditioner: Preconditioner | ArrayLike | None = None
    include_divergence: bool = True
    counter: OperationCounter | None = None
    _resolved_dimension: int | None = field(default=None, init=False, repr=False)
    _geometry: Preconditioner | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")

    def _geometry_for(self, state: Array) -> Preconditioner:
        if self._geometry is None:
            self._geometry = _resolve_preconditioner(self.preconditioner, state.size)
            self._resolved_dimension = state.size
        elif self._resolved_dimension != state.size:
            raise ValueError("state dimension changed after kernel initialization")
        return self._geometry

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        """Advance one ULA step with covariance ``2 h M(x)``."""

        current = _as_vector(state)
        geometry = self._geometry_for(current)
        gradient = np.asarray(
            self.target.grad_log_prob(np.array(current, copy=True)), dtype=np.float64
        )
        if gradient.shape != current.shape or not np.all(np.isfinite(gradient)):
            raise ValueError("target gradient must be a finite vector matching the state")
        evaluation = geometry.evaluate_at(current)
        divergence = evaluation.divergence if self.include_divergence else np.zeros_like(current)
        drift = evaluation.matrix @ gradient + divergence
        if self.counter is not None:
            self.counter.gradient_evaluations += 1
            self.counter.normal_draws += current.size
            if isinstance(geometry, FunctionalPreconditioner):
                self.counter.matrix_factorizations += 1
        noise = rng.normal(size=current.size)
        next_state = (
            current
            + self.step_size * drift
            + np.sqrt(2.0 * self.step_size) * (evaluation.cholesky @ noise)
        )
        return Transition(
            state=np.asarray(next_state, dtype=np.float64),
            diagnostics={
                "step_size": float(self.step_size),
                "gradient_norm": float(np.linalg.norm(gradient)),
                "drift_norm": float(np.linalg.norm(drift)),
                "divergence_norm": float(np.linalg.norm(divergence)),
            },
        )


@dataclass(slots=True)
class MetropolisAdjustedLangevinKernel:
    """Metropolized Euler proposal for preconditioned Langevin dynamics.

    With a constant identity preconditioner this is ordinary MALA. With a matrix
    field it remains a valid Metropolis--Hastings kernel because both state-dependent
    Gaussian proposal densities are evaluated explicitly.
    """

    target: DifferentiableLogDensity
    step_size: float
    preconditioner: Preconditioner | ArrayLike | None = None
    include_divergence: bool = True
    counter: OperationCounter | None = None
    _resolved_dimension: int | None = field(default=None, init=False, repr=False)
    _geometry: Preconditioner | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")

    def _geometry_for(self, state: Array) -> Preconditioner:
        if self._geometry is None:
            self._geometry = _resolve_preconditioner(self.preconditioner, state.size)
            self._resolved_dimension = state.size
        elif self._resolved_dimension != state.size:
            raise ValueError("state dimension changed after kernel initialization")
        return self._geometry

    def _proposal_geometry(
        self,
        state: Array,
        geometry: Preconditioner,
    ) -> tuple[Array, Array, float, Array, Array]:
        gradient = np.asarray(
            self.target.grad_log_prob(np.array(state, copy=True)), dtype=np.float64
        )
        if gradient.shape != state.shape or not np.all(np.isfinite(gradient)):
            raise ValueError("target gradient must be a finite vector matching the state")
        evaluation = geometry.evaluate_at(state)
        divergence = evaluation.divergence if self.include_divergence else np.zeros_like(state)
        drift = evaluation.matrix @ gradient + divergence
        mean = state + self.step_size * drift
        covariance_cholesky = np.sqrt(2.0 * self.step_size) * evaluation.cholesky
        log_determinant_covariance = (
            state.size * np.log(2.0 * self.step_size) + evaluation.log_determinant
        )
        return (
            gradient,
            drift,
            float(log_determinant_covariance),
            mean,
            covariance_cholesky,
        )

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        """Propose one Langevin step and apply the MH correction."""

        current = _as_vector(state)
        geometry = self._geometry_for(current)
        current_log_target = float(self.target.log_prob(np.array(current, copy=True)))
        if not np.isfinite(current_log_target):
            raise ValueError("the current MALA state must have finite target log density")
        (
            current_gradient,
            current_drift,
            current_log_det_covariance,
            forward_mean,
            forward_cholesky,
        ) = self._proposal_geometry(current, geometry)
        if self.counter is not None:
            self.counter.gradient_evaluations += 1
            self.counter.log_density_evaluations += 1
            self.counter.normal_draws += current.size
            if isinstance(geometry, FunctionalPreconditioner):
                self.counter.matrix_factorizations += 1
        proposed = np.asarray(
            forward_mean + forward_cholesky @ rng.normal(size=current.size),
            dtype=np.float64,
        )
        proposed_log_target = float(self.target.log_prob(np.array(proposed, copy=True)))
        if np.isnan(proposed_log_target) or proposed_log_target == float("inf"):
            raise ValueError("proposed target log density must be finite or -inf")
        if self.counter is not None:
            self.counter.log_density_evaluations += 1
            self.counter.proposal_density_evaluations += 1
        forward_log_proposal = gaussian_log_transition_density(
            proposed,
            forward_mean,
            forward_cholesky,
            current_log_det_covariance,
        )

        if proposed_log_target == float("-inf"):
            proposed_gradient = np.full_like(current, np.nan)
            proposed_drift = np.full_like(current, np.nan)
            reverse_log_proposal = float("-inf")
        else:
            (
                proposed_gradient,
                proposed_drift,
                proposed_log_det_covariance,
                reverse_mean,
                reverse_cholesky,
            ) = self._proposal_geometry(proposed, geometry)
            reverse_log_proposal = gaussian_log_transition_density(
                current,
                reverse_mean,
                reverse_cholesky,
                proposed_log_det_covariance,
            )
            if self.counter is not None:
                self.counter.gradient_evaluations += 1
                self.counter.proposal_density_evaluations += 1
                if isinstance(geometry, FunctionalPreconditioner):
                    self.counter.matrix_factorizations += 1

        log_ratio = log_metropolis_hastings_ratio(
            current_log_target=current_log_target,
            proposed_log_target=proposed_log_target,
            forward_log_proposal=forward_log_proposal,
            reverse_log_proposal=reverse_log_proposal,
        )
        if self.counter is not None:
            self.counter.uniform_draws += 1
        draw = max(float(rng.random()), float(np.finfo(np.float64).tiny))
        accepted = math.log(draw) < min(0.0, float(log_ratio))
        next_state = proposed if accepted else current
        return Transition(
            state=np.array(next_state, dtype=np.float64, copy=True),
            accepted=accepted,
            log_acceptance_ratio=log_ratio,
            diagnostics={
                "step_size": float(self.step_size),
                "current_log_target": current_log_target,
                "proposed_log_target": proposed_log_target,
                "forward_log_proposal": forward_log_proposal,
                "reverse_log_proposal": reverse_log_proposal,
                "current_gradient_norm": float(np.linalg.norm(current_gradient)),
                "proposed_gradient_norm": float(np.linalg.norm(proposed_gradient)),
                "current_drift_norm": float(np.linalg.norm(current_drift)),
                "proposed_drift_norm": float(np.linalg.norm(proposed_drift)),
            },
        )
