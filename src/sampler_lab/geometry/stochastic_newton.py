"""Stochastic Newton proposals and their Metropolis correction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import TwiceDifferentiableLogDensity
from sampler_lab.core.results import Transition
from sampler_lab.dynamics.langevin import gaussian_log_transition_density
from sampler_lab.geometry.hessian import RepairMethod, repair_positive_definite
from sampler_lab.mcmc.metropolis import log_metropolis_hastings_ratio

Array = NDArray[np.float64]


def _as_vector(value: ArrayLike, *, name: str = "state") -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


@dataclass(frozen=True, slots=True)
class StochasticNewtonEvaluation:
    """Local Gaussian proposal geometry at one state."""

    mean: Array
    metric: Array
    covariance: Array
    covariance_cholesky: Array
    log_determinant_covariance: float
    minimum_raw_precision_eigenvalue: float
    repair_norm: float


@dataclass(slots=True)
class StochasticNewtonProposal:
    r"""Local Gaussian approximation with explicit overdamped-Langevin scaling.

    For local metric ``S(x) = [-D^2 log pi(x)]^{-1}``, the proposal is

    ``Y = x + h S(x) grad log pi(x) + sqrt(2 h S(x)) Z``.

    The divergence correction is intentionally omitted.  Metropolization restores
    exact invariance and keeps the proposal practical when derivatives of ``S`` are
    unavailable.
    """

    target: TwiceDifferentiableLogDensity
    step_size: float
    repair_method: RepairMethod = "absolute"
    minimum_eigenvalue: float = 1e-6
    counter: OperationCounter | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if not np.isfinite(self.minimum_eigenvalue) or self.minimum_eigenvalue <= 0.0:
            raise ValueError("minimum_eigenvalue must be positive and finite")

    def evaluate_at(self, state: ArrayLike) -> StochasticNewtonEvaluation:
        """Evaluate gradient, repaired local precision, and Gaussian parameters once."""

        current = _as_vector(state)
        gradient = np.asarray(self.target.grad_log_prob(current), dtype=np.float64)
        hessian = np.asarray(self.target.hessian_log_prob(current), dtype=np.float64)
        if gradient.shape != current.shape or not np.all(np.isfinite(gradient)):
            raise ValueError("target gradient must match the finite state")
        if hessian.shape != (current.size, current.size) or not np.all(np.isfinite(hessian)):
            raise ValueError("target Hessian must match the finite state")
        repaired = repair_positive_definite(
            -hessian,
            method=self.repair_method,
            minimum_eigenvalue=self.minimum_eigenvalue,
        )
        identity = np.eye(current.size, dtype=np.float64)
        metric = np.asarray(np.linalg.solve(repaired.matrix, identity), dtype=np.float64)
        covariance = np.asarray(2.0 * self.step_size * metric, dtype=np.float64)
        cholesky = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
        log_determinant = float(2.0 * np.sum(np.log(np.diag(cholesky))))
        mean = np.asarray(current + self.step_size * (metric @ gradient), dtype=np.float64)
        if self.counter is not None:
            self.counter.gradient_evaluations += 1
            self.counter.hessian_evaluations += 1
            self.counter.matrix_factorizations += 1
        return StochasticNewtonEvaluation(
            mean=mean,
            metric=metric,
            covariance=covariance,
            covariance_cholesky=cholesky,
            log_determinant_covariance=log_determinant,
            minimum_raw_precision_eigenvalue=float(np.min(repaired.original_eigenvalues)),
            repair_norm=repaired.correction_frobenius_norm,
        )

    def sample(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        if counter is not None and counter is not self.counter:
            raise ValueError("pass the operation counter when constructing the proposal")
        evaluation = self.evaluate_at(state)
        if self.counter is not None:
            self.counter.normal_draws += evaluation.mean.size
        noise = rng.normal(size=evaluation.mean.size)
        return np.asarray(
            evaluation.mean + evaluation.covariance_cholesky @ noise,
            dtype=np.float64,
        )

    def log_transition_density(
        self,
        to_state: Array,
        from_state: Array,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        if counter is not None and counter is not self.counter:
            raise ValueError("pass the operation counter when constructing the proposal")
        destination = _as_vector(to_state, name="to_state")
        source = _as_vector(from_state, name="from_state")
        if destination.shape != source.shape:
            return float("-inf")
        evaluation = self.evaluate_at(source)
        if self.counter is not None:
            self.counter.proposal_density_evaluations += 1
        return gaussian_log_transition_density(
            destination,
            evaluation.mean,
            evaluation.covariance_cholesky,
            evaluation.log_determinant_covariance,
        )


@dataclass(slots=True)
class MetropolizedStochasticNewtonKernel:
    """Efficient MH kernel that evaluates each endpoint geometry only once."""

    target: TwiceDifferentiableLogDensity
    step_size: float
    repair_method: RepairMethod = "absolute"
    minimum_eigenvalue: float = 1e-6
    counter: OperationCounter | None = None
    _proposal: StochasticNewtonProposal = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._proposal = StochasticNewtonProposal(
            self.target,
            self.step_size,
            repair_method=self.repair_method,
            minimum_eigenvalue=self.minimum_eigenvalue,
            counter=self.counter,
        )

    def _log_target(self, state: Array) -> float:
        if self.counter is not None:
            self.counter.log_density_evaluations += 1
        return float(self.target.log_prob(state))

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = _as_vector(state)
        current_log_target = self._log_target(current)
        if not np.isfinite(current_log_target):
            raise ValueError("current state must lie in the target support")
        current_geometry = self._proposal.evaluate_at(current)
        noise = rng.normal(size=current.size)
        proposal = np.asarray(
            current_geometry.mean + current_geometry.covariance_cholesky @ noise,
            dtype=np.float64,
        )
        if self.counter is not None:
            self.counter.normal_draws += current.size
        proposed_log_target = self._log_target(proposal)
        forward = gaussian_log_transition_density(
            proposal,
            current_geometry.mean,
            current_geometry.covariance_cholesky,
            current_geometry.log_determinant_covariance,
        )
        if proposed_log_target == float("-inf"):
            reverse = float("-inf")
            proposed_geometry = None
        else:
            proposed_geometry = self._proposal.evaluate_at(proposal)
            reverse = gaussian_log_transition_density(
                current,
                proposed_geometry.mean,
                proposed_geometry.covariance_cholesky,
                proposed_geometry.log_determinant_covariance,
            )
        if self.counter is not None:
            self.counter.proposal_density_evaluations += 2 if proposed_geometry is not None else 1
            self.counter.uniform_draws += 1
        log_ratio = log_metropolis_hastings_ratio(
            current_log_target=current_log_target,
            proposed_log_target=proposed_log_target,
            forward_log_proposal=forward,
            reverse_log_proposal=reverse,
        )
        accepted = bool(np.log(float(rng.random())) < min(0.0, log_ratio))
        next_state = proposal if accepted else current
        return Transition(
            state=np.array(next_state, dtype=np.float64, copy=True),
            accepted=accepted,
            log_acceptance_ratio=log_ratio,
            diagnostics={
                "current_log_target": current_log_target,
                "proposed_log_target": proposed_log_target,
                "forward_log_proposal": forward,
                "reverse_log_proposal": reverse,
                "minimum_raw_precision_eigenvalue": (
                    current_geometry.minimum_raw_precision_eigenvalue
                ),
                "hessian_repair_norm": current_geometry.repair_norm,
            },
        )
