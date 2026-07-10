"""Swappable rewards for adaptive and policy-gradient MCMC."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PolicyTransition:
    """One proposal, acceptance decision, and policy-side diagnostics."""

    current_state: Array
    proposed_state: Array
    next_state: Array
    accepted: bool
    log_acceptance_ratio: float
    proposal_entropy: float
    current_features: Array
    next_features: Array
    log_target_ratio: float | None = None
    forward_log_proposal: float | None = None

    def __post_init__(self) -> None:
        if (
            self.current_state.shape != self.proposed_state.shape
            or self.current_state.shape != self.next_state.shape
        ):
            raise ValueError("transition states must have equal shapes")
        if (
            self.current_features.ndim != 1
            or self.next_features.shape != self.current_features.shape
        ):
            raise ValueError("transition features must be equal-length vectors")
        if not np.all(np.isfinite(self.current_state)) or not np.all(
            np.isfinite(self.proposed_state)
        ):
            raise ValueError("transition states must be finite")
        if not np.all(np.isfinite(self.next_state)) or not np.all(
            np.isfinite(self.current_features)
        ):
            raise ValueError("transition state/features must be finite")
        if not np.all(np.isfinite(self.next_features)):
            raise ValueError("transition features must be finite")
        if np.isnan(self.log_acceptance_ratio) or self.log_acceptance_ratio == np.inf:
            raise ValueError("log acceptance ratio must be finite or negative infinity")
        if not np.isfinite(self.proposal_entropy):
            raise ValueError("proposal entropy must be finite")
        if self.log_target_ratio is not None and (
            np.isnan(self.log_target_ratio) or self.log_target_ratio == np.inf
        ):
            raise ValueError("log target ratio must be finite or negative infinity")
        if self.forward_log_proposal is not None and not np.isfinite(self.forward_log_proposal):
            raise ValueError("forward proposal log density must be finite")


class PolicyObjective(Protocol):
    """One-step reward used by a policy-gradient rollout."""

    def reward(
        self,
        transition: PolicyTransition,
        costs: OperationCounter | None = None,
    ) -> float:
        """Return a finite scalar reward."""


@dataclass(frozen=True, slots=True)
class AcceptanceObjective:
    """Acceptance-only baseline, intentionally vulnerable to zero-size proposals."""

    use_probability: bool = False

    def reward(
        self,
        transition: PolicyTransition,
        costs: OperationCounter | None = None,
    ) -> float:
        del costs
        if self.use_probability:
            return float(np.exp(min(0.0, transition.log_acceptance_ratio)))
        return float(transition.accepted)


@dataclass(frozen=True, slots=True)
class AcceptedSquaredJumpObjective:
    """Accepted squared jump in Euclidean or user-supplied Mahalanobis geometry."""

    metric: ArrayLike | None = None

    def reward(
        self,
        transition: PolicyTransition,
        costs: OperationCounter | None = None,
    ) -> float:
        del costs
        if not transition.accepted:
            return 0.0
        jump = transition.proposed_state - transition.current_state
        if self.metric is None:
            return float(jump @ jump)
        metric = np.asarray(self.metric, dtype=np.float64)
        if metric.shape != (jump.size, jump.size) or not np.all(np.isfinite(metric)):
            raise ValueError("metric must be a finite square matrix matching the state")
        return float(jump @ metric @ jump)


@dataclass(frozen=True, slots=True)
class FeatureJumpObjective:
    """Accepted squared displacement in declared diagnostic features."""

    weights: ArrayLike | None = None

    def reward(
        self,
        transition: PolicyTransition,
        costs: OperationCounter | None = None,
    ) -> float:
        del costs
        if not transition.accepted:
            return 0.0
        difference = transition.next_features - transition.current_features
        if self.weights is None:
            return float(difference @ difference)
        weights = np.asarray(self.weights, dtype=np.float64)
        if (
            weights.shape != difference.shape
            or np.any(weights < 0.0)
            or not np.all(np.isfinite(weights))
        ):
            raise ValueError("feature weights must be finite, nonnegative, and match features")
        return float(np.sum(weights * difference * difference))


@dataclass(frozen=True, slots=True)
class GeneralizedSpeedObjective:
    r"""Sampled lower bound on log generalized speed.

    For proposal entropy ``H`` and Metropolis ratio ``r``, this returns
    ``min(0, log r) + beta * H``. Unlike accepted-jump rewards, it learns from
    rejected proposals as well.
    """

    beta: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be nonnegative and finite")

    def reward(
        self,
        transition: PolicyTransition,
        costs: OperationCounter | None = None,
    ) -> float:
        del costs
        return float(
            min(0.0, transition.log_acceptance_ratio) + self.beta * transition.proposal_entropy
        )


@dataclass(frozen=True, slots=True)
class ContrastiveDivergenceLowerBoundObjective:
    r"""One-proposal lower-bound reward for contrastive divergence.

    With Metropolis acceptance probability ``a``, target log-ratio ``delta``,
    and forward proposal log density ``log q(y|x)``, the reward is

    ``a * delta - a * log(a) - a * log q(y|x)``.

    The final two terms lower-bound the entropy of the MH transition kernel;
    the omitted rejection-atom entropy is nonnegative.
    """

    exploitation_weight: float = 1.0
    entropy_weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.exploitation_weight) or self.exploitation_weight < 0.0:
            raise ValueError("exploitation_weight must be nonnegative and finite")
        if not np.isfinite(self.entropy_weight) or self.entropy_weight < 0.0:
            raise ValueError("entropy_weight must be nonnegative and finite")

    def reward(
        self,
        transition: PolicyTransition,
        costs: OperationCounter | None = None,
    ) -> float:
        del costs
        if transition.log_target_ratio is None or transition.forward_log_proposal is None:
            raise ValueError(
                "contrastive-divergence reward requires target and proposal log ratios"
            )
        acceptance = float(np.exp(min(0.0, transition.log_acceptance_ratio)))
        exploitation = acceptance * transition.log_target_ratio
        acceptance_entropy = 0.0 if acceptance == 0.0 else -acceptance * np.log(acceptance)
        proposal_entropy_bound = -acceptance * transition.forward_log_proposal
        return float(
            self.exploitation_weight * exploitation
            + self.entropy_weight * (acceptance_entropy + proposal_entropy_bound)
        )


@dataclass(frozen=True, slots=True)
class OperationCostWeights:
    """Linear cost model for reward normalization."""

    log_density: float = 1.0
    proposal_density: float = 1.0
    gradient: float = 1.0
    hessian: float = 1.0
    factorization: float = 1.0
    policy: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.log_density,
            self.proposal_density,
            self.gradient,
            self.hessian,
            self.factorization,
            self.policy,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("operation weights must be nonnegative and finite")

    def cost(self, counts: OperationCounter) -> float:
        return float(
            self.log_density * counts.log_density_evaluations
            + self.proposal_density * counts.proposal_density_evaluations
            + self.gradient * counts.gradient_evaluations
            + self.hessian * counts.hessian_evaluations
            + self.factorization * counts.matrix_factorizations
            + self.policy * counts.policy_evaluations
        )


@dataclass(frozen=True, slots=True)
class CostNormalizedObjective:
    """Divide another reward by a declared operation-cost model."""

    objective: PolicyObjective
    weights: OperationCostWeights = OperationCostWeights()
    minimum_cost: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_cost) or self.minimum_cost <= 0.0:
            raise ValueError("minimum_cost must be positive and finite")

    def reward(
        self,
        transition: PolicyTransition,
        costs: OperationCounter | None = None,
    ) -> float:
        if costs is None:
            raise ValueError("cost-normalized rewards require operation counts")
        raw = self.objective.reward(transition, costs)
        return float(raw / max(self.minimum_cost, self.weights.cost(costs)))


def make_feature_map(functions: tuple[Callable[[Array], float], ...]) -> Callable[[Array], Array]:
    """Combine scalar observables into one validated feature map."""

    if not functions:
        raise ValueError("at least one feature function is required")

    def feature_map(state: Array) -> Array:
        values = np.asarray([function(state) for function in functions], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("feature functions must return finite values")
        return values

    return feature_map
