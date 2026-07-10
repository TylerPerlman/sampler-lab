"""Exact Metropolis correction for learned mixtures of proposal kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.numerics import logsumexp
from sampler_lab.core.protocols import LogDensity
from sampler_lab.learning.policies import FrozenLinearSoftmaxPolicy
from sampler_lab.mcmc.metropolis import MetropolisHastingsKernel
from sampler_lab.mcmc.proposals import Array, Proposal

FeatureMap = Callable[[Array], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class PolicyMixtureProposal:
    """State-dependent proposal mixture with its marginal density exposed.

    Sampling first draws a latent component action, but the MH correction uses
    the marginal mixture density in both directions. This is the exact way to use
    state-dependent component weights; simply selecting among already-corrected
    kernels with state-dependent probabilities is generally invalid.
    """

    policy: FrozenLinearSoftmaxPolicy
    proposals: tuple[Proposal, ...]
    feature_map: FeatureMap

    def __init__(
        self,
        policy: FrozenLinearSoftmaxPolicy,
        proposals: Sequence[Proposal],
        feature_map: FeatureMap,
    ) -> None:
        proposal_tuple = tuple(proposals)
        if len(proposal_tuple) != policy.n_actions:
            raise ValueError("proposal count must match policy actions")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "proposals", proposal_tuple)
        object.__setattr__(self, "feature_map", feature_map)

    def _probabilities(self, state: Array) -> NDArray[np.float64]:
        features = np.asarray(self.feature_map(np.array(state, copy=True)), dtype=np.float64)
        if features.shape != (self.policy.n_features,) or not np.all(np.isfinite(features)):
            raise ValueError("feature_map returned an invalid policy feature vector")
        return self.policy.probabilities(features)

    def sample(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        probabilities = self._probabilities(state)
        action = int(rng.choice(len(self.proposals), p=probabilities))
        if counter is not None:
            counter.uniform_draws += 1
            counter.increment("policy_evaluations")
        return np.asarray(
            self.proposals[action].sample(state, rng, counter=counter),
            dtype=np.float64,
        )

    def log_transition_density(
        self,
        to_state: Array,
        from_state: Array,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        probabilities = self._probabilities(from_state)
        component_terms = np.empty(len(self.proposals), dtype=np.float64)
        for index, proposal in enumerate(self.proposals):
            component_terms[index] = np.log(probabilities[index]) + proposal.log_transition_density(
                to_state,
                from_state,
                counter=counter,
            )
        if counter is not None:
            counter.increment("policy_evaluations")
        return float(logsumexp(component_terms))

    def corrected_kernel(
        self,
        target: LogDensity,
        *,
        counter: OperationCounter | None = None,
    ) -> MetropolisHastingsKernel:
        return MetropolisHastingsKernel(target, self, counter)
