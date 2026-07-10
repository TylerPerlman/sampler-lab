"""Generic Metropolis--Hastings transitions with explicit proposal accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import LogDensity
from sampler_lab.core.results import Transition
from sampler_lab.mcmc.proposals import Array, Proposal


def log_metropolis_hastings_ratio(
    *,
    current_log_target: float,
    proposed_log_target: float,
    forward_log_proposal: float,
    reverse_log_proposal: float,
) -> float:
    """Return the logarithm of the untruncated MH acceptance ratio.

    The current state must lie in the target support and a sampled proposal must
    have positive forward proposal density. A proposal outside target support or
    without a possible reverse transition is assigned ratio ``-inf``.
    """

    if not np.isfinite(current_log_target):
        raise ValueError("the current state must have finite target log density")
    if np.isnan(proposed_log_target) or proposed_log_target == float("inf"):
        raise ValueError("proposed target log density must be finite or -inf")
    if not np.isfinite(forward_log_proposal):
        raise ValueError("a sampled proposal must have finite forward log density")
    if np.isnan(reverse_log_proposal) or reverse_log_proposal == float("inf"):
        raise ValueError("reverse proposal log density must be finite or -inf")
    if proposed_log_target == float("-inf") or reverse_log_proposal == float("-inf"):
        return float("-inf")
    return float(
        proposed_log_target - current_log_target + reverse_log_proposal - forward_log_proposal
    )


@dataclass(slots=True)
class MetropolisHastingsKernel:
    """One-step Metropolis--Hastings kernel for array-valued states."""

    target: LogDensity
    proposal: Proposal
    counter: OperationCounter | None = None

    def _log_target(self, state: Array) -> float:
        if self.counter is not None:
            self.counter.log_density_evaluations += 1
        return float(self.target.log_prob(state))

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        """Propose once and retain the current state after rejection."""

        current = np.array(state, dtype=np.float64, copy=True)
        if current.size == 0 or not np.all(np.isfinite(current)):
            raise ValueError("state must be nonempty and finite")
        current_log_target = self._log_target(current)
        proposal = np.asarray(
            self.proposal.sample(np.array(current, copy=True), rng, counter=self.counter),
            dtype=np.float64,
        )
        if proposal.shape != current.shape:
            raise ValueError("proposal state shape does not match the current state")
        if not np.all(np.isfinite(proposal)):
            raise ValueError("proposal state must be finite")

        proposed_log_target = self._log_target(proposal)
        forward = self.proposal.log_transition_density(
            proposal,
            current,
            counter=self.counter,
        )
        reverse = self.proposal.log_transition_density(
            current,
            proposal,
            counter=self.counter,
        )
        log_ratio = log_metropolis_hastings_ratio(
            current_log_target=current_log_target,
            proposed_log_target=proposed_log_target,
            forward_log_proposal=forward,
            reverse_log_proposal=reverse,
        )
        if self.counter is not None:
            self.counter.uniform_draws += 1
        draw = float(rng.random())
        accepted = bool(np.log(draw) < min(0.0, log_ratio))
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
            },
        )
