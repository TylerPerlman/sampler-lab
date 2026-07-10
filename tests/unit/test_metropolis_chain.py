from dataclasses import dataclass

import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.core.results import Transition
from sampler_lab.mcmc import (
    MetropolisHastingsKernel,
    Proposal,
    log_metropolis_hastings_ratio,
    run_chain,
)


@dataclass(frozen=True)
class TwoStateTarget:
    probabilities: tuple[float, float] = (0.25, 0.75)

    def log_prob(self, x: np.ndarray) -> float:
        return float(np.log(self.probabilities[int(x[0])]))


@dataclass(frozen=True)
class AsymmetricTwoStateProposal:
    def sample(
        self,
        state: np.ndarray,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> np.ndarray:
        current = int(state[0])
        probability_one = 0.9 if current == 0 else 0.8
        if counter is not None:
            counter.uniform_draws += 1
        return np.array([float(rng.random() < probability_one)])

    def log_transition_density(
        self,
        to_state: np.ndarray,
        from_state: np.ndarray,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        if counter is not None:
            counter.proposal_density_evaluations += 1
        source = int(from_state[0])
        destination = int(to_state[0])
        probability_one = 0.9 if source == 0 else 0.8
        probability = probability_one if destination == 1 else 1.0 - probability_one
        return float(np.log(probability))


@dataclass(frozen=True)
class IrreversibleProposal:
    def sample(
        self,
        state: np.ndarray,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> np.ndarray:
        return np.asarray(state + 1.0)

    def log_transition_density(
        self,
        to_state: np.ndarray,
        from_state: np.ndarray,
        *,
        counter: OperationCounter | None = None,
    ) -> float:
        return 0.0 if np.array_equal(to_state, from_state + 1.0) else float("-inf")


@dataclass(frozen=True)
class StandardNormalTarget:
    def log_prob(self, x: np.ndarray) -> float:
        return float(-0.5 * np.dot(x, x))


def test_log_acceptance_ratio_includes_proposal_asymmetry() -> None:
    ratio = log_metropolis_hastings_ratio(
        current_log_target=np.log(0.25),
        proposed_log_target=np.log(0.75),
        forward_log_proposal=np.log(0.9),
        reverse_log_proposal=np.log(0.2),
    )
    assert np.exp(ratio) == pytest.approx(2.0 / 3.0)


def test_reverse_impossibility_forces_rejection_and_chain_retains_state() -> None:
    kernel = MetropolisHastingsKernel(StandardNormalTarget(), IrreversibleProposal())
    trajectory = run_chain(kernel, np.array([0.0]), np.random.default_rng(3), n_steps=5)

    assert trajectory.n_rejections == 5
    assert trajectory.acceptance_rate == 0.0
    assert trajectory.states[:, 0] == pytest.approx(np.zeros(6))
    assert np.all(np.isneginf(trajectory.log_acceptance_ratios))


def test_asymmetric_metropolis_kernel_recovers_two_state_target() -> None:
    proposal: Proposal = AsymmetricTwoStateProposal()
    kernel = MetropolisHastingsKernel(TwoStateTarget(), proposal)
    trajectory = run_chain(kernel, np.array([0.0]), np.random.default_rng(2022), n_steps=60_000)
    occupancy_one = float(np.mean(trajectory.states[2_000:, 0]))

    assert occupancy_one == pytest.approx(0.75, abs=0.015)
    assert trajectory.acceptance_rate == pytest.approx(0.925, abs=0.015)


def test_chain_runner_preserves_transition_diagnostics() -> None:
    @dataclass
    class IncrementKernel:
        def step(self, state: np.ndarray, rng: np.random.Generator) -> Transition:
            return Transition(state=state + 1.0, diagnostics={"old": float(state[0])})

    trajectory = run_chain(IncrementKernel(), np.array([0.0]), np.random.default_rng(1), n_steps=3)

    assert trajectory.states[:, 0] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert trajectory.acceptance_rate is None
    assert trajectory.transition_diagnostics[2]["old"] == 2.0
    assert trajectory.samples(discard=1, thin=2)[:, 0] == pytest.approx([1.0, 3.0])
