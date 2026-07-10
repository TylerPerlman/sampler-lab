import numpy as np
import pytest

from sampler_lab.markov import simulate_chain
from sampler_lab.models import deterministic_cycle, two_state_chain


@pytest.mark.statistical
def test_stationary_occupancy_matches_two_state_invariant_law() -> None:
    chain = two_state_chain(0.2, 0.3)
    trajectory = simulate_chain(
        chain,
        np.random.default_rng(2022),
        n_steps=100_000,
        initial_state=0,
    )
    assert trajectory.empirical_measure(discard=1_000) == pytest.approx(
        chain.invariant_distribution(),
        abs=0.01,
    )


def test_deterministic_cycle_trajectory_is_exact() -> None:
    chain = deterministic_cycle(4)
    trajectory = simulate_chain(
        chain,
        np.random.default_rng(1),
        n_steps=8,
        initial_state=2,
    )
    assert trajectory.states.tolist() == [2, 3, 0, 1, 2, 3, 0, 1, 2]
