import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.mcmc import (
    CoordinateGaussianRandomWalkProposal,
    GaussianIndependenceProposal,
    GaussianRandomWalkProposal,
    StateDependentGaussianProposal,
)


def test_gaussian_random_walk_density_is_symmetric() -> None:
    proposal = GaussianRandomWalkProposal(scale=np.array([0.5, 2.0]))
    x = np.array([-1.0, 3.0])
    y = np.array([0.25, -2.0])

    assert proposal.log_transition_density(y, x) == pytest.approx(
        proposal.log_transition_density(x, y)
    )


def test_gaussian_proposals_account_for_random_draws_and_density_evaluations() -> None:
    counter = OperationCounter()
    proposal = GaussianIndependenceProposal(mean=[0.0, 1.0], scale=[1.0, 2.0])
    state = np.array([4.0, 5.0])
    sample = proposal.sample(state, np.random.default_rng(1), counter=counter)
    density = proposal.log_transition_density(sample, state, counter=counter)

    assert sample.shape == state.shape
    assert np.isfinite(density)
    assert counter.normal_draws == 2
    assert counter.proposal_density_evaluations == 1


def test_state_dependent_proposal_uses_source_state_for_reverse_density() -> None:
    proposal = StateDependentGaussianProposal(
        mean=lambda x: 0.5 * x,
        scale=lambda x: np.ones_like(x) * (1.0 + 0.1 * np.abs(x)),
    )
    x = np.array([0.0])
    y = np.array([2.0])

    forward = proposal.log_transition_density(y, x)
    reverse = proposal.log_transition_density(x, y)

    assert forward != pytest.approx(reverse)


def test_coordinate_random_walk_changes_exactly_one_coordinate_and_is_symmetric() -> None:
    proposal = CoordinateGaussianRandomWalkProposal(
        scale=[0.5, 1.0, 2.0],
        probabilities=[0.2, 0.3, 0.5],
    )
    x = np.zeros(3)
    y = proposal.sample(x, np.random.default_rng(12))

    assert np.count_nonzero(y != x) == 1
    assert proposal.log_transition_density(y, x) == pytest.approx(
        proposal.log_transition_density(x, y)
    )
    assert proposal.log_transition_density(np.ones(3), x) == float("-inf")


@pytest.mark.parametrize("scale", [0.0, -1.0, [1.0, 0.0]])
def test_invalid_gaussian_scales_are_rejected(scale: object) -> None:
    proposal = GaussianRandomWalkProposal(scale=scale)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        proposal.sample(np.zeros(2), np.random.default_rng(1))


def test_multivariate_gaussian_random_walk_uses_full_covariance() -> None:
    from sampler_lab.mcmc import MultivariateGaussianRandomWalkProposal

    covariance = np.array([[2.0, 0.6], [0.6, 1.0]])
    proposal = MultivariateGaussianRandomWalkProposal(covariance)
    source = np.array([1.0, -2.0])
    rng = np.random.default_rng(5)
    expected_rng = np.random.default_rng(5)
    expected = source + np.linalg.cholesky(covariance) @ expected_rng.normal(size=2)

    sampled = proposal.sample(source, rng)

    np.testing.assert_allclose(sampled, expected)
    assert proposal.log_transition_density(sampled, source) == pytest.approx(
        proposal.log_transition_density(source, sampled)
    )
