import numpy as np
import pytest

from sampler_lab.models import (
    SelfAvoidingWalkProposal,
    available_self_avoiding_neighbors,
    count_self_avoiding_walks,
    is_self_avoiding_walk,
    sample_self_avoiding_walks,
)


def test_small_square_lattice_self_avoiding_walk_counts() -> None:
    known = [1, 4, 12, 36, 100, 284, 780, 2172, 5916]
    assert [count_self_avoiding_walks(n) for n in range(len(known))] == known


def test_available_neighbors_exclude_visited_sites() -> None:
    path = np.array([[0, 0], [1, 0], [1, 1]])
    neighbors = available_self_avoiding_neighbors(path)
    assert {tuple(point) for point in neighbors} == {(2, 1), (0, 1), (1, 2)}


def test_walk_validity_checks_adjacency_and_uniqueness() -> None:
    assert is_self_avoiding_walk([[0, 0], [1, 0], [1, 1], [2, 1]])
    assert not is_self_avoiding_walk([[0, 0], [1, 0], [0, 0]])
    assert not is_self_avoiding_walk([[0, 0], [2, 0]])


def test_rosenbluth_first_two_incremental_weights_are_four_then_three() -> None:
    proposal = SelfAvoidingWalkProposal()
    initial = np.zeros((100, 1, 2))
    first = proposal.propose(initial, 1, np.random.default_rng(10))
    second = proposal.propose(first.particles, 2, np.random.default_rng(11))

    assert np.exp(first.log_incremental_weights) == pytest.approx(np.full(100, 4.0))
    assert np.exp(second.log_incremental_weights) == pytest.approx(np.full(100, 3.0))


def test_positive_weight_sampled_paths_are_valid() -> None:
    result = sample_self_avoiding_walks(
        np.random.default_rng(2022),
        n_steps=9,
        n_particles=2_000,
    )
    cloud = result.final_weighted_cloud
    positive = cloud.weights > 0.0
    assert np.count_nonzero(positive) > 1_500
    assert all(is_self_avoiding_walk(path) for path in cloud.particles[positive])
