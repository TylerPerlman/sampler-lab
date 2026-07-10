import numpy as np
import pytest

from sampler_lab.markov import (
    FiniteStateMarkovChain,
    operator_duality_residual,
    validate_transition_matrix,
)
from sampler_lab.models import deterministic_cycle, ring_random_walk


def test_transition_validation_and_operator_duality() -> None:
    transition = np.array([[0.7, 0.3], [0.2, 0.8]])
    chain = FiniteStateMarkovChain(transition)
    measure = np.array([0.4, 0.6])
    observable = np.array([-1.0, 2.0])

    assert operator_duality_residual(measure, transition, observable) < 1e-15
    assert chain.apply(observable) == pytest.approx(transition @ observable)
    assert chain.pushforward(measure) == pytest.approx(measure @ transition)
    assert chain.apply(observable, steps=0) == pytest.approx(observable)


@pytest.mark.parametrize(
    "transition",
    [
        [[0.5, 0.6], [0.2, 0.8]],
        [[1.1, -0.1], [0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ],
)
def test_invalid_transition_matrices_are_rejected(transition: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        validate_transition_matrix(transition)


def test_reducible_chain_exposes_extreme_invariant_distributions() -> None:
    chain = FiniteStateMarkovChain(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    structure = chain.communication_structure()
    distributions = chain.invariant_distributions()

    assert structure.classes == ((0,), (1,), (2, 3))
    assert structure.closed_classes == ((1,), (2, 3))
    assert structure.periods == (1, 1, 2)
    assert distributions.shape == (2, 4)
    assert distributions[0] == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert distributions[1] == pytest.approx([0.0, 0.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="multiple invariant"):
        chain.invariant_distribution()


def test_unique_stationary_law_can_ignore_transient_states() -> None:
    chain = FiniteStateMarkovChain([[0.25, 0.75], [0.0, 1.0]])
    assert chain.invariant_distribution() == pytest.approx([0.0, 1.0])
    assert chain.global_balance_residual() < 1e-14
    assert not chain.is_irreducible
    assert chain.is_aperiodic


def test_irreducibility_and_periodicity_are_exact_graph_properties() -> None:
    cycle = deterministic_cycle(5)
    lazy = ring_random_walk(5, clockwise=0.4, counterclockwise=0.1, stay=0.5)

    assert cycle.is_irreducible
    assert cycle.period == 5
    assert not cycle.is_aperiodic
    assert not cycle.is_ergodic
    assert lazy.is_irreducible
    assert lazy.period == 1
    assert lazy.is_ergodic


def test_detailed_balance_and_time_reversal() -> None:
    reversible = ring_random_walk(7, clockwise=0.25, counterclockwise=0.25, stay=0.5)
    directed = ring_random_walk(7, clockwise=0.4, counterclockwise=0.1, stay=0.5)
    uniform = np.full(7, 1.0 / 7.0)

    assert reversible.is_reversible(uniform)
    assert reversible.detailed_balance_residual(uniform) < 1e-15
    assert not directed.is_reversible(uniform)
    assert directed.global_balance_residual(uniform) < 1e-15

    reversed_directed = directed.time_reversal(uniform)
    expected = ring_random_walk(7, clockwise=0.1, counterclockwise=0.4, stay=0.5)
    assert reversed_directed.transition == pytest.approx(expected.transition)
    assert reversed_directed.time_reversal(uniform).transition == pytest.approx(directed.transition)


def test_spectral_summary_distinguishes_reversible_and_nonreversible_gaps() -> None:
    reversible = ring_random_walk(8, clockwise=0.25, counterclockwise=0.25, stay=0.5)
    directed = ring_random_walk(8, clockwise=0.4, counterclockwise=0.1, stay=0.5)

    reversible_summary = reversible.spectral_summary()
    directed_summary = directed.spectral_summary()

    assert reversible_summary.reversible
    assert reversible_summary.poincare_gap is not None
    assert reversible_summary.worst_case_iat is not None
    assert reversible_summary.absolute_spectral_gap > 0.0
    assert not directed_summary.reversible
    assert directed_summary.poincare_gap is None
    assert directed_summary.worst_case_iat is None
    assert directed_summary.singular_value_gap > 0.0
