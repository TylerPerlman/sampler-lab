import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.models import (
    IsingModel,
    RandomScanIsingMetropolisKernel,
    deterministic_sweep_ising_gibbs,
    enumerate_ising_states,
    exact_ising_distribution,
    ising_deterministic_sweep_gibbs_transition,
    ising_random_scan_gibbs_transition,
    ising_random_scan_metropolis_transition,
    ising_state_index,
    random_scan_ising_gibbs,
)


def test_enumeration_and_state_indices_are_inverse() -> None:
    states = enumerate_ising_states(2)

    assert states.shape == (16, 2, 2)
    assert [ising_state_index(state) for state in states] == list(range(16))


def test_local_flip_ratio_matches_global_log_density_difference() -> None:
    model = IsingModel(3, beta=0.37, coupling=1.2, field=-0.15)
    state = model.random_state(np.random.default_rng(4))

    for flat_site in range(model.n_sites):
        site = divmod(flat_site, model.length)
        exact_difference = model.log_prob(model.flip(state, site)) - model.log_prob(state)
        assert model.log_flip_ratio(state, site) == pytest.approx(exact_difference)


def test_conditional_probability_matches_two_possible_target_masses() -> None:
    model = IsingModel(3, beta=0.6, field=0.2)
    state = model.random_state(np.random.default_rng(5))
    site = (1, 2)
    plus = np.array(state, copy=True)
    minus = np.array(state, copy=True)
    plus[site] = 1.0
    minus[site] = -1.0
    log_plus = model.log_prob(plus)
    log_minus = model.log_prob(minus)
    exact = 1.0 / (1.0 + np.exp(log_minus - log_plus))

    assert model.conditional_plus_probability(state, site) == pytest.approx(exact)


def test_zero_field_exact_distribution_has_spin_flip_symmetry() -> None:
    model = IsingModel(2, beta=0.4)
    exact = exact_ising_distribution(model)

    assert np.sum(exact.probabilities) == pytest.approx(1.0)
    assert exact.expectation(exact.magnetizations) == pytest.approx(0.0, abs=1e-14)
    assert exact.expectation(np.abs(exact.magnetizations)) > 0.0


def test_exact_random_scan_kernels_are_reversible_and_invariant() -> None:
    model = IsingModel(2, beta=0.4, field=0.1)
    exact = exact_ising_distribution(model)

    for chain in (
        ising_random_scan_gibbs_transition(model),
        ising_random_scan_metropolis_transition(model),
    ):
        assert chain.global_balance_residual(exact.probabilities) < 1e-12
        assert chain.detailed_balance_residual(exact.probabilities) < 1e-12
        assert chain.is_ergodic


def test_deterministic_gibbs_sweep_is_invariant_but_generally_nonreversible() -> None:
    model = IsingModel(2, beta=0.4)
    exact = exact_ising_distribution(model)
    chain = ising_deterministic_sweep_gibbs_transition(model)

    assert chain.global_balance_residual(exact.probabilities) < 1e-12
    assert chain.detailed_balance_residual(exact.probabilities) > 1e-5


def test_ising_kernels_count_spin_updates_and_retain_rejections() -> None:
    model = IsingModel(3, beta=2.0)
    state = np.ones((3, 3))

    gibbs_counter = OperationCounter()
    sweep = deterministic_sweep_ising_gibbs(model, counter=gibbs_counter)
    sweep.step(state, np.random.default_rng(1))
    assert gibbs_counter.spin_updates == model.n_sites
    assert gibbs_counter.conditional_draws == model.n_sites

    random_counter = OperationCounter()
    random_gibbs = random_scan_ising_gibbs(model, counter=random_counter)
    random_gibbs.step(state, np.random.default_rng(2))
    assert random_counter.spin_updates == 1

    metropolis_counter = OperationCounter()
    metropolis = RandomScanIsingMetropolisKernel(model, metropolis_counter)
    transition = metropolis.step(state, np.random.default_rng(3))
    assert not transition.accepted
    assert transition.state == pytest.approx(state)
    assert metropolis_counter.spin_updates == 1
    assert metropolis_counter.extra["local_energy_differences"] == 1


def test_batched_ising_log_density_matches_scalar_evaluation() -> None:
    model = IsingModel(3, beta=0.37, coupling=1.2, field=-0.15)
    rng = np.random.default_rng(44)
    states = np.asarray([model.random_state(rng) for _ in range(7)])

    expected = np.asarray([model.log_prob(state) for state in states])
    np.testing.assert_allclose(model.log_prob_batch(states), expected)


def test_vectorized_annealed_ising_sweep_preserves_shape_and_spin_support() -> None:
    from sampler_lab.models import IsingGibbsPopulationTransition

    rng = np.random.default_rng(45)
    particles = np.asarray(
        2 * rng.integers(0, 2, size=(100, 2, 2)) - 1,
        dtype=np.float64,
    )
    transition = IsingGibbsPopulationTransition(2, final_beta=0.6, n_sweeps=2)

    moved = transition.move(particles, 0.5, rng)

    assert moved.shape == particles.shape
    assert np.all((moved == -1.0) | (moved == 1.0))
    assert not np.shares_memory(moved, particles)
