from __future__ import annotations

import math

import numpy as np
import pytest

from sampler_lab.rare_events import (
    GaussianHalfspaceRareEvent,
    GaussianShiftProposal,
    GaussianTemperedProposal,
    GaussianTwoSidedRareEvent,
    LaplacePoint,
    estimate_from_log_contributions,
    exact_relative_error,
    exact_shifted_log_second_moment,
    exact_symmetric_mixture_log_second_moment,
    exact_tempered_log_second_moment,
    fit_exponential_relative_variance_rate,
    gaussian_linear_event_log_asymptotic,
    laplace_integral,
    select_shift_scale,
    select_temperature,
    standard_normal_log_upper_tail,
    symmetric_twist_mixture,
)


def test_log_normal_tail_matches_erfc_and_remains_finite_far_out() -> None:
    for value in [-3.0, 0.0, 2.0, 7.0]:
        expected = math.log(0.5 * math.erfc(value / math.sqrt(2.0)))
        assert standard_normal_log_upper_tail(value) == pytest.approx(expected, rel=2e-14)
    assert np.isfinite(standard_normal_log_upper_tail(50.0))
    assert standard_normal_log_upper_tail(50.0) < -1200.0


def test_gaussian_halfspace_dominating_point_and_rate() -> None:
    covariance = np.array([[2.0, 0.3], [0.3, 0.8]])
    problem = GaussianHalfspaceRareEvent([1.0, -0.5], 1.3, covariance)
    point = problem.dominant_point
    assert problem.direction_vector @ point == pytest.approx(problem.threshold)
    rate_from_quadratic = 0.5 * point @ problem.precision_matrix @ point
    assert rate_from_quadratic == pytest.approx(problem.rate)
    assert problem.exact_log_probability(0.2) < 0.0


def test_two_sided_probability_is_twice_one_sided_probability() -> None:
    half = GaussianHalfspaceRareEvent([1.0], 1.0, [[1.0]])
    two = GaussianTwoSidedRareEvent([1.0], 1.0, [[1.0]])
    epsilon = 0.3
    assert two.exact_probability(epsilon) == pytest.approx(2.0 * half.exact_probability(epsilon))
    assert two.exact_log_probability(epsilon) == pytest.approx(
        math.log(2.0) + half.exact_log_probability(epsilon)
    )


def test_laplace_formula_is_exact_for_quadratic_integral() -> None:
    epsilon = 0.17
    hessian = np.array([[3.0, 0.4], [0.4, 2.0]])
    minimum = 1.2
    amplitude = 2.5
    result = laplace_integral(
        [LaplacePoint(minimum, hessian, amplitude)],
        epsilon,
    )
    exact = (
        amplitude
        * math.exp(-minimum / epsilon)
        * (2.0 * math.pi * epsilon)
        / math.sqrt(np.linalg.det(hessian))
    )
    assert result.value == pytest.approx(exact, rel=2e-14)
    assert result.exponential_rate == minimum


def test_boundary_laplace_approximation_improves_as_noise_shrinks() -> None:
    problem = GaussianHalfspaceRareEvent([1.0], 1.0, [[1.0]])
    errors = []
    for epsilon in [0.2, 0.1, 0.05, 0.02]:
        approximation = gaussian_linear_event_log_asymptotic(problem, epsilon)
        errors.append(abs(approximation - problem.exact_log_probability(epsilon)))
    assert errors[-1] < errors[0] / 5.0


def test_constant_log_contributions_have_zero_relative_variance() -> None:
    result = estimate_from_log_contributions([math.log(2.0)] * 4)
    assert result.value == pytest.approx(2.0)
    assert result.relative_variance == 0.0
    assert result.relative_standard_error == 0.0
    assert result.contribution_effective_sample_size == pytest.approx(4.0)


def test_log_contribution_diagnostics_match_elementary_values() -> None:
    result = estimate_from_log_contributions([0.0, float("-inf"), math.log(3.0)])
    assert result.value == pytest.approx(4.0 / 3.0)
    assert result.relative_variance == pytest.approx(0.875)
    assert result.relative_standard_error == pytest.approx(math.sqrt(0.875 / 3.0))
    assert result.contribution_effective_sample_size == pytest.approx(1.6)
    assert result.max_normalized_contribution == pytest.approx(0.75)


def test_shift_log_weight_matches_direct_density_difference() -> None:
    proposal = GaussianShiftProposal(
        mean_shift=np.array([0.4, -0.3]),
        covariance=np.array([[1.2, 0.2], [0.2, 0.7]]),
        epsilon=0.25,
    )
    samples = np.array([[0.1, 0.5], [-0.7, 0.2], [1.0, -0.4]])
    np.testing.assert_allclose(
        proposal.log_weights(samples),
        proposal.target_log_density(samples) - proposal.log_density(samples),
        atol=2e-14,
    )


def test_tempered_log_weight_reduces_to_zero_at_unit_temperature() -> None:
    proposal = GaussianTemperedProposal(np.eye(2), epsilon=0.2, temperature=1.0)
    samples = np.array([[0.0, 0.0], [1.0, -2.0]])
    np.testing.assert_allclose(proposal.log_weights(samples), 0.0)


def test_exact_second_moments_reduce_to_crude_monte_carlo() -> None:
    problem = GaussianHalfspaceRareEvent([1.0], 1.0, [[1.0]])
    epsilon = 0.15
    zero = np.zeros(1)
    assert exact_shifted_log_second_moment(problem, zero, epsilon) == pytest.approx(
        problem.exact_log_probability(epsilon)
    )
    assert exact_tempered_log_second_moment(problem, 1.0, epsilon) == pytest.approx(
        problem.exact_log_probability(epsilon)
    )


def test_one_twist_fails_for_two_dominating_points_but_mixture_is_efficient() -> None:
    problem = GaussianTwoSidedRareEvent([1.0], 1.0, [[1.0]])
    epsilon = 0.02
    log_probability = problem.exact_log_probability(epsilon)
    single_second = exact_shifted_log_second_moment(problem, problem.dominant_point, epsilon)
    mixture_second = exact_symmetric_mixture_log_second_moment(problem, epsilon)
    single_rv, _, _ = exact_relative_error(
        log_probability=log_probability,
        log_second_moment=single_second,
        n_samples=1,
    )
    mixture_rv, _, _ = exact_relative_error(
        log_probability=log_probability,
        log_second_moment=mixture_second,
        n_samples=1,
    )
    assert epsilon * math.log(single_rv) > 1.8
    assert epsilon * math.log(mixture_rv) < 0.06


def test_symmetric_mixture_density_is_even() -> None:
    problem = GaussianTwoSidedRareEvent([1.0, 0.5], 1.0, np.eye(2))
    proposal = symmetric_twist_mixture(problem, 0.2)
    points = np.array([[0.3, -0.2], [1.0, 0.5]])
    np.testing.assert_allclose(proposal.log_density(points), proposal.log_density(-points))


def test_small_grid_optimizers_choose_useful_candidates() -> None:
    one_sided = GaussianHalfspaceRareEvent([1.0], 1.0, [[1.0]])
    shift = select_shift_scale(one_sided, 0.1, [0.0, 0.5, 1.0, 1.5])
    temperature = select_temperature(one_sided, 0.1, [1.0, 2.0, 5.0, 10.0])
    assert shift.value in {0.5, 1.0, 1.5}
    assert temperature.value > 1.0


def test_exponential_rate_fit_recovers_constructed_slope() -> None:
    epsilons = np.array([0.5, 0.25, 0.125, 0.0625])
    values = 0.7 / epsilons - 1.3
    fit = fit_exponential_relative_variance_rate(epsilons, values)
    assert fit.slope == pytest.approx(0.7)
    assert fit.intercept == pytest.approx(-1.3)
    assert fit.r_squared == pytest.approx(1.0)
