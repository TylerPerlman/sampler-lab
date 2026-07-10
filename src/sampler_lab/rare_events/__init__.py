"""Rare-event probability estimation and small-noise asymptotics."""

from sampler_lab.rare_events.laplace import (
    LaplaceApproximation,
    LaplacePoint,
    gaussian_linear_event_asymptotic,
    gaussian_linear_event_log_asymptotic,
    laplace_integral,
    laplace_log_integral,
)
from sampler_lab.rare_events.mixtures import (
    GaussianShiftMixtureProposal,
    estimate_with_mixture,
    exact_symmetric_mixture_log_second_moment,
    symmetric_twist_mixture,
)
from sampler_lab.rare_events.normal import (
    standard_normal_log_upper_tail,
    standard_normal_upper_tail,
)
from sampler_lab.rare_events.optimization import (
    ProposalSelection,
    select_shift_scale,
    select_temperature,
)
from sampler_lab.rare_events.problems import (
    GaussianHalfspaceRareEvent,
    GaussianTwoSidedRareEvent,
    RareGaussianProblem,
)
from sampler_lab.rare_events.relative_error import (
    ExponentialRateFit,
    RareEventEstimate,
    estimate_from_log_contributions,
    exact_relative_error,
    fit_exponential_relative_variance_rate,
    scaled_log_relative_variance,
)
from sampler_lab.rare_events.tempering import (
    GaussianTemperedProposal,
    estimate_with_tempering,
    exact_tempered_log_second_moment,
    fixed_scale_temperature,
)
from sampler_lab.rare_events.twisting import (
    GaussianShiftProposal,
    estimate_with_shift,
    exact_shifted_log_second_moment,
    optimal_linear_shift,
)

__all__ = [
    "ExponentialRateFit",
    "GaussianHalfspaceRareEvent",
    "GaussianShiftMixtureProposal",
    "GaussianShiftProposal",
    "GaussianTemperedProposal",
    "GaussianTwoSidedRareEvent",
    "LaplaceApproximation",
    "LaplacePoint",
    "ProposalSelection",
    "RareEventEstimate",
    "RareGaussianProblem",
    "estimate_from_log_contributions",
    "estimate_with_mixture",
    "estimate_with_shift",
    "estimate_with_tempering",
    "exact_relative_error",
    "exact_shifted_log_second_moment",
    "exact_symmetric_mixture_log_second_moment",
    "exact_tempered_log_second_moment",
    "fit_exponential_relative_variance_rate",
    "fixed_scale_temperature",
    "gaussian_linear_event_asymptotic",
    "gaussian_linear_event_log_asymptotic",
    "laplace_integral",
    "laplace_log_integral",
    "optimal_linear_shift",
    "scaled_log_relative_variance",
    "select_shift_scale",
    "select_temperature",
    "standard_normal_log_upper_tail",
    "standard_normal_upper_tail",
    "symmetric_twist_mixture",
]
