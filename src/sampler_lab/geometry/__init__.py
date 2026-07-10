"""Affine geometry, conditioning, Hessian metrics, and Newton proposals."""

from sampler_lab.geometry.affine import (
    AffineMap,
    AffineTransformedTarget,
    affine_equivariance_error,
)
from sampler_lab.geometry.hessian import (
    PositiveDefiniteRepair,
    RepairMethod,
    finite_difference_hessian_from_gradient,
    negative_log_hessian,
    repair_positive_definite,
)
from sampler_lab.geometry.preconditioners import (
    GaussianConditional,
    gaussian_conditional,
    gaussian_whitening_map,
    matrix_condition_number,
    whiten_gaussian_target,
)
from sampler_lab.geometry.stochastic_newton import (
    MetropolizedStochasticNewtonKernel,
    StochasticNewtonEvaluation,
    StochasticNewtonProposal,
)

__all__ = [
    "AffineMap",
    "AffineTransformedTarget",
    "GaussianConditional",
    "MetropolizedStochasticNewtonKernel",
    "PositiveDefiniteRepair",
    "RepairMethod",
    "StochasticNewtonEvaluation",
    "StochasticNewtonProposal",
    "affine_equivariance_error",
    "finite_difference_hessian_from_gradient",
    "gaussian_conditional",
    "gaussian_whitening_map",
    "matrix_condition_number",
    "negative_log_hessian",
    "repair_positive_definite",
    "whiten_gaussian_target",
]
