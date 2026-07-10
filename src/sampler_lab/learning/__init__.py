"""Adaptive, policy-gradient, variational, and Stein proposal learning."""

from sampler_lab.learning.adaptive_mh import (
    AdaptiveRandomWalkResult,
    FrozenAdaptiveRandomWalkKernel,
    evaluate_adaptive_random_walk,
    train_adaptive_random_walk,
)
from sampler_lab.learning.baselines import LinearBaseline, RunningMeanBaseline, ZeroBaseline
from sampler_lab.learning.generalized_speed import (
    GeneralizedSpeedGradient,
    GeneralizedSpeedTrainingResult,
    diagonal_random_walk_generalized_speed_gradient,
    evaluate_generalized_speed,
    train_generalized_speed_random_walk,
)
from sampler_lab.learning.gradients import (
    NaturalGradientResult,
    ReinforceEstimate,
    categorical_kl,
    discounted_returns,
    linear_softmax_fisher,
    natural_gradient_direction,
    reinforce_gradient,
)
from sampler_lab.learning.mixtures import PolicyMixtureProposal
from sampler_lab.learning.objectives import (
    AcceptanceObjective,
    AcceptedSquaredJumpObjective,
    ContrastiveDivergenceLowerBoundObjective,
    CostNormalizedObjective,
    FeatureJumpObjective,
    GeneralizedSpeedObjective,
    OperationCostWeights,
    PolicyTransition,
    make_feature_map,
)
from sampler_lab.learning.optimizers import SGD, Adam
from sampler_lab.learning.policies import (
    FrozenLinearSoftmaxPolicy,
    LinearSoftmaxPolicy,
    PolicyAction,
    SquashedGaussianPolicy,
)
from sampler_lab.learning.stein import (
    IMQKernel,
    SVGDResult,
    kernel_stein_discrepancy,
    run_svgd,
    stein_kernel_value,
    svgd_direction,
)
from sampler_lab.learning.trainer import (
    FrozenKernelMixture,
    PolicyTrainingResult,
    constant_features,
    evaluate_frozen_policy,
    identity_features,
    train_kernel_selection_policy,
)
from sampler_lab.learning.variational import (
    DiagonalGaussianVariational,
    FrozenDiagonalGaussian,
    ReverseKLEstimate,
    VariationalFitResult,
    fit_forward_kl_diagonal_gaussian,
    fit_reverse_kl_diagonal_gaussian,
)

__all__ = [
    "SGD",
    "AcceptanceObjective",
    "AcceptedSquaredJumpObjective",
    "Adam",
    "AdaptiveRandomWalkResult",
    "ContrastiveDivergenceLowerBoundObjective",
    "CostNormalizedObjective",
    "DiagonalGaussianVariational",
    "FeatureJumpObjective",
    "FrozenAdaptiveRandomWalkKernel",
    "FrozenDiagonalGaussian",
    "FrozenKernelMixture",
    "FrozenLinearSoftmaxPolicy",
    "GeneralizedSpeedGradient",
    "GeneralizedSpeedObjective",
    "GeneralizedSpeedTrainingResult",
    "IMQKernel",
    "LinearBaseline",
    "LinearSoftmaxPolicy",
    "NaturalGradientResult",
    "OperationCostWeights",
    "PolicyAction",
    "PolicyMixtureProposal",
    "PolicyTrainingResult",
    "PolicyTransition",
    "ReinforceEstimate",
    "ReverseKLEstimate",
    "RunningMeanBaseline",
    "SVGDResult",
    "SquashedGaussianPolicy",
    "VariationalFitResult",
    "ZeroBaseline",
    "categorical_kl",
    "constant_features",
    "diagonal_random_walk_generalized_speed_gradient",
    "discounted_returns",
    "evaluate_adaptive_random_walk",
    "evaluate_frozen_policy",
    "evaluate_generalized_speed",
    "fit_forward_kl_diagonal_gaussian",
    "fit_reverse_kl_diagonal_gaussian",
    "identity_features",
    "kernel_stein_discrepancy",
    "linear_softmax_fisher",
    "make_feature_map",
    "natural_gradient_direction",
    "reinforce_gradient",
    "run_svgd",
    "stein_kernel_value",
    "svgd_direction",
    "train_adaptive_random_walk",
    "train_generalized_speed_random_walk",
    "train_kernel_selection_policy",
]
