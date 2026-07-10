"""Adaptive-MCMC infrastructure with explicit warmup/evaluation separation."""

from sampler_lab.adaptive.covariance import (
    CovarianceRegularizationResult,
    regularize_covariance,
)
from sampler_lab.adaptive.dual_averaging import DualAveragingStepSize, RobbinsMonroLogScale
from sampler_lab.adaptive.running_moments import RunningMoments, RunningMomentsSnapshot
from sampler_lab.adaptive.schedules import RobbinsMonroSchedule, diminishing_ratio, is_nonincreasing
from sampler_lab.adaptive.warmup import (
    AdaptiveTrainingResult,
    EvaluationTrajectory,
    FrozenPolicy,
    WarmupWindow,
    expanding_warmup_windows,
)

__all__ = [
    "AdaptiveTrainingResult",
    "CovarianceRegularizationResult",
    "DualAveragingStepSize",
    "EvaluationTrajectory",
    "FrozenPolicy",
    "RobbinsMonroLogScale",
    "RobbinsMonroSchedule",
    "RunningMoments",
    "RunningMomentsSnapshot",
    "WarmupWindow",
    "diminishing_ratio",
    "expanding_warmup_windows",
    "is_nonincreasing",
    "regularize_covariance",
]
