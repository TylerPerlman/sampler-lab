"""Capability declarations for honest cross-method benchmark pairings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplerCapabilities:
    """Capabilities and requirements declared by one sampler adapter."""

    produces_samples: bool = True
    produces_weighted_samples: bool = False
    is_markov_chain: bool = True
    is_exact_after_freeze: bool = True
    requires_normalized_density: bool = False
    requires_log_density: bool = True
    requires_gradient: bool = False
    requires_hessian: bool = False
    requires_conditionals: bool = False
    requires_initial_reference_samples: bool = False
    supports_multimodality: bool = True
    supports_unbounded_continuous_space: bool = True


@dataclass(frozen=True, slots=True)
class TargetCapabilities:
    """Access supplied by one benchmark target."""

    normalized_density: bool = True
    log_density: bool = True
    gradient: bool = True
    hessian: bool = True
    conditionals: bool = False
    direct_samples: bool = True
    multimodal: bool = False
    unbounded_continuous_space: bool = True


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Compatibility decision with all explicit exclusion reasons."""

    compatible: bool
    reasons: tuple[str, ...]


def check_compatibility(
    sampler: SamplerCapabilities,
    target: TargetCapabilities,
) -> CompatibilityResult:
    """Check requirements without pretending unlike output semantics are identical."""

    reasons: list[str] = []
    if sampler.requires_normalized_density and not target.normalized_density:
        reasons.append("sampler requires a normalized target density")
    if sampler.requires_log_density and not target.log_density:
        reasons.append("sampler requires log-density evaluation")
    if sampler.requires_gradient and not target.gradient:
        reasons.append("sampler requires target gradients")
    if sampler.requires_hessian and not target.hessian:
        reasons.append("sampler requires target Hessians")
    if sampler.requires_conditionals and not target.conditionals:
        reasons.append("sampler requires exact conditional updates")
    if sampler.requires_initial_reference_samples and not target.direct_samples:
        reasons.append("sampler requires trusted reference samples")
    if target.multimodal and not sampler.supports_multimodality:
        reasons.append("sampler adapter does not support multimodal targets")
    if target.unbounded_continuous_space and not sampler.supports_unbounded_continuous_space:
        reasons.append("sampler does not support unbounded continuous state spaces")
    if not sampler.produces_samples and not sampler.produces_weighted_samples:
        reasons.append("sampler adapter produces no benchmarkable sample representation")
    return CompatibilityResult(not reasons, tuple(reasons))
