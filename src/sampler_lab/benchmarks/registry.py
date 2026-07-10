"""Named sampler and target capability registry."""

from __future__ import annotations

from dataclasses import dataclass

from sampler_lab.benchmarks.capabilities import (
    CompatibilityResult,
    SamplerCapabilities,
    TargetCapabilities,
    check_compatibility,
)


@dataclass(frozen=True, slots=True)
class SamplerRegistration:
    name: str
    capabilities: SamplerCapabilities
    description: str


@dataclass(frozen=True, slots=True)
class TargetRegistration:
    name: str
    capabilities: TargetCapabilities
    description: str


class BenchmarkRegistry:
    """Metadata registry that refuses invalid sampler/target pairings."""

    def __init__(self) -> None:
        self._samplers: dict[str, SamplerRegistration] = {}
        self._targets: dict[str, TargetRegistration] = {}

    def register_sampler(self, registration: SamplerRegistration) -> None:
        if not registration.name:
            raise ValueError("sampler registration requires a nonempty name")
        if registration.name in self._samplers:
            raise ValueError(f"sampler {registration.name!r} is already registered")
        self._samplers[registration.name] = registration

    def register_target(self, registration: TargetRegistration) -> None:
        if not registration.name:
            raise ValueError("target registration requires a nonempty name")
        if registration.name in self._targets:
            raise ValueError(f"target {registration.name!r} is already registered")
        self._targets[registration.name] = registration

    @property
    def sampler_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._samplers))

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._targets))

    def compatibility(self, sampler_name: str, target_name: str) -> CompatibilityResult:
        try:
            sampler = self._samplers[sampler_name]
        except KeyError as error:
            raise KeyError(f"unknown sampler: {sampler_name!r}") from error
        try:
            target = self._targets[target_name]
        except KeyError as error:
            raise KeyError(f"unknown target: {target_name!r}") from error
        return check_compatibility(sampler.capabilities, target.capabilities)

    def compatible_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for sampler_name in self.sampler_names:
            for target_name in self.target_names:
                if self.compatibility(sampler_name, target_name).compatible:
                    pairs.append((sampler_name, target_name))
        return tuple(pairs)


def default_continuous_registry() -> BenchmarkRegistry:
    """Return metadata for the repository's common continuous-method suite."""

    registry = BenchmarkRegistry()
    sampler_specs: dict[str, SamplerCapabilities] = {
        "direct-oracle": SamplerCapabilities(is_markov_chain=False),
        "importance": SamplerCapabilities(
            produces_weighted_samples=True,
            is_markov_chain=False,
            requires_normalized_density=True,
        ),
        "annealed-smc": SamplerCapabilities(
            produces_weighted_samples=True,
            is_markov_chain=False,
            is_exact_after_freeze=False,
        ),
        "random-walk-mh": SamplerCapabilities(),
        "adaptive-random-walk": SamplerCapabilities(),
        "mala": SamplerCapabilities(requires_gradient=True),
        "hmc": SamplerCapabilities(requires_gradient=True),
        "stochastic-newton": SamplerCapabilities(requires_gradient=True, requires_hessian=True),
        "stretch-ensemble": SamplerCapabilities(),
        "walk-ensemble": SamplerCapabilities(),
        "reverse-kl": SamplerCapabilities(
            is_markov_chain=False,
            is_exact_after_freeze=False,
            requires_gradient=True,
            supports_multimodality=False,
        ),
        "reverse-kl-independence-mh": SamplerCapabilities(requires_gradient=True),
        "svgd": SamplerCapabilities(
            is_markov_chain=False,
            is_exact_after_freeze=False,
            requires_gradient=True,
        ),
        "policy-gradient-mh": SamplerCapabilities(),
    }
    for name, sampler_capabilities in sampler_specs.items():
        registry.register_sampler(
            SamplerRegistration(name, sampler_capabilities, name.replace("-", " "))
        )

    target_specs: dict[str, TargetCapabilities] = {
        "correlated-gaussian": TargetCapabilities(multimodal=False),
        "separated-anisotropic-gaussian-mixture": TargetCapabilities(multimodal=True),
        "rotated-anisotropic-funnel": TargetCapabilities(multimodal=False),
        "bimodal-anisotropic-funnel": TargetCapabilities(multimodal=True),
    }
    for name, target_capabilities in target_specs.items():
        registry.register_target(
            TargetRegistration(name, target_capabilities, name.replace("-", " "))
        )
    return registry
