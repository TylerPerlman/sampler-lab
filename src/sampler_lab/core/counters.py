"""Operation accounting for cost-aware Monte Carlo comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Self


@dataclass(slots=True)
class OperationCounter:
    """Mutable counts of stochastic and expensive deterministic operations.

    The named fields cover the common cost units used across the package. ``extra`` permits
    algorithm-specific counts without immediately changing the common interface.
    """

    uniform_draws: int = 0
    normal_draws: int = 0
    log_density_evaluations: int = 0
    proposal_density_evaluations: int = 0
    gradient_evaluations: int = 0
    hessian_evaluations: int = 0
    matrix_factorizations: int = 0
    conditional_draws: int = 0
    spin_updates: int = 0
    particle_propagations: int = 0
    policy_evaluations: int = 0
    training_objective_evaluations: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a named operation by a nonnegative integer amount."""

        if not isinstance(amount, int) or amount < 0:
            raise ValueError("operation increments must be nonnegative integers")
        if hasattr(self, name) and name != "extra":
            current = getattr(self, name)
            if not isinstance(current, int):
                raise TypeError(f"{name!r} is not an integer counter")
            setattr(self, name, current + amount)
            return
        self.extra[name] = self.extra.get(name, 0) + amount

    def merge(self, other: OperationCounter) -> Self:
        """Add another counter into this one and return ``self``."""

        for name in (
            "uniform_draws",
            "normal_draws",
            "log_density_evaluations",
            "proposal_density_evaluations",
            "gradient_evaluations",
            "hessian_evaluations",
            "matrix_factorizations",
            "conditional_draws",
            "spin_updates",
            "particle_propagations",
            "policy_evaluations",
            "training_objective_evaluations",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name, value in other.extra.items():
            self.extra[name] = self.extra.get(name, 0) + value
        return self

    def snapshot(self) -> dict[str, int | dict[str, int]]:
        """Return a detached dictionary suitable for result diagnostics."""

        raw = asdict(self)
        raw["extra"] = dict(self.extra)
        return raw
