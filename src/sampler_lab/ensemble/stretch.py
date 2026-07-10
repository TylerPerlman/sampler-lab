"""Affine-invariant stretch moves on an ensemble product target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import LogDensity
from sampler_lab.ensemble.state import EnsembleState, EnsembleTransition

Schedule = Literal["sequential", "split"]


def sample_stretch_scale(rng: np.random.Generator, scale: float = 2.0) -> float:
    r"""Draw ``Z`` with density proportional to ``z^{-1/2}`` on ``[1/a, a]``."""

    if not np.isfinite(scale) or scale <= 1.0:
        raise ValueError("stretch scale must be finite and greater than one")
    lower = 1.0 / np.sqrt(scale)
    upper = np.sqrt(scale)
    return float(rng.uniform(lower, upper) ** 2)


def stretch_log_density(z: float, scale: float = 2.0) -> float:
    """Normalized log density of the standard stretch factor."""

    if not np.isfinite(scale) or scale <= 1.0:
        raise ValueError("stretch scale must be finite and greater than one")
    if not np.isfinite(z) or z < 1.0 / scale or z > scale:
        return float("-inf")
    normalizer = 2.0 * (np.sqrt(scale) - 1.0 / np.sqrt(scale))
    return float(-0.5 * np.log(z) - np.log(normalizer))


def stretch_symmetry_error(z: float, scale: float = 2.0) -> float:
    """Return ``|log g(1/z) - log z - log g(z)|``."""

    if z <= 0.0:
        raise ValueError("z must be positive")
    return float(
        abs(stretch_log_density(1.0 / z, scale) - float(np.log(z)) - stretch_log_density(z, scale))
    )


@dataclass(slots=True)
class StretchMoveKernel:
    """Goodman--Weare stretch move on the full ensemble Markov state."""

    target: LogDensity
    scale: float = 2.0
    schedule: Schedule = "sequential"
    require_full_affine_span: bool = True
    counter: OperationCounter | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 1.0:
            raise ValueError("scale must be finite and greater than one")
        if self.schedule not in {"sequential", "split"}:
            raise ValueError("schedule must be 'sequential' or 'split'")

    def _validate(self, state: EnsembleState) -> None:
        if self.require_full_affine_span and not state.has_full_affine_span:
            raise ValueError("ensemble walkers do not span the target affine space")
        if self.schedule == "split" and state.n_walkers < 4:
            raise ValueError("split stretch updates require at least four walkers")

    def _attempt(
        self,
        positions: np.ndarray,
        logs: np.ndarray,
        walker: int,
        partner_pool: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[bool, float, int, np.ndarray, float, float]:
        partner = int(rng.choice(partner_pool))
        z = sample_stretch_scale(rng, self.scale)
        proposal = positions[partner] + z * (positions[walker] - positions[partner])
        proposal_log = float(self.target.log_prob(np.asarray(proposal, dtype=np.float64)))
        if self.counter is not None:
            self.counter.uniform_draws += 3
            self.counter.log_density_evaluations += 1
        log_ratio = (
            float("-inf")
            if proposal_log == float("-inf")
            else float((positions.shape[1] - 1) * np.log(z) + proposal_log - logs[walker])
        )
        accepted = bool(np.log(float(rng.random())) < min(0.0, log_ratio))
        return accepted, log_ratio, partner, proposal, proposal_log, z

    def _sequential(self, state: EnsembleState, rng: np.random.Generator) -> EnsembleTransition:
        positions = np.array(state.walkers, dtype=np.float64, copy=True)
        logs = np.array(state.log_probabilities, dtype=np.float64, copy=True)
        accepted = np.zeros(state.n_walkers, dtype=np.bool_)
        ratios = np.empty(state.n_walkers, dtype=np.float64)
        partners = np.empty(state.n_walkers, dtype=np.int64)
        scales = np.empty(state.n_walkers, dtype=np.float64)
        all_indices = np.arange(state.n_walkers, dtype=np.int64)
        for walker in range(state.n_walkers):
            pool = all_indices[all_indices != walker]
            did_accept, ratio, partner, proposal, proposal_log, z = self._attempt(
                positions, logs, walker, pool, rng
            )
            accepted[walker] = did_accept
            ratios[walker] = ratio
            partners[walker] = partner
            scales[walker] = z
            if did_accept:
                positions[walker] = proposal
                logs[walker] = proposal_log
        next_state = EnsembleState(positions, logs)
        return EnsembleTransition(
            next_state,
            accepted,
            ratios,
            partners,
            {
                "mean_stretch": float(np.mean(scales)),
                "acceptance_rate": float(np.mean(accepted)),
                "affine_span_rank": float(next_state.affine_span_rank),
            },
        )

    def _split(self, state: EnsembleState, rng: np.random.Generator) -> EnsembleTransition:
        positions = np.array(state.walkers, dtype=np.float64, copy=True)
        logs = np.array(state.log_probabilities, dtype=np.float64, copy=True)
        accepted = np.zeros(state.n_walkers, dtype=np.bool_)
        ratios = np.empty(state.n_walkers, dtype=np.float64)
        partners = np.empty(state.n_walkers, dtype=np.int64)
        scales = np.empty(state.n_walkers, dtype=np.float64)
        split = state.n_walkers // 2
        groups = (
            np.arange(0, split, dtype=np.int64),
            np.arange(split, state.n_walkers, dtype=np.int64),
        )
        if groups[0].size == 0 or groups[1].size == 0:
            raise ValueError("split schedule requires two nonempty groups")
        for group, complement in ((groups[0], groups[1]), (groups[1], groups[0])):
            base_positions = np.array(positions, copy=True)
            base_logs = np.array(logs, copy=True)
            for raw_walker in group:
                walker = int(raw_walker)
                did_accept, ratio, partner, proposal, proposal_log, z = self._attempt(
                    base_positions, base_logs, walker, complement, rng
                )
                accepted[walker] = did_accept
                ratios[walker] = ratio
                partners[walker] = partner
                scales[walker] = z
                if did_accept:
                    positions[walker] = proposal
                    logs[walker] = proposal_log
        next_state = EnsembleState(positions, logs)
        return EnsembleTransition(
            next_state,
            accepted,
            ratios,
            partners,
            {
                "mean_stretch": float(np.mean(scales)),
                "acceptance_rate": float(np.mean(accepted)),
                "affine_span_rank": float(next_state.affine_span_rank),
            },
        )

    def step(self, state: EnsembleState, rng: np.random.Generator) -> EnsembleTransition:
        self._validate(state)
        if self.schedule == "sequential":
            return self._sequential(state, rng)
        return self._split(state, rng)
