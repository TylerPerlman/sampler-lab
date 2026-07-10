"""Affine-invariant symmetric walk moves for ensemble MCMC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import LogDensity
from sampler_lab.ensemble.state import EnsembleState, EnsembleTransition

Schedule = Literal["sequential", "split"]


@dataclass(slots=True)
class WalkMoveKernel:
    """Symmetric ensemble walk move using centered complementary walkers.

    A proposal for walker ``i`` is ``X_i + scale/sqrt(s) * sum z_j (X_j - mean)``
    where the ``s`` complementary walkers are selected without replacement and
    ``z_j`` are standard normals.  Conditional on the complementary ensemble the
    proposal is symmetric, so the Hastings ratio is just the target-density ratio.
    """

    target: LogDensity
    scale: float = 1.0
    subset_size: int | None = None
    schedule: Schedule = "sequential"
    require_full_affine_span: bool = True
    counter: OperationCounter | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be positive and finite")
        if self.subset_size is not None:
            if isinstance(self.subset_size, bool) or not isinstance(self.subset_size, int):
                raise TypeError("subset_size must be an integer")
            if self.subset_size < 2:
                raise ValueError("subset_size must be at least two")
        if self.schedule not in {"sequential", "split"}:
            raise ValueError("schedule must be 'sequential' or 'split'")

    def _resolved_subset_size(self, dimension: int, complement_size: int) -> int:
        subset = dimension + 1 if self.subset_size is None else self.subset_size
        if subset > complement_size:
            raise ValueError("complement does not contain enough walkers for the walk subset")
        return subset

    def _validate(self, state: EnsembleState) -> None:
        if self.require_full_affine_span and not state.has_full_affine_span:
            raise ValueError("ensemble walkers do not span the target affine space")
        if (
            self.schedule == "split"
            and self.subset_size is None
            and state.n_walkers < 2 * (state.dimension + 1)
        ):
            raise ValueError(
                "default split walk updates require at least 2 * (dimension + 1) walkers"
            )

    def _attempt(
        self,
        positions: np.ndarray,
        logs: np.ndarray,
        walker: int,
        complement: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[bool, float, int, np.ndarray, float, int]:
        subset_size = self._resolved_subset_size(positions.shape[1], int(complement.size))
        selected = np.asarray(
            rng.choice(complement, size=subset_size, replace=False),
            dtype=np.int64,
        )
        companions = positions[selected]
        centered = companions - np.mean(companions, axis=0)
        coefficients = rng.normal(size=subset_size)
        displacement = self.scale * (coefficients @ centered) / np.sqrt(subset_size)
        proposal = positions[walker] + displacement
        proposal_log = float(self.target.log_prob(np.asarray(proposal, dtype=np.float64)))
        if self.counter is not None:
            self.counter.uniform_draws += subset_size + 1
            self.counter.normal_draws += subset_size
            self.counter.log_density_evaluations += 1
        log_ratio = (
            float("-inf") if proposal_log == float("-inf") else float(proposal_log - logs[walker])
        )
        accepted = bool(np.log(float(rng.random())) < min(0.0, log_ratio))
        representative_partner = int(selected[0])
        proposal_rank = int(np.linalg.matrix_rank(centered))
        return accepted, log_ratio, representative_partner, proposal, proposal_log, proposal_rank

    def _run_groups(
        self,
        state: EnsembleState,
        rng: np.random.Generator,
        groups: tuple[tuple[np.ndarray, np.ndarray], ...],
        *,
        freeze_within_group: bool,
    ) -> EnsembleTransition:
        positions = np.array(state.walkers, dtype=np.float64, copy=True)
        logs = np.array(state.log_probabilities, dtype=np.float64, copy=True)
        accepted = np.zeros(state.n_walkers, dtype=np.bool_)
        ratios = np.empty(state.n_walkers, dtype=np.float64)
        partners = np.empty(state.n_walkers, dtype=np.int64)
        proposal_ranks = np.empty(state.n_walkers, dtype=np.float64)
        for group, complement in groups:
            base_positions = np.array(positions, copy=True) if freeze_within_group else positions
            base_logs = np.array(logs, copy=True) if freeze_within_group else logs
            for raw_walker in group:
                walker = int(raw_walker)
                did_accept, ratio, partner, proposal, proposal_log, rank = self._attempt(
                    base_positions,
                    base_logs,
                    walker,
                    complement if freeze_within_group else complement[complement != walker],
                    rng,
                )
                accepted[walker] = did_accept
                ratios[walker] = ratio
                partners[walker] = partner
                proposal_ranks[walker] = rank
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
                "acceptance_rate": float(np.mean(accepted)),
                "minimum_proposal_rank": float(np.min(proposal_ranks)),
                "affine_span_rank": float(next_state.affine_span_rank),
            },
        )

    def step(self, state: EnsembleState, rng: np.random.Generator) -> EnsembleTransition:
        self._validate(state)
        all_indices = np.arange(state.n_walkers, dtype=np.int64)
        if self.schedule == "sequential":
            groups = tuple(
                (np.array([walker], dtype=np.int64), all_indices)
                for walker in range(state.n_walkers)
            )
            return self._run_groups(state, rng, groups, freeze_within_group=False)
        split = state.n_walkers // 2
        first = np.arange(0, split, dtype=np.int64)
        second = np.arange(split, state.n_walkers, dtype=np.int64)
        if first.size == 0 or second.size == 0:
            raise ValueError("split schedule requires two nonempty groups")
        return self._run_groups(
            state,
            rng,
            ((first, second), (second, first)),
            freeze_within_group=True,
        )
