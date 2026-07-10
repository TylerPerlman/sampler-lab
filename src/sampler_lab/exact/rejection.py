"""Generic rejection sampling in the log domain."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.numerics import validate_size
from sampler_lab.core.results import RejectionResult

Array = NDArray[np.float64]
ProposalSampler = Callable[[np.random.Generator, int], Array]
LogDensityFunction = Callable[[Array], float]


def rejection_sample(
    rng: np.random.Generator,
    size: int,
    proposal_sampler: ProposalSampler,
    log_target: LogDensityFunction,
    log_proposal: LogDensityFunction,
    log_envelope: float,
    *,
    batch_size: int = 1_024,
    max_proposals: int | None = None,
    counter: OperationCounter | None = None,
    envelope_tolerance: float = 1e-12,
) -> RejectionResult:
    """Draw from a target using a dominating proposal.

    The caller supplies ``log_envelope = log(M)`` such that
    ``target(x) <= M * proposal(x)`` for every proposal-reachable ``x``. The target
    may be unnormalized provided ``M`` uses the same normalization convention.

    Density callables receive one proposal at a time. Proposal generation is batched
    so model implementations can still use vectorized random-number generation.
    """

    size = validate_size(size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_proposals is not None and max_proposals < size:
        raise ValueError("max_proposals must be at least size")
    if not np.isfinite(log_envelope):
        raise ValueError("log_envelope must be finite")
    if envelope_tolerance < 0.0:
        raise ValueError("envelope_tolerance must be nonnegative")
    if size == 0:
        return RejectionResult(
            samples=np.empty(0, dtype=np.float64),
            n_accepted=0,
            n_proposals=0,
            diagnostics={"acceptance_rate": float("nan")},
        )

    accepted_chunks: list[Array] = []
    n_accepted = 0
    n_proposals = 0
    event_shape: tuple[int, ...] | None = None

    while n_accepted < size:
        remaining_limit = None if max_proposals is None else max_proposals - n_proposals
        if remaining_limit is not None and remaining_limit <= 0:
            raise RuntimeError(
                f"rejection sampler reached max_proposals={max_proposals} "
                f"after accepting {n_accepted}/{size} samples"
            )
        draw_count = min(batch_size, size - n_accepted)
        if remaining_limit is not None:
            draw_count = min(draw_count, remaining_limit)
        proposals = np.asarray(proposal_sampler(rng, draw_count), dtype=np.float64)
        if proposals.ndim == 0 or proposals.shape[0] != draw_count:
            raise ValueError("proposal_sampler must return an array whose first axis equals size")
        if event_shape is None:
            event_shape = proposals.shape[1:]
        elif proposals.shape[1:] != event_shape:
            raise ValueError("proposal_sampler changed event shape between batches")

        if counter is not None:
            counter.increment("extra_proposals", draw_count)

        accepted_in_batch: list[Array] = []
        for proposal in proposals:
            log_target_value = float(log_target(proposal))
            log_proposal_value = float(log_proposal(proposal))
            if counter is not None:
                counter.increment("log_density_evaluations")
                counter.increment("proposal_density_evaluations")

            log_acceptance = log_target_value - log_proposal_value - log_envelope
            if np.isnan(log_acceptance):
                raise ValueError("density evaluation produced an undefined acceptance ratio")
            if log_acceptance > envelope_tolerance:
                raise ValueError(
                    "proposal envelope is invalid: observed log acceptance ratio "
                    f"{log_acceptance:.6g} > 0"
                )

            log_uniform = float(np.log(max(rng.random(), np.nextafter(0.0, 1.0))))
            if counter is not None:
                counter.increment("uniform_draws")
            n_proposals += 1
            if log_uniform <= min(0.0, log_acceptance):
                accepted_in_batch.append(np.array(proposal, copy=True))
                n_accepted += 1
                if n_accepted == size:
                    break

        if accepted_in_batch:
            accepted_chunks.append(np.stack(accepted_in_batch, axis=0))

    samples = np.concatenate(accepted_chunks, axis=0)
    return RejectionResult(
        samples=samples,
        n_accepted=size,
        n_proposals=n_proposals,
        diagnostics={
            "acceptance_rate": size / n_proposals,
            "proposals_per_sample": n_proposals / size,
            "log_envelope": log_envelope,
        },
    )
