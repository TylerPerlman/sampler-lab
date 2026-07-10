"""Uniform distribution on the two-dimensional unit disk."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.numerics import validate_size
from sampler_lab.core.results import RejectionResult
from sampler_lab.exact.rejection import rejection_sample
from sampler_lab.exact.transforms import polar_to_cartesian

Array = NDArray[np.float64]


def sample_unit_disk_direct(
    rng: np.random.Generator,
    size: int,
    *,
    counter: OperationCounter | None = None,
) -> Array:
    """Sample uniformly from the unit disk using ``R = sqrt(U)`` and a uniform angle."""

    size = validate_size(size)
    radius = np.sqrt(rng.random(size))
    angle = 2.0 * np.pi * rng.random(size)
    if counter is not None:
        counter.increment("uniform_draws", 2 * size)
    return polar_to_cartesian(radius, angle)


def sample_unit_disk_rejection(
    rng: np.random.Generator,
    size: int,
    *,
    batch_size: int = 1_024,
    max_proposals: int | None = None,
    counter: OperationCounter | None = None,
) -> RejectionResult:
    """Sample uniformly from the unit disk by rejecting points from ``[-1, 1]^2``."""

    def proposal_sampler(generator: np.random.Generator, n: int) -> Array:
        if counter is not None:
            counter.increment("uniform_draws", 2 * n)
        return generator.uniform(-1.0, 1.0, size=(n, 2))

    def log_target(point: Array) -> float:
        return 0.0 if float(point @ point) <= 1.0 else float("-inf")

    def log_proposal(point: Array) -> float:
        del point
        return float(-np.log(4.0))

    # The unnormalized target is one inside the disk. Since q = 1/4 on the square,
    # the tight envelope constant under this convention is M = 4.
    return rejection_sample(
        rng=rng,
        size=size,
        proposal_sampler=proposal_sampler,
        log_target=log_target,
        log_proposal=log_proposal,
        log_envelope=np.log(4.0),
        batch_size=batch_size,
        max_proposals=max_proposals,
        counter=counter,
    )


def unit_disk_radius_squared(points: Array) -> Array:
    """Return squared radii after validating the final coordinate dimension."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    return np.asarray(np.sum(np.square(array), axis=1), dtype=np.float64)
