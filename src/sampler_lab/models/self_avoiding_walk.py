"""Self-avoiding walks and Rosenbluth sequential proposals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.particles import (
    BernoulliResampler,
    MultinomialResampler,
    PropagationResult,
    SequentialImportanceResult,
    SystematicResampler,
    sequential_importance_sampling,
)

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

_DIRECTIONS = np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=np.int64)


def _validate_steps(n_steps: int) -> int:
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    return n_steps


def _validate_periodic_size(periodic_size: int | None) -> int | None:
    if periodic_size is None:
        return None
    if isinstance(periodic_size, bool) or not isinstance(periodic_size, int):
        raise TypeError("periodic_size must be an integer")
    if periodic_size < 3:
        raise ValueError("periodic_size must be at least 3")
    return periodic_size


def _canonical(point: IntArray, periodic_size: int | None) -> tuple[int, int]:
    x = int(point[0])
    y = int(point[1])
    if periodic_size is not None:
        x %= periodic_size
        y %= periodic_size
    return x, y


def available_self_avoiding_neighbors(
    path: ArrayLike,
    *,
    periodic_size: int | None = None,
) -> IntArray:
    """Return unvisited nearest neighbors of the final point in ``path``."""

    size = _validate_periodic_size(periodic_size)
    points = np.asarray(path, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("path must have shape (n_vertices, 2)")
    visited = {_canonical(point, size) for point in points}
    current = points[-1]
    neighbors: list[tuple[int, int]] = []
    for direction in _DIRECTIONS:
        candidate_array = current + direction
        candidate = _canonical(candidate_array, size)
        if candidate not in visited:
            neighbors.append(candidate)
    if not neighbors:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(neighbors, dtype=np.int64)


def is_self_avoiding_walk(path: ArrayLike, *, periodic_size: int | None = None) -> bool:
    """Check uniqueness and nearest-neighbor adjacency of a lattice path."""

    size = _validate_periodic_size(periodic_size)
    points = np.asarray(path, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        return False
    canonical = [_canonical(point, size) for point in points]
    if len(set(canonical)) != len(canonical):
        return False
    for previous, current in pairwise(canonical):
        dx = abs(current[0] - previous[0])
        dy = abs(current[1] - previous[1])
        if size is not None:
            dx = min(dx, size - dx)
            dy = min(dy, size - dy)
        if dx + dy != 1:
            return False
    return True


def count_self_avoiding_walks(
    n_steps: int,
    *,
    periodic_size: int | None = None,
) -> int:
    """Exactly count fixed-origin square-lattice self-avoiding walks.

    This depth-first enumerator is intended for small validation cases, not
    record-setting enumeration. ``n_steps`` counts edges, so the path contains
    ``n_steps + 1`` vertices.
    """

    steps = _validate_steps(n_steps)
    size = _validate_periodic_size(periodic_size)
    start = (0, 0)
    visited = {start}

    def recurse(point: tuple[int, int], depth: int) -> int:
        if depth == steps:
            return 1
        total = 0
        point_array = np.asarray(point, dtype=np.int64)
        for direction in _DIRECTIONS:
            candidate = _canonical(point_array + direction, size)
            if candidate in visited:
                continue
            visited.add(candidate)
            total += recurse(candidate, depth + 1)
            visited.remove(candidate)
        return total

    return recurse(start, 0)


@dataclass(frozen=True, slots=True)
class SelfAvoidingWalkProposal:
    """Uniformly extend each path among its currently unvisited neighbors."""

    periodic_size: int | None = None

    def __post_init__(self) -> None:
        _validate_periodic_size(self.periodic_size)

    def propose(
        self,
        particles: Array,
        step: int,
        rng: np.random.Generator,
    ) -> PropagationResult:
        paths = np.asarray(particles, dtype=np.float64)
        if paths.ndim != 3 or paths.shape[2] != 2:
            raise ValueError("self-avoiding-walk particles must have shape (N, t, 2)")
        if paths.shape[1] != step:
            raise ValueError("step must match the current number of path vertices")

        integer_paths = np.asarray(paths, dtype=np.int64)
        candidates = integer_paths[:, -1, None, :] + _DIRECTIONS[None, :, :]
        if self.periodic_size is not None:
            candidates %= self.periodic_size

        # Candidate j is available exactly when it differs from every visited
        # point in that particle's path. The temporary boolean tensor has shape
        # (particles, four directions, path length), which stays modest for the
        # educational walk lengths targeted by this module.
        matches_visited = np.all(
            candidates[:, :, None, :] == integer_paths[:, None, :, :],
            axis=3,
        )
        available = ~np.any(matches_visited, axis=2)
        counts = np.asarray(np.sum(available, axis=1, dtype=np.int64), dtype=np.int64)
        alive = counts > 0

        ranks = np.zeros(paths.shape[0], dtype=np.int64)
        ranks[alive] = np.floor(rng.random(np.count_nonzero(alive)) * counts[alive]).astype(
            np.int64
        )
        cumulative_ranks = np.cumsum(available, axis=1) - 1
        selected_directions = np.argmax(
            available & (cumulative_ranks == ranks[:, None]),
            axis=1,
        )
        selected_points = candidates[np.arange(paths.shape[0]), selected_directions]

        extended = np.empty((paths.shape[0], step + 1, 2), dtype=np.float64)
        extended[:, :step] = paths
        extended[:, step] = paths[:, -1]
        extended[alive, step] = selected_points[alive]

        log_incremental_weights = np.full(paths.shape[0], -np.inf, dtype=np.float64)
        log_incremental_weights[alive] = np.log(counts[alive])
        return PropagationResult(extended, log_incremental_weights)


def sample_self_avoiding_walks(
    rng: np.random.Generator,
    *,
    n_steps: int,
    n_particles: int,
    resampling: str | None = None,
    resample_every_step: bool = False,
    resample_ess_fraction: float | None = None,
    periodic_size: int | None = None,
) -> SequentialImportanceResult:
    """Estimate and sample the uniform fixed-origin self-avoiding-walk law.

    The proposal is the Rosenbluth growth rule. Its incremental importance
    weight is the number of currently available neighbors. Consequently the
    returned normalizing-constant estimate targets the exact walk count.
    """

    steps = _validate_steps(n_steps)
    if isinstance(n_particles, bool) or not isinstance(n_particles, int):
        raise TypeError("n_particles must be an integer")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")

    schemes = {
        None: None,
        "multinomial": MultinomialResampler(),
        "systematic": SystematicResampler(),
        "bernoulli": BernoulliResampler(),
    }
    if resampling not in schemes:
        raise ValueError("resampling must be one of None, multinomial, systematic, bernoulli")
    scheme = schemes[resampling]
    initial_paths = np.zeros((n_particles, 1, 2), dtype=np.float64)
    return sequential_importance_sampling(
        initial_paths,
        steps,
        SelfAvoidingWalkProposal(periodic_size),
        rng,
        resampler=scheme,
        resample_every_step=resample_every_step,
        resample_ess_fraction=resample_ess_fraction,
        target_particle_count=n_particles,
    )
