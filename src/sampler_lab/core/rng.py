"""Explicit random-number generator construction and reproducible spawning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SeedLike = int | Sequence[int] | np.random.SeedSequence | None


def make_rng(seed: SeedLike = None) -> np.random.Generator:
    """Create a NumPy generator without touching global random state."""

    return np.random.default_rng(seed)


def spawn_rngs(seed: SeedLike, n_streams: int) -> tuple[np.random.Generator, ...]:
    """Create reproducible, statistically independent child streams.

    Parameters
    ----------
    seed:
        Entropy used for the parent ``SeedSequence``.
    n_streams:
        Number of child generators. Must be nonnegative.
    """

    if n_streams < 0:
        raise ValueError("n_streams must be nonnegative")
    if isinstance(seed, np.random.SeedSequence):
        seed_sequence = seed
    else:
        seed_sequence = np.random.SeedSequence(seed)
    return tuple(np.random.default_rng(child) for child in seed_sequence.spawn(n_streams))
