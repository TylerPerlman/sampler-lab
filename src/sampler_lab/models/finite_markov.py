"""Analytically tractable finite-state Markov-chain examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sampler_lab.markov.finite_state import FiniteStateMarkovChain

Array = NDArray[np.float64]


def two_state_chain(switch_01: float, switch_10: float) -> FiniteStateMarkovChain:
    """Return a two-state chain with specified switching probabilities."""

    for name, probability in (("switch_01", switch_01), ("switch_10", switch_10)):
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    transition = np.asarray(
        [
            [1.0 - switch_01, switch_01],
            [switch_10, 1.0 - switch_10],
        ],
        dtype=np.float64,
    )
    return FiniteStateMarkovChain(transition)


def ring_random_walk(
    n_states: int,
    *,
    clockwise: float,
    counterclockwise: float,
    stay: float,
) -> FiniteStateMarkovChain:
    """Return a translation-invariant nearest-neighbor walk on a ring."""

    if isinstance(n_states, bool) or not isinstance(n_states, int):
        raise TypeError("n_states must be an integer")
    if n_states < 3:
        raise ValueError("n_states must be at least three")
    probabilities = np.asarray([clockwise, counterclockwise, stay], dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("transition probabilities must be finite and nonnegative")
    if not np.isclose(float(np.sum(probabilities)), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("clockwise, counterclockwise, and stay must sum to one")

    transition = np.zeros((n_states, n_states), dtype=np.float64)
    for state in range(n_states):
        transition[state, state] += stay
        transition[state, (state + 1) % n_states] += clockwise
        transition[state, (state - 1) % n_states] += counterclockwise
    return FiniteStateMarkovChain(transition)


def deterministic_cycle(n_states: int) -> FiniteStateMarkovChain:
    """Return the periodic deterministic rotation ``i -> i + 1 mod n``."""

    return ring_random_walk(
        n_states,
        clockwise=1.0,
        counterclockwise=0.0,
        stay=0.0,
    )


def ring_cosine_observable(n_states: int, *, frequency: int = 1) -> Array:
    """Return a real Fourier-mode observable on a ring."""

    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1:
        raise ValueError("n_states must be a positive integer")
    if isinstance(frequency, bool) or not isinstance(frequency, int):
        raise TypeError("frequency must be an integer")
    angles = 2.0 * np.pi * frequency * np.arange(n_states, dtype=np.float64) / n_states
    return np.asarray(np.cos(angles), dtype=np.float64)
