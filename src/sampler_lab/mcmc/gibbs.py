"""Generic Gibbs schedules built from exact conditional updates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import MarkovKernel
from sampler_lab.core.results import Transition
from sampler_lab.mcmc.proposals import Array


class ConditionalUpdate(Protocol):
    """An exact draw from one conditional block given the complement."""

    def apply(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        """Return a state with one conditional block resampled."""


@dataclass(frozen=True, slots=True)
class FunctionalConditionalUpdate:
    """Adapt a function into a conditional-update object."""

    function: Callable[[Array, np.random.Generator, OperationCounter | None], ArrayLike]

    def apply(
        self,
        state: Array,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        updated = np.asarray(self.function(state, rng, counter), dtype=np.float64)
        if updated.shape != state.shape:
            raise ValueError("conditional update changed the state shape")
        return updated


def _validated_updates(updates: Sequence[ConditionalUpdate]) -> tuple[ConditionalUpdate, ...]:
    result = tuple(updates)
    if not result:
        raise ValueError("at least one conditional update is required")
    return result


@dataclass(slots=True, init=False)
class BlockGibbsKernel:
    """Apply one exact conditional block update per transition."""

    update: ConditionalUpdate
    counter: OperationCounter | None

    def __init__(
        self,
        update: ConditionalUpdate,
        *,
        counter: OperationCounter | None = None,
    ) -> None:
        self.update = update
        self.counter = counter

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = np.array(state, dtype=np.float64, copy=True)
        updated = np.asarray(
            self.update.apply(np.array(current, copy=True), rng, counter=self.counter),
            dtype=np.float64,
        )
        if updated.shape != current.shape:
            raise ValueError("conditional update changed the state shape")
        return Transition(state=np.array(updated, copy=True), diagnostics={"n_updates": 1.0})


@dataclass(slots=True, init=False)
class RandomScanGibbsKernel:
    """Choose a conditional block independently at each transition."""

    updates: tuple[ConditionalUpdate, ...]
    probabilities: Array
    counter: OperationCounter | None

    def __init__(
        self,
        updates: Sequence[ConditionalUpdate],
        probabilities: ArrayLike | None = None,
        *,
        counter: OperationCounter | None = None,
    ) -> None:
        validated = _validated_updates(updates)
        if probabilities is None:
            weights = np.full(len(validated), 1.0 / len(validated), dtype=np.float64)
        else:
            weights = np.asarray(probabilities, dtype=np.float64)
            if weights.shape != (len(validated),):
                raise ValueError("scan probabilities must match the number of updates")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("scan probabilities must be finite and nonnegative")
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError("scan probabilities must have positive mass")
            weights = weights / total
        self.updates = validated
        self.probabilities = weights
        self.counter = counter

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = np.array(state, dtype=np.float64, copy=True)
        update_index = int(rng.choice(len(self.updates), p=self.probabilities))
        if self.counter is not None:
            self.counter.uniform_draws += 1
        updated = np.asarray(
            self.updates[update_index].apply(
                np.array(current, copy=True), rng, counter=self.counter
            ),
            dtype=np.float64,
        )
        if updated.shape != current.shape:
            raise ValueError("conditional update changed the state shape")
        return Transition(
            state=np.array(updated, copy=True),
            diagnostics={"update_index": float(update_index), "n_updates": 1.0},
        )


@dataclass(slots=True, init=False)
class DeterministicSweepGibbsKernel:
    """Apply a fixed sequence of conditional blocks in one macro-transition."""

    updates: tuple[ConditionalUpdate, ...]
    counter: OperationCounter | None

    def __init__(
        self,
        updates: Sequence[ConditionalUpdate],
        *,
        counter: OperationCounter | None = None,
    ) -> None:
        self.updates = _validated_updates(updates)
        self.counter = counter

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        updated = np.array(state, dtype=np.float64, copy=True)
        for update in self.updates:
            updated = np.asarray(
                update.apply(updated, rng, counter=self.counter),
                dtype=np.float64,
            )
        return Transition(
            state=np.array(updated, copy=True),
            diagnostics={"n_updates": float(len(self.updates))},
        )


@dataclass(slots=True)
class TransformedGibbsKernel:
    """Conjugate a Gibbs kernel through an invertible coordinate transform."""

    inner_kernel: MarkovKernel
    forward: Callable[[Array], ArrayLike]
    inverse: Callable[[Array], ArrayLike]

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        current = np.asarray(state, dtype=np.float64)
        transformed = np.asarray(self.forward(current), dtype=np.float64)
        if transformed.ndim == 0 or not np.all(np.isfinite(transformed)):
            raise ValueError("forward transform returned an invalid state")
        inner = self.inner_kernel.step(transformed, rng)
        restored = np.asarray(self.inverse(inner.state), dtype=np.float64)
        if restored.shape != current.shape or not np.all(np.isfinite(restored)):
            raise ValueError("inverse transform returned an invalid state")
        diagnostics = dict(inner.diagnostics)
        diagnostics["transformed"] = 1.0
        return Transition(
            state=np.array(restored, copy=True),
            accepted=inner.accepted,
            log_acceptance_ratio=inner.log_acceptance_ratio,
            diagnostics=diagnostics,
        )
