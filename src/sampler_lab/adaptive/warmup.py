"""Warmup windows and explicit training/evaluation result types."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.mcmc.chain import MCMCTrajectory

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WarmupWindow:
    """Half-open warmup interval and the parameters adapted inside it."""

    start: int
    stop: int
    adapt_scale: bool
    adapt_covariance: bool

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("warmup windows require 0 <= start < stop")

    @property
    def length(self) -> int:
        return self.stop - self.start


def expanding_warmup_windows(
    n_warmup: int,
    *,
    initial_buffer: int = 75,
    terminal_buffer: int = 50,
    base_window: int = 25,
) -> tuple[WarmupWindow, ...]:
    """Construct Stan-style expanding covariance windows with safe small-run fallback."""

    for name, value in (
        ("n_warmup", n_warmup),
        ("initial_buffer", initial_buffer),
        ("terminal_buffer", terminal_buffer),
        ("base_window", base_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if n_warmup <= 0 or initial_buffer < 0 or terminal_buffer < 0 or base_window <= 0:
        raise ValueError("warmup lengths must be nonnegative and n_warmup/base_window positive")

    if initial_buffer + terminal_buffer + base_window > n_warmup:
        first = max(1, n_warmup // 5)
        last = max(1, n_warmup // 5)
        middle_start = first
        middle_stop = max(middle_start + 1, n_warmup - last)
        windows: list[WarmupWindow] = []
        if first > 0:
            windows.append(WarmupWindow(0, first, True, False))
        if middle_stop > middle_start:
            windows.append(WarmupWindow(middle_start, middle_stop, True, True))
        if middle_stop < n_warmup:
            windows.append(WarmupWindow(middle_stop, n_warmup, True, False))
        return tuple(windows)

    windows = [WarmupWindow(0, initial_buffer, True, False)] if initial_buffer else []
    start = initial_buffer
    slow_stop = n_warmup - terminal_buffer
    width = base_window
    while start < slow_stop:
        stop = min(start + width, slow_stop)
        windows.append(WarmupWindow(start, stop, True, True))
        start = stop
        width *= 2
    if terminal_buffer:
        windows.append(WarmupWindow(slow_stop, n_warmup, True, False))
    return tuple(windows)


@dataclass(frozen=True, slots=True)
class FrozenPolicy:
    """Detached policy parameters and metadata produced by warmup."""

    name: str
    parameters: Array
    metadata: dict[str, float]

    def __init__(self, name: str, parameters: ArrayLike, metadata: dict[str, float]) -> None:
        values = np.asarray(parameters, dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("frozen policy parameters must be nonempty and finite")
        copied = np.array(values, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "parameters", copied)
        object.__setattr__(self, "metadata", dict(metadata))

    def to_dict(self) -> dict[str, str | list[float] | dict[str, float]]:
        """Return a portable detached representation."""

        return {
            "name": self.name,
            "parameters": [float(value) for value in self.parameters.reshape(-1)],
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the frozen policy without executable code."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, str | list[float] | dict[str, float]],
    ) -> FrozenPolicy:
        """Restore a policy from :meth:`to_dict` output."""

        name = payload.get("name")
        parameters = payload.get("parameters")
        metadata = payload.get("metadata")
        if (
            not isinstance(name, str)
            or not isinstance(parameters, list)
            or not isinstance(metadata, dict)
        ):
            raise ValueError("invalid frozen-policy payload")
        if not all(isinstance(value, (int, float)) for value in parameters):
            raise ValueError("frozen-policy parameters must be numeric")
        if not all(
            isinstance(key, str) and isinstance(value, (int, float))
            for key, value in metadata.items()
        ):
            raise ValueError("frozen-policy metadata must map strings to numbers")
        numeric_metadata = {str(key): float(value) for key, value in metadata.items()}
        return cls(name, np.asarray(parameters, dtype=np.float64), numeric_metadata)

    @classmethod
    def from_json(cls, payload: str) -> FrozenPolicy:
        """Restore a policy from JSON."""

        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("frozen-policy JSON must encode an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class AdaptiveTrainingResult:
    """Warmup states and parameter history, never mislabeled as posterior output."""

    states: Array
    parameter_history: Array
    acceptance_probabilities: Array
    frozen_policy: FrozenPolicy
    diagnostics: dict[str, float]

    def __init__(
        self,
        states: ArrayLike,
        parameter_history: ArrayLike,
        acceptance_probabilities: ArrayLike,
        frozen_policy: FrozenPolicy,
        diagnostics: dict[str, float],
    ) -> None:
        state_values = np.asarray(states, dtype=np.float64)
        parameters = np.asarray(parameter_history, dtype=np.float64)
        acceptance = np.asarray(acceptance_probabilities, dtype=np.float64)
        if state_values.ndim != 2 or state_values.shape[0] < 2:
            raise ValueError("states must contain a vector trajectory")
        if parameters.shape[0] != state_values.shape[0] - 1:
            raise ValueError("parameter history must have one row per warmup transition")
        if acceptance.shape != (state_values.shape[0] - 1,):
            raise ValueError("acceptance probabilities must match warmup transitions")
        if not np.all(np.isfinite(state_values)) or not np.all(np.isfinite(parameters)):
            raise ValueError("warmup states and parameters must be finite")
        if not np.all(np.isfinite(acceptance)) or np.any((acceptance < 0.0) | (acceptance > 1.0)):
            raise ValueError("acceptance probabilities must lie in [0, 1]")
        state_copy = np.array(state_values, copy=True)
        parameter_copy = np.array(parameters, copy=True)
        acceptance_copy = np.array(acceptance, copy=True)
        state_copy.setflags(write=False)
        parameter_copy.setflags(write=False)
        acceptance_copy.setflags(write=False)
        object.__setattr__(self, "states", state_copy)
        object.__setattr__(self, "parameter_history", parameter_copy)
        object.__setattr__(self, "acceptance_probabilities", acceptance_copy)
        object.__setattr__(self, "frozen_policy", frozen_policy)
        object.__setattr__(self, "diagnostics", dict(diagnostics))


@dataclass(frozen=True, slots=True)
class EvaluationTrajectory:
    """Fresh trajectory generated only after adaptation has been frozen."""

    trajectory: MCMCTrajectory
    frozen_policy: FrozenPolicy
    training_steps_excluded: int

    def __post_init__(self) -> None:
        if self.training_steps_excluded < 0:
            raise ValueError("training_steps_excluded must be nonnegative")
