"""Variance-reduction baselines for score-function policy gradients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


class Baseline(Protocol):
    """Predict and update a scalar return baseline."""

    def predict_batch(self, features: ArrayLike) -> Array:
        """Predict one baseline value per feature row."""

    def fit(self, features: ArrayLike, returns: ArrayLike) -> None:
        """Update baseline parameters from a batch."""


@dataclass(slots=True)
class ZeroBaseline:
    """No variance reduction."""

    def predict_batch(self, features: ArrayLike) -> Array:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("features must be a finite matrix")
        return np.zeros(values.shape[0], dtype=np.float64)

    def fit(self, features: ArrayLike, returns: ArrayLike) -> None:
        self.predict_batch(features)
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("returns must be a finite vector")


@dataclass(slots=True)
class RunningMeanBaseline:
    """Online scalar baseline equal to the mean return seen so far."""

    count: int = 0
    mean: float = 0.0

    def predict_batch(self, features: ArrayLike) -> Array:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("features must be a finite matrix")
        return np.full(values.shape[0], self.mean if self.count else 0.0, dtype=np.float64)

    def fit(self, features: ArrayLike, returns: ArrayLike) -> None:
        feature_values = np.asarray(features, dtype=np.float64)
        values = np.asarray(returns, dtype=np.float64)
        if feature_values.ndim != 2 or feature_values.shape[0] != values.size:
            raise ValueError("features and returns must have the same batch length")
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("returns must be a finite vector")
        for value in values:
            self.count += 1
            self.mean += (float(value) - self.mean) / self.count


@dataclass(slots=True)
class LinearBaseline:
    """Ridge-regularized least-squares baseline with an intercept."""

    ridge: float = 1e-6
    coefficients: Array | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative and finite")

    @staticmethod
    def _design(features: ArrayLike) -> Array:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("features must be a finite matrix")
        return np.column_stack((values, np.ones(values.shape[0], dtype=np.float64)))

    def predict_batch(self, features: ArrayLike) -> Array:
        design = self._design(features)
        if self.coefficients is None:
            return np.zeros(design.shape[0], dtype=np.float64)
        if self.coefficients.shape != (design.shape[1],):
            raise ValueError("feature dimension changed after fitting the baseline")
        return np.asarray(design @ self.coefficients, dtype=np.float64)

    def fit(self, features: ArrayLike, returns: ArrayLike) -> None:
        design = self._design(features)
        values = np.asarray(returns, dtype=np.float64)
        if values.shape != (design.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError("returns must match the finite feature batch")
        gram = design.T @ design
        penalty = self.ridge * np.eye(gram.shape[0], dtype=np.float64)
        penalty[-1, -1] = 0.0
        self.coefficients = np.asarray(
            np.linalg.solve(gram + penalty, design.T @ values),
            dtype=np.float64,
        )
