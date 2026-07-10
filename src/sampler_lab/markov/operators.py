"""Finite-state Markov operators acting on functions and measures."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def as_probability_vector(
    probabilities: ArrayLike,
    *,
    n_states: int | None = None,
    atol: float = 1e-12,
) -> Array:
    """Validate, copy, and normalize a finite probability vector.

    Values smaller than zero by at most ``atol`` are clipped to zero. The
    returned vector is always a fresh ``float64`` array whose entries sum to
    one. Larger normalization errors are rejected rather than silently hidden.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a nonempty one-dimensional array")
    if n_states is not None and values.size != n_states:
        raise ValueError("probability vector length does not match the state space")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if np.any(values < -atol):
        raise ValueError("probabilities must be nonnegative")

    clipped = np.maximum(values, 0.0)
    total = float(np.sum(clipped))
    if not np.isclose(total, 1.0, atol=atol, rtol=0.0):
        raise ValueError("probabilities must sum to one")
    return np.asarray(clipped / total, dtype=np.float64)


def as_observable(values: ArrayLike, *, n_states: int | None = None) -> Array:
    """Validate a real-valued observable on a finite state space."""

    observable = np.asarray(values, dtype=np.float64)
    if observable.ndim != 1 or observable.size == 0:
        raise ValueError("observable must be a nonempty one-dimensional array")
    if n_states is not None and observable.size != n_states:
        raise ValueError("observable length does not match the state space")
    if not np.all(np.isfinite(observable)):
        raise ValueError("observable values must be finite")
    return np.array(observable, dtype=np.float64, copy=True)


def apply_transition(transition: ArrayLike, observable: ArrayLike) -> Array:
    """Apply a row-stochastic transition matrix to a column of function values."""

    matrix = np.asarray(transition, dtype=np.float64)
    values = as_observable(observable)
    if matrix.ndim != 2 or matrix.shape != (values.size, values.size):
        raise ValueError("transition must be square and match the observable")
    return np.asarray(matrix @ values, dtype=np.float64)


def pushforward_measure(measure: ArrayLike, transition: ArrayLike) -> Array:
    """Push a row-vector probability measure forward by one Markov step."""

    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be square")
    probabilities = as_probability_vector(measure, n_states=matrix.shape[0])
    return np.asarray(probabilities @ matrix, dtype=np.float64)


def expectation(probabilities: ArrayLike, observable: ArrayLike) -> float:
    """Return the finite-state expectation ``sum_i pi_i f_i``."""

    values = as_observable(observable)
    measure = as_probability_vector(probabilities, n_states=values.size)
    return float(np.dot(measure, values))


def centered_observable(probabilities: ArrayLike, observable: ArrayLike) -> Array:
    """Center an observable under the supplied probability distribution."""

    values = as_observable(observable)
    mean = expectation(probabilities, values)
    return np.asarray(values - mean, dtype=np.float64)


def weighted_inner_product(
    probabilities: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
) -> float:
    """Return the ``L2(pi)`` inner product of two finite-state functions."""

    left_values = as_observable(left)
    right_values = as_observable(right, n_states=left_values.size)
    measure = as_probability_vector(probabilities, n_states=left_values.size)
    return float(np.dot(measure, left_values * right_values))


def variance(probabilities: ArrayLike, observable: ArrayLike) -> float:
    """Return the stationary variance of an observable."""

    centered = centered_observable(probabilities, observable)
    return weighted_inner_product(probabilities, centered, centered)


def covariance(
    probabilities: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
) -> float:
    """Return the covariance of two observables under one measure."""

    left_centered = centered_observable(probabilities, left)
    right_centered = centered_observable(probabilities, right)
    return weighted_inner_product(probabilities, left_centered, right_centered)


def constant_projection(probabilities: ArrayLike) -> Array:
    """Matrix mapping every observable to its constant stationary projection."""

    measure = as_probability_vector(probabilities)
    return np.asarray(np.ones((measure.size, 1)) @ measure[None, :], dtype=np.float64)


def operator_duality_residual(
    measure: ArrayLike,
    transition: ArrayLike,
    observable: ArrayLike,
) -> float:
    """Numerically check ``(mu P)f = mu(Pf)`` for row-vector conventions."""

    matrix = np.asarray(transition, dtype=np.float64)
    values = as_observable(observable)
    probabilities = as_probability_vector(measure, n_states=values.size)
    if matrix.shape != (values.size, values.size):
        raise ValueError("transition must be square and match the state space")
    left = float(np.dot(probabilities @ matrix, values))
    right = float(np.dot(probabilities, matrix @ values))
    return abs(left - right)
