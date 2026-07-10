"""Empirical and analytic generator utilities for small-step Markov dynamics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.protocols import MarkovKernel
from sampler_lab.core.results import IIDEstimate
from sampler_lab.estimators.iid import iid_estimate

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GeneratorEstimate:
    """Monte Carlo estimate of ``(P_h f(x) - f(x)) / h``."""

    value: float
    standard_error: float
    sample_variance: float
    n_replications: int
    step_size: float
    initial_value: float
    mean_next_value: float


@dataclass(frozen=True, slots=True)
class LocalMomentEstimate:
    """Empirical local drift and covariance rates of a small-step kernel."""

    drift: Array
    drift_standard_error: Array
    second_moment_rate: Array
    centered_covariance_rate: Array
    n_replications: int
    step_size: float


def _as_state(x: ArrayLike) -> Array:
    state = np.asarray(x, dtype=np.float64)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("x must be a nonempty one-dimensional state")
    if not np.all(np.isfinite(state)):
        raise ValueError("x must be finite")
    return state


def _validate_sampling_parameters(step_size: float, n_replications: int) -> None:
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be positive and finite")
    if isinstance(n_replications, bool) or not isinstance(n_replications, int):
        raise TypeError("n_replications must be an integer")
    if n_replications < 2:
        raise ValueError("n_replications must be at least two")


def estimate_discrete_generator(
    kernel: MarkovKernel,
    test_function: Callable[[Array], float],
    x: ArrayLike,
    step_size: float,
    n_replications: int,
    rng: np.random.Generator,
) -> GeneratorEstimate:
    """Estimate the scaled one-step generator of ``kernel`` at ``x``.

    Every replication starts from the same state, so this estimates a conditional
    expectation rather than a time average from one trajectory.
    """

    state = _as_state(x)
    _validate_sampling_parameters(step_size, n_replications)
    initial_value = float(test_function(np.array(state, copy=True)))
    if not np.isfinite(initial_value):
        raise ValueError("test_function must be finite at x")

    increments = np.empty(n_replications, dtype=np.float64)
    next_values = np.empty(n_replications, dtype=np.float64)
    for index in range(n_replications):
        transition = kernel.step(np.array(state, copy=True), rng)
        next_state = np.asarray(transition.state, dtype=np.float64)
        if next_state.shape != state.shape or not np.all(np.isfinite(next_state)):
            raise ValueError("kernel returned an invalid state")
        next_value = float(test_function(next_state))
        if not np.isfinite(next_value):
            raise ValueError("test_function returned a nonfinite value")
        next_values[index] = next_value
        increments[index] = (next_value - initial_value) / step_size

    estimate = iid_estimate(increments)
    return GeneratorEstimate(
        value=estimate.value,
        standard_error=estimate.standard_error,
        sample_variance=estimate.sample_variance,
        n_replications=n_replications,
        step_size=float(step_size),
        initial_value=initial_value,
        mean_next_value=float(np.mean(next_values)),
    )


def estimate_local_moments(
    kernel: MarkovKernel,
    x: ArrayLike,
    step_size: float,
    n_replications: int,
    rng: np.random.Generator,
) -> LocalMomentEstimate:
    """Estimate local drift and infinitesimal covariance from repeated one-step moves.

    ``second_moment_rate`` estimates ``E[Delta X Delta X^T | X=x] / h``.
    ``centered_covariance_rate`` removes the empirical mean increment. Both converge
    to the diffusion covariance as ``h -> 0``; the raw second moment is the quantity
    appearing directly in the Kramers--Moyal limit.
    """

    state = _as_state(x)
    _validate_sampling_parameters(step_size, n_replications)
    increments = np.empty((n_replications, state.size), dtype=np.float64)
    for index in range(n_replications):
        transition = kernel.step(np.array(state, copy=True), rng)
        next_state = np.asarray(transition.state, dtype=np.float64)
        if next_state.shape != state.shape or not np.all(np.isfinite(next_state)):
            raise ValueError("kernel returned an invalid state")
        increments[index] = next_state - state

    mean_increment = np.mean(increments, axis=0)
    drift = mean_increment / step_size
    drift_standard_error = np.std(increments / step_size, axis=0, ddof=1) / np.sqrt(n_replications)
    second_moment_rate = increments.T @ increments / (n_replications * step_size)
    centered = increments - mean_increment
    centered_covariance_rate = centered.T @ centered / ((n_replications - 1) * step_size)
    return LocalMomentEstimate(
        drift=np.asarray(drift, dtype=np.float64),
        drift_standard_error=np.asarray(drift_standard_error, dtype=np.float64),
        second_moment_rate=np.asarray(second_moment_rate, dtype=np.float64),
        centered_covariance_rate=np.asarray(centered_covariance_rate, dtype=np.float64),
        n_replications=n_replications,
        step_size=float(step_size),
    )


def diffusion_generator_value(
    drift: Callable[[Array], ArrayLike],
    covariance: Callable[[Array], ArrayLike],
    gradient: Callable[[Array], ArrayLike],
    hessian: Callable[[Array], ArrayLike],
    x: ArrayLike,
) -> float:
    """Evaluate ``b dot grad(f) + 1/2 tr(a Hess(f))`` at one point."""

    state = _as_state(x)
    drift_value = np.asarray(drift(state), dtype=np.float64)
    covariance_value = np.asarray(covariance(state), dtype=np.float64)
    gradient_value = np.asarray(gradient(state), dtype=np.float64)
    hessian_value = np.asarray(hessian(state), dtype=np.float64)
    dimension = state.size
    if drift_value.shape != (dimension,) or gradient_value.shape != (dimension,):
        raise ValueError("drift and gradient must match the state dimension")
    if covariance_value.shape != (dimension, dimension):
        raise ValueError("covariance must be a square state-dimensional matrix")
    if hessian_value.shape != (dimension, dimension):
        raise ValueError("hessian must be a square state-dimensional matrix")
    if not all(
        np.all(np.isfinite(value))
        for value in (drift_value, covariance_value, gradient_value, hessian_value)
    ):
        raise ValueError("generator inputs must be finite")
    return float(drift_value @ gradient_value + 0.5 * np.trace(covariance_value @ hessian_value))


def estimate_poisson_invariant_bias(
    kernel: MarkovKernel,
    poisson_solution: Callable[[Array], float],
    continuous_generator_on_solution: Callable[[Array], float],
    stationary_samples: ArrayLike,
    step_size: float,
    rng: np.random.Generator,
    *,
    replications_per_state: int = 1,
) -> IIDEstimate:
    """Estimate an invariant-measure bias through the Poisson identity.

    If ``L g = f - pi(f)`` and the supplied states follow an invariant law ``pi_h``
    of ``kernel``, then

    ``pi_h(f) - pi(f) = pi_h[(L - (P_h-I)/h) g]``.

    The routine estimates the right-hand side. It does not pretend the input states
    are independent when they come from a chain; callers remain responsible for a
    time-series uncertainty analysis in that case.
    """

    samples = np.asarray(stationary_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("stationary_samples must have shape (n_samples, dimension)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("stationary_samples must be finite")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be positive and finite")
    if isinstance(replications_per_state, bool) or not isinstance(replications_per_state, int):
        raise TypeError("replications_per_state must be an integer")
    if replications_per_state <= 0:
        raise ValueError("replications_per_state must be positive")

    terms = np.empty(samples.shape[0], dtype=np.float64)
    for sample_index, state in enumerate(samples):
        current_value = float(poisson_solution(state))
        continuous_value = float(continuous_generator_on_solution(state))
        if not np.isfinite(current_value) or not np.isfinite(continuous_value):
            raise ValueError("Poisson solution and continuous generator must be finite")
        next_sum = 0.0
        for _ in range(replications_per_state):
            transition = kernel.step(np.array(state, copy=True), rng)
            next_state = np.asarray(transition.state, dtype=np.float64)
            if next_state.shape != state.shape or not np.all(np.isfinite(next_state)):
                raise ValueError("kernel returned an invalid state")
            next_value = float(poisson_solution(next_state))
            if not np.isfinite(next_value):
                raise ValueError("Poisson solution returned a nonfinite value")
            next_sum += next_value
        discrete_value = (next_sum / replications_per_state - current_value) / step_size
        terms[sample_index] = continuous_value - discrete_value
    return iid_estimate(terms)
