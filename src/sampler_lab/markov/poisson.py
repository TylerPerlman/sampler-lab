"""Poisson equations and exact stationary error formulas for finite chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.markov.finite_state import FiniteStateMarkovChain
from sampler_lab.markov.operators import (
    as_observable,
    as_probability_vector,
    centered_observable,
    expectation,
    variance,
    weighted_inner_product,
)

Array = NDArray[np.float64]


def _stationary_measure(
    chain: FiniteStateMarkovChain,
    probabilities: ArrayLike | None,
) -> Array:
    distributions = chain.invariant_distributions()
    if distributions.shape[0] != 1:
        raise ValueError("Poisson analysis requires a unique invariant distribution")
    measure = (
        distributions[0]
        if probabilities is None
        else as_probability_vector(probabilities, n_states=chain.n_states)
    )
    if chain.global_balance_residual(measure) > 10.0 * chain.tolerance:
        raise ValueError("probabilities must be invariant for the chain")
    return np.asarray(measure, dtype=np.float64)


@dataclass(frozen=True, slots=True, init=False)
class PoissonEquationResult:
    """Centered solution of ``(P - I)u = f - pi[f]``."""

    observable: Array
    centered_forcing: Array
    solution: Array
    stationary_distribution: Array
    stationary_mean: float
    residual_norm: float
    centering_residual: float

    def __init__(
        self,
        *,
        observable: Array,
        centered_forcing: Array,
        solution: Array,
        stationary_distribution: Array,
        stationary_mean: float,
        residual_norm: float,
        centering_residual: float,
    ) -> None:
        arrays: list[Array] = []
        for values in (observable, centered_forcing, solution, stationary_distribution):
            copied = np.array(values, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            arrays.append(copied)
        object.__setattr__(self, "observable", arrays[0])
        object.__setattr__(self, "centered_forcing", arrays[1])
        object.__setattr__(self, "solution", arrays[2])
        object.__setattr__(self, "stationary_distribution", arrays[3])
        object.__setattr__(self, "stationary_mean", float(stationary_mean))
        object.__setattr__(self, "residual_norm", float(residual_norm))
        object.__setattr__(self, "centering_residual", float(centering_residual))

    @property
    def fundamental_solution(self) -> Array:
        """Return ``h = -u``, satisfying ``(I - P)h = f - pi[f]``."""

        return np.asarray(-self.solution, dtype=np.float64)


def solve_poisson_equation(
    chain: FiniteStateMarkovChain,
    observable: ArrayLike,
    *,
    probabilities: ArrayLike | None = None,
) -> PoissonEquationResult:
    """Solve the centered finite-state Poisson equation exactly.

    The package convention is ``L = P - I``. The additive constant is fixed by
    ``pi[u] = 0``. Algebraically, the solve uses the nonsingular fundamental
    matrix ``I - P + 1 pi`` and then flips sign.
    """

    values = as_observable(observable, n_states=chain.n_states)
    measure = _stationary_measure(chain, probabilities)
    mean = expectation(measure, values)
    forcing = np.asarray(values - mean, dtype=np.float64)
    fundamental_matrix = (
        np.eye(chain.n_states) - chain.transition + chain.constant_projection(measure)
    )
    try:
        fundamental_solution = np.linalg.solve(fundamental_matrix, forcing)
    except np.linalg.LinAlgError as error:
        raise ValueError("the centered Poisson equation is not uniquely solvable") from error
    solution = np.asarray(-fundamental_solution, dtype=np.float64)
    residual = chain.generator @ solution - forcing
    return PoissonEquationResult(
        observable=values,
        centered_forcing=forcing,
        solution=solution,
        stationary_distribution=measure,
        stationary_mean=mean,
        residual_norm=float(np.max(np.abs(residual))),
        centering_residual=abs(float(np.dot(measure, solution))),
    )


def lag_covariance(
    chain: FiniteStateMarkovChain,
    observable: ArrayLike,
    lag: int,
    *,
    probabilities: ArrayLike | None = None,
) -> float:
    """Return ``Cov_pi(f(X_0), f(X_lag))`` exactly."""

    if isinstance(lag, bool) or not isinstance(lag, int):
        raise TypeError("lag must be an integer")
    if lag < 0:
        raise ValueError("lag must be nonnegative")
    measure = _stationary_measure(chain, probabilities)
    centered = centered_observable(measure, as_observable(observable, n_states=chain.n_states))
    propagated = chain.apply(centered, steps=lag)
    return weighted_inner_product(measure, centered, propagated)


def autocovariances(
    chain: FiniteStateMarkovChain,
    observable: ArrayLike,
    max_lag: int,
    *,
    probabilities: ArrayLike | None = None,
) -> Array:
    """Return exact stationary autocovariances from lag zero through ``max_lag``."""

    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise TypeError("max_lag must be an integer")
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    measure = _stationary_measure(chain, probabilities)
    centered = centered_observable(measure, as_observable(observable, n_states=chain.n_states))
    result = np.empty(max_lag + 1, dtype=np.float64)
    propagated = np.array(centered, dtype=np.float64, copy=True)
    for lag in range(max_lag + 1):
        result[lag] = weighted_inner_product(measure, centered, propagated)
        propagated = np.asarray(chain.transition @ propagated, dtype=np.float64)
    return result


def autocorrelations(
    chain: FiniteStateMarkovChain,
    observable: ArrayLike,
    max_lag: int,
    *,
    probabilities: ArrayLike | None = None,
) -> Array:
    """Return exact stationary autocorrelations through ``max_lag``."""

    covariances = autocovariances(
        chain,
        observable,
        max_lag,
        probabilities=probabilities,
    )
    if covariances[0] <= chain.tolerance:
        raise ValueError("autocorrelation is undefined for a constant observable")
    return np.asarray(covariances / covariances[0], dtype=np.float64)


def asymptotic_variance(
    chain: FiniteStateMarkovChain,
    observable: ArrayLike,
    *,
    probabilities: ArrayLike | None = None,
) -> float:
    """Return the exact CLT variance of the stationary sample mean.

    With ``L = P - I`` and ``Lu = f - pi[f]``, the formula is
    ``-2 <f-pi[f], u>_pi - Var_pi(f)``. It remains valid for irreducible
    periodic chains, where the naive infinite autocovariance series need not
    converge in the ordinary sense.
    """

    result = solve_poisson_equation(chain, observable, probabilities=probabilities)
    stationary_variance = variance(result.stationary_distribution, result.observable)
    value = (
        -2.0
        * weighted_inner_product(
            result.stationary_distribution,
            result.centered_forcing,
            result.solution,
        )
        - stationary_variance
    )
    if abs(value) <= 100.0 * chain.tolerance:
        return 0.0
    if value < 0.0:
        raise ArithmeticError("computed a negative asymptotic variance")
    return float(value)


def integrated_autocorrelation_time(
    chain: FiniteStateMarkovChain,
    observable: ArrayLike,
    *,
    probabilities: ArrayLike | None = None,
) -> float:
    """Return exact IAT as asymptotic variance divided by stationary variance."""

    measure = _stationary_measure(chain, probabilities)
    stationary_variance = variance(measure, observable)
    if stationary_variance <= chain.tolerance:
        raise ValueError("IAT is undefined for a constant observable")
    return asymptotic_variance(chain, observable, probabilities=measure) / stationary_variance


def finite_sample_mean_variance(
    chain: FiniteStateMarkovChain,
    observable: ArrayLike,
    n_samples: int,
    *,
    probabilities: ArrayLike | None = None,
) -> float:
    """Return the exact stationary variance of an ``n_samples`` time average."""

    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    covariances = autocovariances(
        chain,
        observable,
        n_samples - 1,
        probabilities=probabilities,
    )
    lags = np.arange(1, n_samples, dtype=np.float64)
    weighted_sum = float(np.dot(n_samples - lags, covariances[1:]))
    result = (n_samples * covariances[0] + 2.0 * weighted_sum) / (n_samples**2)
    if abs(result) <= 100.0 * chain.tolerance:
        return 0.0
    return float(result)


@dataclass(frozen=True, slots=True, init=False)
class MartingaleDecomposition:
    """Finite-path Poisson decomposition of a centered additive functional."""

    increments: Array
    centered_sum: float
    boundary_term: float
    martingale_term: float
    residual: float

    def __init__(
        self,
        *,
        increments: Array,
        centered_sum: float,
        boundary_term: float,
        martingale_term: float,
        residual: float,
    ) -> None:
        copied = np.array(increments, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "increments", copied)
        object.__setattr__(self, "centered_sum", float(centered_sum))
        object.__setattr__(self, "boundary_term", float(boundary_term))
        object.__setattr__(self, "martingale_term", float(martingale_term))
        object.__setattr__(self, "residual", float(residual))


def poisson_martingale_decomposition(
    chain: FiniteStateMarkovChain,
    observable: ArrayLike,
    states: ArrayLike,
    *,
    probabilities: ArrayLike | None = None,
) -> MartingaleDecomposition:
    """Decompose a path sum using the exact Poisson solution.

    If ``Lu = f - pi[f]`` and
    ``Delta M_k = u(X_{k+1}) - P u(X_k)``, then

    ``sum_{k=0}^{n-1}(f(X_k)-pi[f]) = u(X_n)-u(X_0)-M_n``.

    The returned residual measures only floating-point error; the identity is
    algebraic and does not rely on stationarity of the supplied trajectory.
    """

    state_array = np.asarray(states, dtype=np.int64)
    if state_array.ndim != 1 or state_array.size < 2:
        raise ValueError("states must contain at least two state indices")
    if np.any(state_array < 0) or np.any(state_array >= chain.n_states):
        raise IndexError("states contain an out-of-range index")

    result = solve_poisson_equation(chain, observable, probabilities=probabilities)
    propagated_solution = np.asarray(chain.transition @ result.solution, dtype=np.float64)
    increments = np.asarray(
        result.solution[state_array[1:]] - propagated_solution[state_array[:-1]],
        dtype=np.float64,
    )
    centered_sum = float(np.sum(result.centered_forcing[state_array[:-1]]))
    boundary = float(result.solution[state_array[-1]] - result.solution[state_array[0]])
    martingale = float(np.sum(increments))
    residual = centered_sum - (boundary - martingale)
    return MartingaleDecomposition(
        increments=increments,
        centered_sum=centered_sum,
        boundary_term=boundary,
        martingale_term=martingale,
        residual=residual,
    )
