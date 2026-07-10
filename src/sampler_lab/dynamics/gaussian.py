"""Exact stability, bias, and autocorrelation analysis for Gaussian ULA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.linalg import as_positive_definite
from sampler_lab.models.gaussian import GaussianTarget

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GaussianULAAnalysis:
    """Exact linear-Gaussian analysis of one constant-preconditioned ULA kernel."""

    step_size: float
    transition_matrix: Array
    noise_covariance: Array
    spectral_radius: float
    maximum_stable_step_size: float
    stable: bool
    stationary_covariance: Array | None
    covariance_bias: Array | None
    kl_stationary_to_target: float | None


def _as_vector(value: ArrayLike, dimension: int, *, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (dimension,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {dimension}")
    return array


def solve_discrete_lyapunov(transition: ArrayLike, noise_covariance: ArrayLike) -> Array:
    """Solve ``C = A C A^T + Q`` by vectorization using NumPy only."""

    matrix = np.asarray(transition, dtype=np.float64)
    noise = np.asarray(noise_covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be square")
    if noise.shape != matrix.shape:
        raise ValueError("noise_covariance must match transition")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(noise)):
        raise ValueError("Lyapunov inputs must be finite")
    dimension = matrix.shape[0]
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if spectral_radius >= 1.0:
        raise ValueError("transition must be stable to have a stationary covariance")
    system = np.eye(dimension * dimension) - np.kron(matrix, matrix)
    vectorized_noise = noise.reshape(dimension * dimension, order="F")
    vectorized_covariance = np.linalg.solve(system, vectorized_noise)
    covariance = vectorized_covariance.reshape((dimension, dimension), order="F")
    return np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64)


def gaussian_covariance_kl(
    approximate_covariance: ArrayLike,
    target_covariance: ArrayLike,
) -> float:
    """KL divergence between equal-mean Gaussians in the stated direction."""

    approximate = as_positive_definite(approximate_covariance)
    target = as_positive_definite(target_covariance)
    if approximate.shape != target.shape:
        raise ValueError("covariances must have the same shape")
    dimension = approximate.shape[0]
    trace_term = float(np.trace(np.linalg.solve(target, approximate)))
    sign_approximate, logdet_approximate = np.linalg.slogdet(approximate)
    sign_target, logdet_target = np.linalg.slogdet(target)
    if sign_approximate <= 0.0 or sign_target <= 0.0:
        raise ValueError("covariances must have positive determinant")
    return float(0.5 * (trace_term - dimension + float(logdet_target) - float(logdet_approximate)))


def gaussian_ula_analysis(
    target: GaussianTarget,
    step_size: float,
    *,
    preconditioner: ArrayLike | None = None,
) -> GaussianULAAnalysis:
    """Return exact constant-preconditioned ULA stability and invariant bias."""

    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be positive and finite")
    dimension = target.dimension
    matrix = (
        np.eye(dimension, dtype=np.float64)
        if preconditioner is None
        else as_positive_definite(preconditioner)
    )
    if matrix.shape != (dimension, dimension):
        raise ValueError("preconditioner dimension must match the target")
    precision = target.precision_matrix
    drift_spectrum = np.linalg.eigvals(matrix @ precision)
    if np.max(np.abs(np.imag(drift_spectrum))) > 1e-9:
        raise ValueError("preconditioned Gaussian drift unexpectedly has complex spectrum")
    positive_rates = np.real(drift_spectrum)
    if np.any(positive_rates <= 0.0):
        raise ValueError("preconditioned Gaussian drift rates must be positive")
    maximum_stable_step_size = float(2.0 / np.max(positive_rates))
    transition = np.eye(dimension, dtype=np.float64) - step_size * matrix @ precision
    noise = 2.0 * step_size * matrix
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    stable = bool(spectral_radius < 1.0)
    stationary = solve_discrete_lyapunov(transition, noise) if stable else None
    covariance_bias = None if stationary is None else stationary - target.covariance_matrix
    kl = (
        None if stationary is None else gaussian_covariance_kl(stationary, target.covariance_matrix)
    )
    return GaussianULAAnalysis(
        step_size=float(step_size),
        transition_matrix=np.asarray(transition, dtype=np.float64),
        noise_covariance=np.asarray(noise, dtype=np.float64),
        spectral_radius=spectral_radius,
        maximum_stable_step_size=maximum_stable_step_size,
        stable=stable,
        stationary_covariance=None
        if stationary is None
        else np.asarray(stationary, dtype=np.float64),
        covariance_bias=None
        if covariance_bias is None
        else np.asarray(covariance_bias, dtype=np.float64),
        kl_stationary_to_target=kl,
    )


def linear_gaussian_iat(
    transition: ArrayLike,
    stationary_covariance: ArrayLike,
    observable_vector: ArrayLike,
) -> float:
    """Exact IAT of a centered linear observable for a stable Gaussian AR(1)."""

    matrix = np.asarray(transition, dtype=np.float64)
    covariance = as_positive_definite(stationary_covariance)
    if matrix.shape != covariance.shape:
        raise ValueError("transition and stationary_covariance must have the same shape")
    vector = _as_vector(observable_vector, matrix.shape[0], name="observable_vector")
    if float(np.max(np.abs(np.linalg.eigvals(matrix)))) >= 1.0:
        raise ValueError("transition must be stable")
    variance = float(vector @ covariance @ vector)
    if variance <= 0.0:
        raise ValueError("observable must have positive stationary variance")
    transpose = matrix.T
    lag_sum = (
        covariance
        @ transpose
        @ np.linalg.solve(
            np.eye(matrix.shape[0]) - transpose,
            vector,
        )
    )
    asymptotic_variance = variance + 2.0 * float(vector @ lag_sum)
    iat = asymptotic_variance / variance
    if iat < -1e-10:
        raise ArithmeticError("computed a negative asymptotic variance")
    return float(max(iat, 0.0))


def gaussian_quadratic_expectation(
    mean: ArrayLike,
    covariance: ArrayLike,
    *,
    quadratic: ArrayLike | None = None,
    linear: ArrayLike | None = None,
    constant: float = 0.0,
) -> float:
    """Evaluate ``E[X^T B X + c^T X + constant]`` exactly."""

    covariance_matrix = as_positive_definite(covariance)
    dimension = covariance_matrix.shape[0]
    mean_vector = _as_vector(mean, dimension, name="mean")
    quadratic_matrix = (
        np.zeros_like(covariance_matrix)
        if quadratic is None
        else np.asarray(quadratic, dtype=np.float64)
    )
    if quadratic_matrix.shape != covariance_matrix.shape or not np.all(
        np.isfinite(quadratic_matrix)
    ):
        raise ValueError("quadratic must be a finite square matrix")
    linear_vector = (
        np.zeros(dimension, dtype=np.float64)
        if linear is None
        else _as_vector(linear, dimension, name="linear")
    )
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    return float(
        np.trace(quadratic_matrix @ covariance_matrix)
        + mean_vector @ quadratic_matrix @ mean_vector
        + linear_vector @ mean_vector
        + constant
    )
