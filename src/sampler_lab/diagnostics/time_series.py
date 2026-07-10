"""Empirical autocorrelation and effective-sample-size diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def empirical_autocovariances(values: ArrayLike, *, max_lag: int | None = None) -> Array:
    """Estimate the biased autocovariance sequence using an FFT.

    The denominator is the full sample size at every lag. This choice yields a
    positive-semidefinite empirical covariance sequence and is appropriate for
    the initial-positive-sequence IAT estimator below.
    """

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("values must be a one-dimensional sequence of length at least two")
    if not np.all(np.isfinite(samples)):
        raise ValueError("values must be finite")
    n_samples = int(samples.size)
    if max_lag is None:
        lag_limit = n_samples - 1
    else:
        if isinstance(max_lag, bool) or not isinstance(max_lag, int):
            raise TypeError("max_lag must be an integer")
        if max_lag < 0 or max_lag >= n_samples:
            raise ValueError("max_lag must lie between zero and n_samples - 1")
        lag_limit = max_lag

    centered = samples - np.mean(samples)
    fft_size = 1 << (2 * n_samples - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size)
    raw = np.fft.irfft(transformed * np.conjugate(transformed), n=fft_size)
    return np.asarray(raw[: lag_limit + 1] / n_samples, dtype=np.float64)


def empirical_autocorrelations(values: ArrayLike, *, max_lag: int | None = None) -> Array:
    """Estimate autocorrelations, returning one at lag zero."""

    autocovariances = empirical_autocovariances(values, max_lag=max_lag)
    variance = float(autocovariances[0])
    if variance <= 0.0:
        raise ValueError("autocorrelations are undefined for a constant sequence")
    correlations = np.asarray(autocovariances / variance, dtype=np.float64)
    correlations[0] = 1.0
    return correlations


def empirical_integrated_autocorrelation_time(
    values: ArrayLike,
    *,
    max_lag: int | None = None,
) -> float:
    """Estimate IAT with Geyer's initial-positive paired sequence.

    Adjacent lag pairs ``rho[2k-1] + rho[2k]`` are accumulated until the first
    nonpositive pair. The result is clipped below at one, matching the usual ESS
    convention even when antithetic behavior would imply an IAT below one.
    """

    correlations = empirical_autocorrelations(values, max_lag=max_lag)
    total = 0.0
    lag = 1
    while lag + 1 < correlations.size:
        pair_sum = float(correlations[lag] + correlations[lag + 1])
        if pair_sum <= 0.0:
            break
        total += pair_sum
        lag += 2
    return max(1.0, float(1.0 + 2.0 * total))


def empirical_effective_sample_size(
    values: ArrayLike,
    *,
    max_lag: int | None = None,
) -> float:
    """Return ``N / estimated IAT`` for a scalar time series."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("values must be one-dimensional")
    iat = empirical_integrated_autocorrelation_time(samples, max_lag=max_lag)
    return float(samples.size / iat)
