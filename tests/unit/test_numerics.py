import numpy as np
import pytest

from sampler_lab.core.numerics import logsumexp, normalize_log_weights, validate_size


def test_logsumexp_avoids_overflow() -> None:
    values = np.array([1_000.0, 1_000.0])
    assert logsumexp(values) == pytest.approx(1_000.0 + np.log(2.0))


def test_logsumexp_all_negative_infinity() -> None:
    assert logsumexp([-np.inf, -np.inf]) == -np.inf


def test_normalize_log_weights() -> None:
    weights, log_normalizer = normalize_log_weights(np.log([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(weights, [1 / 6, 2 / 6, 3 / 6])
    assert log_normalizer == pytest.approx(np.log(6.0))


def test_normalize_log_weights_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        normalize_log_weights([-np.inf, -np.inf])


def test_validate_size_rejects_boolean() -> None:
    with pytest.raises(TypeError):
        validate_size(True)
