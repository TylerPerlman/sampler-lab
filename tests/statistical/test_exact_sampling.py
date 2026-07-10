import numpy as np
import pytest

from sampler_lab.core.counters import OperationCounter
from sampler_lab.exact import box_muller, generalized_inverse_discrete
from sampler_lab.models import (
    sample_unit_disk_direct,
    sample_unit_disk_rejection,
    unit_disk_radius_squared,
)

pytestmark = pytest.mark.statistical


def test_box_muller_standard_normal_moments() -> None:
    counter = OperationCounter()
    samples = box_muller(np.random.default_rng(2022), 200_001, counter=counter)
    assert abs(float(np.mean(samples))) < 0.01
    assert abs(float(np.var(samples)) - 1.0) < 0.015
    assert counter.normal_draws == 200_001
    assert counter.uniform_draws == 200_002


def test_discrete_generalized_inverse_probabilities() -> None:
    samples = generalized_inverse_discrete(
        np.random.default_rng(12),
        values=[0.0, 1.0, 2.0],
        probabilities=[0.2, 0.3, 0.5],
        size=100_000,
    )
    frequencies = np.bincount(samples.astype(np.int64), minlength=3) / samples.size
    np.testing.assert_allclose(frequencies, [0.2, 0.3, 0.5], atol=0.005)


def test_direct_unit_disk_radial_law() -> None:
    points = sample_unit_disk_direct(np.random.default_rng(99), 100_000)
    radius_squared = unit_disk_radius_squared(points)
    assert float(np.max(radius_squared)) <= 1.0 + 1e-14
    assert float(np.mean(radius_squared)) == pytest.approx(0.5, abs=0.004)
    np.testing.assert_allclose(np.mean(points, axis=0), [0.0, 0.0], atol=0.006)


def test_rejection_unit_disk_acceptance_and_radial_law() -> None:
    result = sample_unit_disk_rejection(np.random.default_rng(100), 100_000)
    radius_squared = unit_disk_radius_squared(result.samples)
    assert float(np.max(radius_squared)) <= 1.0 + 1e-14
    assert float(np.mean(radius_squared)) == pytest.approx(0.5, abs=0.004)
    assert result.acceptance_rate == pytest.approx(np.pi / 4.0, abs=0.006)
