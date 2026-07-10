from dataclasses import dataclass

import numpy as np
import pytest

from sampler_lab.annealing import (
    AnnealingSchedule,
    FunctionalAnnealingPath,
    GeometricAnnealingPath,
    evaluate_path,
    incremental_log_weights,
)


@dataclass(frozen=True)
class HalfLineDensity:
    positive: bool

    def log_prob(self, x: np.ndarray) -> float:
        supported = bool(x[0] >= 0.0) if self.positive else bool(x[0] <= 0.0)
        return 0.0 if supported else float("-inf")


def test_linear_and_power_schedules_include_endpoints() -> None:
    linear = AnnealingSchedule.linear(4)
    power = AnnealingSchedule.power(4, 2.0)

    assert linear.values == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert linear.increments == pytest.approx(np.full(4, 0.25))
    assert linear.n_steps == 4
    assert power.values == pytest.approx([0.0, 0.0625, 0.25, 0.5625, 1.0])
    assert not linear.values.flags.writeable


@pytest.mark.parametrize(
    "values",
    [
        [0.1, 1.0],
        [0.0, 0.5],
        [0.0, 0.5, 0.5, 1.0],
        [0.0, 0.8, 0.7, 1.0],
        [0.0, np.nan, 1.0],
    ],
)
def test_invalid_custom_schedules_are_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError):
        AnnealingSchedule(values)


def test_geometric_path_handles_zero_times_negative_infinity_at_endpoints() -> None:
    path = GeometricAnnealingPath(HalfLineDensity(False), HalfLineDensity(True))
    positive = np.array([1.0])

    assert path.log_unnormalized(positive, 0.0) == float("-inf")
    assert path.log_unnormalized(positive, 1.0) == 0.0
    assert path.log_unnormalized(positive, 0.5) == float("-inf")


def test_path_evaluation_and_incremental_weights_use_particle_axis() -> None:
    path = FunctionalAnnealingPath(lambda x, beta: beta * float(x[0]))
    particles = np.array([[1.0], [2.0], [-3.0]])

    assert evaluate_path(path, particles, 0.25) == pytest.approx([0.25, 0.5, -0.75])
    assert incremental_log_weights(path, particles, 0.25, 0.75) == pytest.approx([0.5, 1.0, -1.5])


def test_undefined_infinite_path_increment_is_rejected() -> None:
    path = GeometricAnnealingPath(HalfLineDensity(False), HalfLineDensity(True))
    with pytest.raises(ValueError, match="infinite positive"):
        incremental_log_weights(path, np.array([[1.0]]), 0.0, 1.0)
