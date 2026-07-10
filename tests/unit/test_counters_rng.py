import numpy as np
import pytest

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.rng import spawn_rngs


def test_counter_increment_and_merge() -> None:
    first = OperationCounter(uniform_draws=2)
    first.increment("log_density_evaluations", 3)
    first.increment("custom", 4)
    second = OperationCounter(uniform_draws=5, normal_draws=7, extra={"custom": 1})
    first.merge(second)
    assert first.uniform_draws == 7
    assert first.normal_draws == 7
    assert first.log_density_evaluations == 3
    assert first.extra["custom"] == 5


def test_counter_rejects_negative_increment() -> None:
    with pytest.raises(ValueError):
        OperationCounter().increment("uniform_draws", -1)


def test_spawn_rngs_is_reproducible_and_distinct() -> None:
    first = spawn_rngs(1234, 2)
    second = spawn_rngs(1234, 2)
    np.testing.assert_array_equal(first[0].integers(0, 2**32, 10), second[0].integers(0, 2**32, 10))
    stream_a, stream_b = spawn_rngs(1234, 2)
    assert not np.array_equal(stream_a.integers(0, 2**32, 10), stream_b.integers(0, 2**32, 10))
