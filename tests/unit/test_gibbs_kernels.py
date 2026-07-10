import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.mcmc import (
    DeterministicSweepGibbsKernel,
    FunctionalConditionalUpdate,
    RandomScanGibbsKernel,
    TransformedGibbsKernel,
)


def _set_coordinate(index: int, value: float) -> FunctionalConditionalUpdate:
    def update(
        state: np.ndarray,
        rng: np.random.Generator,
        counter: OperationCounter | None,
    ) -> np.ndarray:
        result = np.array(state, copy=True)
        result[index] = value
        if counter is not None:
            counter.conditional_draws += 1
        return result

    return FunctionalConditionalUpdate(update)


def test_deterministic_sweep_applies_updates_in_order() -> None:
    first = _set_coordinate(0, 2.0)

    def copy_first(
        state: np.ndarray,
        rng: np.random.Generator,
        counter: OperationCounter | None,
    ) -> np.ndarray:
        result = np.array(state, copy=True)
        result[1] = result[0]
        return result

    kernel = DeterministicSweepGibbsKernel([first, FunctionalConditionalUpdate(copy_first)])
    transition = kernel.step(np.zeros(2), np.random.default_rng(1))

    assert transition.state == pytest.approx([2.0, 2.0])
    assert transition.diagnostics["n_updates"] == 2.0


def test_random_scan_respects_degenerate_scan_probabilities() -> None:
    counter = OperationCounter()
    kernel = RandomScanGibbsKernel(
        [_set_coordinate(0, 1.0), _set_coordinate(1, 1.0)],
        probabilities=[1.0, 0.0],
        counter=counter,
    )
    transition = kernel.step(np.zeros(2), np.random.default_rng(2))

    assert transition.state == pytest.approx([1.0, 0.0])
    assert transition.diagnostics["update_index"] == 0.0
    assert counter.uniform_draws == 1
    assert counter.conditional_draws == 1


def test_transformed_gibbs_conjugates_inner_kernel() -> None:
    inner = DeterministicSweepGibbsKernel([_set_coordinate(0, 3.0)])
    kernel = TransformedGibbsKernel(
        inner,
        forward=lambda x: 2.0 * x,
        inverse=lambda z: 0.5 * z,
    )
    transition = kernel.step(np.array([5.0]), np.random.default_rng(1))

    assert transition.state == pytest.approx([1.5])
    assert transition.diagnostics["transformed"] == 1.0
