import numpy as np
import pytest

from sampler_lab.ensemble import (
    EnsembleState,
    StretchMoveKernel,
    WalkMoveKernel,
    run_ensemble_chain,
    stretch_log_density,
    stretch_symmetry_error,
)
from sampler_lab.geometry import AffineMap, AffineTransformedTarget, affine_equivariance_error
from sampler_lab.models import GaussianTarget


class FlatTarget:
    def log_prob(self, x: np.ndarray) -> float:
        return 0.0


def test_stretch_density_obeys_required_reciprocal_symmetry() -> None:
    for z in (0.55, 0.8, 1.0, 1.4, 1.9):
        assert stretch_symmetry_error(z, 2.0) < 1e-12
    assert stretch_log_density(0.49, 2.0) == float("-inf")


def test_one_dimensional_flat_stretch_move_accepts_every_walker() -> None:
    state = EnsembleState([[-2.0], [-0.5], [1.0], [3.0]], [0.0, 0.0, 0.0, 0.0])
    transition = StretchMoveKernel(
        FlatTarget(),
        require_full_affine_span=True,
    ).step(state, np.random.default_rng(5))

    assert np.all(transition.accepted)
    np.testing.assert_allclose(transition.log_acceptance_ratios, np.zeros(4))
    assert transition.state.walkers.shape == state.walkers.shape


def test_stretch_move_is_pathwise_affine_equivariant_under_coupled_randomness() -> None:
    target = GaussianTarget([0.5, -1.0], [[2.0, 0.3], [0.3, 0.7]])
    mapping = AffineMap([[1.7, -0.2], [0.6, 1.3]], [2.0, -3.0])
    transformed_target = AffineTransformedTarget(target, mapping)
    walkers = np.random.default_rng(1).normal(size=(10, 2))
    original_state = EnsembleState.from_target(walkers, target)
    transformed_state = EnsembleState.from_target(mapping.forward(walkers), transformed_target)
    original_kernel = StretchMoveKernel(target, schedule="split")
    transformed_kernel = StretchMoveKernel(transformed_target, schedule="split")

    original = original_kernel.step(original_state, np.random.default_rng(123))
    transformed = transformed_kernel.step(transformed_state, np.random.default_rng(123))

    np.testing.assert_array_equal(original.accepted, transformed.accepted)
    np.testing.assert_array_equal(original.partner_indices, transformed.partner_indices)
    assert (
        affine_equivariance_error(
            original.state.walkers,
            transformed.state.walkers,
            mapping,
        )
        < 2e-12
    )


def test_walk_move_is_pathwise_affine_equivariant_under_coupled_randomness() -> None:
    target = GaussianTarget([0.0, 0.0], [[1.0, 0.2], [0.2, 2.0]])
    mapping = AffineMap([[2.0, 0.5], [-0.4, 1.1]], [-1.0, 2.5])
    transformed_target = AffineTransformedTarget(target, mapping)
    walkers = np.random.default_rng(9).normal(size=(12, 2))
    original_state = EnsembleState.from_target(walkers, target)
    transformed_state = EnsembleState.from_target(mapping.forward(walkers), transformed_target)
    original_kernel = WalkMoveKernel(target, schedule="split", subset_size=4)
    transformed_kernel = WalkMoveKernel(
        transformed_target,
        schedule="split",
        subset_size=4,
    )

    original = original_kernel.step(original_state, np.random.default_rng(77))
    transformed = transformed_kernel.step(transformed_state, np.random.default_rng(77))

    np.testing.assert_array_equal(original.accepted, transformed.accepted)
    assert (
        affine_equivariance_error(
            original.state.walkers,
            transformed.state.walkers,
            mapping,
        )
        < 3e-12
    )


def test_moves_reject_degenerate_affine_span_by_default() -> None:
    target = GaussianTarget([0.0, 0.0], np.eye(2))
    state = EnsembleState.from_target([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], target)

    with pytest.raises(ValueError, match="span"):
        StretchMoveKernel(target).step(state, np.random.default_rng(1))
    with pytest.raises(ValueError, match="span"):
        WalkMoveKernel(target).step(state, np.random.default_rng(1))


def test_ensemble_runner_retains_complete_product_states() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    state = EnsembleState.from_target([[-2.0], [-0.5], [0.5], [2.0]], target)
    trajectory = run_ensemble_chain(
        StretchMoveKernel(target),
        state,
        np.random.default_rng(3),
        n_steps=5,
    )

    assert trajectory.walkers.shape == (6, 4, 1)
    assert trajectory.accepted.shape == (5, 4)
    assert trajectory.samples(discard=1, flatten=True).shape == (20, 1)
    assert trajectory.per_walker_acceptance.shape == (4,)
