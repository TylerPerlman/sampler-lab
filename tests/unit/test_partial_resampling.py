import numpy as np
import pytest

from sampler_lab.markov import (
    compose_transitions,
    conditional_component_transition,
    conditional_resampling_transition,
    conjugate_by_permutation,
    mixture_transition,
)


def test_coordinate_resampling_preserves_joint_target() -> None:
    target = np.array([[0.4, 0.1], [0.2, 0.3]])
    flattened = target.ravel()
    first = conditional_resampling_transition(target, axis=0)
    second = conditional_resampling_transition(target, axis=1)

    assert first.is_invariant(flattened)
    assert second.is_invariant(flattened)
    assert not first.is_irreducible
    assert not second.is_irreducible


def test_random_scan_and_deterministic_sweep_preserve_target() -> None:
    target = np.array([[0.4, 0.1], [0.2, 0.3]])
    flattened = target.ravel()
    first = conditional_resampling_transition(target, axis=0)
    second = conditional_resampling_transition(target, axis=1)
    random_scan = mixture_transition([first, second], [0.5, 0.5])
    deterministic_sweep = compose_transitions([first, second])

    assert random_scan.is_invariant(flattened)
    assert random_scan.is_reversible(flattened)
    assert random_scan.is_ergodic
    assert deterministic_sweep.is_invariant(flattened)
    assert not deterministic_sweep.is_reversible(flattened)


def test_custom_conditional_kernel_is_checked_before_lifting() -> None:
    target = np.array([[0.4, 0.1], [0.2, 0.3]])
    # Identity kernels preserve every conditional distribution.
    identity_kernels = np.broadcast_to(np.eye(2), (2, 2, 2)).copy()
    lifted = conditional_component_transition(target, axis=0, conditional_kernels=identity_kernels)
    assert lifted.is_invariant(target.ravel())

    invalid = identity_kernels.copy()
    invalid[0] = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="conditional target"):
        conditional_component_transition(target, axis=0, conditional_kernels=invalid)


def test_permutation_conjugacy_preserves_spectrum_and_invariance() -> None:
    target = np.array([[0.4, 0.1], [0.2, 0.3]])
    chain = mixture_transition(
        [
            conditional_resampling_transition(target, axis=0),
            conditional_resampling_transition(target, axis=1),
        ]
    )
    permutation = np.array([2, 0, 3, 1])
    transformed = conjugate_by_permutation(chain, permutation)
    transformed_target = np.empty(4)
    transformed_target[permutation] = target.ravel()

    assert transformed.is_invariant(transformed_target)
    assert np.sort_complex(transformed.eigenvalues()) == pytest.approx(
        np.sort_complex(chain.eigenvalues())
    )
