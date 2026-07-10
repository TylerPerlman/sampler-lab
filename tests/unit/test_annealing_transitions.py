from dataclasses import dataclass

import numpy as np
import pytest

from sampler_lab.annealing import (
    FunctionalPopulationTransition,
    KernelPopulationTransition,
)
from sampler_lab.core.results import Transition


@dataclass
class AddBetaKernel:
    beta: float

    def step(self, state: np.ndarray, rng: np.random.Generator) -> Transition:
        del rng
        return Transition(state=state + self.beta)


def test_kernel_population_transition_applies_requested_number_of_steps() -> None:
    transition = KernelPopulationTransition(lambda beta: AddBetaKernel(beta), n_steps=3)
    particles = np.array([[0.0], [2.0]])

    moved = transition.move(particles, 0.25, np.random.default_rng(1))
    np.testing.assert_allclose(moved, [[0.75], [2.75]])
    np.testing.assert_allclose(particles, [[0.0], [2.0]])


def test_functional_population_transition_rejects_shape_changes() -> None:
    transition = FunctionalPopulationTransition(lambda particles, beta, rng: particles[:, 0])
    with pytest.raises(ValueError, match="shape"):
        transition.move(np.zeros((3, 1)), 0.5, np.random.default_rng(1))
