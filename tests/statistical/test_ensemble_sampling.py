import numpy as np
import pytest

from sampler_lab.ensemble import (
    EnsembleKernel,
    EnsembleState,
    StretchMoveKernel,
    WalkMoveKernel,
    run_ensemble_chain,
)
from sampler_lab.models import GaussianTarget


@pytest.mark.statistical
@pytest.mark.parametrize("kind", ["stretch", "walk"])
def test_ensemble_moves_recover_correlated_gaussian_moments(kind: str) -> None:
    covariance = np.array([[1.0, 0.75], [0.75, 2.0]])
    target = GaussianTarget([0.5, -1.0], covariance)
    rng = np.random.default_rng(2022)
    initial = rng.normal(size=(20, 2)) * 3.0
    state = EnsembleState.from_target(initial, target)
    kernel: EnsembleKernel
    if kind == "stretch":
        kernel = StretchMoveKernel(target, schedule="split")
    else:
        kernel = WalkMoveKernel(target, schedule="split", subset_size=6, scale=0.8)
    trajectory = run_ensemble_chain(kernel, state, rng, n_steps=1800)
    samples = trajectory.samples(discard=300, flatten=True)

    np.testing.assert_allclose(np.mean(samples, axis=0), target.mean_vector, atol=0.09)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), covariance, atol=0.14)
    assert 0.1 < trajectory.acceptance_rate < 0.9
