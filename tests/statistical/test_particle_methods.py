import numpy as np
import pytest

from sampler_lab.models import count_self_avoiding_walks, sample_self_avoiding_walks

pytestmark = pytest.mark.statistical


@pytest.mark.parametrize("resampling", [None, "multinomial", "systematic", "bernoulli"])
def test_saw_normalizing_constant_estimate_matches_exact_small_count(
    resampling: str | None,
) -> None:
    n_steps = 8
    truth = count_self_avoiding_walks(n_steps)
    result = sample_self_avoiding_walks(
        np.random.default_rng(2022),
        n_steps=n_steps,
        n_particles=20_000,
        resampling=resampling,
        resample_every_step=resampling is not None,
    )
    assert result.normalizing_constant_estimate == pytest.approx(truth, rel=0.025)
