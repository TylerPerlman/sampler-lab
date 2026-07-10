import numpy as np
import pytest

from sampler_lab.exact import rejection_sample


def test_rejection_sampler_detects_invalid_envelope() -> None:
    def proposal(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.random(size)

    with pytest.raises(ValueError, match="envelope"):
        rejection_sample(
            np.random.default_rng(3),
            size=1,
            proposal_sampler=proposal,
            log_target=lambda x: 1.0,
            log_proposal=lambda x: 0.0,
            log_envelope=0.0,
            batch_size=1,
        )


def test_rejection_sampler_respects_max_proposals() -> None:
    def proposal(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.random(size)

    with pytest.raises(RuntimeError, match="max_proposals"):
        rejection_sample(
            np.random.default_rng(5),
            size=2,
            proposal_sampler=proposal,
            log_target=lambda x: -np.inf,
            log_proposal=lambda x: 0.0,
            log_envelope=0.0,
            batch_size=2,
            max_proposals=2,
        )


def test_empty_rejection_request_is_well_defined() -> None:
    result = rejection_sample(
        np.random.default_rng(1),
        size=0,
        proposal_sampler=lambda rng, size: rng.random(size),
        log_target=lambda x: 0.0,
        log_proposal=lambda x: 0.0,
        log_envelope=0.0,
    )
    assert result.samples.size == 0
    assert np.isnan(result.acceptance_rate)
    assert np.isnan(result.proposals_per_sample)
