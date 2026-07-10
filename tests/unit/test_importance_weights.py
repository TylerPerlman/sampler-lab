import numpy as np
import pytest

from sampler_lab.core.counters import OperationCounter
from sampler_lab.importance import log_importance_weights


def test_log_importance_weights_and_operation_counts() -> None:
    samples = np.array([-1.0, 0.0, 2.0])
    counter = OperationCounter()
    result = log_importance_weights(
        samples,
        log_target=lambda x: -float(x**2),
        log_proposal=lambda x: -0.5 * float(x**2),
        counter=counter,
    )
    np.testing.assert_allclose(result, [-0.5, 0.0, -2.0])
    assert counter.log_density_evaluations == 3
    assert counter.proposal_density_evaluations == 3


def test_log_importance_weights_detects_support_violation() -> None:
    with pytest.raises(ValueError, match="support"):
        log_importance_weights(
            [0.0],
            log_target=lambda x: 0.0,
            log_proposal=lambda x: -np.inf,
        )
