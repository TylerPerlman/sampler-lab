import numpy as np
import pytest

from sampler_lab.particles import Ancestry


def test_ancestry_traces_lineages_with_variable_population_sizes() -> None:
    ancestry = Ancestry(
        (
            np.array([0, 0, 2, 3, 3]),
            np.array([1, 1, 3]),
        ),
        (4, 5, 3),
    )

    assert ancestry.trace_lineage(0) == pytest.approx([0, 1, 0])
    assert ancestry.trace_lineage(2) == pytest.approx([3, 3, 2])
    assert ancestry.final_to_initial() == pytest.approx([0, 0, 3])
    assert ancestry.unique_ancestor_counts() == pytest.approx([2, 2, 3])


def test_ancestry_rejects_invalid_parent_maps() -> None:
    with pytest.raises(IndexError):
        Ancestry((np.array([0, 2]),), (2, 2))
