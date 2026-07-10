import numpy as np
import pytest

from sampler_lab.particles import ParticleCloud


def test_particle_cloud_normalizes_log_weights_and_is_shift_invariant() -> None:
    particles = np.array([[1.0], [3.0], [5.0]])
    first = ParticleCloud(particles, np.log([1.0, 2.0, 1.0]))
    second = ParticleCloud(particles, np.log([1.0, 2.0, 1.0]) + 900.0)

    assert first.weights == pytest.approx([0.25, 0.5, 0.25])
    assert second.weights == pytest.approx(first.weights)
    assert np.exp(np.logaddexp.reduce(first.log_weights)) == pytest.approx(1.0)
    assert first.effective_sample_size == pytest.approx(8.0 / 3.0)


def test_particle_cloud_copies_inputs_and_marks_snapshots_read_only() -> None:
    particles = np.array([[1.0], [2.0]])
    cloud = ParticleCloud.uniform(particles)
    particles[0, 0] = 99.0

    assert cloud.particles[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cloud.particles[0, 0] = 7.0
    with pytest.raises(ValueError):
        cloud.log_weights[0] = -1.0


def test_particle_cloud_expectation_supports_scalar_and_vector_observables() -> None:
    cloud = ParticleCloud(np.array([[1.0], [3.0]]), np.log([1.0, 3.0]))

    scalar = cloud.expectation(lambda x: x[:, 0] ** 2)
    vector = cloud.expectation(lambda x: np.column_stack((x[:, 0], x[:, 0] ** 2)))

    assert scalar == pytest.approx(7.0)
    assert vector == pytest.approx([2.5, 7.0])


def test_particle_cloud_selection_resets_weights() -> None:
    cloud = ParticleCloud(np.arange(3.0)[:, None], np.log([0.1, 0.2, 0.7]))
    selected = cloud.select([2, 2, 1, 2])

    assert selected.particles[:, 0] == pytest.approx([2.0, 2.0, 1.0, 2.0])
    assert selected.weights == pytest.approx(np.full(4, 0.25))
