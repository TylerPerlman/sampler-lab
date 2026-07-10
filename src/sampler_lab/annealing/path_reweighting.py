"""Reweight annealing populations between path locations."""

from __future__ import annotations

from sampler_lab.annealing.paths import AnnealingPath, incremental_log_weights
from sampler_lab.particles.cloud import ParticleCloud


def reweight_cloud(
    cloud: ParticleCloud,
    path: AnnealingPath,
    beta_from: float,
    beta_to: float,
) -> ParticleCloud:
    """Reweight a cloud from one path law toward a later path law."""

    increments = incremental_log_weights(path, cloud.particles, beta_from, beta_to)
    return ParticleCloud(cloud.particles, cloud.log_weights + increments)
