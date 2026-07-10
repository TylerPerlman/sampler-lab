"""Console entry points for reproducible repository experiments."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import MarkovKernel
from sampler_lab.core.rng import spawn_rngs
from sampler_lab.importance import (
    gaussian_tail_experiment,
    gaussian_weight_collapse_experiment,
)
from sampler_lab.models.disk import (
    sample_unit_disk_direct,
    sample_unit_disk_rejection,
    unit_disk_radius_squared,
)


@dataclass(frozen=True, slots=True)
class DiskBenchmarkRow:
    method: str
    samples: int
    mean_seconds: float
    samples_per_second: float
    mean_radius_squared: float
    max_radius_squared: float
    acceptance_rate: float
    uniforms_per_sample: float
    proposals_per_sample: float


def run_disk_benchmark(samples: int, repeats: int, seed: int) -> list[DiskBenchmarkRow]:
    """Benchmark direct polar and square-rejection unit-disk samplers."""

    if samples <= 0 or repeats <= 0:
        raise ValueError("samples and repeats must be positive")
    streams = iter(spawn_rngs(seed, 2 * repeats))
    direct_times: list[float] = []
    rejection_times: list[float] = []
    direct_points: np.ndarray | None = None
    rejection_result = None
    direct_uniforms = 0
    rejection_uniforms = 0

    for _ in range(repeats):
        direct_counter = OperationCounter()
        start = time.perf_counter()
        direct_points = sample_unit_disk_direct(next(streams), samples, counter=direct_counter)
        direct_times.append(time.perf_counter() - start)
        direct_uniforms += direct_counter.uniform_draws

        rejection_counter = OperationCounter()
        start = time.perf_counter()
        rejection_result = sample_unit_disk_rejection(
            next(streams), samples, counter=rejection_counter
        )
        rejection_times.append(time.perf_counter() - start)
        rejection_uniforms += rejection_counter.uniform_draws

    assert direct_points is not None
    assert rejection_result is not None
    direct_r2 = unit_disk_radius_squared(direct_points)
    rejection_r2 = unit_disk_radius_squared(rejection_result.samples)
    direct_seconds = float(np.mean(direct_times))
    rejection_seconds = float(np.mean(rejection_times))

    return [
        DiskBenchmarkRow(
            method="direct-polar",
            samples=samples,
            mean_seconds=direct_seconds,
            samples_per_second=samples / direct_seconds,
            mean_radius_squared=float(np.mean(direct_r2)),
            max_radius_squared=float(np.max(direct_r2)),
            acceptance_rate=1.0,
            uniforms_per_sample=direct_uniforms / (repeats * samples),
            proposals_per_sample=1.0,
        ),
        DiskBenchmarkRow(
            method="square-rejection",
            samples=samples,
            mean_seconds=rejection_seconds,
            samples_per_second=samples / rejection_seconds,
            mean_radius_squared=float(np.mean(rejection_r2)),
            max_radius_squared=float(np.max(rejection_r2)),
            acceptance_rate=rejection_result.acceptance_rate,
            uniforms_per_sample=rejection_uniforms / (repeats * samples),
            proposals_per_sample=rejection_result.proposals_per_sample,
        ),
    ]


def _print_disk_rows(rows: list[DiskBenchmarkRow]) -> None:
    headers = (
        "method",
        "seconds",
        "samples/s",
        "E[R^2]",
        "max R^2",
        "acceptance",
        "uniforms/sample",
        "proposals/sample",
    )
    print("  ".join(f"{header:>17}" for header in headers))
    for row in rows:
        print(
            f"{row.method:>17}  {row.mean_seconds:17.6f}  "
            f"{row.samples_per_second:17.1f}  {row.mean_radius_squared:17.6f}  "
            f"{row.max_radius_squared:17.6f}  {row.acceptance_rate:17.6f}  "
            f"{row.uniforms_per_sample:17.4f}  {row.proposals_per_sample:17.4f}"
        )


def disk_benchmark_main() -> None:
    """CLI entry point for the unit-disk exact-sampling benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    rows = run_disk_benchmark(args.samples, args.repeats, args.seed)
    _print_disk_rows(rows)


def importance_sampling_main() -> None:
    """Run Gaussian-tail and product-weight-collapse demonstrations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--proposal-means", type=float, nargs="+", default=[0.0, 2.0, 4.0])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--proposal-scale", type=float, default=1.25)
    args = parser.parse_args()

    tail_rng, collapse_rng = spawn_rngs(args.seed, 2)
    tail_rows = gaussian_tail_experiment(
        tail_rng,
        threshold=args.threshold,
        n_samples=args.samples,
        proposal_means=args.proposal_means,
    )
    print("Gaussian upper-tail estimation")
    print(
        f"{'proposal mean':>14}  {'estimate':>12}  {'truth':>12}  "
        f"{'std. error':>12}  {'rel. SE':>10}  {'events':>9}  {'weight ESS':>12}"
    )
    for tail_row in tail_rows:
        print(
            f"{tail_row.proposal_mean:14.3f}  {tail_row.estimate:12.6g}  "
            f"{tail_row.truth:12.6g}  {tail_row.standard_error:12.6g}  "
            f"{tail_row.relative_standard_error:10.4f}  {tail_row.event_count:9d}  "
            f"{tail_row.effective_sample_size:12.1f}"
        )

    collapse_rows = gaussian_weight_collapse_experiment(
        collapse_rng,
        dimensions=args.dimensions,
        n_samples=args.samples,
        proposal_scale=args.proposal_scale,
    )
    print("\nGaussian product-target weight collapse")
    print(
        f"{'dimension':>10}  {'ESS/N':>10}  {'max weight':>12}  "
        f"{'empirical chi^2':>16}  {'exact chi^2':>14}  {'asymptotic ESS/N':>18}"
    )
    for collapse_row in collapse_rows:
        print(
            f"{collapse_row.dimension:10d}  {collapse_row.ess_fraction:10.5f}  "
            f"{collapse_row.max_normalized_weight:12.6g}  "
            f"{collapse_row.empirical_chi_squared:16.6g}  "
            f"{collapse_row.theoretical_chi_squared:14.6g}  "
            f"{collapse_row.asymptotic_ess_fraction:18.6g}"
        )


@dataclass(frozen=True, slots=True)
class SelfAvoidingWalkExperimentRow:
    """One self-avoiding-walk particle-method comparison row."""

    method: str
    estimated_count: float
    exact_count: int | None
    relative_error: float | None
    final_population: int
    positive_final_weights: int
    resampling_steps: int
    minimum_ess_fraction: float
    unique_initial_ancestors: int


def run_self_avoiding_walk_experiment(
    *,
    n_steps: int,
    n_particles: int,
    seed: int,
    methods: tuple[str, ...] = (
        "sis",
        "multinomial",
        "systematic",
        "bernoulli",
        "ess-systematic",
    ),
    exact_limit: int = 12,
    ess_threshold: float = 0.85,
) -> list[SelfAvoidingWalkExperimentRow]:
    """Compare SIS and resampled particle methods on walk-count estimation."""

    from sampler_lab.models.self_avoiding_walk import (
        count_self_avoiding_walks,
        sample_self_avoiding_walks,
    )

    if n_steps < 0 or n_particles <= 0:
        raise ValueError("n_steps must be nonnegative and n_particles must be positive")
    if not 0.0 <= ess_threshold <= 1.0:
        raise ValueError("ess_threshold must lie in [0, 1]")
    allowed = {"sis", "multinomial", "systematic", "bernoulli", "ess-systematic"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"unknown particle methods: {sorted(unknown)}")
    exact = count_self_avoiding_walks(n_steps) if n_steps <= exact_limit else None
    streams = spawn_rngs(seed, len(methods))
    rows: list[SelfAvoidingWalkExperimentRow] = []

    for method, rng in zip(methods, streams, strict=True):
        if method == "sis":
            result = sample_self_avoiding_walks(
                rng,
                n_steps=n_steps,
                n_particles=n_particles,
            )
        elif method == "ess-systematic":
            result = sample_self_avoiding_walks(
                rng,
                n_steps=n_steps,
                n_particles=n_particles,
                resampling="systematic",
                resample_ess_fraction=ess_threshold,
            )
        else:
            result = sample_self_avoiding_walks(
                rng,
                n_steps=n_steps,
                n_particles=n_particles,
                resampling=method,
                resample_every_step=True,
            )

        estimate = result.normalizing_constant_estimate
        relative_error = None if exact is None else abs(estimate - exact) / exact
        ess_history = result.ess_history
        minimum_ess_fraction = (
            1.0
            if ess_history.size == 0
            else float(
                np.min(
                    ess_history
                    / np.asarray(
                        [cloud.n_particles for cloud in result.weighted_clouds],
                        dtype=np.float64,
                    )
                )
            )
        )
        rows.append(
            SelfAvoidingWalkExperimentRow(
                method=method,
                estimated_count=estimate,
                exact_count=exact,
                relative_error=relative_error,
                final_population=result.final_cloud.n_particles,
                positive_final_weights=result.final_weighted_cloud.diagnostics.n_positive,
                resampling_steps=int(np.count_nonzero(result.resampled)),
                minimum_ess_fraction=minimum_ess_fraction,
                unique_initial_ancestors=int(result.ancestry.unique_ancestor_counts()[0]),
            )
        )
    return rows


def particle_methods_main() -> None:
    """Run the self-avoiding-walk SIS and resampling comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--particles", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--exact-limit", type=int, default=12)
    parser.add_argument("--ess-threshold", type=float, default=0.85)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sis", "multinomial", "systematic", "bernoulli", "ess-systematic"],
    )
    args = parser.parse_args()
    rows = run_self_avoiding_walk_experiment(
        n_steps=args.steps,
        n_particles=args.particles,
        seed=args.seed,
        methods=tuple(args.methods),
        exact_limit=args.exact_limit,
        ess_threshold=args.ess_threshold,
    )

    print(f"Fixed-origin square-lattice self-avoiding walks ({args.steps} steps)")
    print(
        f"{'method':>16}  {'count estimate':>15}  {'exact':>12}  {'rel. error':>11}  "
        f"{'population':>11}  {'positive':>9}  {'resamples':>9}  "
        f"{'min ESS/N':>10}  {'unique roots':>12}"
    )
    for row in rows:
        exact_text = "-" if row.exact_count is None else str(row.exact_count)
        error_text = "-" if row.relative_error is None else f"{row.relative_error:.5f}"
        print(
            f"{row.method:>16}  {row.estimated_count:15.4f}  {exact_text:>12}  "
            f"{error_text:>11}  {row.final_population:11d}  "
            f"{row.positive_final_weights:9d}  {row.resampling_steps:9d}  "
            f"{row.minimum_ess_fraction:10.4f}  {row.unique_initial_ancestors:12d}"
        )


if __name__ == "__main__":
    disk_benchmark_main()


@dataclass(frozen=True, slots=True)
class MarkovTheoryExperimentRow:
    """Exact and replicated diagnostics for one finite-state ring chain."""

    method: str
    reversible: bool
    period: int
    invariant_residual: float
    detailed_balance_residual: float
    integrated_autocorrelation_time: float
    asymptotic_variance: float
    exact_finite_sample_standard_error: float
    empirical_standard_error: float
    absolute_spectral_gap: float
    singular_value_gap: float
    poincare_gap: float | None


def _replicated_stationary_mean_standard_error(
    transition: np.ndarray,
    stationary: np.ndarray,
    observable: np.ndarray,
    *,
    n_samples: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> float:
    """Estimate finite-sample standard error from independent stationary runs."""

    states = rng.choice(transition.shape[0], size=n_replicates, p=stationary)
    sums = np.asarray(observable[states], dtype=np.float64)
    cumulative = np.cumsum(transition, axis=1)
    for _ in range(1, n_samples):
        draws = rng.random(n_replicates)
        state_cumulative = cumulative[states]
        states = np.sum(draws[:, None] > state_cumulative, axis=1, dtype=np.int64)
        states = np.minimum(states, transition.shape[0] - 1)
        sums += observable[states]
    means = sums / n_samples
    return float(np.std(means, ddof=1))


def run_markov_theory_experiment(
    *,
    n_states: int,
    n_samples: int,
    n_replicates: int,
    seed: int,
) -> list[MarkovTheoryExperimentRow]:
    """Compare reversible, irreversible, and periodic ring transitions."""

    from sampler_lab.markov import (
        asymptotic_variance,
        finite_sample_mean_variance,
        integrated_autocorrelation_time,
    )
    from sampler_lab.models import (
        deterministic_cycle,
        ring_cosine_observable,
        ring_random_walk,
    )

    if n_states < 3:
        raise ValueError("n_states must be at least three")
    if n_samples <= 0 or n_replicates <= 1:
        raise ValueError("n_samples must be positive and n_replicates must exceed one")

    chains = (
        (
            "lazy reversible",
            ring_random_walk(
                n_states,
                clockwise=0.25,
                counterclockwise=0.25,
                stay=0.5,
            ),
        ),
        (
            "lazy directed",
            ring_random_walk(
                n_states,
                clockwise=0.4,
                counterclockwise=0.1,
                stay=0.5,
            ),
        ),
        ("deterministic cycle", deterministic_cycle(n_states)),
    )
    observable = ring_cosine_observable(n_states)
    rngs = spawn_rngs(seed, len(chains))
    rows: list[MarkovTheoryExperimentRow] = []

    for (name, chain), rng in zip(chains, rngs, strict=True):
        stationary = chain.invariant_distribution()
        spectrum = chain.spectral_summary(stationary)
        exact_variance = finite_sample_mean_variance(
            chain,
            observable,
            n_samples,
            probabilities=stationary,
        )
        rows.append(
            MarkovTheoryExperimentRow(
                method=name,
                reversible=spectrum.reversible,
                period=chain.period,
                invariant_residual=chain.global_balance_residual(stationary),
                detailed_balance_residual=chain.detailed_balance_residual(stationary),
                integrated_autocorrelation_time=integrated_autocorrelation_time(
                    chain,
                    observable,
                    probabilities=stationary,
                ),
                asymptotic_variance=asymptotic_variance(
                    chain,
                    observable,
                    probabilities=stationary,
                ),
                exact_finite_sample_standard_error=float(np.sqrt(max(0.0, exact_variance))),
                empirical_standard_error=_replicated_stationary_mean_standard_error(
                    chain.transition,
                    stationary,
                    observable,
                    n_samples=n_samples,
                    n_replicates=n_replicates,
                    rng=rng,
                ),
                absolute_spectral_gap=spectrum.absolute_spectral_gap,
                singular_value_gap=spectrum.singular_value_gap,
                poincare_gap=spectrum.poincare_gap,
            )
        )
    return rows


def markov_theory_main() -> None:
    """Run the exact finite-state Markov laboratory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    rows = run_markov_theory_experiment(
        n_states=args.states,
        n_samples=args.samples,
        n_replicates=args.replicates,
        seed=args.seed,
    )

    print(
        f"Finite-state ring laboratory ({args.states} states, "
        f"{args.samples} stationary observations per run)"
    )
    print(
        f"{'method':>20}  {'rev.':>5}  {'period':>6}  {'IAT':>10}  "
        f"{'asym. var.':>11}  {'exact SE':>10}  {'emp. SE':>10}  "
        f"{'abs gap':>9}  {'sv gap':>9}  {'Poincare':>9}"
    )
    for row in rows:
        poincare = "-" if row.poincare_gap is None else f"{row.poincare_gap:.5f}"
        print(
            f"{row.method:>20}  {row.reversible!s:>5}  {row.period:6d}  "
            f"{row.integrated_autocorrelation_time:10.5f}  "
            f"{row.asymptotic_variance:11.6f}  "
            f"{row.exact_finite_sample_standard_error:10.6f}  "
            f"{row.empirical_standard_error:10.6f}  "
            f"{row.absolute_spectral_gap:9.5f}  {row.singular_value_gap:9.5f}  "
            f"{poincare:>9}"
        )
    print("\nAll three chains preserve the uniform law; only the first satisfies detailed balance.")


@dataclass(frozen=True, slots=True)
class IsingExperimentRow:
    """Cost-normalized diagnostics for one Ising MCMC method."""

    method: str
    lattice_size: int
    beta: float
    sampling_sweeps: int
    spin_updates: int
    elapsed_seconds: float
    acceptance_rate: float | None
    mean_magnetization: float
    mean_absolute_magnetization: float
    exact_mean_absolute_magnetization: float | None
    mean_energy_per_site: float
    magnetization_iat_sweeps: float
    effective_sample_size: float
    ess_per_spin_update: float
    ess_per_second: float


def _run_one_ising_method(
    *,
    method: str,
    model: object,
    n_sweeps: int,
    burn_in_sweeps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, OperationCounter, float, float | None]:
    from sampler_lab.models import (
        IsingModel,
        RandomScanIsingMetropolisKernel,
        deterministic_sweep_ising_gibbs,
        random_scan_ising_gibbs,
    )

    if not isinstance(model, IsingModel):
        raise TypeError("model must be an IsingModel")
    counter = OperationCounter()
    kernel: MarkovKernel
    if method == "random-scan Gibbs":
        kernel = random_scan_ising_gibbs(model, counter=counter)
        updates_per_sweep = model.n_sites
    elif method == "deterministic-sweep Gibbs":
        kernel = deterministic_sweep_ising_gibbs(model, counter=counter)
        updates_per_sweep = 1
    elif method == "single-spin Metropolis":
        kernel = RandomScanIsingMetropolisKernel(model, counter=counter)
        updates_per_sweep = model.n_sites
    else:
        raise ValueError(f"unknown Ising method: {method}")

    state = model.random_state(rng)
    magnetizations = np.empty(n_sweeps, dtype=np.float64)
    energies = np.empty(n_sweeps, dtype=np.float64)
    accepted = 0
    observed_acceptance = 0
    start = time.perf_counter()
    total_sweeps = burn_in_sweeps + n_sweeps
    for sweep in range(total_sweeps):
        for _ in range(updates_per_sweep):
            transition = kernel.step(state, rng)
            state = np.asarray(transition.state, dtype=np.float64)
            if transition.accepted is not None:
                observed_acceptance += 1
                accepted += int(transition.accepted)
        if sweep >= burn_in_sweeps:
            output_index = sweep - burn_in_sweeps
            magnetizations[output_index] = model.magnetization(state, normalized=True)
            energies[output_index] = model.energy(state) / model.n_sites
    elapsed = time.perf_counter() - start
    acceptance_rate = None if observed_acceptance == 0 else accepted / observed_acceptance
    return magnetizations, energies, counter, elapsed, acceptance_rate


def run_ising_experiment(
    *,
    lattice_sizes: tuple[int, ...],
    betas: tuple[float, ...],
    n_sweeps: int,
    burn_in_sweeps: int,
    seed: int,
) -> list[IsingExperimentRow]:
    """Compare Gibbs scan schedules and single-spin Metropolis on Ising targets."""

    from sampler_lab.diagnostics import empirical_integrated_autocorrelation_time
    from sampler_lab.models import IsingModel, exact_ising_distribution

    if not lattice_sizes or any(size < 2 for size in lattice_sizes):
        raise ValueError("lattice_sizes must contain integers at least two")
    if not betas or any(not np.isfinite(beta) or beta < 0.0 for beta in betas):
        raise ValueError("betas must be finite and nonnegative")
    if n_sweeps < 4 or burn_in_sweeps < 0:
        raise ValueError("n_sweeps must be at least four and burn_in_sweeps nonnegative")

    methods = (
        "random-scan Gibbs",
        "deterministic-sweep Gibbs",
        "single-spin Metropolis",
    )
    streams = iter(spawn_rngs(seed, len(lattice_sizes) * len(betas) * len(methods)))
    rows: list[IsingExperimentRow] = []
    for size in lattice_sizes:
        for beta in betas:
            model = IsingModel(size, beta)
            exact_absolute = None
            if model.n_sites <= 16:
                exact = exact_ising_distribution(model)
                exact_absolute = exact.expectation(np.abs(exact.magnetizations) / model.n_sites)
            for method in methods:
                magnetizations, energies, counter, elapsed, acceptance_rate = _run_one_ising_method(
                    method=method,
                    model=model,
                    n_sweeps=n_sweeps,
                    burn_in_sweeps=burn_in_sweeps,
                    rng=next(streams),
                )
                try:
                    iat = empirical_integrated_autocorrelation_time(magnetizations)
                    effective_sample_size = n_sweeps / iat
                except ValueError:
                    iat = float("inf")
                    effective_sample_size = 0.0
                rows.append(
                    IsingExperimentRow(
                        method=method,
                        lattice_size=size,
                        beta=beta,
                        sampling_sweeps=n_sweeps,
                        spin_updates=counter.spin_updates,
                        elapsed_seconds=elapsed,
                        acceptance_rate=acceptance_rate,
                        mean_magnetization=float(np.mean(magnetizations)),
                        mean_absolute_magnetization=float(np.mean(np.abs(magnetizations))),
                        exact_mean_absolute_magnetization=exact_absolute,
                        mean_energy_per_site=float(np.mean(energies)),
                        magnetization_iat_sweeps=iat,
                        effective_sample_size=effective_sample_size,
                        ess_per_spin_update=effective_sample_size / counter.spin_updates,
                        ess_per_second=effective_sample_size / elapsed,
                    )
                )
    return rows


def ising_mcmc_main() -> None:
    """Run the Ising Gibbs-versus-Metropolis benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.3, 0.44, 0.6])
    parser.add_argument("--sweeps", type=int, default=4_000)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    rows = run_ising_experiment(
        lattice_sizes=tuple(args.sizes),
        betas=tuple(args.betas),
        n_sweeps=args.sweeps,
        burn_in_sweeps=args.burn_in,
        seed=args.seed,
    )

    print("Periodic square-lattice Ising MCMC benchmark")
    print(
        f"{'L':>3}  {'beta':>6}  {'method':>27}  {'accept':>8}  "
        f"{'E[m]':>9}  {'E[|m|]':>9}  {'exact':>9}  {'E/N':>9}  "
        f"{'IAT':>9}  {'ESS':>9}  {'ESS/1k upd':>12}  {'ESS/s':>10}"
    )
    for row in rows:
        acceptance = "-" if row.acceptance_rate is None else f"{row.acceptance_rate:.4f}"
        exact = (
            "-"
            if row.exact_mean_absolute_magnetization is None
            else f"{row.exact_mean_absolute_magnetization:.5f}"
        )
        print(
            f"{row.lattice_size:3d}  {row.beta:6.3f}  {row.method:>27}  "
            f"{acceptance:>8}  {row.mean_magnetization:9.5f}  "
            f"{row.mean_absolute_magnetization:9.5f}  {exact:>9}  "
            f"{row.mean_energy_per_site:9.5f}  {row.magnetization_iat_sweeps:9.3f}  "
            f"{row.effective_sample_size:9.1f}  "
            f"{1000.0 * row.ess_per_spin_update:12.5f}  {row.ess_per_second:10.1f}"
        )
    print("\nIAT is measured in recorded sweeps; cost-normalized ESS includes burn-in updates.")


@dataclass(frozen=True, slots=True)
class AnnealingExperimentRow:
    """One Ising annealing path-length and resampling comparison."""

    method: str
    n_steps: int
    n_particles: int
    target_beta: float
    log_ratio_estimate: float
    exact_log_ratio: float
    log_ratio_error: float
    delta_free_energy_estimate: float
    exact_delta_free_energy: float
    mean_absolute_magnetization: float
    exact_mean_absolute_magnetization: float
    minimum_ess_fraction: float
    resampling_steps: int
    unique_initial_ancestors: int
    spin_updates: int


def run_annealing_experiment(
    *,
    lattice_size: int,
    target_beta: float,
    path_steps: tuple[int, ...],
    n_particles: int,
    sweeps_per_stage: int,
    ess_threshold: float,
    seed: int,
) -> list[AnnealingExperimentRow]:
    """Compare Jarzynski/AIS and resampled annealed SMC on a finite Ising model."""

    from sampler_lab.annealing import (
        AnnealingSchedule,
        GeometricAnnealingPath,
        annealed_importance_sampling,
    )
    from sampler_lab.models import (
        IsingGibbsPopulationTransition,
        IsingModel,
        exact_ising_distribution,
    )
    from sampler_lab.particles import SystematicResampler

    if lattice_size < 2:
        raise ValueError("lattice_size must be at least two")
    if lattice_size * lattice_size > 16:
        raise ValueError("the exact annealing experiment is limited to at most 16 sites")
    if not np.isfinite(target_beta) or target_beta < 0.0:
        raise ValueError("target_beta must be finite and nonnegative")
    if not path_steps or any(step <= 0 for step in path_steps):
        raise ValueError("path_steps must contain positive integers")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if sweeps_per_stage < 0:
        raise ValueError("sweeps_per_stage must be nonnegative")
    if not 0.0 <= ess_threshold <= 1.0:
        raise ValueError("ess_threshold must lie in [0, 1]")

    initial_model = IsingModel(lattice_size, 0.0)
    target_model = IsingModel(lattice_size, target_beta)
    exact = exact_ising_distribution(target_model, max_sites=16)
    exact_log_ratio = exact.log_partition - target_model.n_sites * np.log(2.0)
    exact_abs_magnetization = exact.expectation(np.abs(exact.magnetizations) / target_model.n_sites)
    path = GeometricAnnealingPath(initial_model, target_model)
    methods = ("Jarzynski/AIS", "ESS-resampled SMC")
    streams = iter(spawn_rngs(seed, len(path_steps) * len(methods)))
    rows: list[AnnealingExperimentRow] = []

    for n_steps in path_steps:
        schedule = AnnealingSchedule.linear(n_steps)

        transition = IsingGibbsPopulationTransition(
            lattice_size,
            target_beta,
            n_sweeps=sweeps_per_stage,
        )
        for method in methods:
            rng = next(streams)
            initial_particles = np.asarray(
                2
                * rng.integers(
                    0,
                    2,
                    size=(n_particles, lattice_size, lattice_size),
                )
                - 1,
                dtype=np.float64,
            )
            if method == "Jarzynski/AIS":
                result = annealed_importance_sampling(
                    initial_particles,
                    path,
                    schedule,
                    transition,
                    rng,
                )
            else:
                result = annealed_importance_sampling(
                    initial_particles,
                    path,
                    schedule,
                    transition,
                    rng,
                    resampler=SystematicResampler(),
                    resample_ess_fraction=ess_threshold,
                )
            absolute_magnetization = result.final_cloud.expectation(
                lambda particles: np.abs(np.sum(particles, axis=(1, 2))) / target_model.n_sites
            )
            assert isinstance(absolute_magnetization, float)
            ess_fractions = result.ess_history / np.asarray(
                [cloud.n_particles for cloud in result.weighted_clouds],
                dtype=np.float64,
            )
            rows.append(
                AnnealingExperimentRow(
                    method=method,
                    n_steps=n_steps,
                    n_particles=n_particles,
                    target_beta=target_beta,
                    log_ratio_estimate=result.log_normalizing_constant_ratio,
                    exact_log_ratio=float(exact_log_ratio),
                    log_ratio_error=result.log_normalizing_constant_ratio - float(exact_log_ratio),
                    delta_free_energy_estimate=result.delta_free_energy,
                    exact_delta_free_energy=-float(exact_log_ratio),
                    mean_absolute_magnetization=absolute_magnetization,
                    exact_mean_absolute_magnetization=exact_abs_magnetization,
                    minimum_ess_fraction=float(np.min(ess_fractions)),
                    resampling_steps=int(np.count_nonzero(result.resampled)),
                    unique_initial_ancestors=int(result.ancestry.unique_ancestor_counts()[0]),
                    spin_updates=n_particles * target_model.n_sites * sweeps_per_stage * n_steps,
                )
            )
    return rows


def annealing_main() -> None:
    """Run the finite-Ising Jarzynski and annealed-SMC path study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--target-beta", type=float, default=0.6)
    parser.add_argument("--path-steps", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--particles", type=int, default=5_000)
    parser.add_argument("--sweeps-per-stage", type=int, default=1)
    parser.add_argument("--ess-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    rows = run_annealing_experiment(
        lattice_size=args.size,
        target_beta=args.target_beta,
        path_steps=tuple(args.path_steps),
        n_particles=args.particles,
        sweeps_per_stage=args.sweeps_per_stage,
        ess_threshold=args.ess_threshold,
        seed=args.seed,
    )

    print(
        f"Ising annealing study (L={args.size}, beta={args.target_beta}, "
        f"particles={args.particles})"
    )
    print(
        f"{'method':>20}  {'steps':>5}  {'log Z ratio':>12}  {'exact':>12}  "
        f"{'error':>10}  {'Delta F':>11}  {'E|m|':>9}  {'exact':>9}  "
        f"{'min ESS/N':>10}  {'resamp.':>7}  {'roots':>7}  {'spin upd.':>10}"
    )
    for row in rows:
        print(
            f"{row.method:>20}  {row.n_steps:5d}  {row.log_ratio_estimate:12.6f}  "
            f"{row.exact_log_ratio:12.6f}  {row.log_ratio_error:10.6f}  "
            f"{row.delta_free_energy_estimate:11.6f}  "
            f"{row.mean_absolute_magnetization:9.5f}  "
            f"{row.exact_mean_absolute_magnetization:9.5f}  "
            f"{row.minimum_ess_fraction:10.4f}  {row.resampling_steps:7d}  "
            f"{row.unique_initial_ancestors:7d}  {row.spin_updates:10d}"
        )


@dataclass(frozen=True, slots=True)
class MetropolisGeneratorExperimentRow:
    """Empirical small-step random-walk Metropolis generator estimate."""

    step_size: float
    estimate: float
    standard_error: float
    limiting_generator: float
    z_score: float


@dataclass(frozen=True, slots=True)
class LangevinGaussianExperimentRow:
    """Conditioning, bias, and time-series diagnostics for one Gaussian method."""

    method: str
    condition_number: float
    step_size: float
    acceptance_rate: float | None
    exact_iat: float | None
    empirical_iat: float
    target_variance: float
    exact_stationary_variance: float
    empirical_variance: float
    covariance_kl: float
    gradient_evaluations: int
    elapsed_seconds: float


def run_metropolis_generator_experiment(
    *,
    step_sizes: tuple[float, ...],
    n_replications: int,
    state: float,
    seed: int,
) -> list[MetropolisGeneratorExperimentRow]:
    """Estimate the Langevin generator limit of small-step random-walk MH."""

    from sampler_lab.dynamics import estimate_discrete_generator
    from sampler_lab.mcmc import GaussianRandomWalkProposal, MetropolisHastingsKernel
    from sampler_lab.models import GaussianTarget

    if not step_sizes or any(not np.isfinite(step) or step <= 0.0 for step in step_sizes):
        raise ValueError("step_sizes must contain positive finite values")
    if n_replications < 2:
        raise ValueError("n_replications must be at least two")
    if not np.isfinite(state):
        raise ValueError("state must be finite")

    target = GaussianTarget([0.0], [[1.0]])
    streams = spawn_rngs(seed, len(step_sizes))
    limiting_generator = -float(state)
    rows: list[MetropolisGeneratorExperimentRow] = []
    for step_size, rng in zip(step_sizes, streams, strict=True):
        kernel = MetropolisHastingsKernel(
            target,
            GaussianRandomWalkProposal(np.sqrt(2.0 * step_size)),
        )
        estimate = estimate_discrete_generator(
            kernel,
            lambda x: float(x[0]),
            np.array([state], dtype=np.float64),
            step_size,
            n_replications,
            rng,
        )
        rows.append(
            MetropolisGeneratorExperimentRow(
                step_size=step_size,
                estimate=estimate.value,
                standard_error=estimate.standard_error,
                limiting_generator=limiting_generator,
                z_score=(estimate.value - limiting_generator) / estimate.standard_error,
            )
        )
    return rows


def run_langevin_gaussian_experiment(
    *,
    condition_numbers: tuple[float, ...],
    n_samples: int,
    burn_in: int,
    step_fraction: float,
    seed: int,
) -> list[LangevinGaussianExperimentRow]:
    """Compare identity and covariance-preconditioned ULA/MALA on Gaussians."""

    from sampler_lab.diagnostics import empirical_integrated_autocorrelation_time
    from sampler_lab.dynamics import (
        MetropolisAdjustedLangevinKernel,
        UnadjustedLangevinKernel,
        gaussian_ula_analysis,
        linear_gaussian_iat,
    )
    from sampler_lab.mcmc import run_chain
    from sampler_lab.models import GaussianTarget

    if not condition_numbers or any(
        not np.isfinite(condition) or condition < 1.0 for condition in condition_numbers
    ):
        raise ValueError("condition_numbers must contain finite values at least one")
    if n_samples < 4 or burn_in < 0:
        raise ValueError("n_samples must be at least four and burn_in nonnegative")
    if not np.isfinite(step_fraction) or not 0.0 < step_fraction < 1.0:
        raise ValueError("step_fraction must lie strictly between zero and one")

    methods = ("ULA identity", "ULA covariance", "MALA covariance")
    streams = iter(spawn_rngs(seed, len(condition_numbers) * len(methods)))
    rows: list[LangevinGaussianExperimentRow] = []
    observable = np.array([1.0, 0.0])

    for condition_number in condition_numbers:
        covariance = np.diag([1.0, 1.0 / condition_number])
        target = GaussianTarget([0.0, 0.0], covariance)
        identity_step = 2.0 * step_fraction / condition_number
        covariance_step = 2.0 * step_fraction
        configurations: tuple[tuple[str, float, np.ndarray | None, bool], ...] = (
            ("ULA identity", identity_step, None, False),
            ("ULA covariance", covariance_step, covariance, False),
            ("MALA covariance", covariance_step, covariance, True),
        )
        for method, step_size, preconditioner, adjusted in configurations:
            rng = next(streams)
            counter = OperationCounter()
            kernel: MarkovKernel
            if adjusted:
                kernel = MetropolisAdjustedLangevinKernel(
                    target,
                    step_size,
                    preconditioner=preconditioner,
                    counter=counter,
                )
                analysis = None
            else:
                kernel = UnadjustedLangevinKernel(
                    target,
                    step_size,
                    preconditioner=preconditioner,
                    counter=counter,
                )
                analysis = gaussian_ula_analysis(
                    target,
                    step_size,
                    preconditioner=preconditioner,
                )
            initial_state = np.asarray(
                np.linalg.cholesky(covariance) @ rng.normal(size=2),
                dtype=np.float64,
            )
            start = time.perf_counter()
            trajectory = run_chain(
                kernel,
                initial_state,
                rng,
                n_steps=burn_in + n_samples,
            )
            elapsed = time.perf_counter() - start
            samples = np.asarray(trajectory.states[burn_in + 1 :, 0], dtype=np.float64)
            empirical_iat = empirical_integrated_autocorrelation_time(samples)
            empirical_variance = float(np.var(samples))
            if analysis is None:
                exact_iat = None
                exact_stationary_variance = 1.0
                covariance_kl = 0.0
            else:
                assert analysis.stationary_covariance is not None
                exact_iat = linear_gaussian_iat(
                    analysis.transition_matrix,
                    analysis.stationary_covariance,
                    observable,
                )
                exact_stationary_variance = float(analysis.stationary_covariance[0, 0])
                assert analysis.kl_stationary_to_target is not None
                covariance_kl = analysis.kl_stationary_to_target
            rows.append(
                LangevinGaussianExperimentRow(
                    method=method,
                    condition_number=condition_number,
                    step_size=step_size,
                    acceptance_rate=trajectory.acceptance_rate,
                    exact_iat=exact_iat,
                    empirical_iat=empirical_iat,
                    target_variance=1.0,
                    exact_stationary_variance=exact_stationary_variance,
                    empirical_variance=empirical_variance,
                    covariance_kl=covariance_kl,
                    gradient_evaluations=counter.gradient_evaluations,
                    elapsed_seconds=elapsed,
                )
            )
    return rows


def langevin_main() -> None:
    """Run generator-limit and conditioned-Gaussian Langevin studies."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-numbers", type=float, nargs="+", default=[1, 10, 100])
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--step-fraction", type=float, default=0.25)
    parser.add_argument("--generator-steps", type=float, nargs="+", default=[0.1, 0.03, 0.01])
    parser.add_argument("--generator-replications", type=int, default=20_000)
    parser.add_argument("--generator-state", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    generator_rows = run_metropolis_generator_experiment(
        step_sizes=tuple(args.generator_steps),
        n_replications=args.generator_replications,
        state=args.generator_state,
        seed=args.seed,
    )
    print("Small-step random-walk Metropolis generator")
    print(f"{'h':>10}  {'estimate':>12}  {'SE':>10}  {'limit':>12}  {'z':>9}")
    for generator_row in generator_rows:
        print(
            f"{generator_row.step_size:10.5g}  {generator_row.estimate:12.6f}  "
            f"{generator_row.standard_error:10.6f}  "
            f"{generator_row.limiting_generator:12.6f}  "
            f"{generator_row.z_score:9.3f}"
        )

    rows = run_langevin_gaussian_experiment(
        condition_numbers=tuple(args.condition_numbers),
        n_samples=args.samples,
        burn_in=args.burn_in,
        step_fraction=args.step_fraction,
        seed=args.seed + 1,
    )
    print("\nConditioned Gaussian Langevin comparison")
    print(
        f"{'kappa':>8}  {'method':>17}  {'h':>9}  {'accept':>8}  "
        f"{'IAT exact':>10}  {'IAT emp':>9}  {'var exact':>10}  "
        f"{'var emp':>9}  {'KL':>10}  {'grad/s':>11}"
    )
    for langevin_row in rows:
        acceptance = (
            "-" if langevin_row.acceptance_rate is None else f"{langevin_row.acceptance_rate:.4f}"
        )
        exact_iat = "-" if langevin_row.exact_iat is None else f"{langevin_row.exact_iat:.3f}"
        gradient_rate = langevin_row.gradient_evaluations / langevin_row.elapsed_seconds
        print(
            f"{langevin_row.condition_number:8.1f}  {langevin_row.method:>17}  "
            f"{langevin_row.step_size:9.5f}  {acceptance:>8}  {exact_iat:>10}  "
            f"{langevin_row.empirical_iat:9.3f}  "
            f"{langevin_row.exact_stationary_variance:10.5f}  "
            f"{langevin_row.empirical_variance:9.5f}  "
            f"{langevin_row.covariance_kl:10.4g}  {gradient_rate:11.1f}"
        )
    print(
        "\nIdentity ULA pays for the stiff coordinate in its step size; covariance "
        "preconditioning removes that condition-number penalty. MALA then removes "
        "the remaining Euler invariant-law bias."
    )


@dataclass(frozen=True, slots=True)
class HamiltonianGaussianExperimentRow:
    """Conditioning and cost diagnostics for one Hamiltonian method."""

    method: str
    condition_number: float
    step_size: float
    n_leapfrog_steps: int
    acceptance_rate: float | None
    exact_ideal_iat: float | None
    empirical_iat: float
    empirical_variance: float
    root_mean_square_energy_error: float | None
    gradient_evaluations: int
    ess_per_thousand_gradients: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class XYDynamicsExperimentRow:
    """Single-site XY / von Mises benchmark row."""

    method: str
    mean_cosine: float
    exact_mean_cosine: float
    absolute_error: float
    mean_sine: float
    empirical_iat: float
    acceptance_rate: float | None
    gradient_evaluations: int
    ess_per_thousand_gradients: float
    elapsed_seconds: float


def _root_mean_square_transition_diagnostic(
    diagnostics: tuple[dict[str, float], ...],
    name: str,
) -> float | None:
    values = np.asarray(
        [item[name] for item in diagnostics if name in item],
        dtype=np.float64,
    )
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values * values)))


def run_hamiltonian_gaussian_experiment(
    *,
    condition_numbers: tuple[float, ...],
    n_samples: int,
    burn_in: int,
    trajectory_time: float,
    step_fraction: float,
    friction: float,
    seed: int,
) -> list[HamiltonianGaussianExperimentRow]:
    """Compare identity/precision HMC and underdamped methods on Gaussians."""

    from sampler_lab.diagnostics import empirical_integrated_autocorrelation_time
    from sampler_lab.dynamics import (
        HamiltonianMonteCarloKernel,
        MassMatrix,
        MetropolizedUnderdampedLangevinKernel,
        UnderdampedLangevinKernel,
        gaussian_hamiltonian_analysis,
        linear_gaussian_iat,
    )
    from sampler_lab.mcmc import run_chain
    from sampler_lab.models import GaussianTarget

    if not condition_numbers or any(
        not np.isfinite(condition) or condition < 1.0 for condition in condition_numbers
    ):
        raise ValueError("condition_numbers must contain finite values at least one")
    if n_samples < 4 or burn_in < 0:
        raise ValueError("n_samples must be at least four and burn_in nonnegative")
    if not np.isfinite(trajectory_time) or trajectory_time <= 0.0:
        raise ValueError("trajectory_time must be positive and finite")
    if not np.isfinite(step_fraction) or not 0.0 < step_fraction < 1.0:
        raise ValueError("step_fraction must lie strictly between zero and one")
    if not np.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be nonnegative and finite")

    method_count = 4
    streams = iter(spawn_rngs(seed, len(condition_numbers) * method_count))
    rows: list[HamiltonianGaussianExperimentRow] = []
    observable = np.array([1.0, 0.0])

    for condition_number in condition_numbers:
        covariance = np.diag([1.0, 1.0 / condition_number])
        target = GaussianTarget([0.0, 0.0], covariance)
        precision_mass = MassMatrix(target.precision_matrix)
        maximum_frequency = float(np.sqrt(condition_number))
        identity_step = 2.0 * step_fraction / maximum_frequency
        precision_step = 2.0 * step_fraction
        identity_steps = max(1, round(trajectory_time / identity_step))
        precision_steps = max(1, round(trajectory_time / precision_step))
        configurations: tuple[
            tuple[str, str, float, int, MassMatrix, bool],
            ...,
        ] = (
            (
                "HMC identity mass",
                "hmc",
                identity_step,
                identity_steps,
                MassMatrix.identity(2),
                False,
            ),
            (
                "HMC precision mass",
                "hmc",
                precision_step,
                precision_steps,
                precision_mass,
                False,
            ),
            (
                "BAOAB precision mass",
                "underdamped",
                min(0.25, precision_step),
                1,
                precision_mass,
                True,
            ),
            (
                "Metropolized underdamped",
                "metropolized",
                min(0.5, precision_step),
                1,
                precision_mass,
                True,
            ),
        )
        for method, family, step_size, n_steps, mass, phase_valued in configurations:
            rng = next(streams)
            counter = OperationCounter()
            if family == "hmc":
                kernel: MarkovKernel = HamiltonianMonteCarloKernel(
                    target,
                    step_size,
                    n_steps,
                    mass_matrix=mass,
                    counter=counter,
                )
                initial_state = np.asarray(
                    np.linalg.cholesky(covariance) @ rng.normal(size=2),
                    dtype=np.float64,
                )
                analysis = gaussian_hamiltonian_analysis(
                    target,
                    step_size,
                    n_steps,
                    mass=mass,
                )
                exact_iat = linear_gaussian_iat(
                    analysis.exact_position_transition,
                    covariance,
                    observable,
                )
            elif family == "underdamped":
                kernel = UnderdampedLangevinKernel(
                    target,
                    step_size,
                    friction,
                    mass_matrix=mass,
                    counter=counter,
                )
                initial_position = np.asarray(
                    np.linalg.cholesky(covariance) @ rng.normal(size=2),
                    dtype=np.float64,
                )
                initial_state = np.concatenate(
                    (initial_position, mass.sample_momentum(rng))
                ).astype(np.float64)
                exact_iat = None
            else:
                kernel = MetropolizedUnderdampedLangevinKernel(
                    target,
                    step_size,
                    friction,
                    n_leapfrog_steps=n_steps,
                    mass_matrix=mass,
                    counter=counter,
                )
                initial_position = np.asarray(
                    np.linalg.cholesky(covariance) @ rng.normal(size=2),
                    dtype=np.float64,
                )
                initial_state = np.concatenate(
                    (initial_position, mass.sample_momentum(rng))
                ).astype(np.float64)
                exact_iat = None

            start = time.perf_counter()
            trajectory = run_chain(
                kernel,
                initial_state,
                rng,
                n_steps=burn_in + n_samples,
            )
            elapsed = time.perf_counter() - start
            position_values = np.asarray(
                trajectory.states[burn_in + 1 :, 0],
                dtype=np.float64,
            )
            empirical_iat = empirical_integrated_autocorrelation_time(position_values)
            effective_samples = n_samples / empirical_iat
            ess_per_thousand = (
                float("inf")
                if counter.gradient_evaluations == 0
                else 1000.0 * effective_samples / counter.gradient_evaluations
            )
            rms_energy_error = _root_mean_square_transition_diagnostic(
                trajectory.transition_diagnostics,
                "energy_error",
            )
            if phase_valued and family == "underdamped":
                rms_energy_error = None
            rows.append(
                HamiltonianGaussianExperimentRow(
                    method=method,
                    condition_number=condition_number,
                    step_size=step_size,
                    n_leapfrog_steps=n_steps,
                    acceptance_rate=trajectory.acceptance_rate,
                    exact_ideal_iat=exact_iat,
                    empirical_iat=empirical_iat,
                    empirical_variance=float(np.var(position_values)),
                    root_mean_square_energy_error=rms_energy_error,
                    gradient_evaluations=counter.gradient_evaluations,
                    ess_per_thousand_gradients=ess_per_thousand,
                    elapsed_seconds=elapsed,
                )
            )
    return rows


def run_xy_dynamics_experiment(
    *,
    n_samples: int,
    burn_in: int,
    concentration: float,
    seed: int,
) -> list[XYDynamicsExperimentRow]:
    """Compare Hamiltonian and underdamped methods on a one-site XY law."""

    from sampler_lab.diagnostics import empirical_integrated_autocorrelation_time
    from sampler_lab.dynamics import (
        HamiltonianMonteCarloKernel,
        MetropolizedUnderdampedLangevinKernel,
        UnderdampedLangevinKernel,
    )
    from sampler_lab.mcmc import run_chain
    from sampler_lab.models import XYModel, wrap_angles

    if n_samples < 4 or burn_in < 0:
        raise ValueError("n_samples must be at least four and burn_in nonnegative")
    if not np.isfinite(concentration) or concentration < 0.0:
        raise ValueError("concentration must be nonnegative and finite")
    model = XYModel(
        size=1,
        inverse_temperature=1.0,
        coupling=0.0,
        external_field=concentration,
    )
    exact = model.exact_single_site_mean_cosine()
    streams = iter(spawn_rngs(seed, 3))
    counters = [OperationCounter() for _ in range(3)]
    configurations: tuple[tuple[str, MarkovKernel, np.ndarray, OperationCounter], ...] = (
        (
            "HMC",
            HamiltonianMonteCarloKernel(
                model,
                0.35,
                4,
                position_map=wrap_angles,
                counter=counters[0],
            ),
            np.array([0.0]),
            counters[0],
        ),
        (
            "BAOAB",
            UnderdampedLangevinKernel(
                model,
                0.15,
                1.0,
                position_map=wrap_angles,
                counter=counters[1],
            ),
            np.array([0.0, 0.0]),
            counters[1],
        ),
        (
            "Metropolized underdamped",
            MetropolizedUnderdampedLangevinKernel(
                model,
                0.45,
                1.0,
                position_map=wrap_angles,
                counter=counters[2],
            ),
            np.array([0.0, 0.0]),
            counters[2],
        ),
    )
    rows: list[XYDynamicsExperimentRow] = []
    for (method, kernel, initial_state, counter), rng in zip(
        configurations,
        streams,
        strict=True,
    ):
        start = time.perf_counter()
        trajectory = run_chain(
            kernel,
            initial_state,
            rng,
            n_steps=burn_in + n_samples,
        )
        elapsed = time.perf_counter() - start
        angles = np.asarray(trajectory.states[burn_in + 1 :, 0], dtype=np.float64)
        cosine_values = np.cos(angles)
        empirical_iat = empirical_integrated_autocorrelation_time(cosine_values)
        effective_samples = n_samples / empirical_iat
        rows.append(
            XYDynamicsExperimentRow(
                method=method,
                mean_cosine=float(np.mean(cosine_values)),
                exact_mean_cosine=exact,
                absolute_error=abs(float(np.mean(cosine_values)) - exact),
                mean_sine=float(np.mean(np.sin(angles))),
                empirical_iat=empirical_iat,
                acceptance_rate=trajectory.acceptance_rate,
                gradient_evaluations=counter.gradient_evaluations,
                ess_per_thousand_gradients=(
                    1000.0 * effective_samples / counter.gradient_evaluations
                ),
                elapsed_seconds=elapsed,
            )
        )
    return rows


def hamiltonian_main() -> None:
    """Run Gaussian conditioning and one-site XY Hamiltonian studies."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-numbers", type=float, nargs="+", default=[1, 10, 100])
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--trajectory-time", type=float, default=1.0)
    parser.add_argument("--step-fraction", type=float, default=0.25)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--xy-concentration", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    gaussian_rows = run_hamiltonian_gaussian_experiment(
        condition_numbers=tuple(args.condition_numbers),
        n_samples=args.samples,
        burn_in=args.burn_in,
        trajectory_time=args.trajectory_time,
        step_fraction=args.step_fraction,
        friction=args.friction,
        seed=args.seed,
    )
    print("Conditioned Gaussian Hamiltonian comparison")
    print(
        f"{'kappa':>8}  {'method':>25}  {'h':>8}  {'L':>4}  {'accept':>8}  "
        f"{'IAT ideal':>10}  {'IAT emp':>9}  {'variance':>9}  {'RMS dH':>9}  "
        f"{'ESS/1k grad':>12}"
    )
    for gaussian_row in gaussian_rows:
        acceptance = (
            "-" if gaussian_row.acceptance_rate is None else f"{gaussian_row.acceptance_rate:.4f}"
        )
        ideal_iat = (
            "-" if gaussian_row.exact_ideal_iat is None else f"{gaussian_row.exact_ideal_iat:.3f}"
        )
        energy_error = (
            "-"
            if gaussian_row.root_mean_square_energy_error is None
            else f"{gaussian_row.root_mean_square_energy_error:.4g}"
        )
        print(
            f"{gaussian_row.condition_number:8.1f}  {gaussian_row.method:>25}  "
            f"{gaussian_row.step_size:8.4f}  {gaussian_row.n_leapfrog_steps:4d}  "
            f"{acceptance:>8}  {ideal_iat:>10}  {gaussian_row.empirical_iat:9.3f}  "
            f"{gaussian_row.empirical_variance:9.5f}  {energy_error:>9}  "
            f"{gaussian_row.ess_per_thousand_gradients:12.3f}"
        )

    xy_rows = run_xy_dynamics_experiment(
        n_samples=args.samples,
        burn_in=args.burn_in,
        concentration=args.xy_concentration,
        seed=args.seed + 1,
    )
    print("\nOne-site XY / von Mises benchmark")
    print(
        f"{'method':>25}  {'E cos':>10}  {'exact':>10}  {'abs err':>10}  "
        f"{'E sin':>10}  {'IAT':>9}  {'accept':>8}  {'ESS/1k grad':>12}"
    )
    for xy_row in xy_rows:
        acceptance = "-" if xy_row.acceptance_rate is None else f"{xy_row.acceptance_rate:.4f}"
        print(
            f"{xy_row.method:>25}  {xy_row.mean_cosine:10.6f}  "
            f"{xy_row.exact_mean_cosine:10.6f}  {xy_row.absolute_error:10.6f}  "
            f"{xy_row.mean_sine:10.6f}  {xy_row.empirical_iat:9.3f}  "
            f"{acceptance:>8}  {xy_row.ess_per_thousand_gradients:12.3f}"
        )
    print(
        "\nPrecision mass equalizes Gaussian frequencies. BAOAB is unadjusted; "
        "the metropolized phase-space scheme restores the target through a "
        "momentum-flipped rejection state."
    )


@dataclass(frozen=True, slots=True)
class ConditioningExperimentRow:
    """One conditioned-Gaussian MCMC comparison row."""

    method: str
    condition_number: float
    acceptance_rate: float
    empirical_iat: float
    empirical_variance: float
    exact_variance: float
    target_evaluations: int
    ess_per_thousand_target_evaluations: float


@dataclass(frozen=True, slots=True)
class RosenbrockExperimentRow:
    """One Rosenbrock chain or ensemble comparison row."""

    method: str
    acceptance_rate: float
    mean_x: float
    mean_y: float
    exact_mean_x: float
    exact_mean_y: float
    x_iat: float
    effective_sample_size: float
    target_evaluations: int
    ess_per_thousand_target_evaluations: float


def _total_target_evaluations(counter: OperationCounter) -> int:
    return (
        counter.log_density_evaluations + counter.gradient_evaluations + counter.hessian_evaluations
    )


def run_conditioning_geometry_experiment(
    *,
    condition_numbers: tuple[float, ...],
    n_samples: int,
    burn_in: int,
    seed: int,
) -> list[ConditioningExperimentRow]:
    """Compare isotropic, covariance-aware, and stochastic-Newton Gaussian chains."""

    from sampler_lab.diagnostics import empirical_integrated_autocorrelation_time
    from sampler_lab.geometry import MetropolizedStochasticNewtonKernel
    from sampler_lab.mcmc import (
        GaussianRandomWalkProposal,
        MetropolisHastingsKernel,
        MultivariateGaussianRandomWalkProposal,
        run_chain,
    )
    from sampler_lab.models import GaussianTarget

    if n_samples < 4 or burn_in < 0:
        raise ValueError("n_samples must be at least four and burn_in nonnegative")
    if not condition_numbers:
        raise ValueError("condition_numbers must be nonempty")
    if any(not np.isfinite(value) or value < 1.0 for value in condition_numbers):
        raise ValueError("condition numbers must be finite and at least one")

    streams = iter(spawn_rngs(seed, 3 * len(condition_numbers)))
    rows: list[ConditioningExperimentRow] = []
    for condition_number in condition_numbers:
        covariance = np.diag([condition_number, 1.0]).astype(np.float64)
        target = GaussianTarget([0.0, 0.0], covariance)
        for method in ("isotropic RWM", "covariance RWM", "stochastic Newton"):
            rng = next(streams)
            counter = OperationCounter()
            if method == "isotropic RWM":
                kernel: MarkovKernel = MetropolisHastingsKernel(
                    target,
                    GaussianRandomWalkProposal(0.9),
                    counter=counter,
                )
            elif method == "covariance RWM":
                kernel = MetropolisHastingsKernel(
                    target,
                    MultivariateGaussianRandomWalkProposal(0.8**2 * covariance),
                    counter=counter,
                )
            else:
                kernel = MetropolizedStochasticNewtonKernel(
                    target,
                    0.45,
                    repair_method="raise",
                    counter=counter,
                )
            initial = np.asarray(np.linalg.cholesky(covariance) @ rng.normal(size=2))
            trajectory = run_chain(
                kernel,
                initial,
                rng,
                n_steps=burn_in + n_samples,
            )
            values = np.asarray(trajectory.states[burn_in + 1 :, 0], dtype=np.float64)
            iat = empirical_integrated_autocorrelation_time(values)
            effective = n_samples / iat
            target_evaluations = _total_target_evaluations(counter)
            rows.append(
                ConditioningExperimentRow(
                    method=method,
                    condition_number=float(condition_number),
                    acceptance_rate=float(trajectory.acceptance_rate or 0.0),
                    empirical_iat=iat,
                    empirical_variance=float(np.var(values)),
                    exact_variance=float(condition_number),
                    target_evaluations=target_evaluations,
                    ess_per_thousand_target_evaluations=(
                        float("inf")
                        if target_evaluations == 0
                        else 1000.0 * effective / target_evaluations
                    ),
                )
            )
    return rows


def run_rosenbrock_geometry_experiment(
    *,
    n_samples: int,
    burn_in: int,
    n_walkers: int,
    seed: int,
) -> list[RosenbrockExperimentRow]:
    """Compare local, curvature-aware, and ensemble methods on Rosenbrock."""

    from sampler_lab.diagnostics import empirical_integrated_autocorrelation_time
    from sampler_lab.ensemble import (
        EnsembleState,
        StretchMoveKernel,
        WalkMoveKernel,
        ensemble_effective_sample_size,
        run_ensemble_chain,
    )
    from sampler_lab.geometry import MetropolizedStochasticNewtonKernel
    from sampler_lab.mcmc import (
        GaussianRandomWalkProposal,
        MetropolisHastingsKernel,
        MultivariateGaussianRandomWalkProposal,
        run_chain,
    )
    from sampler_lab.models import RosenbrockTarget

    if n_samples < 4 or burn_in < 0:
        raise ValueError("n_samples must be at least four and burn_in nonnegative")
    if n_walkers < 6:
        raise ValueError("n_walkers must be at least six for two-dimensional split ensembles")
    target = RosenbrockTarget()
    exact_mean = target.exact_mean()
    streams = iter(spawn_rngs(seed, 5))
    rows: list[RosenbrockExperimentRow] = []

    single_configurations = (
        (
            "isotropic RWM",
            lambda counter: MetropolisHastingsKernel(
                target,
                GaussianRandomWalkProposal(0.35),
                counter=counter,
            ),
        ),
        (
            "fixed-covariance RWM",
            lambda counter: MetropolisHastingsKernel(
                target,
                MultivariateGaussianRandomWalkProposal(0.015 * target.exact_covariance()),
                counter=counter,
            ),
        ),
        (
            "stochastic Newton",
            lambda counter: MetropolizedStochasticNewtonKernel(
                target,
                0.08,
                repair_method="absolute",
                minimum_eigenvalue=1e-3,
                counter=counter,
            ),
        ),
    )
    for method, build_kernel in single_configurations:
        rng = next(streams)
        counter = OperationCounter()
        kernel = build_kernel(counter)
        initial = target.sample_exact(rng, 1)[0]
        single_trajectory = run_chain(
            kernel,
            initial,
            rng,
            n_steps=burn_in + n_samples,
        )
        samples = np.asarray(single_trajectory.states[burn_in + 1 :], dtype=np.float64)
        x_iat = empirical_integrated_autocorrelation_time(samples[:, 0])
        effective = n_samples / x_iat
        target_evaluations = _total_target_evaluations(counter)
        rows.append(
            RosenbrockExperimentRow(
                method=method,
                acceptance_rate=float(single_trajectory.acceptance_rate or 0.0),
                mean_x=float(np.mean(samples[:, 0])),
                mean_y=float(np.mean(samples[:, 1])),
                exact_mean_x=float(exact_mean[0]),
                exact_mean_y=float(exact_mean[1]),
                x_iat=x_iat,
                effective_sample_size=effective,
                target_evaluations=target_evaluations,
                ess_per_thousand_target_evaluations=(
                    float("inf")
                    if target_evaluations == 0
                    else 1000.0 * effective / target_evaluations
                ),
            )
        )

    for method, kernel_builder in (
        (
            "stretch ensemble",
            lambda counter: StretchMoveKernel(
                target,
                schedule="split",
                counter=counter,
            ),
        ),
        (
            "walk ensemble",
            lambda counter: WalkMoveKernel(
                target,
                schedule="split",
                subset_size=min(6, n_walkers // 2),
                scale=0.2,
                counter=counter,
            ),
        ),
    ):
        rng = next(streams)
        counter = OperationCounter()
        initial_walkers = target.sample_exact(rng, n_walkers)
        initial_state = EnsembleState.from_target(initial_walkers, target)
        ensemble_trajectory = run_ensemble_chain(
            kernel_builder(counter),
            initial_state,
            rng,
            n_steps=burn_in + n_samples,
        )
        retained = np.asarray(
            ensemble_trajectory.walkers[burn_in + 1 :],
            dtype=np.float64,
        )
        efficiency = ensemble_effective_sample_size(retained[:, :, 0])
        samples = retained.reshape(-1, 2)
        target_evaluations = _total_target_evaluations(counter)
        rows.append(
            RosenbrockExperimentRow(
                method=method,
                acceptance_rate=ensemble_trajectory.acceptance_rate,
                mean_x=float(np.mean(samples[:, 0])),
                mean_y=float(np.mean(samples[:, 1])),
                exact_mean_x=float(exact_mean[0]),
                exact_mean_y=float(exact_mean[1]),
                x_iat=efficiency.integrated_autocorrelation_time,
                effective_sample_size=efficiency.effective_sample_size,
                target_evaluations=target_evaluations,
                ess_per_thousand_target_evaluations=(
                    float("inf")
                    if target_evaluations == 0
                    else 1000.0 * efficiency.effective_sample_size / target_evaluations
                ),
            )
        )
    return rows


def geometry_main() -> None:
    """Run conditioned-Gaussian and Rosenbrock geometry experiments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-numbers", type=float, nargs="+", default=[1, 10, 100])
    parser.add_argument("--samples", type=int, default=3_000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--walkers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    conditioning_rows = run_conditioning_geometry_experiment(
        condition_numbers=tuple(args.condition_numbers),
        n_samples=args.samples,
        burn_in=args.burn_in,
        seed=args.seed,
    )
    print("Conditioned Gaussian geometry comparison")
    print(
        f"{'kappa':>8}  {'method':>22}  {'accept':>8}  {'IAT':>9}  "
        f"{'variance':>10}  {'exact':>10}  {'ESS/1k eval':>12}"
    )
    for conditioning_row in conditioning_rows:
        print(
            f"{conditioning_row.condition_number:8.1f}  "
            f"{conditioning_row.method:>22}  "
            f"{conditioning_row.acceptance_rate:8.4f}  "
            f"{conditioning_row.empirical_iat:9.3f}  "
            f"{conditioning_row.empirical_variance:10.4f}  "
            f"{conditioning_row.exact_variance:10.4f}  "
            f"{conditioning_row.ess_per_thousand_target_evaluations:12.3f}"
        )

    rosenbrock_rows = run_rosenbrock_geometry_experiment(
        n_samples=args.samples,
        burn_in=args.burn_in,
        n_walkers=args.walkers,
        seed=args.seed + 1,
    )
    print("\nRosenbrock comparison")
    print(
        f"{'method':>24}  {'accept':>8}  {'mean x':>10}  {'mean y':>10}  "
        f"{'IAT x':>9}  {'ESS':>10}  {'ESS/1k eval':>12}"
    )
    for rosenbrock_row in rosenbrock_rows:
        print(
            f"{rosenbrock_row.method:>24}  {rosenbrock_row.acceptance_rate:8.4f}  "
            f"{rosenbrock_row.mean_x:10.4f}  {rosenbrock_row.mean_y:10.4f}  "
            f"{rosenbrock_row.x_iat:9.3f}  "
            f"{rosenbrock_row.effective_sample_size:10.1f}  "
            f"{rosenbrock_row.ess_per_thousand_target_evaluations:12.3f}"
        )
