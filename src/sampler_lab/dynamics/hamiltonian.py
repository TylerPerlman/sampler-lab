"""Hamiltonian phase-space geometry, mass matrices, and Gaussian analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.linalg import as_positive_definite
from sampler_lab.core.protocols import DifferentiableLogDensity
from sampler_lab.models.gaussian import GaussianTarget

Array = NDArray[np.float64]


def _as_vector(value: ArrayLike, *, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True, slots=True, init=False)
class PhaseSpaceState:
    """Immutable position--momentum state with matching vector dimensions."""

    position: Array
    momentum: Array

    def __init__(self, position: ArrayLike, momentum: ArrayLike) -> None:
        q = _as_vector(position, name="position")
        p = _as_vector(momentum, name="momentum")
        if q.shape != p.shape:
            raise ValueError("position and momentum must have the same shape")
        q_copy = np.array(q, dtype=np.float64, copy=True)
        p_copy = np.array(p, dtype=np.float64, copy=True)
        q_copy.setflags(write=False)
        p_copy.setflags(write=False)
        object.__setattr__(self, "position", q_copy)
        object.__setattr__(self, "momentum", p_copy)

    @property
    def dimension(self) -> int:
        return int(self.position.size)

    def as_array(self) -> Array:
        """Return ``[q, p]`` as a detached phase-space vector."""

        return np.concatenate((self.position, self.momentum)).astype(np.float64, copy=False)

    @classmethod
    def from_array(cls, state: ArrayLike) -> PhaseSpaceState:
        """Split an even-length phase vector into position and momentum."""

        array = _as_vector(state, name="phase state")
        if array.size % 2 != 0:
            raise ValueError("phase state must have even length")
        dimension = array.size // 2
        return cls(array[:dimension], array[dimension:])


@dataclass(slots=True)
class MassMatrix:
    """Constant positive-definite momentum covariance ``M``.

    The kinetic energy is ``K(p) = p^T M^{-1} p / 2`` and momenta are sampled
    from ``N(0, M)``.  For a Gaussian target with precision ``Kappa``, choosing
    ``M = Kappa`` equalizes Hamiltonian frequencies; this convention is worth
    stating because some software exposes the inverse mass instead.
    """

    matrix: ArrayLike
    _matrix: Array = field(init=False, repr=False)
    _inverse: Array = field(init=False, repr=False)
    _cholesky: Array = field(init=False, repr=False)
    _log_determinant: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        matrix = np.array(as_positive_definite(self.matrix), dtype=np.float64, copy=True)
        cholesky = np.asarray(np.linalg.cholesky(matrix), dtype=np.float64)
        inverse = np.asarray(np.linalg.solve(matrix, np.eye(matrix.shape[0])), dtype=np.float64)
        self._matrix = matrix
        self._inverse = inverse
        self._cholesky = cholesky
        self._log_determinant = float(2.0 * np.sum(np.log(np.diag(cholesky))))

    @classmethod
    def identity(cls, dimension: int) -> MassMatrix:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        return cls(np.eye(dimension, dtype=np.float64))

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def covariance(self) -> Array:
        return self._matrix.copy()

    @property
    def inverse(self) -> Array:
        return self._inverse.copy()

    @property
    def cholesky(self) -> Array:
        return self._cholesky.copy()

    @property
    def log_determinant(self) -> float:
        return self._log_determinant

    def _check_momentum(self, momentum: ArrayLike) -> Array:
        value = _as_vector(momentum, name="momentum")
        if value.shape != (self.dimension,):
            raise ValueError("momentum dimension does not match the mass matrix")
        return value

    def kinetic_energy(self, momentum: ArrayLike) -> float:
        p = self._check_momentum(momentum)
        return float(0.5 * p @ self._inverse @ p)

    def velocity(self, momentum: ArrayLike) -> Array:
        p = self._check_momentum(momentum)
        return np.asarray(self._inverse @ p, dtype=np.float64)

    def sample_momentum(
        self,
        rng: np.random.Generator,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        if counter is not None:
            counter.normal_draws += self.dimension
        return np.asarray(self._cholesky @ rng.normal(size=self.dimension), dtype=np.float64)


@dataclass(slots=True)
class HamiltonianSystem:
    """Separable Hamiltonian ``H(q,p) = -log pi(q) + K(p)``."""

    target: DifferentiableLogDensity
    mass: MassMatrix
    counter: OperationCounter | None = None

    def _check_position(self, position: ArrayLike) -> Array:
        q = _as_vector(position, name="position")
        if q.shape != (self.mass.dimension,):
            raise ValueError("position dimension does not match the mass matrix")
        return q

    def potential_energy(self, position: ArrayLike) -> float:
        q = self._check_position(position)
        if self.counter is not None:
            self.counter.log_density_evaluations += 1
        log_density = float(self.target.log_prob(np.array(q, copy=True)))
        if np.isnan(log_density) or log_density == float("inf"):
            raise ValueError("target log density must be finite or -inf")
        return float(-log_density)

    def potential_gradient(self, position: ArrayLike) -> Array:
        q = self._check_position(position)
        if self.counter is not None:
            self.counter.gradient_evaluations += 1
        gradient = np.asarray(self.target.grad_log_prob(np.array(q, copy=True)), dtype=np.float64)
        if gradient.shape != q.shape or not np.all(np.isfinite(gradient)):
            raise ValueError("target gradient must be a finite vector matching position")
        return -gradient

    def log_density_gradient(self, position: ArrayLike) -> Array:
        """Return ``grad log pi(q)`` with shared validation and accounting."""

        return -self.potential_gradient(position)

    def energy(self, state: PhaseSpaceState) -> float:
        if state.dimension != self.mass.dimension:
            raise ValueError("phase-state dimension does not match the mass matrix")
        return self.potential_energy(state.position) + self.mass.kinetic_energy(state.momentum)

    def vector_field(self, state: PhaseSpaceState) -> PhaseSpaceState:
        """Return canonical ``(q_dot, p_dot)`` at one phase point."""

        if state.dimension != self.mass.dimension:
            raise ValueError("phase-state dimension does not match the mass matrix")
        return PhaseSpaceState(
            self.mass.velocity(state.momentum),
            self.log_density_gradient(state.position),
        )


@dataclass(frozen=True, slots=True)
class SkewMatrixEvaluation:
    """One evaluation of a skew field and its row-wise divergence."""

    matrix: Array
    divergence: Array


class SkewMatrixField(Protocol):
    """Skew-symmetric matrix field used in conservative nonreversible flows."""

    def evaluate_at(self, state: Array) -> SkewMatrixEvaluation:
        """Return ``A(x)`` and ``div A(x)``."""


@dataclass(frozen=True, slots=True)
class ConstantSkewMatrix:
    """Constant skew-symmetric matrix field with zero divergence."""

    matrix: ArrayLike

    def evaluate_at(self, state: Array) -> SkewMatrixEvaluation:
        x = _as_vector(state, name="state")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (x.size, x.size) or not np.all(np.isfinite(matrix)):
            raise ValueError("skew matrix must be finite and match the state dimension")
        if not np.allclose(matrix, -matrix.T, atol=1e-12, rtol=0.0):
            raise ValueError("matrix must be skew-symmetric")
        return SkewMatrixEvaluation(
            matrix=np.array(matrix, dtype=np.float64, copy=True),
            divergence=np.zeros(x.size, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class FunctionalSkewMatrix:
    """Position-dependent skew field with an explicit row divergence."""

    matrix_function: Callable[[Array], ArrayLike]
    divergence_function: Callable[[Array], ArrayLike]

    def evaluate_at(self, state: Array) -> SkewMatrixEvaluation:
        x = _as_vector(state, name="state")
        matrix = np.asarray(self.matrix_function(np.array(x, copy=True)), dtype=np.float64)
        divergence = np.asarray(self.divergence_function(np.array(x, copy=True)), dtype=np.float64)
        if matrix.shape != (x.size, x.size) or not np.all(np.isfinite(matrix)):
            raise ValueError("skew matrix must be finite and match the state dimension")
        if not np.allclose(matrix, -matrix.T, atol=1e-12, rtol=0.0):
            raise ValueError("matrix must be skew-symmetric")
        if divergence.shape != x.shape or not np.all(np.isfinite(divergence)):
            raise ValueError("skew-field divergence must match the finite state vector")
        return SkewMatrixEvaluation(matrix=matrix, divergence=divergence)


def conservative_skew_drift(
    state: ArrayLike,
    energy_gradient: ArrayLike,
    skew_field: SkewMatrixField,
) -> Array:
    """Return ``A grad(H) - div(A)`` for a skew-symmetric field ``A``.

    Under ordinary boundary conditions this drift preserves density proportional
    to ``exp(-H)``.  The divergence correction vanishes for the canonical constant
    symplectic matrix used by ordinary Hamiltonian mechanics.
    """

    x = _as_vector(state, name="state")
    gradient = _as_vector(energy_gradient, name="energy_gradient")
    if gradient.shape != x.shape:
        raise ValueError("energy_gradient must match the state")
    evaluation = skew_field.evaluate_at(x)
    return np.asarray(evaluation.matrix @ gradient - evaluation.divergence, dtype=np.float64)


def canonical_symplectic_matrix(dimension: int) -> Array:
    """Return the canonical ``2d x 2d`` symplectic matrix."""

    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    identity = np.eye(dimension, dtype=np.float64)
    zeros = np.zeros_like(identity)
    return np.block([[zeros, identity], [-identity, zeros]]).astype(np.float64)


def finite_difference_jacobian(
    mapping: Callable[[Array], ArrayLike],
    state: ArrayLike,
    *,
    epsilon: float = 1e-6,
) -> Array:
    """Central finite-difference Jacobian for deterministic-map diagnostics."""

    x = _as_vector(state, name="state")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    baseline = np.asarray(mapping(np.array(x, copy=True)), dtype=np.float64)
    if baseline.shape != x.shape or not np.all(np.isfinite(baseline)):
        raise ValueError("mapping must return a finite vector with unchanged shape")
    jacobian = np.empty((x.size, x.size), dtype=np.float64)
    for index in range(x.size):
        direction = np.zeros_like(x)
        direction[index] = epsilon
        forward = np.asarray(mapping(x + direction), dtype=np.float64)
        backward = np.asarray(mapping(x - direction), dtype=np.float64)
        if forward.shape != x.shape or backward.shape != x.shape:
            raise ValueError("mapping changed shape during Jacobian evaluation")
        jacobian[:, index] = (forward - backward) / (2.0 * epsilon)
    return jacobian


def volume_preservation_error(jacobian: ArrayLike) -> float:
    """Return ``abs(log |det J|)``; zero characterizes local volume preservation."""

    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("jacobian must be square")
    sign, log_abs_determinant = np.linalg.slogdet(matrix)
    if sign == 0.0:
        return float("inf")
    return float(abs(log_abs_determinant))


def symplecticity_error(jacobian: ArrayLike) -> float:
    """Frobenius residual of ``J.T Omega J = Omega``."""

    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2:
        raise ValueError("jacobian must be square with even dimension")
    omega = canonical_symplectic_matrix(matrix.shape[0] // 2)
    return float(np.linalg.norm(matrix.T @ omega @ matrix - omega, ord="fro"))


def _symmetric_matrix_square_roots(matrix: Array) -> tuple[Array, Array]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("matrix must be positive definite")
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return np.asarray(root, dtype=np.float64), np.asarray(inverse_root, dtype=np.float64)


def gaussian_hamiltonian_frequencies(
    target: GaussianTarget,
    mass: MassMatrix | ArrayLike | None = None,
) -> Array:
    """Return normal-mode frequencies for a Gaussian Hamiltonian system."""

    resolved = (
        MassMatrix.identity(target.dimension)
        if mass is None
        else (mass if isinstance(mass, MassMatrix) else MassMatrix(mass))
    )
    if resolved.dimension != target.dimension:
        raise ValueError("mass dimension must match the Gaussian target")
    _, inverse_root = _symmetric_matrix_square_roots(resolved.covariance)
    transformed_precision = inverse_root @ target.precision_matrix @ inverse_root
    eigenvalues = np.linalg.eigvalsh(transformed_precision)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Gaussian Hamiltonian frequencies must be positive")
    return np.asarray(np.sqrt(eigenvalues), dtype=np.float64)


def gaussian_leapfrog_matrix(
    target: GaussianTarget,
    step_size: float,
    n_steps: int = 1,
    *,
    mass: MassMatrix | ArrayLike | None = None,
) -> Array:
    """Exact linear phase map of leapfrog on a centered Gaussian target."""

    if not np.isfinite(step_size) or step_size == 0.0:
        raise ValueError("step_size must be finite and nonzero")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    resolved = (
        MassMatrix.identity(target.dimension)
        if mass is None
        else (mass if isinstance(mass, MassMatrix) else MassMatrix(mass))
    )
    if resolved.dimension != target.dimension:
        raise ValueError("mass dimension must match the Gaussian target")
    dimension = target.dimension
    identity = np.eye(dimension, dtype=np.float64)
    inverse_mass = resolved.inverse
    precision = target.precision_matrix
    h = float(step_size)
    a = identity - 0.5 * h * h * inverse_mass @ precision
    b = h * inverse_mass
    c = -h * precision + 0.25 * h**3 * precision @ inverse_mass @ precision
    d = identity - 0.5 * h * h * precision @ inverse_mass
    one_step = np.block([[a, b], [c, d]]).astype(np.float64)
    return np.asarray(np.linalg.matrix_power(one_step, n_steps), dtype=np.float64)


def gaussian_exact_flow_matrix(
    target: GaussianTarget,
    integration_time: float,
    *,
    mass: MassMatrix | ArrayLike | None = None,
) -> Array:
    """Exact Gaussian Hamiltonian flow matrix in centered phase coordinates."""

    if not np.isfinite(integration_time):
        raise ValueError("integration_time must be finite")
    resolved = (
        MassMatrix.identity(target.dimension)
        if mass is None
        else (mass if isinstance(mass, MassMatrix) else MassMatrix(mass))
    )
    if resolved.dimension != target.dimension:
        raise ValueError("mass dimension must match the Gaussian target")
    mass_root, inverse_mass_root = _symmetric_matrix_square_roots(resolved.covariance)
    transformed_precision = inverse_mass_root @ target.precision_matrix @ inverse_mass_root
    rates, eigenvectors = np.linalg.eigh(transformed_precision)
    frequencies = np.sqrt(rates)
    cosine = (eigenvectors * np.cos(frequencies * integration_time)) @ eigenvectors.T
    sine_over_frequency = (
        eigenvectors * (np.sin(frequencies * integration_time) / frequencies)
    ) @ eigenvectors.T
    frequency_sine = (
        eigenvectors * (frequencies * np.sin(frequencies * integration_time))
    ) @ eigenvectors.T
    qq = inverse_mass_root @ cosine @ mass_root
    qp = inverse_mass_root @ sine_over_frequency @ inverse_mass_root
    pq = -mass_root @ frequency_sine @ mass_root
    pp = mass_root @ cosine @ inverse_mass_root
    return np.block([[qq, qp], [pq, pp]]).astype(np.float64)


@dataclass(frozen=True, slots=True)
class GaussianHamiltonianAnalysis:
    """Frequency and leapfrog-stability summary for a Gaussian system."""

    frequencies: Array
    maximum_stable_step_size: float
    step_size: float
    n_steps: int
    leapfrog_spectral_radius: float
    stable: bool
    integration_time: float
    exact_position_transition: Array


def gaussian_hamiltonian_analysis(
    target: GaussianTarget,
    step_size: float,
    n_steps: int,
    *,
    mass: MassMatrix | ArrayLike | None = None,
) -> GaussianHamiltonianAnalysis:
    """Analyze leapfrog stability and the corresponding exact-flow HMC map."""

    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be positive and finite")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    frequencies = gaussian_hamiltonian_frequencies(target, mass)
    maximum = float(2.0 / np.max(frequencies))
    leapfrog = gaussian_leapfrog_matrix(target, step_size, n_steps, mass=mass)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(leapfrog))))
    integration_time = float(step_size * n_steps)
    exact = gaussian_exact_flow_matrix(target, integration_time, mass=mass)
    return GaussianHamiltonianAnalysis(
        frequencies=frequencies,
        maximum_stable_step_size=maximum,
        step_size=float(step_size),
        n_steps=n_steps,
        leapfrog_spectral_radius=spectral_radius,
        stable=bool(step_size < maximum and spectral_radius <= 1.0 + 1e-10),
        integration_time=integration_time,
        exact_position_transition=np.asarray(
            exact[: target.dimension, : target.dimension], dtype=np.float64
        ),
    )
