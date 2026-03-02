"""1D Brusselator reaction-diffusion PDE simulation (finite differences).

PDE system on periodic domain [0, L]:

    du/dt = D_u * u_xx + a - (b+1)*u + u^2*v
    dv/dt = D_v * v_xx + b*u - u^2*v

Solver: explicit Euler with finite-difference Laplacian (periodic BC).

Homogeneous steady state: (u*, v*) = (a, b/a).

Turing instability requires:
    1. b > 1 + a^2 (necessary for diffusion-driven instability)
    2. D_v/D_u sufficiently large (inhibitor diffuses faster than activator)

The critical Turing wavelength scales as:
    lambda_c ~ 2*pi * sqrt(D_u * D_v) / sqrt(b - 1 - a^2)

Default parameters: a=1.0, b=3.0, D_u=0.01, D_v=0.1, L=20.0, N=256.
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


def _laplacian_1d_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    """Compute 1D Laplacian with periodic boundary conditions (finite diff)."""
    return (np.roll(field, 1) + np.roll(field, -1) - 2.0 * field) / (dx * dx)


class Brusselator1DSimulation(SimulationEnvironment):
    """1D Brusselator reaction-diffusion PDE with finite-difference Laplacian.

    State vector: [u_0, ..., u_{N-1}, v_0, ..., v_{N-1}] of shape (2*N,).

    Parameters:
        a: production rate (default 1.0)
        b: control parameter (default 3.0, above Turing threshold for a=1)
        D_u: activator diffusion coefficient (default 0.01)
        D_v: inhibitor diffusion coefficient (default 0.1)
        N: number of spatial grid points (default 256)
        L: domain length (default 20.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Reaction parameters
        self.a = p.get("a", 1.0)
        self.b = p.get("b", 3.0)

        # Diffusion coefficients
        self.D_u = p.get("D_u", 0.01)
        self.D_v = p.get("D_v", 0.1)

        # Spatial grid
        self.N = int(p.get("N", 256))
        self.L = p.get("L", 20.0)
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # CFL stability check: dt < dx^2 / (2 * max(D_u, D_v))
        D_max = max(self.D_u, self.D_v)
        if D_max > 0:
            self.cfl_limit = self.dx ** 2 / (2.0 * D_max)
            if config.dt > self.cfl_limit:
                raise ValueError(
                    f"dt={config.dt} exceeds CFL limit {self.cfl_limit:.6f} "
                    f"for dx={self.dx:.4f}, D_max={D_max}. "
                    f"Reduce dt or increase N."
                )
        else:
            self.cfl_limit = float("inf")

        # Internal fields (initialized in reset)
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None

    @property
    def fixed_point(self) -> tuple[float, float]:
        """Homogeneous steady state (u*, v*) = (a, b/a)."""
        if self.a == 0:
            return (0.0, 0.0)
        return (self.a, self.b / self.a)

    @property
    def turing_threshold(self) -> float:
        """Critical b for Turing instability: b_c = 1 + a^2 (necessary condition).

        The full Turing condition also requires a sufficient diffusion ratio
        D_v/D_u, but b > 1 + a^2 is necessary for the homogeneous state
        to be unstable to spatial perturbations.
        """
        return 1.0 + self.a ** 2

    @property
    def is_turing_unstable(self) -> bool:
        """True if b > 1 + a^2 (necessary condition for pattern formation)."""
        return self.b > self.turing_threshold

    @property
    def total_u(self) -> float:
        """Total u concentration (integral over domain)."""
        return float(np.sum(self._u) * self.dx)

    @property
    def total_v(self) -> float:
        """Total v concentration (integral over domain)."""
        return float(np.sum(self._v) * self.dx)

    @property
    def mean_u(self) -> float:
        """Spatial mean of u."""
        return float(np.mean(self._u))

    @property
    def mean_v(self) -> float:
        """Spatial mean of v."""
        return float(np.mean(self._v))

    @property
    def spatial_heterogeneity_u(self) -> float:
        """Coefficient of variation of u (std/mean). Higher = more pattern."""
        return self._spatial_cv(self._u)

    @property
    def spatial_heterogeneity_v(self) -> float:
        """Coefficient of variation of v (std/mean). Higher = more pattern."""
        return self._spatial_cv(self._v)

    @property
    def u_field(self) -> np.ndarray:
        """Copy of activator concentration field u(x)."""
        return self._u.copy()

    @property
    def v_field(self) -> np.ndarray:
        """Copy of inhibitor concentration field v(x)."""
        return self._v.copy()

    def _spatial_cv(self, field: np.ndarray) -> float:
        """Coefficient of variation (std / mean) of a spatial field."""
        mean = np.mean(field)
        if mean < 1e-15:
            return 0.0
        return float(np.std(field) / mean)

    def dominant_wavelength(self) -> float:
        """Measure the dominant spatial wavelength of u via FFT.

        Returns the wavelength corresponding to the largest-amplitude
        Fourier mode (excluding the DC component, mode 0).
        """
        u_hat = np.fft.fft(self._u)
        power = np.abs(u_hat[1:self.N // 2]) ** 2
        if np.max(power) < 1e-30:
            return float("inf")
        dominant_mode = np.argmax(power) + 1  # +1 because we excluded mode 0
        return float(self.L / dominant_mode)

    def count_peaks(self, field: str = "u", threshold: float | None = None) -> int:
        """Count spatial peaks (local maxima) in a field.

        Args:
            field: "u" or "v".
            threshold: Minimum value for a peak. If None, uses 50% of max.

        Returns:
            Number of detected peaks.
        """
        f = self._u if field == "u" else self._v
        if threshold is None:
            threshold = 0.5 * np.max(f)
        left = np.roll(f, 1)
        right = np.roll(f, -1)
        is_peak = (f > left) & (f > right) & (f > threshold)
        return int(np.sum(is_peak))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize near the homogeneous steady state with small perturbation.

        u is set to a + noise, v is set to b/a + noise. The perturbation
        seeds pattern formation when the system is Turing-unstable.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        u_star, v_star = self.fixed_point

        self._u = np.full(self.N, u_star, dtype=np.float64)
        self._v = np.full(self.N, v_star, dtype=np.float64)

        # Small random perturbation to seed instability
        self._u += 0.05 * rng.standard_normal(self.N)
        self._v += 0.05 * rng.standard_normal(self.N)

        # Ensure positive concentrations
        self._u = np.maximum(self._u, 0.01)
        self._v = np.maximum(self._v, 0.01)

        self._step_count = 0
        self._state = np.concatenate([self._u, self._v])
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep: finite-difference diffusion + Euler reaction.

        Brusselator kinetics:
            du/dt = D_u * u_xx + a - (b+1)*u + u^2*v
            dv/dt = D_v * v_xx + b*u - u^2*v
        """
        dt = self.config.dt

        # Diffusion via finite differences (periodic BC)
        lap_u = _laplacian_1d_periodic(self._u, self.dx)
        lap_v = _laplacian_1d_periodic(self._v, self.dx)

        # Reaction terms
        u2v = self._u ** 2 * self._v
        du = self.D_u * lap_u + self.a - (self.b + 1.0) * self._u + u2v
        dv = self.D_v * lap_v + self.b * self._u - u2v

        # Euler update
        self._u = self._u + dt * du
        self._v = self._v + dt * dv

        # Clamp to non-negative
        self._u = np.maximum(self._u, 0.0)
        self._v = np.maximum(self._v, 0.0)

        self._step_count += 1
        self._state = np.concatenate([self._u, self._v])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state: [u_0..u_{N-1}, v_0..v_{N-1}]."""
        return self._state
