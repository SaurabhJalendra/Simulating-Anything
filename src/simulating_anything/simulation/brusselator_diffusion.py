"""1D spatial Brusselator (reaction-diffusion PDE) simulation.

Extends the well-mixed Brusselator ODE to 1D space with diffusion:

    du/dt = D_u * u_xx + a - (b+1)*u + u^2*v
    dv/dt = D_v * v_xx + b*u - u^2*v

with periodic boundary conditions on [0, L].

Solver: operator splitting -- spectral (FFT) exact diffusion + explicit Euler
for reaction terms. Periodic boundary conditions are enforced automatically
by the FFT.

Target rediscoveries:
- Turing instability: patterns form when D_v/D_u large enough and b > 1+a^2
- Turing wavelength: lambda_c ~ 2*pi*sqrt(D_u*D_v) / sqrt(b-1-a^2)
- Homogeneous steady state: (u*, v*) = (a, b/a)
- Without diffusion, reduces to classical Brusselator ODE at each point
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BrusselatorDiffusion(SimulationEnvironment):
    """1D spatial Brusselator with periodic boundary conditions.

    State: chemical concentrations u(x) and v(x) on a 1D grid.
    Observe shape: (2*N,) = [u_1..u_N, v_1..v_N].

    Parameters:
        a: production rate (default 1.0)
        b: control parameter (default 3.0, above Turing threshold for a=1)
        D_u: activator diffusion coefficient (default 0.01)
        D_v: inhibitor diffusion coefficient (default 0.1)
        N_grid: number of spatial grid points (default 128)
        L_domain: domain length (default 20.0)
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
        self.N = int(p.get("N_grid", 128))
        self.L = p.get("L_domain", 20.0)
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers for spectral Laplacian (periodic domain)
        self.k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        self.k_sq = self.k ** 2

        # CFL stability check: dt < dx^2 / (4 * max(D_u, D_v))
        D_max = max(self.D_u, self.D_v)
        if D_max > 0:
            dt_cfl = self.dx ** 2 / (4.0 * D_max)
            if config.dt > dt_cfl:
                raise ValueError(
                    f"dt={config.dt} exceeds CFL limit {dt_cfl:.6f} "
                    f"for dx={self.dx:.4f}, D_max={D_max}. "
                    f"Reduce dt or increase N_grid."
                )

        # Initial state placeholders
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None

    @property
    def fixed_point(self) -> tuple[float, float]:
        """The homogeneous steady state (u*, v*) = (a, b/a)."""
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
        """Total u concentration (integral of u over domain)."""
        return float(np.sum(self._u) * self.dx)

    @property
    def total_v(self) -> float:
        """Total v concentration (integral of v over domain)."""
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
        return self._spatial_heterogeneity(self._u)

    @property
    def spatial_heterogeneity_v(self) -> float:
        """Coefficient of variation of v (std/mean). Higher = more pattern."""
        return self._spatial_heterogeneity(self._v)

    @property
    def u_field(self) -> np.ndarray:
        """Activator concentration field u(x)."""
        return self._u.copy()

    @property
    def v_field(self) -> np.ndarray:
        """Inhibitor concentration field v(x)."""
        return self._v.copy()

    def _spatial_heterogeneity(self, field: np.ndarray) -> float:
        """Coefficient of variation of a field (std/mean)."""
        mean = np.mean(field)
        if mean < 1e-15:
            return 0.0
        return float(np.std(field) / mean)

    def dominant_wavelength(self) -> float:
        """Measure the dominant spatial wavelength of u via FFT.

        Returns the wavelength corresponding to the largest-amplitude
        Fourier mode (excluding mode 0).
        """
        u_hat = np.fft.fft(self._u)
        # Exclude DC component (mode 0)
        power = np.abs(u_hat[1:self.N // 2]) ** 2
        if np.max(power) < 1e-30:
            return float("inf")
        dominant_mode = np.argmax(power) + 1  # +1 because we excluded mode 0
        wavelength = self.L / dominant_mode
        return float(wavelength)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize u and v near the homogeneous steady state with perturbation.

        u is initialized near a with small random perturbation to seed
        pattern formation. v is initialized near b/a.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        u_star, v_star = self.fixed_point

        self._u = np.full(self.N, u_star, dtype=np.float64)
        self._v = np.full(self.N, v_star, dtype=np.float64)

        # Add small spatial perturbation to seed pattern formation
        self._u += 0.05 * rng.standard_normal(self.N)
        self._v += 0.05 * rng.standard_normal(self.N)

        # Ensure positive concentrations
        self._u = np.maximum(self._u, 0.01)
        self._v = np.maximum(self._v, 0.01)

        self._step_count = 0
        self._state = np.concatenate([self._u, self._v])
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep: spectral diffusion + explicit Euler reaction.

        Operator splitting:
        1. Exact diffusion in Fourier space: u_hat *= exp(-D*k^2*dt)
        2. Reaction step in physical space: Euler on Brusselator kinetics
        """
        dt = self.config.dt
        u = self._u
        v = self._v

        # Step 1: Spectral diffusion (exact in Fourier space)
        u_hat = np.fft.fft(u)
        v_hat = np.fft.fft(v)

        u_hat *= np.exp(-self.D_u * self.k_sq * dt)
        v_hat *= np.exp(-self.D_v * self.k_sq * dt)

        u_diffused = np.real(np.fft.ifft(u_hat))
        v_diffused = np.real(np.fft.ifft(v_hat))

        # Step 2: Reaction terms (explicit Euler on diffused state)
        # du/dt = a - (b+1)*u + u^2*v
        # dv/dt = b*u - u^2*v
        u2v = u_diffused ** 2 * v_diffused
        du_react = self.a - (self.b + 1.0) * u_diffused + u2v
        dv_react = self.b * u_diffused - u2v

        u_new = u_diffused + dt * du_react
        v_new = v_diffused + dt * dv_react

        # Ensure positive concentrations
        self._u = np.maximum(u_new, 0.0)
        self._v = np.maximum(v_new, 0.0)

        self._step_count += 1
        self._state = np.concatenate([self._u, self._v])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state: [u_1..u_N, v_1..v_N]."""
        return self._state
