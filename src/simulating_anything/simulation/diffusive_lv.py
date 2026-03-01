"""1D diffusive Lotka-Volterra (reaction-diffusion predator-prey) simulation.

Combines Lotka-Volterra population dynamics with spatial diffusion on a 1D
periodic domain [0, L]:

    du/dt = D_u * u_xx + alpha*u - beta*u*v   (prey)
    dv/dt = D_v * v_xx - gamma*v + delta*u*v   (predator)

Solver: spectral (FFT) Laplacian + explicit Euler for reaction terms.
Periodic boundary conditions are enforced automatically by the FFT.

Target rediscoveries:
- Traveling wave speed depends on diffusion coefficients
- Spatial heterogeneity emerges from diffusion-driven instability
- Without diffusion, reduces to classical Lotka-Volterra ODE at each point
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DiffusiveLotkaVolterra(SimulationEnvironment):
    """1D diffusive predator-prey model with periodic boundary conditions.

    State: prey u(x) and predator v(x) concentrations on a 1D grid.
    Observe shape: (2*N,) = [u_1..u_N, v_1..v_N].

    Parameters:
        alpha: prey birth rate (default 1.0)
        beta: predation rate (default 0.5)
        gamma: predator death rate (default 0.5)
        delta: predator conversion efficiency (default 0.2)
        D_u: prey diffusion coefficient (default 0.1)
        D_v: predator diffusion coefficient (default 0.05)
        N_grid: number of spatial grid points (default 64)
        L_domain: domain length (default 20.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Reaction parameters
        self.alpha = p.get("alpha", 1.0)
        self.beta = p.get("beta", 0.5)
        self.gamma = p.get("gamma", 0.5)
        self.delta = p.get("delta", 0.2)

        # Diffusion coefficients
        self.D_u = p.get("D_u", 0.1)
        self.D_v = p.get("D_v", 0.05)

        # Spatial grid
        self.N = int(p.get("N_grid", 64))
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

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize prey and predator concentrations.

        Prey: uniform near equilibrium + small random perturbation.
        Predator: uniform near equilibrium.
        Equilibrium: u* = gamma/delta, v* = alpha/beta.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        u_eq = self.gamma / self.delta if self.delta > 0 else 1.0
        v_eq = self.alpha / self.beta if self.beta > 0 else 1.0

        self._u = np.full(self.N, u_eq, dtype=np.float64)
        self._v = np.full(self.N, v_eq, dtype=np.float64)

        # Add small spatial perturbation to prey to seed pattern formation
        self._u += 0.1 * u_eq * rng.standard_normal(self.N)
        self._v += 0.05 * v_eq * rng.standard_normal(self.N)

        # Ensure non-negative concentrations
        self._u = np.maximum(self._u, 0.0)
        self._v = np.maximum(self._v, 0.0)

        self._step_count = 0
        self._state = np.concatenate([self._u, self._v])
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep: spectral diffusion + explicit Euler reaction."""
        dt = self.config.dt
        u = self._u
        v = self._v

        # Spectral diffusion: apply exact diffusion in Fourier space
        u_hat = np.fft.fft(u)
        v_hat = np.fft.fft(v)

        u_hat *= np.exp(-self.D_u * self.k_sq * dt)
        v_hat *= np.exp(-self.D_v * self.k_sq * dt)

        u_diffused = np.real(np.fft.ifft(u_hat))
        v_diffused = np.real(np.fft.ifft(v_hat))

        # Reaction terms (explicit Euler on the diffused state)
        uv = u_diffused * v_diffused
        du_react = self.alpha * u_diffused - self.beta * uv
        dv_react = -self.gamma * v_diffused + self.delta * uv

        u_new = u_diffused + dt * du_react
        v_new = v_diffused + dt * dv_react

        # Ensure non-negative concentrations
        self._u = np.maximum(u_new, 0.0)
        self._v = np.maximum(v_new, 0.0)

        self._step_count += 1
        self._state = np.concatenate([self._u, self._v])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state: [u_1..u_N, v_1..v_N]."""
        return self._state

    @property
    def total_prey(self) -> float:
        """Total prey biomass (integral of u over domain)."""
        return float(np.sum(self._u) * self.dx)

    @property
    def total_predator(self) -> float:
        """Total predator biomass (integral of v over domain)."""
        return float(np.sum(self._v) * self.dx)

    @property
    def total_biomass(self) -> float:
        """Total biomass (prey + predator)."""
        return self.total_prey + self.total_predator

    @property
    def prey_field(self) -> np.ndarray:
        """Prey concentration field u(x)."""
        return self._u.copy()

    @property
    def predator_field(self) -> np.ndarray:
        """Predator concentration field v(x)."""
        return self._v.copy()

    def spatial_heterogeneity(self, field: np.ndarray) -> float:
        """Coefficient of variation of a field (std/mean).

        Higher values indicate more spatial structure.
        """
        mean = np.mean(field)
        if mean < 1e-15:
            return 0.0
        return float(np.std(field) / mean)
