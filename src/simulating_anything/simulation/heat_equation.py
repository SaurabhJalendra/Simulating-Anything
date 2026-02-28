"""1D heat equation simulation.

Target rediscoveries:
- Diffusion: u_t = D * u_xx
- Analytical: Gaussian spreading u(x,t) = (1/sqrt(4*pi*D*t)) * exp(-x^2/(4*D*t))
- Decay rate of Fourier modes: a_k(t) = a_k(0) * exp(-D*k^2*t)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class HeatEquation1DSimulation(SimulationEnvironment):
    """1D heat equation: u_t = D * u_xx with periodic boundary conditions.

    State: u(x) on a 1D grid.
    Solver: spectral method (FFT) for exact linear diffusion.

    Parameters:
        D: diffusion coefficient (default 0.1)
        N: number of grid points (default 128)
        L: domain length (default 2*pi)
        init_type: 'gaussian', 'sine', 'step' (set via attribute)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.D = p.get("D", 0.1)
        self.N = int(p.get("N", 128))
        self.L = p.get("L", 2 * np.pi)
        self.init_type = "gaussian"

        # Grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers for spectral solver
        self.k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        self.k_sq = self.k**2

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize temperature profile."""
        if self.init_type == "sine":
            self._state = np.sin(2 * np.pi * self.x / self.L)
        elif self.init_type == "step":
            self._state = np.where(
                (self.x > self.L / 4) & (self.x < 3 * self.L / 4), 1.0, 0.0
            )
        else:  # gaussian
            center = self.L / 2
            sigma = self.L / 20
            self._state = np.exp(-((self.x - center) ** 2) / (2 * sigma**2))

        self._state = self._state.astype(np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using exact spectral method."""
        dt = self.config.dt
        u_hat = np.fft.fft(self._state)
        # Exact solution for each mode: u_hat_k(t+dt) = u_hat_k(t) * exp(-D*k^2*dt)
        u_hat *= np.exp(-self.D * self.k_sq * dt)
        self._state = np.real(np.fft.ifft(u_hat))
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current temperature profile."""
        return self._state

    @property
    def total_heat(self) -> float:
        """Total heat content (integral of u)."""
        return float(np.sum(self._state) * self.dx)

    @property
    def max_temperature(self) -> float:
        """Maximum temperature."""
        return float(np.max(self._state))

    def decay_rate_of_mode(self, mode: int) -> float:
        """Theoretical decay rate for Fourier mode k: D * (2*pi*mode/L)^2."""
        k_mode = 2 * np.pi * mode / self.L
        return self.D * k_mode**2
