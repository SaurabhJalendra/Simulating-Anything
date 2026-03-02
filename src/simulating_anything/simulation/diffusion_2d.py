"""2D diffusion equation simulation.

du/dt = D * (d2u/dx2 + d2u/dy2) on periodic [0, L] x [0, L] domain.

Target rediscoveries:
- Mode (kx, ky) decays as exp(-D * (kx^2 + ky^2) * t)
- Total energy E = integral(u^2) dx dy decays monotonically
- Total heat integral(u) dx dy is conserved
- Exact analytical solution available via spectral method
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class Diffusion2DSimulation(SimulationEnvironment):
    """2D heat/diffusion equation: u_t = D * (u_xx + u_yy) with periodic BCs.

    State: u(x, y) on an N x N grid, flattened to N^2 vector.
    Solver: spectral method (FFT2) for exact linear diffusion.

    Parameters:
        D: diffusion coefficient (default 0.1)
        N_grid: number of grid points per side (default 64)
        L: domain side length (default 2*pi)
        init_type: 'gaussian', 'modes', 'step' (set via attribute)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.D = p.get("D", 0.1)
        self.N_grid = int(p.get("N_grid", 64))
        self.L = p.get("L", 2 * np.pi)
        self.init_type = "gaussian"

        # Grid spacing
        self.dx = self.L / self.N_grid
        self.dy = self.L / self.N_grid

        # Spatial grid
        x = np.linspace(0, self.L, self.N_grid, endpoint=False)
        y = np.linspace(0, self.L, self.N_grid, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

        # 2D wavenumbers for spectral solver
        kx = np.fft.fftfreq(self.N_grid, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N_grid, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self.k_sq = KX**2 + KY**2

        # Internal 2D state (reset initializes this)
        self._u: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize temperature profile on the 2D grid.

        Returns the state as a flattened N^2 vector.
        """
        if self.init_type == "modes":
            # Sum of a few Fourier modes
            self._u = (
                np.sin(2 * np.pi * self.X / self.L)
                * np.sin(2 * np.pi * self.Y / self.L)
                + 0.5 * np.sin(4 * np.pi * self.X / self.L)
                * np.cos(2 * np.pi * self.Y / self.L)
            )
        elif self.init_type == "step":
            # Square block in the center quarter
            cx, cy = self.L / 2, self.L / 2
            hw = self.L / 4
            self._u = np.where(
                (np.abs(self.X - cx) < hw) & (np.abs(self.Y - cy) < hw),
                1.0,
                0.0,
            )
        elif self.init_type == "single_mode":
            # Single Fourier mode (kx=1, ky=1) for exact verification
            self._u = np.sin(
                2 * np.pi * self.X / self.L
            ) * np.sin(
                2 * np.pi * self.Y / self.L
            )
        else:  # gaussian (default)
            cx, cy = self.L / 2, self.L / 2
            sigma = self.L / 10
            self._u = np.exp(
                -((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2 * sigma**2)
            )

        self._u = self._u.astype(np.float64)
        self._state = self._u.ravel()
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using exact spectral method.

        Each Fourier mode (kx, ky) evolves as:
            u_hat(kx, ky, t+dt) = u_hat(kx, ky, t) * exp(-D*(kx^2+ky^2)*dt)
        """
        dt = self.config.dt
        u_hat = np.fft.fft2(self._u)
        u_hat *= np.exp(-self.D * self.k_sq * dt)
        self._u = np.real(np.fft.ifft2(u_hat))
        self._state = self._u.ravel()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current temperature profile as flattened N^2 vector."""
        return self._state

    def observe_2d(self) -> np.ndarray:
        """Return current temperature profile as N x N array."""
        return self._u.copy()

    @property
    def total_heat(self) -> float:
        """Total heat content: integral(u) dx dy.

        Conserved quantity for diffusion with periodic BCs.
        """
        return float(np.sum(self._u) * self.dx * self.dy)

    @property
    def energy(self) -> float:
        """L2 energy: integral(u^2) dx dy.

        Monotonically decreasing for the diffusion equation.
        """
        return float(np.sum(self._u**2) * self.dx * self.dy)

    @property
    def max_temperature(self) -> float:
        """Maximum temperature on the grid."""
        return float(np.max(self._u))

    @property
    def min_temperature(self) -> float:
        """Minimum temperature on the grid."""
        return float(np.min(self._u))

    def decay_rate_of_mode(self, kx_mode: int, ky_mode: int) -> float:
        """Theoretical decay rate for Fourier mode (kx, ky).

        lambda = D * ((2*pi*kx_mode/L)^2 + (2*pi*ky_mode/L)^2)
        """
        kx = 2 * np.pi * kx_mode / self.L
        ky = 2 * np.pi * ky_mode / self.L
        return self.D * (kx**2 + ky**2)

    def mode_amplitude(self, kx_mode: int, ky_mode: int) -> float:
        """Current amplitude of Fourier mode (kx_mode, ky_mode).

        Returns the absolute value of the FFT coefficient at the given mode.
        """
        u_hat = np.fft.fft2(self._u)
        return float(np.abs(u_hat[kx_mode, ky_mode]))

    def dominant_mode(self) -> tuple[int, int, float]:
        """Find the dominant (largest amplitude) non-zero Fourier mode.

        Returns:
            (kx_idx, ky_idx, amplitude) of the dominant mode.
        """
        u_hat = np.fft.fft2(self._u)
        amp = np.abs(u_hat)
        # Zero out the DC component to find the dominant non-zero mode
        amp[0, 0] = 0.0
        idx = np.unravel_index(np.argmax(amp), amp.shape)
        return int(idx[0]), int(idx[1]), float(amp[idx])

    def analytical_solution(self, t: float) -> np.ndarray:
        """Compute the exact analytical solution at time t.

        Starting from the initial condition, each Fourier mode decays exactly.
        This resets, computes the FFT, applies decay, and returns the result.
        """
        # Save current state
        saved_u = self._u.copy()
        saved_state = self._state.copy()
        saved_count = self._step_count

        # Reset to get initial condition
        self.reset()
        u0 = self._u.copy()

        # Restore state
        self._u = saved_u
        self._state = saved_state
        self._step_count = saved_count

        # Apply exact decay to initial condition
        u0_hat = np.fft.fft2(u0)
        u0_hat *= np.exp(-self.D * self.k_sq * t)
        return np.real(np.fft.ifft2(u0_hat))
