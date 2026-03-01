"""1D damped wave equation simulation.

Target rediscoveries:
- Dispersion relation: omega_k = sqrt(c^2 * k^2 - gamma^2/4)
- Damping rate: all modes decay at rate gamma/2
- Wave speed: pulse propagates at speed c
- Energy decay: total energy decreases monotonically with damping
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DampedWave1D(SimulationEnvironment):
    """1D damped wave equation: u_tt + gamma*u_t = c^2 * u_xx.

    Periodic boundary conditions on [0, L].

    Solver: spectral (FFT) for spatial derivatives, RK4 for time integration.
    The system is rewritten as two first-order equations:
        du/dt = v
        dv/dt = c^2 * u_xx - gamma * v

    State vector: [u_1, ..., u_N, v_1, ..., v_N] where v = du/dt.

    Parameters:
        c: wave speed (default 1.0)
        gamma: damping coefficient (default 0.1)
        N: number of grid points (default 64)
        L: domain length (default 2*pi)
        init_type: 'gaussian', 'sine', 'standing' (set via attribute)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.c = p.get("c", 1.0)
        self.gamma = p.get("gamma", 0.1)
        self.N = int(p.get("N", 64))
        self.L = p.get("L", 2 * np.pi)
        self.init_type = "gaussian"

        # Spatial grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers for spectral derivatives
        self.k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        self.k_sq = self.k ** 2

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize displacement and velocity fields."""
        if self.init_type == "sine":
            # Single sine mode
            u = np.sin(2 * np.pi * self.x / self.L)
            v = np.zeros(self.N)
        elif self.init_type == "standing":
            # Standing wave: u = sin(2*pi*x/L), v = 0
            u = np.sin(2 * np.pi * self.x / self.L)
            v = np.zeros(self.N)
        else:  # gaussian
            center = self.L / 2
            sigma = self.L / 20
            u = np.exp(-((self.x - center) ** 2) / (2 * sigma ** 2))
            v = np.zeros(self.N)

        self._state = np.concatenate([u, v]).astype(np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with spectral spatial derivatives."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [u_1,...,u_N, v_1,...,v_N]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute derivatives: du/dt = v, dv/dt = c^2 * u_xx - gamma * v.

        Uses spectral (FFT) method for u_xx.
        """
        N = self.N
        u = y[:N]
        v = y[N:]

        # Spectral second derivative: u_xx in Fourier space = -k^2 * u_hat
        u_hat = np.fft.fft(u)
        u_xx_hat = -self.k_sq * u_hat
        u_xx = np.real(np.fft.ifft(u_xx_hat))

        du_dt = v
        dv_dt = self.c ** 2 * u_xx - self.gamma * v

        return np.concatenate([du_dt, dv_dt])

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy: 0.5 * integral(v^2) dx."""
        v = self._state[self.N:]
        return 0.5 * np.sum(v ** 2) * self.dx

    @property
    def potential_energy(self) -> float:
        """Potential energy: 0.5 * c^2 * integral(u_x^2) dx.

        Computed spectrally: u_x in Fourier space = i*k * u_hat.
        """
        u = self._state[:self.N]
        u_hat = np.fft.fft(u)
        u_x_hat = 1j * self.k * u_hat
        u_x = np.real(np.fft.ifft(u_x_hat))
        return 0.5 * self.c ** 2 * np.sum(u_x ** 2) * self.dx

    @property
    def total_energy(self) -> float:
        """Total mechanical energy (kinetic + potential)."""
        return self.kinetic_energy + self.potential_energy

    @property
    def dominant_wavenumber(self) -> int:
        """Index of the dominant Fourier mode of the displacement field."""
        u = self._state[:self.N]
        u_hat = np.fft.fft(u)
        # Skip the DC component (index 0) and look at positive frequencies
        magnitudes = np.abs(u_hat[1:self.N // 2])
        return int(np.argmax(magnitudes)) + 1

    def mode_amplitudes(self) -> np.ndarray:
        """FFT of the displacement field u.

        Returns complex Fourier coefficients for modes 0..N-1.
        """
        u = self._state[:self.N]
        return np.fft.fft(u)

    def theoretical_frequency(self, mode: int) -> float:
        """Theoretical angular frequency for Fourier mode number.

        omega_k = sqrt(c^2 * k^2 - gamma^2/4) where k = 2*pi*mode/L.
        Returns 0 if the mode is overdamped (argument under sqrt is negative).
        """
        k_mode = 2 * np.pi * mode / self.L
        arg = self.c ** 2 * k_mode ** 2 - self.gamma ** 2 / 4
        if arg <= 0:
            return 0.0
        return float(np.sqrt(arg))

    def theoretical_decay_rate(self) -> float:
        """Theoretical amplitude decay rate: gamma/2 for all modes."""
        return self.gamma / 2.0
