"""2D Navier-Stokes simulation -- incompressible viscous flow.

Solves the vorticity-streamfunction formulation on a periodic domain
using spectral methods (FFT-based Poisson solver).

  omega_t + u * omega_x + v * omega_y = nu * (omega_xx + omega_yy)
  Lap(psi) = -omega
  u = psi_y, v = -psi_x

Target rediscoveries:
- Viscous decay rate: energy ~ exp(-2*nu*k^2*t) for single mode
- Vortex merging timescale
- Energy spectrum scaling (inverse cascade in 2D)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class NavierStokes2DSimulation(SimulationEnvironment):
    """2D incompressible Navier-Stokes via vorticity-streamfunction method.

    State: vorticity field omega on NxN periodic grid.
    Uses FFT-based Poisson solver for the streamfunction.

    Parameters:
        nu: kinematic viscosity (default: 0.001)
        N: grid resolution (default: 64)
        L: domain size (default: 2*pi)
        init_type: "taylor_green", "random", "double_vortex" (default: taylor_green)
        init_amplitude: initial vortex strength (default: 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.nu = p.get("nu", 0.001)
        self.N = int(p.get("N", 64))
        self.L = p.get("L", 2.0 * np.pi)
        # init_type is a string, not stored in parameters (dict[str, float])
        self.init_type = "taylor_green"
        self.init_amplitude = p.get("init_amplitude", 1.0)

        self.dx = self.L / self.N
        self._setup_spectral()

    def _setup_spectral(self) -> None:
        """Setup wavenumber arrays for spectral differentiation."""
        N = self.N
        # Wavenumber arrays
        k = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.kx, self.ky = np.meshgrid(k, k)
        self.k_sq = self.kx**2 + self.ky**2
        # Avoid division by zero at k=0
        self.k_sq_inv = np.zeros_like(self.k_sq)
        mask = self.k_sq > 0
        self.k_sq_inv[mask] = 1.0 / self.k_sq[mask]
        # Dealiasing mask (2/3 rule)
        kmax = N // 3
        self.dealias = np.ones((N, N), dtype=bool)
        self.dealias[np.abs(self.kx) > kmax * 2 * np.pi / self.L] = False
        self.dealias[np.abs(self.ky) > kmax * 2 * np.pi / self.L] = False

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize vorticity field."""
        N = self.N
        x = np.linspace(0, self.L, N, endpoint=False)
        X, Y = np.meshgrid(x, x)

        if self.init_type == "taylor_green":
            # Taylor-Green vortex: omega = 2k*cos(kx)*cos(ky)
            k = 2 * np.pi / self.L
            self._omega = self.init_amplitude * 2 * k * np.cos(k * X) * np.cos(k * Y)
        elif self.init_type == "double_vortex":
            # Two Gaussian vortices
            cx1, cy1 = self.L * 0.35, self.L * 0.5
            cx2, cy2 = self.L * 0.65, self.L * 0.5
            r = self.L * 0.08
            self._omega = self.init_amplitude * (
                np.exp(-((X - cx1)**2 + (Y - cy1)**2) / (2 * r**2))
                - np.exp(-((X - cx2)**2 + (Y - cy2)**2) / (2 * r**2))
            )
        elif self.init_type == "random":
            rng = np.random.default_rng(seed or 42)
            omega_hat = np.zeros((N, N), dtype=complex)
            # Random low-wavenumber modes
            for kk in range(1, 5):
                for ll in range(1, 5):
                    phase = rng.uniform(0, 2 * np.pi)
                    amp = self.init_amplitude / (kk**2 + ll**2)
                    omega_hat[ll, kk] = amp * np.exp(1j * phase)
                    omega_hat[N - ll, N - kk] = np.conj(omega_hat[ll, kk])
            self._omega = np.real(np.fft.ifft2(omega_hat))
        else:
            self._omega = np.zeros((N, N))

        self._state = self._omega.flatten()
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 in spectral space."""
        dt = self.config.dt
        omega = self._omega

        k1 = self._rhs(omega)
        k2 = self._rhs(omega + 0.5 * dt * k1)
        k3 = self._rhs(omega + 0.5 * dt * k2)
        k4 = self._rhs(omega + dt * k3)

        self._omega = omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = self._omega.flatten()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current vorticity field (flattened)."""
        return self._state

    def _rhs(self, omega: np.ndarray) -> np.ndarray:
        """Compute right-hand side: -u*omega_x - v*omega_y + nu*Lap(omega)."""
        omega_hat = np.fft.fft2(omega)

        # Streamfunction: Lap(psi) = -omega => psi_hat = omega_hat / k^2
        psi_hat = omega_hat * self.k_sq_inv

        # Velocity: u = psi_y, v = -psi_x
        u = np.real(np.fft.ifft2(1j * self.ky * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self.kx * psi_hat))

        # Vorticity gradients
        omega_x = np.real(np.fft.ifft2(1j * self.kx * omega_hat))
        omega_y = np.real(np.fft.ifft2(1j * self.ky * omega_hat))

        # Nonlinear term (dealiased)
        nlterm_hat = np.fft.fft2(u * omega_x + v * omega_y)
        nlterm_hat *= self.dealias

        # Diffusion
        diffusion_hat = -self.nu * self.k_sq * omega_hat

        rhs_hat = -nlterm_hat + diffusion_hat
        return np.real(np.fft.ifft2(rhs_hat))

    @property
    def velocity_field(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute velocity field (u, v) from current vorticity."""
        omega_hat = np.fft.fft2(self._omega)
        psi_hat = omega_hat * self.k_sq_inv
        u = np.real(np.fft.ifft2(1j * self.ky * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self.kx * psi_hat))
        return u, v

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy: 0.5 * integral(u^2 + v^2)."""
        u, v = self.velocity_field
        return 0.5 * np.mean(u**2 + v**2) * self.L**2

    @property
    def enstrophy(self) -> float:
        """Total enstrophy: 0.5 * integral(omega^2)."""
        return 0.5 * np.mean(self._omega**2) * self.L**2

    def energy_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the 1D energy spectrum E(k).

        Returns:
            (k_bins, E_k) where E_k[i] is energy in wavenumber shell i.
        """
        omega_hat = np.fft.fft2(self._omega)
        psi_hat = omega_hat * self.k_sq_inv

        # Energy density in spectral space
        energy_hat = 0.5 * np.abs(psi_hat)**2 * self.k_sq

        k_mag = np.sqrt(self.k_sq)
        k_max = self.N // 2
        k_bins = np.arange(1, k_max + 1, dtype=float)
        E_k = np.zeros(k_max)

        for i in range(k_max):
            mask = (k_mag >= k_bins[i] - 0.5) & (k_mag < k_bins[i] + 0.5)
            E_k[i] = np.sum(energy_hat[mask].real)

        # Normalize
        E_k *= (2 * np.pi / self.L)**2 / self.N**4
        return k_bins * (2 * np.pi / self.L), E_k

    def taylor_green_analytical_energy(self, t: float) -> float:
        """Analytical energy decay for Taylor-Green vortex.

        E(t) = E_0 * exp(-2*nu*k^2*t) where k = 2*pi/L.
        """
        k = 2 * np.pi / self.L
        E_0 = 0.5 * self.init_amplitude**2 * self.L**2 / 2  # 0.5 * A^2 * L^2 / 2
        return E_0 * np.exp(-2 * self.nu * k**2 * t)
