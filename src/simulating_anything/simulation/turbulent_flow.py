"""2D turbulent flow simulation -- high Reynolds number Navier-Stokes.

Extends the basic NS solver to explore turbulence transition and statistics.
Uses spectral methods with hyperviscosity for stability at high Re.

Discovery targets:
- Critical Reynolds number for turbulence onset
- Energy spectrum scaling E(k) ~ k^(-3) (2D enstrophy cascade)
- Inverse energy cascade E(k) ~ k^(-5/3) at large scales
- Kolmogorov microscale eta ~ Re^(-3/4) * L
- Enstrophy decay rate
- Vortex statistics (number, size distribution)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TurbulentFlow2DSimulation(SimulationEnvironment):
    """2D turbulent flow via spectral Navier-Stokes with forcing.

    Uses vorticity-streamfunction formulation with random forcing
    to sustain turbulence. Includes hyperviscosity for de-aliasing.

    State: vorticity field omega on NxN periodic grid (flattened).

    Parameters:
        nu: kinematic viscosity (default: 1e-4, gives Re ~ 10^4)
        N: grid resolution per side (default: 128)
        L: domain size (default: 2*pi)
        forcing_k: forcing wavenumber band center (default: 4)
        forcing_amplitude: forcing strength (default: 0.1)
        hyper_nu: hyperviscosity coefficient (default: 1e-10)
        hyper_order: hyperviscosity order (default: 4, i.e., nu_h * nabla^8)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.nu = p.get("nu", 1e-4)
        self.N = int(p.get("N", 128))
        self.L = p.get("L", 2 * np.pi)
        self.forcing_k = int(p.get("forcing_k", 4))
        self.forcing_amp = p.get("forcing_amplitude", 0.1)
        self.hyper_nu = p.get("hyper_nu", 1e-10)
        self.hyper_order = int(p.get("hyper_order", 4))
        self.dt = config.dt

        dx = self.L / self.N
        # Wavenumbers
        k = np.fft.fftfreq(self.N, d=dx / (2 * np.pi))
        self._kx, self._ky = np.meshgrid(k, k, indexing="ij")
        self._k2 = self._kx ** 2 + self._ky ** 2
        self._k2[0, 0] = 1.0  # avoid division by zero

        # Dealiasing mask (2/3 rule)
        kmax = self.N // 3
        self._dealias = np.ones((self.N, self.N), dtype=np.float64)
        self._dealias[np.abs(self._kx) > kmax] = 0.0
        self._dealias[np.abs(self._ky) > kmax] = 0.0

        # Hyperviscous operator: nu*k^2 + hyper_nu*k^(2*hyper_order)
        self._dissipation = self.nu * self._k2 + self.hyper_nu * self._k2 ** self.hyper_order

        # Forcing mask: wavenumbers in a band around forcing_k
        k_mag = np.sqrt(self._kx ** 2 + self._ky ** 2)
        self._forcing_mask = (
            (k_mag >= self.forcing_k - 1) & (k_mag <= self.forcing_k + 1)
        ).astype(np.float64)

        self._rng = np.random.RandomState(config.seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._step_count = 0

        # Initialize with random vorticity concentrated at forcing scale
        omega_hat = np.zeros((self.N, self.N), dtype=np.complex128)
        k_mag = np.sqrt(self._kx ** 2 + self._ky ** 2)
        init_mask = (k_mag >= 2) & (k_mag <= 8)
        phases = self._rng.uniform(0, 2 * np.pi, (self.N, self.N))
        amplitudes = self._rng.exponential(1.0, (self.N, self.N))
        omega_hat[init_mask] = amplitudes[init_mask] * np.exp(1j * phases[init_mask])
        omega_hat *= self._dealias

        self._omega = np.real(np.fft.ifft2(omega_hat))
        self._state = self._omega.ravel()
        return self._state.copy()

    def step(self) -> np.ndarray:
        omega = self._omega
        dt = self.dt

        # ETDRK2 (exponential time differencing) for stiff dissipation
        omega_hat = np.fft.fft2(omega) * self._dealias

        # Nonlinear term: -J(psi, omega) = -(u * omega_x + v * omega_y)
        nl1 = self._nonlinear(omega_hat)

        # Random forcing
        forcing = self._generate_forcing()

        # Half step: exponential integrator
        E = np.exp(-self._dissipation * dt)
        E2 = np.exp(-self._dissipation * dt / 2)

        # Predictor
        omega_hat_star = E2 * omega_hat + (dt / 2) * E2 * (nl1 + forcing)
        omega_hat_star *= self._dealias

        # Corrector
        nl2 = self._nonlinear(omega_hat_star)
        omega_hat_new = E * omega_hat + dt * E2 * (nl2 + forcing)
        omega_hat_new *= self._dealias

        self._omega = np.real(np.fft.ifft2(omega_hat_new))
        self._state = self._omega.ravel()
        self._step_count += 1
        return self._state.copy()

    def observe(self) -> np.ndarray:
        return self._state.copy()

    def _nonlinear(self, omega_hat: np.ndarray) -> np.ndarray:
        """Compute spectral nonlinear advection term."""
        # Streamfunction: psi_hat = -omega_hat / k^2
        psi_hat = -omega_hat / self._k2

        # Velocity: u = dpsi/dy, v = -dpsi/dx
        u_hat = 1j * self._ky * psi_hat
        v_hat = -1j * self._kx * psi_hat
        omega_x_hat = 1j * self._kx * omega_hat
        omega_y_hat = 1j * self._ky * omega_hat

        # Transform to physical, multiply, transform back
        u = np.real(np.fft.ifft2(u_hat * self._dealias))
        v = np.real(np.fft.ifft2(v_hat * self._dealias))
        omega_x = np.real(np.fft.ifft2(omega_x_hat * self._dealias))
        omega_y = np.real(np.fft.ifft2(omega_y_hat * self._dealias))

        nl = -(u * omega_x + v * omega_y)
        return np.fft.fft2(nl) * self._dealias

    def _generate_forcing(self) -> np.ndarray:
        """Generate random forcing in the forcing wavenumber band."""
        phases = self._rng.uniform(0, 2 * np.pi, (self.N, self.N))
        forcing_hat = self.forcing_amp * self._forcing_mask * np.exp(1j * phases)
        return forcing_hat

    def energy_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute isotropic energy spectrum E(k).

        Returns:
            (k_bins, E_k): wavenumber bins and energy per wavenumber.
        """
        omega_hat = np.fft.fft2(self._omega)
        psi_hat = -omega_hat / self._k2

        # Energy density in spectral space: 0.5 * |k|^2 * |psi_hat|^2 / N^4
        energy_2d = 0.5 * self._k2 * np.abs(psi_hat) ** 2 / self.N ** 4

        # Bin by wavenumber magnitude
        k_mag = np.sqrt(self._kx ** 2 + self._ky ** 2)
        k_max = self.N // 2
        k_bins = np.arange(1, k_max + 1, dtype=np.float64)
        E_k = np.zeros(k_max)

        for i, k_val in enumerate(k_bins):
            mask = (k_mag >= k_val - 0.5) & (k_mag < k_val + 0.5)
            E_k[i] = np.sum(energy_2d[mask])

        return k_bins, E_k

    def total_energy(self) -> float:
        """Total kinetic energy."""
        omega_hat = np.fft.fft2(self._omega)
        psi_hat = -omega_hat / self._k2
        return 0.5 * np.sum(self._k2 * np.abs(psi_hat) ** 2) / self.N ** 4

    def total_enstrophy(self) -> float:
        """Total enstrophy (integral of omega^2 / 2)."""
        return 0.5 * np.mean(self._omega ** 2) * self.L ** 2

    def reynolds_number(self) -> float:
        """Estimate Reynolds number from RMS velocity and domain scale."""
        omega_hat = np.fft.fft2(self._omega)
        psi_hat = -omega_hat / self._k2
        u_hat = 1j * self._ky * psi_hat
        v_hat = -1j * self._kx * psi_hat
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        u_rms = np.sqrt(np.mean(u ** 2 + v ** 2))
        return u_rms * self.L / self.nu if self.nu > 0 else float("inf")
