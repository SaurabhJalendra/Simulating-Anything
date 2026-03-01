"""Quantum harmonic oscillator simulation via split-operator FFT.

Solves the time-dependent Schrodinger equation:
    i*hbar * d|psi>/dt = H|psi>
    H = p^2/(2m) + (1/2)*m*omega^2*x^2

Uses Trotter (split-operator) decomposition:
    psi(t+dt) = exp(-i*V*dt/2) * FFT^-1[ exp(-i*K*dt) * FFT[ exp(-i*V*dt/2) * psi(t) ] ]

Target rediscoveries:
- Energy spectrum: E_n = hbar*omega*(n + 1/2)
- Ground state energy: E_0 = 0.5*hbar*omega
- Coherent state oscillation: <x>(t) = x_0*cos(omega*t)
- Norm conservation: integral |psi|^2 dx = 1
"""
from __future__ import annotations

import math

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class QuantumHarmonicOscillator(SimulationEnvironment):
    """Quantum harmonic oscillator with split-operator FFT propagation.

    The internal state stores the complex wavefunction psi(x).
    The observe() method returns |psi(x)|^2 (probability density).

    Parameters:
        m: particle mass (default 1.0)
        omega: oscillator frequency (default 1.0)
        hbar: reduced Planck constant (default 1.0)
        N: number of grid points (default 128)
        x_max: domain half-width, grid spans [-x_max, x_max] (default 10.0)
        x_0: initial wavepacket displacement from origin (default 2.0)
        p_0: initial wavepacket momentum (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.m = p.get("m", 1.0)
        self.omega = p.get("omega", 1.0)
        self.hbar = p.get("hbar", 1.0)
        self.N = int(p.get("N", 128))
        self.x_max = p.get("x_max", 10.0)
        self.x_0 = p.get("x_0", 2.0)
        self.p_0 = p.get("p_0", 0.0)

        # Spatial grid: N points on [-x_max, x_max)
        self.dx = 2.0 * self.x_max / self.N
        self.x = np.linspace(-self.x_max, self.x_max, self.N, endpoint=False)

        # Momentum-space wavenumbers (FFT convention)
        self.k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi

        # Potential energy on the grid: V(x) = 0.5 * m * omega^2 * x^2
        self._V = 0.5 * self.m * self.omega**2 * self.x**2

        # Kinetic energy in momentum space: K(k) = hbar^2 * k^2 / (2m)
        self._K = self.hbar * self.k**2 / (2.0 * self.m)

        # Precompute propagators for the current dt
        self._precompute_propagators()

        # Complex wavefunction (set by reset)
        self._psi: np.ndarray = np.zeros(self.N, dtype=np.complex128)

    def _precompute_propagators(self) -> None:
        """Precompute the split-operator propagators for efficiency."""
        dt = self.config.dt
        # Half-step potential propagator: exp(-i * V * dt / (2*hbar))
        self._exp_V_half = np.exp(-1j * self._V * dt / (2.0 * self.hbar))
        # Full-step kinetic propagator: exp(-i * K * dt / hbar)
        # K(k) = hbar*k^2/(2m), so exponent = -i * k^2 * hbar * dt / (2m)
        self._exp_K = np.exp(-1j * self._K * dt / self.hbar)

    def _coherent_state(self, x_0: float, p_0: float) -> np.ndarray:
        """Create a coherent state (displaced Gaussian wavepacket).

        psi(x) = (m*omega/(pi*hbar))^(1/4)
                 * exp(-m*omega*(x-x_0)^2 / (2*hbar))
                 * exp(i*p_0*x/hbar)

        This is the minimum-uncertainty state with width sigma = sqrt(hbar/(m*omega)).
        """
        sigma = np.sqrt(self.hbar / (self.m * self.omega))
        norm = (self.m * self.omega / (np.pi * self.hbar)) ** 0.25
        psi = norm * np.exp(
            -(self.x - x_0) ** 2 / (2.0 * sigma**2)
        ) * np.exp(1j * p_0 * self.x / self.hbar)
        return psi

    def _eigenstate(self, n: int) -> np.ndarray:
        """Compute the n-th energy eigenstate of the harmonic oscillator.

        psi_n(x) = (1/sqrt(2^n * n!)) * (m*omega/(pi*hbar))^(1/4)
                   * exp(-m*omega*x^2/(2*hbar)) * H_n(x*sqrt(m*omega/hbar))

        where H_n is the physicist's Hermite polynomial.
        """
        xi = self.x * np.sqrt(self.m * self.omega / self.hbar)
        prefactor = (self.m * self.omega / (np.pi * self.hbar)) ** 0.25
        prefactor /= np.sqrt(2.0**n * float(math.factorial(n)))
        gauss = np.exp(-xi**2 / 2.0)

        # Hermite polynomial via recurrence: H_0=1, H_1=2*xi, H_{n+1}=2*xi*H_n - 2*n*H_{n-1}
        if n == 0:
            H_n = np.ones_like(xi)
        elif n == 1:
            H_n = 2.0 * xi
        else:
            H_prev2 = np.ones_like(xi)
            H_prev1 = 2.0 * xi
            for k in range(2, n + 1):
                H_curr = 2.0 * xi * H_prev1 - 2.0 * (k - 1) * H_prev2
                H_prev2 = H_prev1
                H_prev1 = H_curr
            H_n = H_prev1

        return prefactor * gauss * H_n

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize wavefunction as a coherent state displaced from origin."""
        self._psi = self._coherent_state(self.x_0, self.p_0)
        # Normalize to ensure integral |psi|^2 dx = 1
        norm = np.sqrt(np.sum(np.abs(self._psi) ** 2) * self.dx)
        self._psi /= norm
        self._step_count = 0
        self._state = np.abs(self._psi) ** 2
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using split-operator FFT (Strang splitting)."""
        # Half-step in potential
        self._psi *= self._exp_V_half
        # Transform to momentum space, full kinetic step, transform back
        psi_k = np.fft.fft(self._psi)
        psi_k *= self._exp_K
        self._psi = np.fft.ifft(psi_k)
        # Half-step in potential
        self._psi *= self._exp_V_half

        self._step_count += 1
        self._state = np.abs(self._psi) ** 2
        return self._state

    def observe(self) -> np.ndarray:
        """Return probability density |psi(x)|^2."""
        return self._state

    @property
    def norm(self) -> float:
        """Norm of the wavefunction: integral |psi|^2 dx. Should be ~1.0."""
        return float(np.sum(np.abs(self._psi) ** 2) * self.dx)

    @property
    def energy(self) -> float:
        """Expectation value <H> = <T> + <V>.

        <V> = integral psi* V psi dx
        <T> = integral psi* (-hbar^2/(2m) d^2/dx^2) psi dx
            = (hbar^2/(2m)) * integral |d psi/dx|^2 dx  (integration by parts)
            = sum_k (hbar^2 k^2 / (2m)) |psi_k|^2  (Parseval)
        """
        # Potential energy
        V_expect = float(np.sum(np.abs(self._psi) ** 2 * self._V) * self.dx)

        # Kinetic energy via FFT
        psi_k = np.fft.fft(self._psi)
        # Parseval: sum |psi_k|^2 / N = integral |psi|^2 dx / L
        # Energy in k-space: T = sum_k (hbar^2 k^2 / (2m)) |psi_k|^2 * dx / N
        # But we need proper normalization: integral = sum * dx
        # FFT gives psi_k = sum_j psi_j exp(-i k_j x_j), so |psi_k|^2 ~ N * integral
        T_expect = float(
            np.sum(self._K * np.abs(psi_k) ** 2) * self.dx / self.N
        )

        return V_expect + T_expect

    @property
    def position_expectation(self) -> float:
        """Expectation value <x> = integral psi* x psi dx."""
        return float(np.sum(np.abs(self._psi) ** 2 * self.x) * self.dx)

    @property
    def momentum_expectation(self) -> float:
        """Expectation value <p> = -i*hbar * integral psi* (d psi/dx) dx.

        Computed in momentum space: <p> = sum_k hbar*k |psi_k|^2 / N * dx.
        """
        psi_k = np.fft.fft(self._psi)
        return float(
            np.sum(self.hbar * self.k * np.abs(psi_k) ** 2) * self.dx / self.N
        )

    def energy_eigenvalue(self, n: int) -> float:
        """Theoretical energy of the n-th eigenstate: E_n = hbar*omega*(n + 0.5)."""
        return self.hbar * self.omega * (n + 0.5)

    def project_onto_eigenstate(self, n: int) -> float:
        """Compute |<n|psi>|^2 -- overlap probability with n-th eigenstate."""
        phi_n = self._eigenstate(n)
        overlap = np.sum(np.conj(phi_n) * self._psi) * self.dx
        return float(np.abs(overlap) ** 2)

    def measure_energy_from_eigenstate(self, n: int) -> float:
        """Prepare the n-th eigenstate and measure its energy expectation."""
        self._psi = self._eigenstate(n).astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(self._psi) ** 2) * self.dx)
        self._psi /= norm
        self._state = np.abs(self._psi) ** 2
        return self.energy
