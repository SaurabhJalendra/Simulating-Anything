"""Kuramoto-Sivashinsky equation simulation -- 1D spatiotemporal chaos PDE.

The KS equation:  u_t + u*u_x + u_xx + u_xxxx = 0

Equivalently:     u_t = -u*u_x - u_xx - u_xxxx

This is a canonical model for spatiotemporal chaos arising in flame-front
instabilities, thin film flows, and chemical turbulence. The second-order
term is destabilizing (anti-diffusion) while the fourth-order term provides
short-wavelength stabilization. Competition between these produces chaos
when the domain is sufficiently large (L > 2*pi*sqrt(2)).

Solver: Fourier spectral method with ETDRK4 time integration.
  - Linear terms (u_xx + u_xxxx) integrated exactly via exponential integrating factor
  - Nonlinear term (u*u_x = 0.5 * d(u^2)/dx) via 2/3 dealiasing
  - ETDRK4 allows stable integration with moderate timestep despite stiffness

Target rediscoveries:
- Energy (L2 norm) saturates to a finite value (no blowup)
- Lyapunov exponent estimation from trajectory divergence
- Correlation length vs domain size L
- Spatial mean conservation (integral property)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class KuramotoSivashinsky(SimulationEnvironment):
    """1D Kuramoto-Sivashinsky PDE: u_t + u*u_x + u_xx + u_xxxx = 0.

    State: u(x) discretized on N grid points, shape (N,).
    Solver: Fourier spectral + ETDRK4 time integration on periodic [0, L].

    The ETDRK4 scheme (Cox & Matthews 2002, with Kassam & Trefethen 2005
    contour integral improvements) treats the stiff linear part exactly,
    enabling stable integration with dt = O(1) instead of dt = O(dx^4).

    Parameters:
        L: domain length (default: 32*pi for well-developed chaos)
        N: number of grid points (default: 128)
        viscosity: coefficient of fourth-order term (default: 1.0)
        init_type: 'random' or 'sine' (set via attribute, default: 'random')
        init_amplitude: initial perturbation amplitude (default: 0.01)
        seed: random seed for initial condition
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.L = p.get("L", 32.0 * np.pi)
        self.N = int(p.get("N", 128))
        self.viscosity = p.get("viscosity", 1.0)
        self.init_amplitude = p.get("init_amplitude", 0.01)
        self.init_type = "random"

        # Grid spacing
        self.dx = self.L / self.N

        # Spatial grid
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers: k = 2*pi*n/L for FFT ordering
        self.k = (2 * np.pi / self.L) * np.fft.fftfreq(self.N) * self.N
        self.ik = 1j * self.k  # for d/dx in Fourier space

        # Linear operator L_hat: u_t = L_hat * u_hat + N_hat(u)
        # KS: u_t = -u_xx - viscosity*u_xxxx - u*u_x
        # In Fourier: L_hat = k^2 - viscosity * k^4
        self.linear_op = self.k ** 2 - self.viscosity * self.k ** 4

        # Dealiasing mask (2/3 rule): zero out modes above N/3
        kmax_index = self.N // 3
        self.dealias_mask = np.ones(self.N)
        self.dealias_mask[kmax_index + 1: self.N - kmax_index] = 0.0

        # Precompute ETDRK4 coefficients
        self._setup_etdrk4()

    def _setup_etdrk4(self) -> None:
        """Precompute ETDRK4 coefficients using contour integrals.

        Uses the Kassam & Trefethen (2005) contour integral approach for
        numerically stable evaluation of the phi functions.
        """
        dt = self.config.dt
        L = self.linear_op

        # E and E2 are the integrating factors
        self.E = np.exp(L * dt)
        self.E2 = np.exp(L * dt / 2)

        # Contour integral for stable computation of ETDRK4 coefficients
        # Evaluate on a circle of radius 1 around each L*dt value
        M = 32  # Number of contour points
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

        # L*dt reshaped for broadcasting: (N, 1) + (1, M) circle points
        LR = dt * L[:, np.newaxis] + r[np.newaxis, :]

        # phi functions via contour integrals
        # f1 = (e^z - 1) / z
        # f2 = (e^z - z - 1) / z^2
        # f3 = (e^z - z^2/2 - z - 1) / z^3
        Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))

        self.f1 = dt * np.real(
            np.mean(
                (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3,
                axis=1,
            )
        )
        self.f2 = dt * np.real(
            np.mean(
                (2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3,
                axis=1,
            )
        )
        self.f3 = dt * np.real(
            np.mean(
                (-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3,
                axis=1,
            )
        )
        self.Q = Q

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize the field u(x).

        Args:
            seed: Random seed for reproducible initial conditions.

        Returns:
            Initial state u(x) as shape (N,) array.
        """
        if seed is None:
            seed = self.config.seed

        if self.init_type == "sine":
            # Single-mode sine perturbation
            k1 = 2 * np.pi / self.L
            self._state = self.init_amplitude * np.sin(k1 * self.x)
        else:
            # Small random perturbation (low-wavenumber modes)
            rng = np.random.default_rng(seed)
            u_hat = np.zeros(self.N, dtype=complex)
            # Populate low-wavenumber modes with random phases
            n_modes = min(self.N // 4, 16)
            for j in range(1, n_modes + 1):
                phase = rng.uniform(0, 2 * np.pi)
                amp = self.init_amplitude / j
                u_hat[j] = amp * np.exp(1j * phase)
                u_hat[self.N - j] = np.conj(u_hat[j])
            self._state = np.real(np.fft.ifft(u_hat))

        self._state = self._state.astype(np.float64)
        self._u_hat = np.fft.fft(self._state)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using ETDRK4 in spectral space.

        The ETDRK4 scheme integrates the linear part exactly and uses
        a fourth-order Runge-Kutta method for the nonlinear part.

        Returns:
            Updated state u(x) as shape (N,) array.
        """
        v = self._u_hat

        Nv = self._nonlinear_hat(v)
        a = self.E2 * v + self.Q * Nv
        Na = self._nonlinear_hat(a)
        b = self.E2 * v + self.Q * Na
        Nb = self._nonlinear_hat(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = self._nonlinear_hat(c)

        self._u_hat = (
            self.E * v
            + Nv * self.f1
            + 2 * (Na + Nb) * self.f2
            + Nc * self.f3
        )

        self._state = np.real(np.fft.ifft(self._u_hat))
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current field u(x)."""
        return self._state

    def _nonlinear_hat(self, u_hat: np.ndarray) -> np.ndarray:
        """Compute the nonlinear term in Fourier space.

        N(u) = -u*u_x = -0.5 * d(u^2)/dx

        Uses 2/3 dealiasing rule: zero high-frequency modes before
        computing the product in physical space.
        """
        u_hat_d = u_hat * self.dealias_mask
        u = np.real(np.fft.ifft(u_hat_d))
        u_sq = u ** 2
        return -0.5 * self.ik * np.fft.fft(u_sq)

    @property
    def energy(self) -> float:
        """L2 energy: (1/N) * sum(u^2), equivalent to spatial average."""
        return float(np.mean(self._state ** 2))

    @property
    def spatial_mean(self) -> float:
        """Spatial mean: (1/N) * sum(u)."""
        return float(np.mean(self._state))

    @property
    def max_amplitude(self) -> float:
        """Maximum absolute value of u(x)."""
        return float(np.max(np.abs(self._state)))

    def energy_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the energy spectrum |u_hat(k)|^2.

        Returns:
            (wavenumbers, spectrum) arrays for positive wavenumbers.
        """
        u_hat = np.fft.fft(self._state)
        n_pos = self.N // 2
        k_pos = np.abs(self.k[:n_pos])
        spectrum = (np.abs(u_hat[:n_pos]) ** 2) / self.N ** 2
        return k_pos, spectrum

    def correlation_length(self) -> float:
        """Estimate the spatial correlation length from the autocorrelation.

        The correlation length is defined as the first zero crossing of the
        normalized spatial autocorrelation function.
        """
        u = self._state - np.mean(self._state)
        if np.all(np.abs(u) < 1e-15):
            return float(self.L / 2)

        u_hat = np.fft.fft(u)
        acf = np.real(np.fft.ifft(u_hat * np.conj(u_hat)))
        if acf[0] < 1e-15:
            return float(self.L / 2)
        acf = acf / acf[0]  # Normalize

        # Find first zero crossing
        for i in range(1, len(acf) // 2):
            if acf[i] <= 0:
                # Linear interpolation for zero crossing
                frac = acf[i - 1] / (acf[i - 1] - acf[i])
                return float((i - 1 + frac) * self.dx)

        return float(self.L / 2)
