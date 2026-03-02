"""Swift-Hohenberg equation simulation -- 1D pattern formation PDE.

The Swift-Hohenberg equation:
    du/dt = r*u - (1 + nabla^2)^2 * u + g*u^2 - u^3

In 1D with periodic boundary conditions and spectral method:
    du_hat/dt = [r - (1 - k^2)^2] * u_hat + FFT(g*u^2 - u^3)

The linear operator L(k) = r - (1 - k^2)^2 has its maximum at k = +/-1
(the critical wavenumber k_c = 1). For r > 0, modes near k_c = 1 are
linearly unstable, driving pattern formation with wavelength lambda = 2*pi.

Solver: Fourier spectral method with Strang splitting.
  - Linear part integrated exactly via exponential: u_hat *= exp(L*dt/2)
  - Nonlinear part (g*u^2 - u^3) integrated with RK4 in physical space
  - 2/3 dealiasing rule to prevent aliasing from the cubic term
  - Internal substepping for accuracy at larger user-facing timesteps

Target rediscoveries:
- Pattern wavelength lambda ~ 2*pi (from k_c = 1)
- Amplitude scaling A ~ sqrt(r) near onset (supercritical bifurcation, g=0)
- Subcritical bifurcation for g != 0
"""
from __future__ import annotations

import math

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

# Maximum internal timestep for RK4 stability of the cubic nonlinear ODE.
# For |u| ~ 2, the Lipschitz constant of -u^3 is ~3*u^2 ~ 12,
# so RK4 requires dt < 2.8/12 ~ 0.23. We use 0.05 for safety margin.
_MAX_INTERNAL_DT = 0.05


class SwiftHohenbergSimulation(SimulationEnvironment):
    """1D Swift-Hohenberg PDE: du/dt = r*u - (1+nabla^2)^2 u + g*u^2 - u^3.

    State: u(x) discretized on N grid points, shape (N,).
    Solver: Fourier spectral + Strang splitting on periodic [0, L].

    Strang splitting (2nd order):
        1. Half-step exact linear: u_hat *= exp(L * dt/2)
        2. Full-step RK4 nonlinear: du/dt = g*u^2 - u^3
        3. Half-step exact linear: u_hat *= exp(L * dt/2)

    The linear part is integrated exactly (no timestep restriction),
    while the cubic nonlinearity is stable under standard RK4 CFL.

    Parameters:
        r: control parameter (default 0.5). r > 0 for pattern formation.
        g: quadratic coupling (default 0.0). g=0 is supercritical, g!=0
           introduces subcritical behavior.
        N: number of grid points (default 256)
        L: domain length (default 20*pi)
        init_amplitude: amplitude of initial perturbation (default 0.01)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 0.5)
        self.g = p.get("g", 0.0)
        self.N = int(p.get("N", 256))
        self.L = p.get("L", 20.0 * np.pi)
        self.init_amplitude = p.get("init_amplitude", 0.01)

        # Grid spacing
        self.dx = self.L / self.N

        # Spatial grid
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers for FFT: k = 2*pi*n/L
        self.k = (2 * np.pi / self.L) * np.fft.fftfreq(self.N) * self.N

        # Linear operator: L_hat(k) = r - (1 - k^2)^2
        self.linear_op = self.r - (1.0 - self.k ** 2) ** 2

        # Dealiasing mask (2/3 rule): zero out modes above N/3
        kmax_index = self.N // 3
        self.dealias_mask = np.ones(self.N)
        self.dealias_mask[kmax_index + 1: self.N - kmax_index] = 0.0

        # Internal substepping for accuracy of the nonlinear RK4
        dt_external = config.dt
        if dt_external <= _MAX_INTERNAL_DT:
            self._n_substeps = 1
            self._dt_internal = dt_external
        else:
            self._n_substeps = math.ceil(dt_external / _MAX_INTERNAL_DT)
            self._dt_internal = dt_external / self._n_substeps

        # Precompute exponential integrating factors for Strang splitting
        dt_int = self._dt_internal
        self._exp_half = np.exp(self.linear_op * dt_int / 2)
        self._exp_full = np.exp(self.linear_op * dt_int)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize the field u(x) with a small random perturbation.

        Args:
            seed: Random seed for reproducible initial conditions.

        Returns:
            Initial state u(x) as shape (N,) array.
        """
        if seed is None:
            seed = self.config.seed

        rng = np.random.default_rng(seed)

        # Small random perturbation seeded in low-wavenumber modes
        u_hat = np.zeros(self.N, dtype=complex)
        n_modes = min(self.N // 4, 20)
        for j in range(1, n_modes + 1):
            phase = rng.uniform(0, 2 * np.pi)
            amp = self.init_amplitude / j
            u_hat[j] = amp * np.exp(1j * phase)
            u_hat[self.N - j] = np.conj(u_hat[j])

        self._state = np.real(np.fft.ifft(u_hat)).astype(np.float64)
        self._u_hat = np.fft.fft(self._state)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one external timestep (config.dt) via Strang splitting.

        Returns:
            Updated state u(x) as shape (N,) array.
        """
        for _ in range(self._n_substeps):
            self._strang_substep()

        self._state = np.real(np.fft.ifft(self._u_hat))
        self._step_count += 1
        return self._state

    def _strang_substep(self) -> None:
        """Perform one Strang splitting substep.

        1. Half-step linear (exact in Fourier)
        2. Full-step nonlinear (RK4 in physical space)
        3. Half-step linear (exact in Fourier)
        """
        dt = self._dt_internal

        # Step 1: Half-step linear propagation
        self._u_hat = self._u_hat * self._exp_half * self.dealias_mask

        # Step 2: Full-step nonlinear RK4 in physical space
        u = np.real(np.fft.ifft(self._u_hat))
        u = self._rk4_nonlinear(u, dt)
        self._u_hat = np.fft.fft(u) * self.dealias_mask

        # Step 3: Half-step linear propagation
        self._u_hat = self._u_hat * self._exp_half * self.dealias_mask

    def _rk4_nonlinear(self, u: np.ndarray, dt: float) -> np.ndarray:
        """RK4 integration of the nonlinear ODE: du/dt = g*u^2 - u^3.

        This is a pointwise ODE (no spatial coupling), so it operates
        independently at each grid point.
        """
        g = self.g

        def f(u_val: np.ndarray) -> np.ndarray:
            return g * u_val ** 2 - u_val ** 3

        k1 = f(u)
        k2 = f(u + 0.5 * dt * k1)
        k3 = f(u + 0.5 * dt * k2)
        k4 = f(u + dt * k3)

        return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def observe(self) -> np.ndarray:
        """Return current field u(x)."""
        return self._state

    def pattern_wavelength(self, state: np.ndarray | None = None) -> float:
        """Measure the dominant pattern wavelength via FFT peak detection.

        Computes the power spectrum, finds the peak wavenumber k_peak
        among positive wavenumbers (excluding k=0), and returns
        lambda = 2*pi / k_peak.

        Args:
            state: Field to analyze. Uses current state if None.

        Returns:
            Dominant wavelength. Returns L (domain length) if no pattern.
        """
        if state is None:
            state = self._state

        u_hat = np.fft.fft(state)
        n_pos = self.N // 2
        k_pos = np.abs(self.k[:n_pos])
        spectrum = np.abs(u_hat[:n_pos]) ** 2

        # Skip k=0 (DC component)
        if n_pos < 2:
            return float(self.L)

        spectrum[0] = 0.0
        peak_idx = np.argmax(spectrum[1:]) + 1

        if k_pos[peak_idx] > 0 and spectrum[peak_idx] > 0:
            return float(2 * np.pi / k_pos[peak_idx])
        return float(self.L)

    def pattern_amplitude(self, state: np.ndarray | None = None) -> float:
        """Measure the pattern amplitude as sqrt(2 * spatial variance).

        For a sinusoidal pattern u = A*sin(kx), the RMS is A/sqrt(2),
        so A = sqrt(2 * var(u)).

        Args:
            state: Field to analyze. Uses current state if None.

        Returns:
            Estimated pattern amplitude.
        """
        if state is None:
            state = self._state
        variance = np.var(state)
        return float(np.sqrt(2.0 * variance))

    @property
    def energy(self) -> float:
        """L2 energy: (1/N) * sum(u^2), spatial average of u^2."""
        return float(np.mean(self._state ** 2))

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

    @property
    def max_amplitude(self) -> float:
        """Maximum absolute value of u(x)."""
        return float(np.max(np.abs(self._state)))

    @property
    def spatial_mean(self) -> float:
        """Spatial mean: (1/N) * sum(u)."""
        return float(np.mean(self._state))
