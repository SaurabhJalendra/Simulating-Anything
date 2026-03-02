"""Korteweg-de Vries (KdV) equation simulation -- nonlinear dispersive PDE with solitons.

The KdV equation in standard form: u_t + 6*u*u_x + u_xxx = 0

Target rediscoveries:
- Soliton solution: u(x,t) = (c/2)*sech^2(sqrt(c)/2 * (x - c*t - x0))
- Speed-amplitude relation: c = 2*A (where A = peak height)
- Elastic soliton collisions (two solitons pass through each other)
- Conserved quantities: mass = integral(u), momentum = integral(u^2),
  energy = integral(u^3 - (u_x)^2 / 2)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class KdVSimulation(SimulationEnvironment):
    """KdV equation: u_t + 6*u*u_x + u_xxx = 0.

    Uses pseudo-spectral method (FFT) for spatial derivatives with RK4 in time
    on a periodic 1D domain.

    State vector: u(x) on a 1D grid of N points.

    Parameters:
        c: soliton speed / parameter (default 2.0)
        x0: initial soliton position (default 10.0)
        N: number of grid points (default 256)
        L: domain length (default 40.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.c = p.get("c", 2.0)
        self.x0 = p.get("x0", 10.0)
        self.N = int(p.get("N", 256))
        self.L = p.get("L", 40.0)

        # Spatial grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers for spectral derivatives
        self.k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        # Precompute ik and (ik)^3 for d/dx and d^3/dx^3
        self._ik = 1j * self.k
        self._ik3 = (1j * self.k) ** 3

        # Default init type
        self.init_type = "soliton"

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize the field u(x) based on init_type."""
        if self.init_type == "soliton":
            self._state = self._soliton_profile(self.c, self.x0)
        elif self.init_type == "two_soliton":
            self._state = self._two_soliton_profile()
        elif self.init_type == "gaussian":
            center = self.L / 2
            sigma = self.L / 20
            self._state = np.exp(-((self.x - center) ** 2) / (2 * sigma ** 2))
        elif self.init_type == "zero":
            self._state = np.zeros(self.N, dtype=np.float64)
        else:
            self._state = self._soliton_profile(self.c, self.x0)

        self._state = self._state.astype(np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with pseudo-spectral spatial derivatives.

        The KdV equation u_t = -6*u*u_x - u_xxx is integrated in physical space
        with spatial derivatives computed via FFT. The 2/3 dealiasing rule is
        applied to the nonlinear term.
        """
        dt = self.config.dt
        u = self._state

        k1 = self._rhs(u)
        k2 = self._rhs(u + 0.5 * dt * k1)
        k3 = self._rhs(u + 0.5 * dt * k2)
        k4 = self._rhs(u + dt * k3)

        self._state = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current field u(x)."""
        return self._state

    def _rhs(self, u: np.ndarray) -> np.ndarray:
        """Compute right-hand side: du/dt = -6*u*u_x - u_xxx.

        Uses FFT for spatial derivatives with 2/3 dealiasing on the nonlinear term.
        """
        u_hat = np.fft.fft(u)

        # 2/3 dealiasing mask for the nonlinear term
        n_dealias = self.N // 3
        mask = np.ones(self.N, dtype=np.float64)
        mask[n_dealias: self.N - n_dealias + 1] = 0.0

        # Nonlinear term with dealiasing: 6*u*u_x
        u_dealiased = np.real(np.fft.ifft(mask * u_hat))
        u_x_dealiased = np.real(np.fft.ifft(mask * self._ik * u_hat))
        nonlinear = 6.0 * u_dealiased * u_x_dealiased

        # Dispersive term: u_xxx
        u_xxx = np.real(np.fft.ifft(self._ik3 * u_hat))

        return -nonlinear - u_xxx

    # --- Soliton profile generators ---

    def _soliton_profile(self, c: float, x0: float) -> np.ndarray:
        """Single soliton: u(x,0) = (c/2)*sech^2(sqrt(c)/2 * (x - x0)).

        Args:
            c: soliton speed (must be > 0).
            x0: center position.

        Returns:
            u(x) array.
        """
        c = max(c, 1e-12)
        arg = np.sqrt(c) / 2.0 * (self.x - x0)
        return (c / 2.0) / np.cosh(arg) ** 2

    def _two_soliton_profile(self) -> np.ndarray:
        """Two-soliton initial condition for collision studies.

        Places a fast soliton (c=2.0) behind a slow soliton (c=0.5).
        """
        c1 = max(self.c, 1.0)
        c2 = c1 / 4.0  # Slower soliton
        x0_fast = self.L * 0.25
        x0_slow = self.L * 0.55
        return self._soliton_profile(c1, x0_fast) + self._soliton_profile(c2, x0_slow)

    def set_soliton(self, c: float, x0: float) -> None:
        """Set the state to a single soliton with given speed and position."""
        self._state = self._soliton_profile(c, x0).astype(np.float64)
        self._step_count = 0

    def set_two_solitons(
        self, c1: float, x01: float, c2: float, x02: float,
    ) -> None:
        """Set the state to two solitons with given speeds and positions."""
        u1 = self._soliton_profile(c1, x01)
        u2 = self._soliton_profile(c2, x02)
        self._state = (u1 + u2).astype(np.float64)
        self._step_count = 0

    # --- Conserved quantities ---

    def compute_mass(self) -> float:
        """First conserved quantity: I1 = integral(u) dx."""
        return float(np.sum(self._state) * self.dx)

    def compute_momentum(self) -> float:
        """Second conserved quantity: I2 = integral(u^2) dx."""
        return float(np.sum(self._state ** 2) * self.dx)

    def compute_energy(self) -> float:
        """Third conserved quantity: I3 = integral(u^3 - (u_x)^2 / 2) dx.

        Also known as the Hamiltonian of the KdV equation.
        """
        u = self._state
        u_hat = np.fft.fft(u)
        u_x = np.real(np.fft.ifft(self._ik * u_hat))

        integrand = u ** 3 - 0.5 * u_x ** 2
        return float(np.sum(integrand) * self.dx)

    # --- Analysis methods ---

    def measure_peak_amplitude(self) -> float:
        """Measure the peak amplitude of the current state."""
        return float(np.max(self._state))

    def measure_peak_position(self) -> float:
        """Measure the position of the peak of the current state.

        Returns the grid point with the maximum value.
        """
        idx = np.argmax(self._state)
        return float(self.x[idx])

    def measure_center_of_mass(self) -> float:
        """Measure the center of mass of the field: x_cm = integral(x*u) / integral(u).

        This provides a sub-grid-accurate position measurement that is much
        more robust than tracking the discrete peak for speed estimation.
        Handles periodic wrapping by unwinding relative to the peak.
        """
        u = self._state
        peak_idx = np.argmax(u)
        peak_x = self.x[peak_idx]

        # Shift coordinates so peak is at center, handling periodicity
        x_shifted = (self.x - peak_x + self.L / 2) % self.L - self.L / 2

        total = np.sum(u) * self.dx
        if abs(total) < 1e-15:
            return peak_x
        x_cm = np.sum(x_shifted * u) * self.dx / total
        # Map back to domain
        return float((peak_x + x_cm) % self.L)

    def measure_soliton_speed(
        self, n_steps: int = 100, dt: float | None = None,
    ) -> float:
        """Measure soliton speed by tracking center of mass over time.

        Uses center-of-mass tracking for sub-grid accuracy.
        Handles periodic wrapping of the domain.

        Args:
            n_steps: number of steps to evolve for measurement.
            dt: override timestep (uses config.dt by default).

        Returns:
            Measured speed.
        """
        if dt is None:
            dt = self.config.dt

        x0 = self.measure_center_of_mass()
        for _ in range(n_steps):
            self.step()
        x1 = self.measure_center_of_mass()

        total_time = n_steps * dt
        # Handle periodic wrapping
        displacement = x1 - x0
        if displacement < -self.L / 2:
            displacement += self.L
        elif displacement > self.L / 2:
            displacement -= self.L

        return displacement / total_time

    def speed_amplitude_sweep(
        self, c_values: np.ndarray | list[float],
    ) -> dict[str, np.ndarray]:
        """Sweep soliton speed parameter c and measure peak amplitude.

        Theory: peak amplitude A = c/2, so speed = 2*A.

        Returns dict with 'c_values', 'amplitudes', 'theoretical_amplitudes'.
        """
        c_values = np.asarray(c_values, dtype=np.float64)
        amplitudes = []

        for c in c_values:
            self.set_soliton(c, self.L / 2)
            amplitudes.append(self.measure_peak_amplitude())

        return {
            "c_values": c_values,
            "amplitudes": np.array(amplitudes),
            "theoretical_amplitudes": c_values / 2.0,
        }

    @staticmethod
    def analytical_soliton_amplitude(c: float) -> float:
        """Analytical peak amplitude for speed c: A = c/2."""
        return c / 2.0

    @staticmethod
    def analytical_soliton_width(c: float) -> float:
        """Analytical soliton width (FWHM): w = 2*arccosh(sqrt(2)) / (sqrt(c)/2).

        The sech^2 function has FWHM = 2*arccosh(sqrt(2)) / kappa
        where kappa = sqrt(c)/2.
        """
        if c <= 0:
            return float("inf")
        kappa = np.sqrt(c) / 2.0
        return 2.0 * np.arccosh(np.sqrt(2.0)) / kappa
