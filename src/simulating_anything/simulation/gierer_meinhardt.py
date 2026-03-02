"""Gierer-Meinhardt reaction-diffusion simulation on a 1D periodic grid.

Classic activator-inhibitor system for biological pattern formation (Turing
patterns). The model describes short-range activation and long-range inhibition:

    da/dt = rho_a * a^2/h - mu_a*a + D_a * d^2a/dx^2 + rho_0
    dh/dt = rho_h * a^2     - mu_h*h + D_h * d^2h/dx^2

with periodic boundary conditions on [0, L].

Solver: operator splitting -- spectral (FFT) exact diffusion in Fourier space
plus explicit Euler for the reaction terms.

Target rediscoveries:
- Homogeneous steady state: a0 = (rho_a*rho_0 + mu_a*rho_0) ...
  Exact: a0 = rho_a/(mu_a*mu_h) * rho_h * a0^2 ... solve for a0, h0
- Turing instability when D_h >> D_a (long-range inhibition, short-range activation)
- Pattern wavelength scaling with diffusion parameters
- SINDy ODE recovery in the well-mixed limit
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GiererMeinhardtSimulation(SimulationEnvironment):
    """Gierer-Meinhardt reaction-diffusion on a 1D periodic domain.

    State: activator a(x) and inhibitor h(x) on a 1D grid.
    Observe shape: (2*N_grid,) = [a_0..a_N, h_0..h_N].

    Parameters:
        rho_a: activator production rate (default 0.1)
        rho_h: inhibitor production rate (default 0.1)
        mu_a: activator decay rate (default 0.02)
        mu_h: inhibitor decay rate (default 0.03)
        D_a: activator diffusion coefficient (default 0.005)
        D_h: inhibitor diffusion coefficient (default 0.2)
        rho_0: basal activator production (default 0.001)
        N_grid: number of spatial grid points (default 128)
        L: domain length (default 50.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Reaction parameters
        self.rho_a = p.get("rho_a", 0.1)
        self.rho_h = p.get("rho_h", 0.1)
        self.mu_a = p.get("mu_a", 0.02)
        self.mu_h = p.get("mu_h", 0.03)
        self.rho_0 = p.get("rho_0", 0.001)

        # Diffusion coefficients
        self.D_a = p.get("D_a", 0.005)
        self.D_h = p.get("D_h", 0.2)

        # Spatial grid
        self.N = int(p.get("N_grid", 128))
        self.L = p.get("L", 50.0)
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers for spectral Laplacian (periodic domain)
        self.k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        self.k_sq = self.k ** 2

        # Precompute spectral diffusion operators for one timestep
        dt = config.dt
        self._diff_a = np.exp(-self.D_a * self.k_sq * dt)
        self._diff_h = np.exp(-self.D_h * self.k_sq * dt)

        # State placeholders
        self._a: np.ndarray | None = None
        self._h: np.ndarray | None = None

    def homogeneous_steady_state(self) -> tuple[float, float]:
        """Return the homogeneous steady state (a0, h0).

        Setting da/dt = 0, dh/dt = 0 with no spatial variation:
            rho_a * a0^2 / h0 - mu_a * a0 + rho_0 = 0
            rho_h * a0^2 - mu_h * h0 = 0

        From the second equation: h0 = rho_h * a0^2 / mu_h
        Substituting into the first:
            rho_a * a0^2 / (rho_h * a0^2 / mu_h) - mu_a * a0 + rho_0 = 0
            rho_a * mu_h / rho_h - mu_a * a0 + rho_0 = 0
            a0 = (rho_a * mu_h / rho_h + rho_0) / mu_a
            h0 = rho_h * a0^2 / mu_h
        """
        if self.mu_a == 0:
            return (0.0, 0.0)
        a0 = (self.rho_a * self.mu_h / self.rho_h + self.rho_0) / self.mu_a
        if self.mu_h == 0:
            return (a0, 0.0)
        h0 = self.rho_h * a0 ** 2 / self.mu_h
        return (a0, h0)

    def turing_analysis(self) -> dict:
        """Compute Turing instability analysis for current parameters.

        The Jacobian of the reaction kinetics at (a0, h0) is:
            J = [[rho_a * a0 / h0 - mu_a,  -rho_a * a0^2 / h0^2],
                 [2 * rho_h * a0,           -mu_h]]

        Using h0 = rho_h * a0^2 / mu_h:
            J11 = mu_a - mu_a = rho_a*mu_h/rho_h / a0 - mu_a
                = mu_a * (... - 1) -- depends on rho_0

        For Turing instability: homogeneous state stable (tr(J)<0, det(J)>0)
        but unstable to diffusive perturbations.
        """
        a0, h0 = self.homogeneous_steady_state()

        if h0 <= 0 or a0 <= 0:
            return {
                "a0": a0, "h0": h0,
                "homogeneous_stable": False,
                "turing_unstable": False,
            }

        # Jacobian elements for GM reaction kinetics
        # f(a,h) = rho_a * a^2 / h - mu_a * a + rho_0
        # g(a,h) = rho_h * a^2 - mu_h * h
        fa = 2 * self.rho_a * a0 / h0 - self.mu_a  # df/da
        fh = -self.rho_a * a0 ** 2 / h0 ** 2        # df/dh
        ga = 2 * self.rho_h * a0                     # dg/da
        gh = -self.mu_h                              # dg/dh

        tr_J = fa + gh
        det_J = fa * gh - fh * ga

        homogeneous_stable = (tr_J < 0) and (det_J > 0)

        # Turing condition: D_h * fa + D_a * gh > 0
        # and (D_h * fa + D_a * gh)^2 > 4 * D_a * D_h * det_J
        cross = self.D_h * fa + self.D_a * gh
        discriminant = cross ** 2 - 4 * self.D_a * self.D_h * det_J

        turing_unstable = homogeneous_stable and (cross > 0) and (discriminant > 0)

        result: dict = {
            "a0": a0,
            "h0": h0,
            "fa": fa,
            "fh": fh,
            "ga": ga,
            "gh": gh,
            "trace_J": tr_J,
            "det_J": det_J,
            "homogeneous_stable": homogeneous_stable,
            "cross_diffusion": cross,
            "discriminant": discriminant,
            "turing_unstable": turing_unstable,
        }

        if turing_unstable and discriminant > 0:
            sqrt_disc = np.sqrt(discriminant)
            k_sq_minus = (cross - sqrt_disc) / (2 * self.D_a * self.D_h)
            k_sq_plus = (cross + sqrt_disc) / (2 * self.D_a * self.D_h)

            if k_sq_minus > 0:
                result["k_sq_minus"] = float(k_sq_minus)
                result["k_sq_plus"] = float(k_sq_plus)

                # Most unstable wavenumber
                k_sq_max = cross / (2 * self.D_a * self.D_h)
                result["k_sq_most_unstable"] = float(k_sq_max)
                result["wavelength_most_unstable"] = float(
                    2 * np.pi / np.sqrt(k_sq_max)
                )

        return result

    @property
    def a_field(self) -> np.ndarray:
        """Activator concentration field a(x)."""
        return self._a.copy()

    @property
    def h_field(self) -> np.ndarray:
        """Inhibitor concentration field h(x)."""
        return self._h.copy()

    @property
    def mean_a(self) -> float:
        """Spatial mean of activator."""
        return float(np.mean(self._a))

    @property
    def mean_h(self) -> float:
        """Spatial mean of inhibitor."""
        return float(np.mean(self._h))

    @property
    def total_a(self) -> float:
        """Total activator concentration (integral over domain)."""
        return float(np.sum(self._a) * self.dx)

    @property
    def total_h(self) -> float:
        """Total inhibitor concentration (integral over domain)."""
        return float(np.sum(self._h) * self.dx)

    @property
    def spatial_heterogeneity_a(self) -> float:
        """Coefficient of variation of a (std/mean). Higher = more pattern."""
        mean = np.mean(self._a)
        if mean < 1e-15:
            return 0.0
        return float(np.std(self._a) / mean)

    @property
    def spatial_heterogeneity_h(self) -> float:
        """Coefficient of variation of h (std/mean). Higher = more pattern."""
        mean = np.mean(self._h)
        if mean < 1e-15:
            return 0.0
        return float(np.std(self._h) / mean)

    def dominant_wavelength(self) -> float:
        """Measure the dominant spatial wavelength of a via FFT.

        Returns the wavelength corresponding to the largest-amplitude
        Fourier mode (excluding mode 0).
        """
        a_hat = np.fft.fft(self._a)
        power = np.abs(a_hat[1:self.N // 2]) ** 2
        if np.max(power) < 1e-30:
            return float("inf")
        dominant_mode = np.argmax(power) + 1
        wavelength = self.L / dominant_mode
        return float(wavelength)

    def compute_pattern_energy(self) -> float:
        """Pattern energy: spatial variance of the activator field."""
        if self._a is None:
            return 0.0
        return float(np.var(self._a))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize a and h near the homogeneous steady state with perturbation.

        Small random noise seeds pattern formation via Turing instability.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        a0, h0 = self.homogeneous_steady_state()

        self._a = np.full(self.N, a0, dtype=np.float64)
        self._h = np.full(self.N, h0, dtype=np.float64)

        # Add small spatial perturbation to seed pattern formation
        self._a += 0.01 * a0 * rng.standard_normal(self.N)
        self._h += 0.01 * h0 * rng.standard_normal(self.N)

        # Ensure positive concentrations
        self._a = np.maximum(self._a, 1e-10)
        self._h = np.maximum(self._h, 1e-10)

        self._step_count = 0
        self._state = np.concatenate([self._a, self._h])
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep: spectral diffusion + explicit Euler reaction.

        Operator splitting:
        1. Exact diffusion in Fourier space: a_hat *= exp(-D_a * k^2 * dt)
        2. Reaction step in physical space: Euler on GM kinetics
        """
        dt = self.config.dt

        # Step 1: Spectral diffusion (exact in Fourier space)
        a_hat = np.fft.fft(self._a)
        h_hat = np.fft.fft(self._h)

        a_hat *= self._diff_a
        h_hat *= self._diff_h

        a_diffused = np.real(np.fft.ifft(a_hat))
        h_diffused = np.real(np.fft.ifft(h_hat))

        # Step 2: Reaction terms (explicit Euler)
        # da/dt = rho_a * a^2/h - mu_a*a + rho_0
        # dh/dt = rho_h * a^2 - mu_h*h
        # Avoid division by zero in a^2/h
        h_safe = np.maximum(h_diffused, 1e-15)
        a2_over_h = a_diffused ** 2 / h_safe

        da_react = self.rho_a * a2_over_h - self.mu_a * a_diffused + self.rho_0
        dh_react = self.rho_h * a_diffused ** 2 - self.mu_h * h_diffused

        self._a = a_diffused + dt * da_react
        self._h = h_diffused + dt * dh_react

        # Ensure positive concentrations
        self._a = np.maximum(self._a, 1e-10)
        self._h = np.maximum(self._h, 1e-10)

        self._step_count += 1
        self._state = np.concatenate([self._a, self._h])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state: [a_0..a_N, h_0..h_N] of shape (2*N_grid,)."""
        return self._state
