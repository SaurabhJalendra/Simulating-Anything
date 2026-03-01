"""2D Brusselator reaction-diffusion simulation.

Extends the well-mixed Brusselator ODE to 2D space with diffusion:

    du/dt = D_u * nabla^2(u) + a - (b+1)*u + u^2*v
    dv/dt = D_v * nabla^2(v) + b*u - u^2*v

with periodic boundary conditions on [0, L] x [0, L].

Solver: forward Euler with finite-difference Laplacian (np.roll).

Target rediscoveries:
- Turing instability: b > 1 + a^2 and D_v/D_u > ((a+1)/(a-1))^2 (for a>1)
- Pattern wavelength: lambda ~ 2*pi*sqrt(D_v / (b - 1 - a^2))
- Steady state: (u*, v*) = (a, b/a)
- Pattern selection: hexagonal spots vs stripes depending on parameters
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


def _laplacian_2d_np(field: np.ndarray, dx: float) -> np.ndarray:
    """Compute 2D Laplacian with periodic boundary conditions via np.roll."""
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


class Brusselator2DSimulation(SimulationEnvironment):
    """2D Brusselator reaction-diffusion model.

    State: two chemical concentration fields u(x,y) and v(x,y) on an N x N grid.
    Internal shape: (2, N, N). Observed as flattened (2*N*N,).

    The homogeneous steady state is (u*, v*) = (a, b/a).
    Turing instability occurs when:
        b > 1 + a^2   (necessary condition)
        D_v/D_u > ((a+1)/(a-1))^2   (sufficient diffusion ratio, for a > 1)

    Parameters:
        a: production rate (default 4.5)
        b: control parameter (default 7.0, inside Turing regime for a=4.5)
        D_u: activator diffusion coefficient (default 1.0)
        D_v: inhibitor diffusion coefficient (default 8.0)
        N_grid: grid resolution per side (default 64)
        L_domain: spatial domain length (default 64.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Reaction parameters
        self.a = p.get("a", 4.5)
        self.b = p.get("b", 7.0)

        # Diffusion coefficients
        self.D_u = p.get("D_u", 1.0)
        self.D_v = p.get("D_v", 8.0)

        # Spatial grid
        self.N = int(p.get("N_grid", 64))
        self.L = p.get("L_domain", 64.0)
        self.dx = self.L / self.N

        # CFL stability check: dt < dx^2 / (4 * max(D_u, D_v))
        D_max = max(self.D_u, self.D_v)
        if D_max > 0:
            self.dt_cfl = self.dx ** 2 / (4.0 * D_max)
            if config.dt > self.dt_cfl:
                raise ValueError(
                    f"dt={config.dt} exceeds CFL limit {self.dt_cfl:.6f} "
                    f"for dx={self.dx:.4f}, D_max={D_max}. "
                    f"Reduce dt or increase N_grid."
                )
        else:
            self.dt_cfl = float("inf")

        # State fields
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None

    @property
    def fixed_point(self) -> tuple[float, float]:
        """The homogeneous steady state (u*, v*) = (a, b/a)."""
        if self.a == 0:
            return (0.0, 0.0)
        return (self.a, self.b / self.a)

    @property
    def turing_threshold_b(self) -> float:
        """Critical b for Turing instability: b_c = 1 + a^2."""
        return 1.0 + self.a ** 2

    @property
    def turing_threshold_diffusion_ratio(self) -> float:
        """Critical D_v/D_u ratio for Turing instability.

        For the Brusselator, the diffusion-driven instability requires
        D_v/D_u > ((a+1)/(a-1))^2  (valid for a > 1).
        """
        if self.a <= 1.0:
            return float("inf")
        return ((self.a + 1.0) / (self.a - 1.0)) ** 2

    @property
    def is_turing_unstable(self) -> bool:
        """True if both Turing conditions are satisfied."""
        if self.b <= self.turing_threshold_b:
            return False
        if self.a <= 1.0:
            # For a <= 1, use the b condition only as an approximation
            return True
        return (self.D_v / self.D_u) > self.turing_threshold_diffusion_ratio

    @property
    def theoretical_wavelength(self) -> float:
        """Theoretical dominant Turing wavelength.

        lambda ~ 2*pi * sqrt(D_v / (b - 1 - a^2))
        Only valid when b > 1 + a^2 (Turing regime).
        """
        gap = self.b - 1.0 - self.a ** 2
        if gap <= 0:
            return float("inf")
        return 2.0 * np.pi * np.sqrt(self.D_v / gap)

    @property
    def mean_u(self) -> float:
        """Spatial mean of u field."""
        return float(np.mean(self._u))

    @property
    def mean_v(self) -> float:
        """Spatial mean of v field."""
        return float(np.mean(self._v))

    @property
    def spatial_heterogeneity_u(self) -> float:
        """Coefficient of variation of u (std/mean). Higher = more pattern."""
        return self._spatial_heterogeneity(self._u)

    @property
    def spatial_heterogeneity_v(self) -> float:
        """Coefficient of variation of v (std/mean). Higher = more pattern."""
        return self._spatial_heterogeneity(self._v)

    @property
    def u_field(self) -> np.ndarray:
        """Activator concentration field u(x, y). Shape: (N, N)."""
        return self._u.copy()

    @property
    def v_field(self) -> np.ndarray:
        """Inhibitor concentration field v(x, y). Shape: (N, N)."""
        return self._v.copy()

    def _spatial_heterogeneity(self, field: np.ndarray) -> float:
        """Coefficient of variation of a field (std/mean)."""
        mean = np.mean(field)
        if abs(mean) < 1e-15:
            return 0.0
        return float(np.std(field) / abs(mean))

    def compute_pattern_wavelength(self) -> float:
        """Measure the dominant spatial wavelength of u via 2D FFT.

        Returns the wavelength corresponding to the largest-amplitude
        Fourier mode (excluding the DC component at k=0).
        """
        u_hat = np.fft.fft2(self._u)
        power = np.abs(u_hat) ** 2
        # Zero out the DC component
        power[0, 0] = 0.0

        # Find the peak in the power spectrum
        if np.max(power) < 1e-30:
            return float("inf")

        # Get 2D wavenumber magnitudes
        kx = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K_mag = np.sqrt(KX ** 2 + KY ** 2)

        # Find peak wavenumber (excluding DC)
        peak_idx = np.unravel_index(np.argmax(power), power.shape)
        k_peak = K_mag[peak_idx]

        if k_peak < 1e-15:
            return float("inf")

        return float(2.0 * np.pi / k_peak)

    def compute_radial_power_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the radially averaged power spectrum of u.

        Returns:
            Tuple of (k_bins, power) where k_bins are wavenumber bin centers
            and power is the azimuthally averaged power at each k.
        """
        u_hat = np.fft.fft2(self._u)
        power_2d = np.abs(u_hat) ** 2

        kx = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K_mag = np.sqrt(KX ** 2 + KY ** 2)

        # Bin by wavenumber magnitude
        k_max = np.pi / self.dx  # Nyquist
        n_bins = self.N // 2
        k_bins = np.linspace(0, k_max, n_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        power_radial = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (K_mag >= k_bins[i]) & (K_mag < k_bins[i + 1])
            if np.any(mask):
                power_radial[i] = np.mean(power_2d[mask])

        return k_centers, power_radial

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize u and v near homogeneous steady state with perturbation.

        u is initialized near a with small random noise to seed pattern
        formation. v is initialized near b/a.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        u_star, v_star = self.fixed_point

        self._u = np.full((self.N, self.N), u_star, dtype=np.float64)
        self._v = np.full((self.N, self.N), v_star, dtype=np.float64)

        # Add small spatial perturbation to seed pattern formation
        self._u += 0.05 * rng.standard_normal((self.N, self.N))
        self._v += 0.05 * rng.standard_normal((self.N, self.N))

        # Ensure positive concentrations
        self._u = np.maximum(self._u, 0.01)
        self._v = np.maximum(self._v, 0.01)

        self._step_count = 0
        self._state = np.concatenate([self._u.ravel(), self._v.ravel()])
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep: finite-difference diffusion + Euler reaction."""
        dt = self.config.dt
        u = self._u
        v = self._v

        # Laplacian (periodic BCs via np.roll)
        lap_u = _laplacian_2d_np(u, self.dx)
        lap_v = _laplacian_2d_np(v, self.dx)

        # Reaction terms: du/dt = a - (b+1)*u + u^2*v
        #                 dv/dt = b*u - u^2*v
        u2v = u ** 2 * v
        du = self.D_u * lap_u + self.a - (self.b + 1.0) * u + u2v
        dv = self.D_v * lap_v + self.b * u - u2v

        # Forward Euler
        self._u = u + dt * du
        self._v = v + dt * dv

        # Ensure positive concentrations
        self._u = np.maximum(self._u, 0.0)
        self._v = np.maximum(self._v, 0.0)

        self._step_count += 1
        self._state = np.concatenate([self._u.ravel(), self._v.ravel()])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state: [u_flat, v_flat] with shape (2*N*N,)."""
        return self._state
