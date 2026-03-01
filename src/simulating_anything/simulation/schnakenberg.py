"""Schnakenberg reaction-diffusion simulation on a 2D periodic grid.

The Schnakenberg model is a minimal two-component activator-inhibitor system
that produces Turing patterns (spots, stripes, hexagonal structures).

    du/dt = D_u * nabla^2(u) + a - u + u^2 * v
    dv/dt = D_v * nabla^2(v) + b - u^2 * v

with periodic boundary conditions on [0, L]^2.

Solver: operator splitting -- spectral (FFT) exact diffusion in Fourier space
plus explicit Euler for the reaction terms. This avoids CFL restrictions from
diffusion while keeping the reaction terms simple.

Target rediscoveries:
- Homogeneous steady state: u* = a + b, v* = b / (a + b)^2
- Turing instability onset: requires D_v / D_u >> 1
- Pattern wavelength scaling with diffusion ratio
- Phase diagram across (a, b) or D_v parameter space
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SchnakenbergSimulation(SimulationEnvironment):
    """Schnakenberg reaction-diffusion on a 2D periodic domain.

    State: chemical concentrations u(x, y) and v(x, y) on a 2D grid.
    Observe shape: (2*N*N,) = [u_flat, v_flat].

    Parameters:
        D_u: activator diffusion coefficient (default 1.0)
        D_v: inhibitor diffusion coefficient (default 40.0)
        a: kinetic parameter for u production (default 0.1)
        b: kinetic parameter for v production (default 0.9)
        N: grid size in each dimension (default 64)
        L: domain length in each dimension (default 50.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Reaction parameters
        self.a = p.get("a", 0.1)
        self.b = p.get("b", 0.9)

        # Diffusion coefficients
        self.D_u = p.get("D_u", 1.0)
        self.D_v = p.get("D_v", 40.0)

        # Spatial grid
        self.N = int(p.get("N", 64))
        self.L = p.get("L", 50.0)
        self.dx = self.L / self.N

        # 2D wavenumbers for spectral Laplacian (periodic domain)
        kx = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self._k_sq = KX**2 + KY**2

        # Precompute spectral diffusion operators for one timestep
        dt = config.dt
        self._diff_u = np.exp(-self.D_u * self._k_sq * dt)
        self._diff_v = np.exp(-self.D_v * self._k_sq * dt)

        # State placeholders
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None

    def homogeneous_steady_state(self) -> tuple[float, float]:
        """Return the homogeneous steady state (u*, v*).

        Setting du/dt = 0, dv/dt = 0 with no spatial variation:
            a - u + u^2*v = 0
            b - u^2*v = 0
        From the second equation: u^2*v = b, substituting into the first:
            a - u + b = 0 => u* = a + b
            v* = b / u*^2 = b / (a + b)^2
        """
        u_star = self.a + self.b
        v_star = self.b / (u_star**2)
        return (u_star, v_star)

    def turing_analysis(self) -> dict:
        """Compute Turing instability analysis for current parameters.

        The Jacobian of the reaction kinetics at (u*, v*) is:
            J = [[-1 + 2*u*v,  u^2],
                 [-2*u*v,      -u^2]]

        For Turing instability, we need the homogeneous state to be stable
        (tr(J) < 0, det(J) > 0) but unstable to diffusive perturbations.

        Returns a dict with critical wavenumber, unstable range, and
        whether the system is Turing unstable.
        """
        u_star, v_star = self.homogeneous_steady_state()

        # Jacobian elements
        fu = -1.0 + 2 * u_star * v_star  # df/du
        fv = u_star**2                     # df/dv
        gu = -2 * u_star * v_star          # dg/du
        gv = -(u_star**2)                  # dg/dv

        tr_J = fu + gv
        det_J = fu * gv - fv * gu

        # Homogeneous stability requires tr_J < 0 and det_J > 0
        homogeneous_stable = (tr_J < 0) and (det_J > 0)

        # For Turing instability with wavenumber k:
        # The dispersion relation is:
        # sigma(k^2) has roots from:
        #   det(J - D*k^2*I) = 0 where D = diag(D_u, D_v)
        #   => D_u*D_v*k^4 - (D_v*fu + D_u*gv)*k^2 + det_J = 0
        # Turing instability requires D_v*fu + D_u*gv > 0 (since fu > 0, gv < 0)
        # and (D_v*fu + D_u*gv)^2 > 4*D_u*D_v*det_J

        h = self.D_v * fu + self.D_u * gv
        discriminant = h**2 - 4 * self.D_u * self.D_v * det_J

        turing_unstable = homogeneous_stable and (h > 0) and (discriminant > 0)

        result = {
            "u_star": u_star,
            "v_star": v_star,
            "trace_J": tr_J,
            "det_J": det_J,
            "homogeneous_stable": homogeneous_stable,
            "h": h,
            "discriminant": discriminant,
            "turing_unstable": turing_unstable,
        }

        if turing_unstable and discriminant > 0:
            # Critical wavenumber range: k^2 in [k_minus^2, k_plus^2]
            sqrt_disc = np.sqrt(discriminant)
            k_sq_minus = (h - sqrt_disc) / (2 * self.D_u * self.D_v)
            k_sq_plus = (h + sqrt_disc) / (2 * self.D_u * self.D_v)

            if k_sq_minus > 0:
                result["k_sq_minus"] = float(k_sq_minus)
                result["k_sq_plus"] = float(k_sq_plus)

                # Most unstable wavenumber: k^2 = h / (2*D_u*D_v)
                k_sq_max = h / (2 * self.D_u * self.D_v)
                result["k_sq_most_unstable"] = float(k_sq_max)
                result["wavelength_most_unstable"] = float(
                    2 * np.pi / np.sqrt(k_sq_max)
                )

        return result

    def compute_pattern_wavelength(self) -> float:
        """Measure the dominant spatial wavelength of u via 2D FFT.

        Computes the radial power spectrum and returns the wavelength
        corresponding to the peak wavenumber (excluding the DC component).

        Returns:
            Dominant wavelength, or 0.0 if no pattern is detected.
        """
        if self._u is None:
            return 0.0

        u_centered = self._u - np.mean(self._u)
        if np.std(u_centered) < 1e-10:
            return 0.0

        fft2 = np.fft.fft2(u_centered)
        power = np.abs(fft2)**2
        power_shifted = np.fft.fftshift(power)

        cx, cy = self.N // 2, self.N // 2
        Y, X = np.ogrid[-cx:self.N - cx, -cy:self.N - cy]
        R = np.sqrt(X**2 + Y**2).astype(int)
        max_r = min(cx, cy)

        radial_power = np.zeros(max_r)
        for r_val in range(1, max_r):
            mask = R == r_val
            if np.any(mask):
                radial_power[r_val] = np.mean(power_shifted[mask])

        if np.max(radial_power[2:]) > 0:
            peak_r = np.argmax(radial_power[2:]) + 2
            wavelength = self.L / peak_r
            return float(wavelength)

        return 0.0

    def compute_pattern_energy(self) -> float:
        """Pattern energy: spatial variance of the u field."""
        if self._u is None:
            return 0.0
        return float(np.var(self._u))

    def diffusivity_sweep(
        self,
        D_v_values: np.ndarray,
        n_steps: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep D_v and measure pattern formation for each value.

        Args:
            D_v_values: Array of D_v values to sweep.
            n_steps: Number of simulation steps per sweep point.

        Returns:
            Dict with D_v values, wavelengths, and pattern energies.
        """
        wavelengths = []
        energies = []

        for D_v in D_v_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    **self.config.parameters,
                    "D_v": float(D_v),
                },
                seed=self.config.seed,
            )
            sim = SchnakenbergSimulation(config)
            sim.reset()
            for _ in range(n_steps):
                sim.step()

            wavelengths.append(sim.compute_pattern_wavelength())
            energies.append(sim.compute_pattern_energy())

        return {
            "D_v": np.array(D_v_values),
            "wavelength": np.array(wavelengths),
            "energy": np.array(energies),
        }

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize u and v near the homogeneous steady state with perturbation.

        Small random noise seeds pattern formation via Turing instability.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        u_star, v_star = self.homogeneous_steady_state()

        self._u = np.full((self.N, self.N), u_star, dtype=np.float64)
        self._v = np.full((self.N, self.N), v_star, dtype=np.float64)

        # Add small perturbation to seed pattern formation
        self._u += 0.01 * rng.standard_normal((self.N, self.N))
        self._v += 0.01 * rng.standard_normal((self.N, self.N))

        # Ensure positive concentrations
        self._u = np.maximum(self._u, 0.0)
        self._v = np.maximum(self._v, 0.0)

        self._step_count = 0
        self._state = np.concatenate([self._u.ravel(), self._v.ravel()])
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep: spectral diffusion + explicit Euler reaction.

        Operator splitting:
        1. Exact diffusion in Fourier space: u_hat *= exp(-D_u * k^2 * dt)
        2. Reaction step in physical space: Euler on Schnakenberg kinetics
        """
        dt = self.config.dt

        # Step 1: Spectral diffusion (exact in Fourier space)
        u_hat = np.fft.fft2(self._u)
        v_hat = np.fft.fft2(self._v)

        u_hat *= self._diff_u
        v_hat *= self._diff_v

        u_diffused = np.real(np.fft.ifft2(u_hat))
        v_diffused = np.real(np.fft.ifft2(v_hat))

        # Step 2: Reaction terms (explicit Euler)
        # du/dt = a - u + u^2 * v
        # dv/dt = b - u^2 * v
        u2v = u_diffused**2 * v_diffused
        du_react = self.a - u_diffused + u2v
        dv_react = self.b - u2v

        self._u = u_diffused + dt * du_react
        self._v = v_diffused + dt * dv_react

        # Ensure positive concentrations
        self._u = np.maximum(self._u, 0.0)
        self._v = np.maximum(self._v, 0.0)

        self._step_count += 1
        self._state = np.concatenate([self._u.ravel(), self._v.ravel()])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state: [u_flat, v_flat] of shape (2*N*N,)."""
        return self._state

    @property
    def u_field(self) -> np.ndarray:
        """Current u concentration field (N x N)."""
        return self._u.copy()

    @property
    def v_field(self) -> np.ndarray:
        """Current v concentration field (N x N)."""
        return self._v.copy()
