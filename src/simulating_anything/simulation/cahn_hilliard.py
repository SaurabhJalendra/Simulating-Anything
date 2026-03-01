"""Cahn-Hilliard equation simulation -- phase separation in binary mixtures.

Solves the Cahn-Hilliard equation on a 2D periodic domain using a semi-implicit
spectral method (FFT):

    du/dt = M * nabla^2 (f'(u) - epsilon^2 * nabla^2 u)

where f(u) = (u^2 - 1)^2 / 4 is the symmetric double-well potential, so
f'(u) = u^3 - u.

Target rediscoveries:
- Mass conservation: integral(u) dx = const
- Energy decrease: E[u] = integral(f(u) + eps^2/2 |grad u|^2) dx decreases
- Coarsening law: domain size L(t) ~ t^(1/3) (Lifshitz-Slyozov)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CahnHilliardSimulation(SimulationEnvironment):
    """2D Cahn-Hilliard equation via semi-implicit spectral method.

    State: concentration field u(x,y) on an NxN periodic grid, flattened to N*N.
    The order parameter u is in [-1, 1], where u=+1 and u=-1 are the two phases.

    Semi-implicit spectral scheme:
    - Nonlinear chemical potential f'(u) = u^3 - u treated explicitly
    - Biharmonic (4th order) diffusion treated implicitly for stability

    In Fourier space, the Cahn-Hilliard equation reads:
        du_hat/dt = -M*k^2*FFT(f'(u)) - M*eps^2*k^4*u_hat

    Semi-implicit update:
        u_hat^{n+1} = (u_hat^n - dt*M*k^2*FFT(f'(u^n))) / (1 + dt*M*eps^2*k^4)

    Parameters:
        M: mobility coefficient (default 1.0)
        epsilon: interfacial width parameter (default 0.05)
        N: grid resolution per side (default 64)
        L: domain length (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.M = p.get("M", 1.0)
        self.epsilon = p.get("epsilon", 0.05)
        self.N = int(p.get("N", 64))
        self.L = p.get("L", 1.0)

        self.dx = self.L / self.N
        self._setup_spectral()

    def _setup_spectral(self) -> None:
        """Precompute wavenumber arrays and semi-implicit coefficients."""
        N = self.N
        k = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.kx, self.ky = np.meshgrid(k, k)
        self.k_sq = self.kx**2 + self.ky**2
        self.k_fourth = self.k_sq**2

        # Semi-implicit scheme:
        # Denominator: 1 + dt*M*eps^2*k^4 (implicit biharmonic)
        dt = self.config.dt
        self._denom = 1.0 + dt * self.M * self.epsilon**2 * self.k_fourth
        # Numerator coefficient for explicit nonlinear term: dt * M * k^2
        self._num_coeff = dt * self.M * self.k_sq

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize concentration field.

        Default: small random perturbation around u=0 (spinodal region).
        """
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        self._u = 0.05 * rng.standard_normal((self.N, self.N))
        self._u = self._u.astype(np.float64)
        self._state = self._u.flatten()
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using semi-implicit spectral method.

        The Cahn-Hilliard equation in Fourier space:
            du_hat/dt = -M*k^2*fprime_hat - M*eps^2*k^4*u_hat

        Semi-implicit (biharmonic implicit, nonlinear explicit):
            u_hat^{n+1} = (u_hat^n - dt*M*k^2*FFT(f'(u^n)))
                          / (1 + dt*M*eps^2*k^4)
        """
        u = self._u
        # Nonlinear chemical potential: f'(u) = u^3 - u
        fprime = u**3 - u

        u_hat = np.fft.fft2(u)
        fprime_hat = np.fft.fft2(fprime)

        # Semi-implicit update (note the minus sign on the explicit term)
        u_hat_new = (u_hat - self._num_coeff * fprime_hat) / self._denom

        self._u = np.real(np.fft.ifft2(u_hat_new))
        self._state = self._u.flatten()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current concentration field (flattened)."""
        return self._state

    def spinodal_initial_condition(
        self, mean: float = 0.0, noise: float = 0.05, seed: int | None = None
    ) -> np.ndarray:
        """Set initial condition as random perturbation around a mean value.

        Args:
            mean: Mean concentration (0.0 = critical quench, symmetric phases).
            noise: Standard deviation of the Gaussian noise.
            seed: Random seed.

        Returns:
            Initial state array.
        """
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        self._u = mean + noise * rng.standard_normal((self.N, self.N))
        self._u = self._u.astype(np.float64)
        self._state = self._u.flatten()
        self._step_count = 0
        return self._state

    def compute_free_energy(self) -> float:
        """Compute the Ginzburg-Landau free energy functional.

        E[u] = integral( f(u) + eps^2/2 * |grad u|^2 ) dx

        where f(u) = (u^2 - 1)^2 / 4 is the double-well potential.
        """
        u = self._u
        # Double-well potential: f(u) = (u^2 - 1)^2 / 4
        f_bulk = (u**2 - 1.0)**2 / 4.0

        # Gradient energy via spectral differentiation
        u_hat = np.fft.fft2(u)
        ux = np.real(np.fft.ifft2(1j * self.kx * u_hat))
        uy = np.real(np.fft.ifft2(1j * self.ky * u_hat))
        grad_sq = ux**2 + uy**2

        f_gradient = 0.5 * self.epsilon**2 * grad_sq

        # Integrate over domain (mean * area)
        energy = np.mean(f_bulk + f_gradient) * self.L**2
        return float(energy)

    def compute_total_mass(self) -> float:
        """Compute total mass: integral of u over the domain.

        Should be conserved by the Cahn-Hilliard dynamics.
        """
        return float(np.mean(self._u) * self.L**2)

    def compute_interface_length(self, threshold: float = 0.5) -> float:
        """Estimate total interface perimeter.

        Counts grid cells where |u| < threshold as interface region,
        then estimates the length from the area fraction.

        Args:
            threshold: Value of |u| below which a cell is considered interfacial.

        Returns:
            Estimated interface length.
        """
        interface_mask = np.abs(self._u) < threshold
        interface_fraction = np.mean(interface_mask)
        # Rough estimate: perimeter ~ interface_area / interface_width
        # Interface width ~ epsilon, interface area ~ fraction * L^2
        interface_area = interface_fraction * self.L**2
        return float(interface_area)

    def coarsening_analysis(self, n_snapshots: int = 20) -> dict[str, np.ndarray]:
        """Measure domain size L(t) over time via structure factor peak.

        Runs the simulation and records the characteristic length scale at
        evenly-spaced snapshots. The length scale is extracted from the peak
        of the radially-averaged structure factor S(k).

        Args:
            n_snapshots: Number of measurement points.

        Returns:
            Dict with 'times' and 'length_scales' arrays.
        """
        total_steps = self.config.n_steps
        snapshot_interval = max(1, total_steps // n_snapshots)

        times = []
        length_scales = []

        for step_idx in range(total_steps):
            self.step()
            if (step_idx + 1) % snapshot_interval == 0:
                t = (step_idx + 1) * self.config.dt
                L_char = self._characteristic_length()
                times.append(t)
                length_scales.append(L_char)

        return {
            "times": np.array(times),
            "length_scales": np.array(length_scales),
        }

    def _characteristic_length(self) -> float:
        """Compute characteristic domain size from structure factor peak.

        Uses the radially-averaged structure factor S(k) and finds the
        wavenumber k* at the peak. The characteristic length is 2*pi/k*.
        """
        u_hat = np.fft.fft2(self._u)
        S_k = np.abs(u_hat)**2 / self.N**4

        k_mag = np.sqrt(self.k_sq)
        k_max = self.N // 2
        dk = 2 * np.pi / self.L

        # Radial binning
        k_bins = np.arange(1, k_max + 1) * dk
        S_radial = np.zeros(k_max)
        for i in range(k_max):
            k_low = (i + 0.5) * dk
            k_high = (i + 1.5) * dk
            mask = (k_mag >= k_low) & (k_mag < k_high)
            if np.any(mask):
                S_radial[i] = np.mean(S_k[mask])

        # Find peak (skip k=0 bin)
        if np.max(S_radial) > 0:
            peak_idx = np.argmax(S_radial)
            k_star = k_bins[peak_idx]
            if k_star > 0:
                return float(2 * np.pi / k_star)

        return float(self.L)

    def energy_vs_time(self, n_steps: int | None = None) -> dict[str, np.ndarray]:
        """Track free energy over time.

        Args:
            n_steps: Number of steps to run (default: config.n_steps).

        Returns:
            Dict with 'times' and 'energies' arrays.
        """
        if n_steps is None:
            n_steps = self.config.n_steps

        times = [self._step_count * self.config.dt]
        energies = [self.compute_free_energy()]

        for _ in range(n_steps):
            self.step()
            times.append(self._step_count * self.config.dt)
            energies.append(self.compute_free_energy())

        return {
            "times": np.array(times),
            "energies": np.array(energies),
        }
