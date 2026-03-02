"""1D viscous Burgers equation simulation.

PDE: du/dt + u * du/dx = nu * d^2u/dx^2

Combines nonlinear advection (u*u_x) with linear diffusion (nu*u_xx).
This is the simplest PDE with both nonlinearity and dissipation, making it
a fundamental test case for numerical methods and turbulence modeling.

Target rediscoveries:
- Shock formation time: t_s = 1 / max(du0/dx) for inviscid limit
- Cole-Hopf analytical solution for sin(x) IC on [0, 2*pi]
- Energy dissipation: dE/dt = -nu * integral(u_x^2) dx
- Viscosity scaling: sharper gradients as nu -> 0
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class Burgers1DSimulation(SimulationEnvironment):
    """1D viscous Burgers equation: du/dt + u * du/dx = nu * d^2u/dx^2.

    Periodic boundary conditions on [0, L].

    Solver: method-of-lines with
      - Upwind finite differences for the advection term (u * du/dx)
      - Central differences for the diffusion term (nu * d^2u/dx^2)
      - RK4 time integration

    CFL condition: dt < min(dx / max|u|, dx^2 / (2*nu))

    State: u(x) on a 1D grid of N points.

    Parameters:
        nu: kinematic viscosity (default 0.01)
        N: number of grid points (default 256)
        L: domain length (default 2*pi)
        amplitude: initial condition amplitude (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.nu = p.get("nu", 0.01)
        self.N = int(p.get("N", 256))
        self.L = p.get("L", 2 * np.pi)
        self.amplitude = p.get("amplitude", 1.0)

        # Spatial grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize with u(x, 0) = amplitude * sin(x) on [0, 2*pi].

        This IC develops a sawtooth profile (shock) before viscous decay
        smooths the solution. It also admits the Cole-Hopf exact solution.
        """
        self._state = (self.amplitude * np.sin(
            2 * np.pi * self.x / self.L
        )).astype(np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current u field."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, u: np.ndarray) -> np.ndarray:
        """Compute du/dt = -u * du/dx + nu * d^2u/dx^2.

        Uses upwind differences for the nonlinear advection term and
        central differences for the diffusion term.
        """
        dx = self.dx

        # Diffusion: central differences for u_xx
        # u_xx[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx^2
        u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)

        # Advection: upwind scheme for u * du/dx
        # Where u > 0, use backward difference; where u < 0, forward difference
        du_backward = (u - np.roll(u, 1)) / dx   # (u[i] - u[i-1]) / dx
        du_forward = (np.roll(u, -1) - u) / dx   # (u[i+1] - u[i]) / dx

        # Upwind selection based on sign of u
        advection = np.where(u >= 0, u * du_backward, u * du_forward)

        return -advection + self.nu * u_xx

    @property
    def total_energy(self) -> float:
        """Total kinetic energy: 0.5 * integral(u^2) dx."""
        return 0.5 * float(np.sum(self._state ** 2) * self.dx)

    @property
    def enstrophy(self) -> float:
        """Enstrophy (gradient energy): integral(u_x^2) dx.

        Related to the energy dissipation rate: dE/dt = -nu * enstrophy.
        """
        u_x = (np.roll(self._state, -1) - np.roll(self._state, 1)) / (
            2 * self.dx
        )
        return float(np.sum(u_x ** 2) * self.dx)

    @property
    def energy_dissipation_rate(self) -> float:
        """Instantaneous energy dissipation rate: nu * integral(u_x^2) dx."""
        return self.nu * self.enstrophy

    @property
    def max_gradient(self) -> float:
        """Maximum absolute spatial gradient |du/dx|."""
        u_x = (np.roll(self._state, -1) - np.roll(self._state, 1)) / (
            2 * self.dx
        )
        return float(np.max(np.abs(u_x)))

    @property
    def total_mass(self) -> float:
        """Total mass (integral of u over domain). Conserved for periodic BC."""
        return float(np.sum(self._state) * self.dx)

    @property
    def max_value(self) -> float:
        """Maximum value of the field."""
        return float(np.max(self._state))

    @property
    def min_value(self) -> float:
        """Minimum value of the field."""
        return float(np.min(self._state))

    def cfl_advection(self) -> float:
        """CFL number for the advection term: max|u| * dt / dx."""
        u_max = np.max(np.abs(self._state))
        if u_max < 1e-15:
            return 0.0
        return float(u_max * self.config.dt / self.dx)

    def cfl_diffusion(self) -> float:
        """CFL number for the diffusion term: nu * dt / dx^2.

        Stability requires this to be < 0.5 for explicit methods.
        """
        return float(self.nu * self.config.dt / (self.dx ** 2))

    def cole_hopf_exact(self, t: float) -> np.ndarray:
        """Cole-Hopf exact solution for u(x, 0) = amplitude * sin(2*pi*x/L).

        The Cole-Hopf transformation u = -2*nu * (d/dx) ln(phi) reduces the
        viscous Burgers equation to the heat equation for phi. The exact
        solution uses a Fourier series of modified Bessel functions.

        This is computed via the ratio of two Fourier sums:
            phi(x,t) = sum_{n=-inf}^{inf} a_n * exp(-n^2*(2*pi/L)^2*nu*t)
                        * exp(i*n*2*pi*x/L)

        For the sin(x) IC the coefficients involve Bessel functions I_n.
        """
        from scipy.special import iv as bessel_iv

        k0 = 2 * np.pi / self.L
        R = self.amplitude / (2 * self.nu * k0)

        # Number of Fourier terms (truncate when Bessel function is negligible)
        n_max = max(int(R) + 30, 50)

        # Compute phi(x,t) = sum_n I_n(R) * exp(-n^2*k0^2*nu*t) * exp(i*n*k0*x)
        # where I_n is the modified Bessel function of the first kind.
        # From Cole-Hopf with sin IC: a_n = I_n(R) (with appropriate sign)
        phi = np.zeros_like(self.x)
        dphi_dx = np.zeros_like(self.x)

        for n in range(-n_max, n_max + 1):
            # The coefficient is I_n(R) with sign (-1)^n for sin IC convention
            coeff = bessel_iv(n, R) * np.exp(-n ** 2 * k0 ** 2 * self.nu * t)
            basis = np.exp(1j * n * k0 * self.x)
            phi = phi + coeff * np.real(basis)
            dphi_dx = dphi_dx + coeff * np.real(1j * n * k0 * basis)

        # u = -2*nu * dphi_dx / phi
        u_exact = -2.0 * self.nu * dphi_dx / phi
        return u_exact

    def shock_formation_time_inviscid(self) -> float:
        """Theoretical shock formation time for the inviscid Burgers equation.

        For u(x,0) = A * sin(k*x), the inviscid shock time is t_s = 1/(A*k).
        """
        k0 = 2 * np.pi / self.L
        return 1.0 / (self.amplitude * k0)

    def reynolds_number(self) -> float:
        """Effective Reynolds number: amplitude * L / (2*pi*nu).

        Higher Re means sharper gradients and more nonlinear dynamics.
        """
        if self.nu < 1e-15:
            return float("inf")
        return self.amplitude * self.L / (2 * np.pi * self.nu)

    def mode_amplitudes(self) -> np.ndarray:
        """FFT of the current u field.

        Returns complex Fourier coefficients for modes 0..N-1.
        """
        return np.fft.fft(self._state)
