"""1D Shallow Water Equations simulation.

Target rediscoveries:
- Gravity wave speed: c = sqrt(g * h)
- Mass conservation: integral(h * dx) = const
- Energy conservation: integral(0.5*h*u^2 + 0.5*g*h^2) * dx ~ const

The 1D shallow water equations in conservative form:
    dh/dt + d(h*u)/dx = 0               [mass conservation]
    d(hu)/dt + d(h*u^2 + 0.5*g*h^2)/dx = 0  [momentum conservation]

Solver: Lax-Friedrichs finite difference scheme on a periodic domain [0, L].
State vector: [h_1..h_N, u_1..u_N] shape (2*N,).
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ShallowWater(SimulationEnvironment):
    """1D shallow water equations with periodic boundary conditions.

    Solver: Lax-Friedrichs scheme for the flux-conservative form.
    CFL condition: dt < dx / max(|u| + sqrt(g*h)).

    State vector: [h_1, ..., h_N, u_1, ..., u_N].

    Parameters:
        g: gravitational acceleration (default 9.81)
        L: domain length (default 10.0)
        N: number of grid points (default 128)
        h0: background water depth (default 1.0)
        perturbation_amplitude: initial bump height (default 0.1)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.g = p.get("g", 9.81)
        self.L = p.get("L", 10.0)
        self.N = int(p.get("N", 128))
        self.h0 = p.get("h0", 1.0)
        self.perturbation_amplitude = p.get("perturbation_amplitude", 0.1)

        # Spatial grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Internal fields
        self._h: np.ndarray | None = None
        self._u: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize with flat surface plus a Gaussian bump perturbation."""
        # Background depth with a Gaussian bump
        center = self.L / 2
        sigma = self.L / 20
        bump = self.perturbation_amplitude * np.exp(
            -((self.x - center) ** 2) / (2 * sigma ** 2)
        )

        self._h = np.full(self.N, self.h0, dtype=np.float64) + bump
        self._u = np.zeros(self.N, dtype=np.float64)

        self._step_count = 0
        self._state = np.concatenate([self._h, self._u])
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using Lax-Friedrichs scheme."""
        dt = self.config.dt
        dx = self.dx
        g = self.g
        N = self.N
        h = self._h.copy()
        u = self._u.copy()

        # Conservative variables: q1 = h, q2 = h*u
        q1 = h
        q2 = h * u

        # Fluxes: f1 = h*u, f2 = h*u^2 + 0.5*g*h^2
        f1 = q2
        f2 = q2 * u + 0.5 * g * h ** 2

        # Lax-Friedrichs update with periodic boundary (numpy roll)
        # q_new = 0.5*(q[i+1] + q[i-1]) - dt/(2*dx) * (f[i+1] - f[i-1])
        q1_new = (
            0.5 * (np.roll(q1, -1) + np.roll(q1, 1))
            - (dt / (2 * dx)) * (np.roll(f1, -1) - np.roll(f1, 1))
        )
        q2_new = (
            0.5 * (np.roll(q2, -1) + np.roll(q2, 1))
            - (dt / (2 * dx)) * (np.roll(f2, -1) - np.roll(f2, 1))
        )

        # Recover primitive variables
        self._h = q1_new
        # Avoid division by zero in dry regions
        self._u = np.where(q1_new > 1e-12, q2_new / q1_new, 0.0)

        self._step_count += 1
        self._state = np.concatenate([self._h, self._u])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [h_1,...,h_N, u_1,...,u_N]."""
        return self._state

    @property
    def total_mass(self) -> float:
        """Total mass: integral of h over domain (h * dx summed)."""
        return float(np.sum(self._h) * self.dx)

    @property
    def total_energy(self) -> float:
        """Total energy: kinetic + potential.

        E = integral(0.5*h*u^2 + 0.5*g*h^2) dx.
        """
        kinetic = 0.5 * self._h * self._u ** 2
        potential = 0.5 * self.g * self._h ** 2
        return float(np.sum(kinetic + potential) * self.dx)

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy: integral(0.5 * h * u^2) dx."""
        return float(np.sum(0.5 * self._h * self._u ** 2) * self.dx)

    @property
    def potential_energy(self) -> float:
        """Potential energy: integral(0.5 * g * h^2) dx."""
        return float(np.sum(0.5 * self.g * self._h ** 2) * self.dx)

    @property
    def mean_height(self) -> float:
        """Mean water depth across domain."""
        return float(np.mean(self._h))

    @property
    def max_height(self) -> float:
        """Maximum water depth across domain."""
        return float(np.max(self._h))

    @property
    def wave_speed(self) -> float:
        """Theoretical linear gravity wave speed: sqrt(g * h0)."""
        return float(np.sqrt(self.g * self.h0))

    @property
    def height_field(self) -> np.ndarray:
        """Water depth field h(x)."""
        return self._h.copy()

    @property
    def velocity_field(self) -> np.ndarray:
        """Velocity field u(x)."""
        return self._u.copy()

    def cfl_number(self) -> float:
        """Current CFL number: dt * max(|u| + sqrt(g*h)) / dx.

        Must be < 1 for stability.
        """
        max_speed = np.max(np.abs(self._u) + np.sqrt(self.g * np.maximum(self._h, 0)))
        return float(self.config.dt * max_speed / self.dx)
