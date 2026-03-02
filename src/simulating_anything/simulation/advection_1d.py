"""1D linear advection equation simulation.

Target rediscoveries:
- Exact solution: u(x,t) = u0(x - c*t) (translation at speed c)
- Pulse speed equals wave speed c
- Mass conservation: integral of u is constant
- CFL stability: requires dt <= dx / |c| (Courant number <= 1)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class Advection1DSimulation(SimulationEnvironment):
    """1D linear advection equation: du/dt + c * du/dx = 0.

    Periodic boundary conditions on [0, L].

    Solver: first-order upwind finite difference scheme.
      - c > 0: backward difference (information flows right)
      - c < 0: forward difference (information flows left)

    The CFL condition dt <= dx / |c| must be satisfied for stability.

    State: u(x) on a 1D grid of N points.

    Parameters:
        c: wave speed (default 1.0)
        N: number of grid points (default 256)
        L: domain length (default 10.0)
        sigma: Gaussian pulse width (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.c = p.get("c", 1.0)
        self.N = int(p.get("N", 256))
        self.L = p.get("L", 10.0)
        self.sigma = p.get("sigma", 0.5)

        # Spatial grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # CFL stability: dt must satisfy dt <= dx / |c|
        if abs(self.c) > 0:
            dt_max = self.dx / abs(self.c)
            if self.config.dt > dt_max:
                # Override dt to satisfy CFL with safety factor
                self.config = SimulationConfig(
                    domain=config.domain,
                    backend=config.backend,
                    grid_resolution=config.grid_resolution,
                    domain_size=config.domain_size,
                    dt=0.8 * dt_max,
                    n_steps=config.n_steps,
                    parameters=config.parameters,
                    boundary_conditions=config.boundary_conditions,
                    initial_conditions=config.initial_conditions,
                    seed=config.seed,
                )

        self.courant = abs(self.c) * self.config.dt / self.dx if self.dx > 0 else 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize with a Gaussian pulse at x = L/4."""
        center = self.L / 4.0
        # Use shortest periodic distance so the pulse is consistent
        # with the exact_solution method on a periodic domain
        dx_arr = self.x - center
        dx_arr = dx_arr - self.L * np.round(dx_arr / self.L)
        self._state = np.exp(
            -(dx_arr ** 2) / (2 * self.sigma ** 2)
        ).astype(np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using upwind finite difference."""
        dt = self.config.dt
        u = self._state
        c = self.c

        if c >= 0:
            # Backward difference: du/dx ~ (u[i] - u[i-1]) / dx
            # u[i]^{n+1} = u[i]^n - c * dt/dx * (u[i]^n - u[i-1]^n)
            u_new = u - c * dt / self.dx * (u - np.roll(u, 1))
        else:
            # Forward difference: du/dx ~ (u[i+1] - u[i]) / dx
            # u[i]^{n+1} = u[i]^n - c * dt/dx * (u[i+1]^n - u[i]^n)
            u_new = u - c * dt / self.dx * (np.roll(u, -1) - u)

        self._state = u_new
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current u field."""
        return self._state

    @property
    def total_mass(self) -> float:
        """Total mass (integral of u over domain)."""
        return float(np.sum(self._state) * self.dx)

    @property
    def peak_value(self) -> float:
        """Maximum value of the field."""
        return float(np.max(self._state))

    @property
    def peak_position(self) -> float:
        """Position of the field maximum."""
        idx = int(np.argmax(self._state))
        return float(self.x[idx])

    def exact_solution(self, t: float) -> np.ndarray:
        """Exact solution: u0(x - c*t) with periodic wrapping.

        The initial condition is a Gaussian centered at L/4.
        The exact advection solution shifts the center to (L/4 + c*t) mod L.
        We compute the shortest periodic distance from each grid point to
        the shifted center.
        """
        center = self.L / 4.0
        shifted_center = (center + self.c * t) % self.L
        # Shortest periodic distance from each x to the shifted center
        dx_arr = self.x - shifted_center
        dx_arr = dx_arr - self.L * np.round(dx_arr / self.L)
        return np.exp(-(dx_arr ** 2) / (2 * self.sigma ** 2))

    def l2_error(self, t: float) -> float:
        """L2 norm of the error relative to the exact solution."""
        exact = self.exact_solution(t)
        return float(np.sqrt(np.sum((self._state - exact) ** 2) * self.dx))

    @property
    def cfl_number(self) -> float:
        """Current Courant-Friedrichs-Lewy number: |c| * dt / dx."""
        return self.courant
