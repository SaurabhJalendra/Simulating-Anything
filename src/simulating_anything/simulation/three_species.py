"""Three-species food chain (Lotka-Volterra 3-species) simulation.

Target rediscoveries:
- ODE recovery via SINDy:
    dx/dt = a1*x - b1*x*y
    dy/dt = -a2*y + b1*x*y - b2*y*z
    dz/dt = -a3*z + b2*y*z
- Predator-free equilibrium: x* = a2/b1, y* = a1/b1
- Coexistence dynamics
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ThreeSpecies(SimulationEnvironment):
    """Three-species food chain: grass -> herbivore -> predator.

    State vector: [x, y, z] where x = grass, y = herbivore, z = predator.

    Equations:
        dx/dt = a1*x - b1*x*y         (grass grows, eaten by herbivore)
        dy/dt = -a2*y + b1*x*y - b2*y*z  (herbivore eats grass, eaten by predator)
        dz/dt = -a3*z + b2*y*z        (predator eats herbivore)

    Parameters:
        a1: grass growth rate (default 1.0)
        b1: herbivore predation rate on grass (default 0.5)
        a2: herbivore natural death rate (default 0.5)
        b2: predator predation rate on herbivore (default 0.2)
        a3: predator natural death rate (default 0.3)
        x0: initial grass population (default 1.0)
        y0: initial herbivore population (default 0.5)
        z0: initial predator population (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a1 = p.get("a1", 1.0)
        self.b1 = p.get("b1", 0.5)
        self.a2 = p.get("a2", 0.5)
        self.b2 = p.get("b2", 0.2)
        self.a3 = p.get("a3", 0.3)
        self.x0 = p.get("x0", 1.0)
        self.y0 = p.get("y0", 0.5)
        self.z0 = p.get("z0", 0.5)

    @property
    def total_population(self) -> float:
        """Sum of all three species populations."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    @property
    def is_coexisting(self) -> bool:
        """True if all three species are above the extinction threshold."""
        if self._state is None:
            return False
        threshold = 1e-6
        return bool(np.all(self._state > threshold))

    def equilibrium_point(self) -> np.ndarray:
        """Compute the boundary equilibrium (predator-free steady state).

        The 3-species food chain generically has no interior fixed point
        where all three species coexist. An interior equilibrium requires
        a1/b1 = a3/b2, which is a measure-zero condition in parameter space.

        The predator-free boundary equilibrium always exists:
            x* = a2/b1  (prey population at equilibrium)
            y* = a1/b1  (herbivore population at equilibrium)
            z* = 0

        At this point, the predator subsystem has growth rate
        b2*y* - a3 = b2*a1/b1 - a3. If positive, the predator can invade
        and the boundary equilibrium is unstable (leading to oscillations
        or chaos). If negative, predator goes extinct.

        Returns:
            numpy array [x*, y*, z*] of the predator-free equilibrium.
        """
        x_star = self.a2 / self.b1
        y_star = self.a1 / self.b1
        z_star = 0.0
        return np.array([x_star, y_star, z_star], dtype=np.float64)

    def predator_invasion_rate(self) -> float:
        """Growth rate of predator at the predator-free equilibrium.

        If positive, the predator can invade and coexistence dynamics emerge.
        Value: b2 * (a1/b1) - a3
        """
        y_star = self.a1 / self.b1
        return self.b2 * y_star - self.a3

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [x, y, z]."""
        self._state = np.array([self.x0, self.y0, self.z0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [x, y, z]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Ensure non-negative populations
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Three-species food chain right-hand side."""
        x, yh, z = y
        dx = self.a1 * x - self.b1 * x * yh
        dy = -self.a2 * yh + self.b1 * x * yh - self.b2 * yh * z
        dz = -self.a3 * z + self.b2 * yh * z
        return np.array([dx, dy, dz])
