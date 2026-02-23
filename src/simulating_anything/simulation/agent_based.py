"""Lotka-Volterra predator-prey ODE simulation."""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

try:
    import diffrax
    import jax.numpy as jnp

    _HAS_DIFFRAX = True
except ImportError:
    _HAS_DIFFRAX = False


class LotkaVolterraSimulation(SimulationEnvironment):
    """Lotka-Volterra predator-prey model.

    State vector: [prey, predator]

    Equations:
        d(prey)/dt = alpha * prey - beta * prey * predator
        d(predator)/dt = delta * prey * predator - gamma * predator
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 1.1)
        self.beta = p.get("beta", 0.4)
        self.gamma = p.get("gamma", 0.4)
        self.delta = p.get("delta", 0.1)
        self.prey_0 = p.get("prey_0", 40.0)
        self.predator_0 = p.get("predator_0", 9.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations."""
        self._state = np.array([self.prey_0, self.predator_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 or diffrax."""
        if _HAS_DIFFRAX:
            self._diffrax_step()
        else:
            self._rk4_step()
        self._step_count += 1
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step (numpy fallback)."""
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
        """Lotka-Volterra right-hand side."""
        prey, pred = y
        dprey = self.alpha * prey - self.beta * prey * pred
        dpred = self.delta * prey * pred - self.gamma * pred
        return np.array([dprey, dpred])

    def _diffrax_step(self) -> None:
        """Single step using diffrax ODE solver."""
        alpha, beta, gamma, delta = self.alpha, self.beta, self.gamma, self.delta

        def vector_field(t, y, args):
            prey, pred = y
            dprey = alpha * prey - beta * prey * pred
            dpred = delta * prey * pred - gamma * pred
            return jnp.array([dprey, dpred])

        t0 = self._step_count * self.config.dt
        t1 = t0 + self.config.dt
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Dopri5()
        y0 = jnp.array(self._state)

        sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=self.config.dt, y0=y0)
        self._state = np.asarray(sol.ys[-1])
        self._state = np.maximum(self._state, 0.0)

    def observe(self) -> np.ndarray:
        """Return current populations [prey, predator]."""
        return self._state
