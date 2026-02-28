"""Brusselator chemical oscillator simulation.

Target rediscoveries:
- Hopf bifurcation: b_c = 1 + a^2 (oscillation onset)
- Limit cycle amplitude and period as functions of (a, b)
- Fixed point: (u*, v*) = (a, b/a)
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BrusselatorSimulation(SimulationEnvironment):
    """Brusselator: du/dt = a - (b+1)*u + u^2*v, dv/dt = b*u - u^2*v.

    State vector: [u, v] where u, v are chemical concentrations.

    The system has a unique fixed point at (a, b/a).
    Hopf bifurcation occurs at b = 1 + a^2:
    - b < 1 + a^2: stable fixed point
    - b > 1 + a^2: stable limit cycle (oscillations)

    Parameters:
        a: production rate (default 1.0)
        b: control parameter (default 3.0, above Hopf for a=1)
        u_0: initial u concentration
        v_0: initial v concentration
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.0)
        self.b = p.get("b", 3.0)
        self.u_0 = p.get("u_0", 1.0)
        self.v_0 = p.get("v_0", 1.0)

    @property
    def fixed_point(self) -> tuple[float, float]:
        """The unique fixed point (u*, v*) = (a, b/a)."""
        return (self.a, self.b / self.a)

    @property
    def hopf_threshold(self) -> float:
        """Critical b for Hopf bifurcation: b_c = 1 + a^2."""
        return 1.0 + self.a**2

    @property
    def is_oscillatory(self) -> bool:
        """True if b > b_c (above Hopf bifurcation)."""
        return self.b > self.hopf_threshold

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize concentrations."""
        self._state = np.array([self.u_0, self.v_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [u, v]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        u, v = y
        du = self.a - (self.b + 1) * u + u**2 * v
        dv = self.b * u - u**2 * v
        return np.array([du, dv])

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period via zero crossings of u - u*."""
        if not self.is_oscillatory:
            return float("inf")

        dt = self.config.dt
        u_star = self.a  # fixed point u

        # Transient
        transient_steps = int(200 / dt)
        for _ in range(transient_steps):
            self.step()

        # Detect crossings of u = u_star (upward)
        crossings = []
        prev_u = self._state[0]
        for _ in range(int(n_periods * 50 / dt)):
            self.step()
            u = self._state[0]
            if prev_u < u_star and u >= u_star:
                t_cross = (self._step_count - 1) * dt + dt * (u_star - prev_u) / (u - prev_u)
                crossings.append(t_cross)
            prev_u = u

        if len(crossings) < 2:
            return float("inf")

        return float(np.mean(np.diff(crossings)))
