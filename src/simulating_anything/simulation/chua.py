"""Chua's circuit simulation -- double-scroll strange attractor.

Target rediscoveries:
- SINDy recovery of Chua ODEs: x'=alpha*(y-x-f(x)), y'=x-y+z, z'=-beta*y
- Period-doubling route to chaos as alpha varies
- Lyapunov exponent estimation from trajectory divergence
- Three fixed points: origin + two symmetric points
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ChuaCircuit(SimulationEnvironment):
    """Chua's circuit: the simplest electronic circuit exhibiting chaos.

    State vector: [x, y, z]

    ODEs:
        dx/dt = alpha * (y - x - f(x))
        dy/dt = x - y + z
        dz/dt = -beta * y

    where f(x) is Chua's piecewise-linear diode characteristic:
        f(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|)

    This simplifies to:
        f(x) = m0*x   if |x| >= 1
        f(x) = m1*x   if |x| < 1

    Parameters:
        alpha: capacitor ratio (classic: 15.6 for double-scroll)
        beta: inductance parameter (classic: 28.0)
        m0: outer slope of Chua diode (classic: -1.143)
        m1: inner slope of Chua diode (classic: -0.714)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 15.6)
        self.beta = p.get("beta", 28.0)
        self.m0 = p.get("m0", -1.143)
        self.m1 = p.get("m1", -0.714)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Chua circuit state."""
        self._state = np.array(
            [self.x_0, self.y_0, self.z_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y, z]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def chua_diode(self, x: float) -> float:
        """Chua's piecewise-linear diode function.

        f(x) = m0*x + 0.5*(m1 - m0)*(|x+1| - |x-1|)

        Equivalent to:
            f(x) = m1*x               if |x| < 1  (inner region)
            f(x) = m0*x + (m1 - m0)   if x >= 1   (outer region, positive)
            f(x) = m0*x - (m1 - m0)   if x <= -1  (outer region, negative)
        """
        return self.m0 * x + 0.5 * (self.m1 - self.m0) * (
            abs(x + 1.0) - abs(x - 1.0)
        )

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Chua circuit equations."""
        x, y, z = state
        fx = self.chua_diode(x)
        dx = self.alpha * (y - x - fx)
        dy = x - y + z
        dz = -self.beta * y
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the three fixed points of Chua's circuit.

        At equilibrium: dy/dt=0 => x-y+z=0, dz/dt=0 => y=0.
        So y=0, z=-x, and dx/dt=0 => alpha*(0-x-f(x))=0 => x+f(x)=0.

        For the inner region (|x|<1): x + m1*x = 0 => x*(1+m1)=0 => x=0.
        For the outer region (|x|>=1 with x>0):
          f(x) = m0*x + (m1-m0), so x + m0*x + (m1-m0) = 0
          => x = (m0-m1)/(1+m0)
        For |x|>=1 with x<0:
          f(x) = m0*x - (m1-m0), so x + m0*x - (m1-m0) = 0
          => x = (m1-m0)/(1+m0) = -x_pos
        """
        points = [np.array([0.0, 0.0, 0.0])]

        # Non-zero fixed points exist when 1 + m0 != 0
        if abs(1.0 + self.m0) > 1e-12:
            x_pos = (self.m0 - self.m1) / (1.0 + self.m0)
            x_neg = -x_pos

            # These are valid only if |x| >= 1 (outer region)
            if abs(x_pos) >= 1.0:
                points.append(np.array([x_pos, 0.0, -x_pos]))
                points.append(np.array([x_neg, 0.0, -x_neg]))

        return points

    def estimate_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the method of Wolf et al. (1985): track two nearby trajectories,
        renormalize when they diverge too far.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0, 0])

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance both states with RK4
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Compute distance
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                # Renormalize
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)
