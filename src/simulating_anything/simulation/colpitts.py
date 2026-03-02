"""Colpitts oscillator simulation -- chaotic electronic oscillator.

Dimensionless jerk form (Kennedy 1994, piecewise-linear model):
    dx/dt = y
    dy/dt = z
    dz/dt = -g_d*z - y + V_cc - Q*max(0, x)

Parameters:
    Q: transistor forward gain (controls chaos onset, classic: 8.0)
    g_d: damping coefficient (parasitic resistance, classic: 0.3)
    V_cc: DC supply voltage bias (classic: 1.0)

The nonlinearity max(0, x) models the piecewise-linear BJT
characteristic in the forward-active region.

Target rediscoveries:
- SINDy recovery of Colpitts ODEs
- Lyapunov exponent estimation for chaos detection
- Q sweep for chaos transition (Q_c ~ 7)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ColpittsSimulation(SimulationEnvironment):
    """Colpitts oscillator: chaotic electronic circuit in jerk form.

    State vector: [x, y, z]

    ODEs:
        dx/dt = y
        dy/dt = z
        dz/dt = -g_d*z - y + V_cc - Q*max(0, x)

    The piecewise-linear nonlinearity max(0, x) models the transistor
    characteristic. For Q > Q_c ~ 7 (with g_d=0.3), the system exhibits
    chaotic oscillations.

    Parameters:
        Q: transistor forward gain (classic value: 8.0 for chaos)
        g_d: damping coefficient (classic value: 0.3)
        V_cc: DC supply voltage bias (classic value: 1.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.Q = p.get("Q", 8.0)
        self.g_d = p.get("g_d", 0.3)
        self.V_cc = p.get("V_cc", 1.0)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Colpitts state."""
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

    @staticmethod
    def _h(x: float) -> float:
        """Piecewise-linear transistor characteristic.

        h(x) = max(0, x) = 0 for x < 0, x for x >= 0
        Models the BJT forward-active region.
        """
        return max(0.0, x)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Colpitts jerk equations."""
        x, y, z = state
        dx = y
        dy = z
        dz = -self.g_d * z - y + self.V_cc - self.Q * self._h(x)
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Colpitts system.

        At equilibrium: dx/dt = dy/dt = dz/dt = 0
            y = 0, z = 0
            -g_d*0 - 0 + V_cc - Q*max(0, x) = 0

        Case 1 (x < 0): V_cc = 0, impossible if V_cc > 0
        Case 2 (x >= 0): V_cc - Q*x = 0 => x = V_cc/Q

        So the unique fixed point is (V_cc/Q, 0, 0).
        """
        x_eq = self.V_cc / self.Q
        return [np.array([x_eq, 0.0, 0.0], dtype=np.float64)]

    def estimate_lyapunov(
        self, n_steps: int = 20000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the method of Wolf et al. (1985): track two nearby
        trajectories, renormalize when they diverge too far.
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

    def measure_period(
        self, n_transient: int = 5000, n_measure: int = 20000
    ) -> float:
        """Measure the oscillation period by detecting zero crossings of x.

        Returns the average period, or np.inf if no complete cycle detected.
        """
        dt = self.config.dt
        # The attractor oscillates around x = V_cc/Q
        x_center = self.V_cc / self.Q

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Detect positive-going crossings of x through x_center
        crossings: list[float] = []
        prev_x = self._state[0] - x_center
        for _ in range(n_measure):
            self.step()
            curr_x = self._state[0] - x_center
            if prev_x < 0 and curr_x >= 0:
                frac = (
                    -prev_x / (curr_x - prev_x)
                    if curr_x != prev_x
                    else 0.5
                )
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_x = curr_x

        if len(crossings) < 2:
            return np.inf

        periods = np.diff(crossings)
        return float(np.mean(periods))
