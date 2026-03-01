"""Rabinovich-Fabrikant system simulation -- multiscroll strange attractor.

Models stochasticity arising from modulation instability in nonequilibrium
dissipative media. Describes nonlinear wave interaction in plasma physics.

Target rediscoveries:
- SINDy recovery of RF ODEs:
    dx/dt = y(z - 1 + x^2) + gamma*x
    dy/dt = x(3z + 1 - x^2) + gamma*y
    dz/dt = -2z(alpha + xy)
- Lyapunov exponent estimation (positive for chaotic regime)
- Chaos transition as gamma varies
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RabinovichFabrikantSimulation(SimulationEnvironment):
    """Rabinovich-Fabrikant system: multiscroll strange attractor.

    State vector: [x, y, z]

    ODEs:
        dx/dt = y(z - 1 + x^2) + gamma*x
        dy/dt = x(3z + 1 - x^2) + gamma*y
        dz/dt = -2z(alpha + xy)

    Parameters:
        alpha: dissipation parameter (classic value: 1.1)
        gamma: coupling parameter (classic value: 0.87)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 1.1)
        self.gamma = p.get("gamma", 0.87)
        self.x_0 = p.get("x_0", -1.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Rabinovich-Fabrikant state."""
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

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Rabinovich-Fabrikant equations.

        dx/dt = y(z - 1 + x^2) + gamma*x
        dy/dt = x(3z + 1 - x^2) + gamma*y
        dz/dt = -2z(alpha + xy)
        """
        x, y, z = state
        dx = y * (z - 1.0 + x**2) + self.gamma * x
        dy = x * (3.0 * z + 1.0 - x**2) + self.gamma * y
        dz = -2.0 * z * (self.alpha + x * y)
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute fixed points of the Rabinovich-Fabrikant system.

        The origin is always a fixed point. Additional fixed points exist
        when certain parameter conditions are met. Setting derivatives to zero:
            y(z - 1 + x^2) + gamma*x = 0
            x(3z + 1 - x^2) + gamma*y = 0
            -2z(alpha + xy) = 0

        From the third equation, either z=0 or alpha + xy = 0.

        Case z=0:
            y(-1 + x^2) + gamma*x = 0
            x(1 - x^2) + gamma*y = 0
            From the second: y = -x(1 - x^2)/gamma (if gamma != 0)
            Substituting: -x(1-x^2)(-1+x^2)/gamma + gamma*x = 0
                          x[(1-x^2)(1-x^2)/gamma + gamma] = 0  (note: -(−1+x^2)=(1-x^2))
                          x = 0, or (1-x^2)^2/gamma + gamma = 0
            The latter gives (1-x^2)^2 = -gamma^2, which has no real solution.
            So for z=0, only the origin x=y=z=0.

        Case alpha + xy = 0, i.e. y = -alpha/x (x != 0):
            Non-trivial fixed points exist but depend on parameter values.
        """
        points = [np.array([0.0, 0.0, 0.0])]

        # For the case alpha + xy = 0 => y = -alpha/x
        # Substituting into the first two equations with z != 0 generally
        # requires numerical root finding. We provide the origin as the
        # analytically accessible fixed point.
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

    def measure_period(
        self, n_transient: int = 5000, n_measure: int = 20000
    ) -> float:
        """Measure the oscillation period by detecting zero crossings of x.

        Returns the average period, or np.inf if no complete cycle is detected.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Detect positive-going zero crossings of x
        crossings = []
        prev_x = self._state[0]
        for _ in range(n_measure):
            self.step()
            curr_x = self._state[0]
            if prev_x < 0 and curr_x >= 0:
                frac = -prev_x / (curr_x - prev_x) if curr_x != prev_x else 0.5
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_x = curr_x

        if len(crossings) < 2:
            return np.inf

        periods = np.diff(crossings)
        return float(np.mean(periods))
