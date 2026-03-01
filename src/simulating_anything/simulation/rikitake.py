"""Rikitake dynamo simulation -- geomagnetic field reversals.

Models the Rikitake two-disc dynamo system, a minimal model for
geomagnetic polarity reversals:
    dx/dt = -mu*x + z*y
    dy/dt = -mu*y + (z - a)*x
    dz/dt = 1 - x*y

Target rediscoveries:
- SINDy recovery of the Rikitake ODEs
- Chaotic polarity reversals (sign changes in x or y)
- Lyapunov exponent estimation as a function of a
- Fixed point computation and verification
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RikitakeSimulation(SimulationEnvironment):
    """Rikitake two-disc dynamo system.

    State vector: [x, y, z]
        x, y: currents in the two disc dynamos
        z: angular velocity difference between the two discs

    Parameters:
        mu: viscous dissipation coefficient (default: 1.0)
        a: asymmetry parameter (default: 5.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.mu = p.get("mu", 1.0)
        self.a = p.get("a", 5.0)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 1.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Rikitake state."""
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
        """Rikitake equations.

        dx/dt = -mu*x + z*y
        dy/dt = -mu*y + (z - a)*x
        dz/dt = 1 - x*y
        """
        x, y, z = state
        dx = -self.mu * x + z * y
        dy = -self.mu * y + (z - self.a) * x
        dz = 1.0 - x * y
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the two fixed points of the Rikitake system.

        At equilibrium:
            -mu*x + z*y = 0  =>  z = mu*x/y
            -mu*y + (z-a)*x = 0  =>  z = a + mu*y/x
            1 - x*y = 0  =>  y = 1/x

        Substituting y = 1/x into the z equations:
            z = mu*x^2
            z = a + mu/x^2
        Equating: mu*x^2 = a + mu/x^2
            mu*x^4 - a*x^2 - mu = 0

        Solving the quadratic in x^2:
            x^2 = (a + sqrt(a^2 + 4*mu^2)) / (2*mu)
        """
        discriminant = self.a**2 + 4.0 * self.mu**2
        x2 = (self.a + np.sqrt(discriminant)) / (2.0 * self.mu)
        x_pos = np.sqrt(x2)

        points = []
        for x_val in [x_pos, -x_pos]:
            y_val = 1.0 / x_val
            z_val = self.mu * x_val**2
            points.append(np.array([x_val, y_val, z_val]))

        return points

    def count_reversals(
        self,
        n_transient: int = 5000,
        n_measure: int = 50000,
    ) -> dict[str, int | float]:
        """Count polarity reversals (sign changes in x) after transient.

        Returns:
            Dict with reversal counts and mean interval between reversals.
        """
        state = self._state.copy()

        # Skip transient
        for _ in range(n_transient):
            k1 = self._derivatives(state)
            k2 = self._derivatives(state + 0.5 * self.config.dt * k1)
            k3 = self._derivatives(state + 0.5 * self.config.dt * k2)
            k4 = self._derivatives(state + self.config.dt * k3)
            state = state + (self.config.dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )

        # Count sign changes in x
        x_signs = []
        reversal_times = []
        prev_sign = np.sign(state[0])

        for step_idx in range(n_measure):
            k1 = self._derivatives(state)
            k2 = self._derivatives(state + 0.5 * self.config.dt * k1)
            k3 = self._derivatives(state + 0.5 * self.config.dt * k2)
            k4 = self._derivatives(state + self.config.dt * k3)
            state = state + (self.config.dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )

            curr_sign = np.sign(state[0])
            if curr_sign != 0 and curr_sign != prev_sign:
                reversal_times.append(step_idx * self.config.dt)
                prev_sign = curr_sign
            x_signs.append(state[0])

        n_reversals = len(reversal_times)
        mean_interval = 0.0
        if n_reversals > 1:
            intervals = np.diff(reversal_times)
            mean_interval = float(np.mean(intervals))

        return {
            "n_reversals": n_reversals,
            "mean_interval": mean_interval,
            "total_time": n_measure * self.config.dt,
            "x_std": float(np.std(x_signs)),
        }

    def estimate_lyapunov(
        self,
        n_steps: int = 50000,
        dt: float | None = None,
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the Wolf et al. (1985) renormalization method.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0, 0])

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance state1
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance state2
            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Compute distance and renormalize
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)
