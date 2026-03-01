"""Rossler system simulation -- strange attractor.

Target rediscoveries:
- SINDy recovery of Rossler ODEs: x'=-y-z, y'=x+a*y, z'=b+z*(x-c)
- Period-doubling route to chaos as c increases
- Lyapunov exponent estimation from trajectory divergence
- Fixed point computation and verification
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RosslerSimulation(SimulationEnvironment):
    """Rossler system: a simpler 3D chaotic attractor.

    State vector: [x, y, z]

    ODEs:
        dx/dt = -y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x - c)

    Parameters:
        a: controls the frequency and shape of oscillation (classic: 0.2)
        b: controls the z-dynamics (classic: 0.2)
        c: controls chaos onset (classic: 5.7 for chaos)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.2)
        self.b = p.get("b", 0.2)
        self.c = p.get("c", 5.7)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 1.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Rossler state."""
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
        """Rossler equations: dx/dt=-y-z, dy/dt=x+a*y, dz/dt=b+z*(x-c)."""
        x, y, z = state
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the two fixed points of the Rossler system.

        Setting derivatives to zero:
            -y - z = 0  =>  z = -y
            x + a*y = 0  =>  x = -a*y
            b + z*(x - c) = 0  =>  b + (-y)*(-a*y - c) = 0
                => b + a*y^2 + c*y = 0
                => a*y^2 + c*y + b = 0
                => y = (-c +/- sqrt(c^2 - 4*a*b)) / (2*a)

        Then x = -a*y, z = -y.

        Fixed points exist when c^2 >= 4*a*b (discriminant >= 0).
        """
        discriminant = self.c**2 - 4.0 * self.a * self.b
        if discriminant < 0:
            return []

        sqrt_disc = np.sqrt(discriminant)
        points = []

        # FP1: y = (-c + sqrt(c^2 - 4ab)) / (2a)
        y1 = (-self.c + sqrt_disc) / (2.0 * self.a)
        x1 = -self.a * y1
        z1 = -y1
        points.append(np.array([x1, y1, z1], dtype=np.float64))

        # FP2: y = (-c - sqrt(c^2 - 4ab)) / (2a)
        y2 = (-self.c - sqrt_disc) / (2.0 * self.a)
        x2 = -self.a * y2
        z2 = -y2
        points.append(np.array([x2, y2, z2], dtype=np.float64))

        return points

    @property
    def is_chaotic(self) -> bool:
        """Heuristic: standard chaotic regime is c > ~5.0 with a=0.2, b=0.2.

        For the standard a=0.2, b=0.2 case, chaos emerges around c ~ 5.0.
        This is a rough heuristic, not an exact boundary.
        """
        # The classic chaotic regime has c large enough relative to a, b
        return self.c > 4.5 and self.a > 0 and self.b > 0

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
        for i in range(n_measure):
            self.step()
            curr_x = self._state[0]
            if prev_x < 0 and curr_x >= 0:
                # Linear interpolation for crossing time
                frac = -prev_x / (curr_x - prev_x) if curr_x != prev_x else 0.5
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_x = curr_x

        if len(crossings) < 2:
            return np.inf

        # Average period from consecutive crossings
        periods = np.diff(crossings)
        return float(np.mean(periods))
