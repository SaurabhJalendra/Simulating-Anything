"""Burke-Shaw chaotic system simulation -- 3D chaotic attractor.

The Burke-Shaw system is a 3D autonomous ODE exhibiting chaotic behavior:
    dx/dt = -s*(x + y)
    dy/dt = -y - s*x*z
    dz/dt = s*x*y + v

Classic parameters: s=10.0, v=4.272

Target rediscoveries:
- SINDy recovery of Burke-Shaw ODEs
- Positive Lyapunov exponent confirming chaos
- Fixed point computation and verification
- Parameter sweep of s for chaos transition
- Attractor boundedness verification
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BurkeShawSimulation(SimulationEnvironment):
    """Burke-Shaw chaotic attractor: a 3D ODE system with quadratic nonlinearity.

    State vector: [x, y, z]

    ODEs:
        dx/dt = -s*(x + y)
        dy/dt = -y - s*x*z
        dz/dt = s*x*y + v

    The system exhibits chaotic behavior for the classic parameter values
    s=10.0, v=4.272. The attractor has a characteristic double-scroll
    structure.

    Parameters:
        s: coupling/nonlinearity parameter (classic: 10.0)
        v: offset parameter controlling z-dynamics (classic: 4.272)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.s = p.get("s", 10.0)
        self.v = p.get("v", 4.272)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Burke-Shaw state."""
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
        """Fourth-order Runge-Kutta integration step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Burke-Shaw equations.

        dx/dt = -s*(x + y)
        dy/dt = -y - s*x*z
        dz/dt = s*x*y + v
        """
        x, y, z = state
        dx = -self.s * (x + y)
        dy = -y - self.s * x * z
        dz = self.s * x * y + self.v
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Burke-Shaw system.

        Setting derivatives to zero:
            -s*(x + y) = 0          => y = -x
            -y - s*x*z = 0          => x - s*x*z = 0 (since y=-x)
            s*x*y + v = 0           => -s*x^2 + v = 0

        From eq1: y = -x
        From eq3: s*x*(-x) + v = 0 => -s*x^2 + v = 0 => x^2 = v/s

        If v/s > 0, two fixed points exist:
            x = +/- sqrt(v/s), y = -x

        From eq2: -y - s*x*z = 0 => x - s*x*z = 0
            If x != 0: 1 - s*z = 0 => z = 1/s

        Special case: x=0 gives y=0, and eq3 gives v=0 (only if v=0).
        """
        ratio = self.v / self.s
        if ratio < 0:
            # No real fixed points when v/s < 0
            if abs(self.v) < 1e-15:
                return [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
            return []

        if abs(ratio) < 1e-15:
            # v/s ~ 0: single fixed point at origin
            return [np.array([0.0, 0.0, 0.0], dtype=np.float64)]

        x_mag = np.sqrt(ratio)
        z_fp = 1.0 / self.s

        fp1 = np.array([x_mag, -x_mag, z_fp], dtype=np.float64)
        fp2 = np.array([-x_mag, x_mag, z_fp], dtype=np.float64)
        return [fp1, fp2]

    @property
    def is_chaotic(self) -> bool:
        """Heuristic: standard chaotic regime is s ~ 10, v ~ 4.272.

        Chaos generally occurs for large enough s with positive v.
        This is a rough heuristic based on known parameter studies.
        """
        return self.s > 5.0 and self.v > 1.0

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

    def compute_trajectory_statistics(
        self, n_steps: int = 10000, n_transient: int = 2000
    ) -> dict[str, float]:
        """Compute time-averaged statistics of the trajectory.

        Args:
            n_steps: Number of steps to measure after transient.
            n_transient: Steps to skip for transient.

        Returns:
            Dict with mean, std, min, max for each component.
        """
        self.reset()

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect data
        xs, ys, zs = [], [], []
        for _ in range(n_steps):
            state = self.step()
            xs.append(state[0])
            ys.append(state[1])
            zs.append(state[2])

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        return {
            "x_mean": float(np.mean(xs)),
            "y_mean": float(np.mean(ys)),
            "z_mean": float(np.mean(zs)),
            "x_std": float(np.std(xs)),
            "y_std": float(np.std(ys)),
            "z_std": float(np.std(zs)),
            "x_min": float(np.min(xs)),
            "y_min": float(np.min(ys)),
            "z_min": float(np.min(zs)),
            "x_max": float(np.max(xs)),
            "y_max": float(np.max(ys)),
            "z_max": float(np.max(zs)),
        }
