"""Aizawa attractor simulation -- mushroom-shaped 3D strange attractor.

The Aizawa system is a 3D chaotic ODE with a distinctive mushroom-shaped attractor:
    dx/dt = (z - b)*x - d*y
    dy/dt = d*x + (z - b)*y
    dz/dt = c + a*z - z^3/3 - (x^2 + y^2)*(1 + e*z) + f*z*x^3

Parameters:
    a: controls z-dynamics growth (classic: 0.95)
    b: determines rotational axis offset (classic: 0.7)
    c: constant forcing term (classic: 0.6)
    d: rotation rate in the xy-plane (classic: 3.5)
    e: nonlinear coupling strength (classic: 0.25)
    f: cubic coupling coefficient (classic: 0.1)

Target rediscoveries:
- SINDy recovery of Aizawa ODEs
- Positive Lyapunov exponent confirming chaos
- Attractor boundedness and mushroom geometry
- Parameter sweep for chaos transition
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AizawaSimulation(SimulationEnvironment):
    """Aizawa attractor: mushroom-shaped chaotic strange attractor.

    State vector: [x, y, z]

    ODEs:
        dx/dt = (z - b)*x - d*y
        dy/dt = d*x + (z - b)*y
        dz/dt = c + a*z - z^3/3 - (x^2 + y^2)*(1 + e*z) + f*z*x^3

    The attractor has a distinctive mushroom or torus-like shape with
    chaotic wandering. The xy-dynamics exhibit rotation (controlled by d)
    around an axis that shifts with z (controlled by b).

    Parameters:
        a: z-dynamics growth rate (default 0.95)
        b: rotational axis offset (default 0.7)
        c: constant forcing in z (default 0.6)
        d: xy-plane rotation rate (default 3.5)
        e: nonlinear coupling (default 0.25)
        f: cubic coupling (default 0.1)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.95)
        self.b = p.get("b", 0.7)
        self.c = p.get("c", 0.6)
        self.d = p.get("d", 3.5)
        self.e = p.get("e", 0.25)
        self.f = p.get("f", 0.1)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Aizawa state."""
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
        """Aizawa equations."""
        x, y, z = state
        r_sq = x**2 + y**2

        dx = (z - self.b) * x - self.d * y
        dy = self.d * x + (z - self.b) * y
        dz = (
            self.c
            + self.a * z
            - z**3 / 3.0
            - r_sq * (1.0 + self.e * z)
            + self.f * z * x**3
        )
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute fixed points of the Aizawa system numerically.

        Setting dx=0 and dy=0 simultaneously:
            (z-b)*x - d*y = 0
            d*x + (z-b)*y = 0

        For d != 0, the only solution to the xy-subsystem is x=y=0
        (the determinant of the 2x2 matrix is (z-b)^2 + d^2 > 0).

        With x=y=0, the z-equation becomes:
            c + a*z - z^3/3 = 0
            => z^3 - 3*a*z - 3*c = 0

        We solve this cubic numerically.
        """
        # Cubic: z^3 - 3*a*z - 3*c = 0
        coeffs = [1.0, 0.0, -3.0 * self.a, -3.0 * self.c]
        roots = np.roots(coeffs)

        points = []
        for root in roots:
            if np.isreal(root):
                z_val = float(np.real(root))
                points.append(
                    np.array([0.0, 0.0, z_val], dtype=np.float64)
                )

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

    def compute_trajectory_statistics(
        self, n_steps: int = 10000, n_transient: int = 2000
    ) -> dict[str, float]:
        """Compute time-averaged statistics of the trajectory.

        Args:
            n_steps: Number of steps to measure after transient.
            n_transient: Steps to skip for transient.

        Returns:
            Dict with mean, std, min, max for each component and r = sqrt(x^2+y^2).
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
        rs = np.sqrt(xs**2 + ys**2)

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
            "r_mean": float(np.mean(rs)),
            "r_max": float(np.max(rs)),
        }
