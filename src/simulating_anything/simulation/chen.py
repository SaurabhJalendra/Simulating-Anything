"""Chen attractor simulation -- dual of the Lorenz system.

The Chen system is a 3D chaotic ODE discovered by Guanrong Chen (1999):
    dx/dt = a*(y - x)
    dy/dt = (c - a)*x - x*z + c*y
    dz/dt = x*y - b*z

It is the dual of the Lorenz system in the sense of Celikovsky and Vanecek's
classification of the generalized Lorenz system family. While Lorenz satisfies
a12*a21 > 0 (where a12, a21 are the (1,2) and (2,1) entries of the linear
part), Chen satisfies a12*a21 < 0.

Target rediscoveries:
- SINDy recovery of Chen ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin + two symmetric)
- Chaos transition as c varies
- Comparison with Lorenz coefficients
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ChenSimulation(SimulationEnvironment):
    """Chen attractor: the dual of the Lorenz system.

    State vector: [x, y, z]

    ODEs:
        dx/dt = a*(y - x)
        dy/dt = (c - a)*x - x*z + c*y
        dz/dt = x*y - b*z

    Parameters:
        a: diffusion parameter (default 35.0)
        b: damping parameter (default 3.0)
        c: driving parameter (default 28.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 35.0)
        self.b = p.get("b", 3.0)
        self.c = p.get("c", 28.0)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 1.0)
        self.z_0 = p.get("z_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Chen state."""
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
        """Chen equations: dx/dt=a*(y-x), dy/dt=(c-a)*x-x*z+c*y, dz/dt=x*y-b*z."""
        x, y, z = state
        dx = self.a * (y - x)
        dy = (self.c - self.a) * x - x * z + self.c * y
        dz = x * y - self.b * z
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the three fixed points of the Chen system.

        Setting derivatives to zero:
            a*(y - x) = 0         =>  y = x
            (c - a)*x - x*z + c*y = 0  with y=x  =>  (2c - a)*x - x*z = 0
            x*y - b*z = 0         with y=x  =>  x^2 = b*z

        From the second equation: x*((2c - a) - z) = 0
        Case 1: x = 0  =>  origin
        Case 2: z = 2c - a, then x^2 = b*(2c - a)

        Non-origin fixed points exist when b*(2c - a) > 0.
        """
        points = [np.array([0.0, 0.0, 0.0])]
        z_eq = 2.0 * self.c - self.a
        if self.b * z_eq > 0:
            x_eq = np.sqrt(self.b * z_eq)
            points.append(np.array([x_eq, x_eq, z_eq]))
            points.append(np.array([-x_eq, -x_eq, z_eq]))
        return points

    @property
    def is_chaotic(self) -> bool:
        """Heuristic for chaotic regime.

        For standard a=35, b=3, chaos occurs when c is sufficiently large.
        The condition c > 2a - b gives an approximate boundary, though the
        actual transition depends on the specific parameter combination.
        """
        return self.c > 2.0 * self.a - self.b

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

    def compare_with_lorenz(self) -> dict[str, float]:
        """Compare Chen and Lorenz linear coefficient structures.

        The generalized Lorenz system family is parameterized by the sign of
        a12*a21 in the linear part. Lorenz has a12*a21 > 0, Chen has a12*a21 < 0.

        For Chen: linear part has a12 = a (from dx/dt = a*(y-x)), and
        a21 = (c - a) (from dy/dt = (c-a)*x + c*y - x*z).
        Product a12*a21 = a*(c - a).
        """
        a12 = self.a
        a21 = self.c - self.a
        product = a12 * a21

        return {
            "a12": float(a12),
            "a21": float(a21),
            "a12_times_a21": float(product),
            "is_chen_type": bool(product < 0),
            "is_lorenz_type": bool(product > 0),
        }
