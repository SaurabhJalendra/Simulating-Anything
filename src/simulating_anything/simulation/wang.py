"""Wang chaotic attractor simulation.

The Wang system is a 3D autonomous chaotic system:
    dx/dt = x - a*y
    dy/dt = -b*y + x*z
    dz/dt = -c*z + d*x*y

where a, b, c, d are positive parameters. It displays a single-wing chaotic
attractor with non-standard topology compared to the Lorenz family, and
undergoes period-doubling route to chaos.

Target rediscoveries:
- SINDy recovery of Wang ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin + two symmetric)
- Divergence = 1 - b - c (trace of Jacobian, dissipative when < 0)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class WangSimulation(SimulationEnvironment):
    """Wang chaotic attractor: single-wing 3D quadratic system.

    State vector: [x, y, z]

    ODEs:
        dx/dt = x - a*y
        dy/dt = -b*y + x*z
        dz/dt = -c*z + d*x*y

    Parameters:
        a: coupling parameter (default 1.0)
        b: damping parameter (default 1.0)
        c: damping parameter (default 0.7)
        d: nonlinear coupling (default 0.5)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.0)
        self.b = p.get("b", 1.0)
        self.c = p.get("c", 0.7)
        self.d = p.get("d", 0.5)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.2)
        self.z_0 = p.get("z_0", 0.3)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Wang state."""
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
        """Wang equations: dx/dt=x-a*y, dy/dt=-b*y+x*z, dz/dt=-c*z+d*x*y."""
        x, y, z = state
        dx = x - self.a * y
        dy = -self.b * y + x * z
        dz = -self.c * z + self.d * x * y
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Wang system.

        Setting derivatives to zero:
            x - a*y = 0            =>  y = x/a
            -b*y + x*z = 0         =>  x*z = b*x/a  =>  z = b/a (if x != 0)
            -c*z + d*x*y = 0       =>  d*x*(x/a) = c*(b/a)  =>  x^2 = b*c/d

        Case 1: x = 0  =>  origin (0, 0, 0)
        Case 2: x = +/-sqrt(b*c/d), y = x/a, z = b/a
        """
        points = [np.array([0.0, 0.0, 0.0])]
        if self.d > 0 and self.b * self.c > 0:
            x_eq = np.sqrt(self.b * self.c / self.d)
            y_eq = x_eq / self.a
            z_eq = self.b / self.a
            points.append(np.array([x_eq, y_eq, z_eq]))
            points.append(np.array([-x_eq, -y_eq, z_eq]))
        return points

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[1,    -a,   0  ],
             [z,    -b,   x  ],
             [d*y,  d*x,  -c ]]
        """
        x, y, z = state
        return np.array([
            [1.0, -self.a, 0.0],
            [z, -self.b, x],
            [self.d * y, self.d * x, -self.c],
        ])

    def compute_divergence(self) -> float:
        """Compute the divergence of the Wang vector field.

        div(F) = dF1/dx + dF2/dy + dF3/dz = 1 - b - c

        This is constant (state-independent). The system is dissipative
        when 1 - b - c < 0 (i.e., b + c > 1).
        """
        return 1.0 - self.b - self.c

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
