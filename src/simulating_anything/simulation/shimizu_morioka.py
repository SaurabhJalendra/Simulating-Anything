"""Shimizu-Morioka attractor simulation -- simplified Lorenz-like system.

The Shimizu-Morioka system is a simplified model related to the Lorenz system,
describing fluid convection:
    dx/dt = y
    dy/dt = (1 - z)*x - a*y
    dz/dt = -b*z + x^2

where a (default 0.75) and b (default 0.45) are parameters.

Key physics:
- Lorenz-like chaos with simpler ODE structure (only 2 parameters)
- Period-doubling route to chaos
- Two symmetric equilibria at (+/-sqrt(b), 0, 1)
- Displays butterfly-like strange attractor

Target rediscoveries:
- SINDy recovery of Shimizu-Morioka ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin + two symmetric)
- Chaos transition as a varies
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ShimizuMoriokaSimulation(SimulationEnvironment):
    """Shimizu-Morioka attractor: a simplified Lorenz-like system.

    State vector: [x, y, z]

    ODEs:
        dx/dt = y
        dy/dt = (1 - z)*x - a*y
        dz/dt = -b*z + x^2

    Parameters:
        a: damping parameter (default 0.75)
        b: damping parameter for z (default 0.45)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.75)
        self.b = p.get("b", 0.45)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Shimizu-Morioka state."""
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
        """Shimizu-Morioka equations: dx/dt=y, dy/dt=(1-z)*x-a*y, dz/dt=-b*z+x^2."""
        x, y, z = state
        dx = y
        dy = (1.0 - z) * x - self.a * y
        dz = -self.b * z + x * x
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the three fixed points of the Shimizu-Morioka system.

        Setting derivatives to zero:
            y = 0
            (1 - z)*x - a*y = 0  =>  (1 - z)*x = 0
            -b*z + x^2 = 0       =>  z = x^2/b

        Case 1: x = 0  =>  z = 0  =>  origin (0, 0, 0)
        Case 2: z = 1, then x^2 = b  =>  x = +/-sqrt(b)

        Non-origin fixed points exist when b > 0.
        """
        points = [np.array([0.0, 0.0, 0.0])]
        if self.b > 0:
            x_eq = np.sqrt(self.b)
            points.append(np.array([x_eq, 0.0, 1.0]))
            points.append(np.array([-x_eq, 0.0, 1.0]))
        return points

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[0,      1,  0   ],
             [1 - z, -a,  -x  ],
             [2*x,    0,  -b  ]]
        """
        x, _y, z = state
        return np.array([
            [0.0, 1.0, 0.0],
            [1.0 - z, -self.a, -x],
            [2.0 * x, 0.0, -self.b],
        ])

    def compute_divergence(self, state: np.ndarray) -> float:
        """Compute the divergence of the vector field at a given state.

        div(F) = d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz
               = 0 + (-a) + (-b)
               = -(a + b)

        The divergence is constant (state-independent), indicating uniform
        volume contraction in phase space.
        """
        return -(self.a + self.b)

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
