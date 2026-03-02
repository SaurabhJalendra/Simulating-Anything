"""Lu-Chen attractor simulation -- unified Lorenz-Chen-Lu system.

The Lu-Chen system is a 3D chaotic ODE that unifies the Lorenz, Chen, and Lu
attractors through continuous parameter variation:
    dx/dt = a*(y - x)
    dy/dt = x - x*z + c*y
    dz/dt = x*y - b*z

When c=20 (classic): Lu-Chen chaotic attractor
When c=28: Chen-like attractor
When c=12: Lu-like attractor
Varying c continuously transitions between the three regimes.

Target rediscoveries:
- SINDy recovery of Lu-Chen ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- c-parameter sweep mapping Lorenz/Lu/Chen transitions
- Fixed point analysis (origin + two symmetric)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LuChenSimulation(SimulationEnvironment):
    """Lu-Chen system: a unified 3D chaotic attractor.

    State vector: [x, y, z]

    ODEs:
        dx/dt = a*(y - x)
        dy/dt = x - x*z + c*y
        dz/dt = x*y - b*z

    Parameters:
        a: diffusion parameter (default 36.0)
        b: damping parameter (default 3.0)
        c: driving/unification parameter (default 20.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 36.0)
        self.b = p.get("b", 3.0)
        self.c = p.get("c", 20.0)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.3)
        self.z_0 = p.get("z_0", -0.6)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Lu-Chen state."""
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
        """Lu-Chen equations: dx/dt=a*(y-x), dy/dt=x-x*z+c*y, dz/dt=x*y-b*z."""
        x, y, z = state
        dx = self.a * (y - x)
        dy = x - x * z + self.c * y
        dz = x * y - self.b * z
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Lu-Chen system.

        Setting derivatives to zero:
            a*(y - x) = 0              =>  y = x
            x - x*z + c*y = 0  with y=x  =>  x*(1 + c - z) = 0
            x*y - b*z = 0      with y=x  =>  x^2 = b*z

        Case 1: x = 0  =>  origin (0, 0, 0)
        Case 2: z = 1 + c, then x^2 = b*(1 + c)

        Non-origin fixed points exist when b*(1 + c) > 0.
        """
        points = [np.array([0.0, 0.0, 0.0])]
        z_eq = 1.0 + self.c
        if self.b * z_eq > 0:
            x_eq = np.sqrt(self.b * z_eq)
            points.append(np.array([x_eq, x_eq, z_eq]))
            points.append(np.array([-x_eq, -x_eq, z_eq]))
        return points

    @property
    def is_chaotic(self) -> bool:
        """Heuristic for chaotic regime.

        The Lu-Chen system is chaotic for a wide range of c values with
        the standard a=36, b=3. The classic chaotic regime is near c=20.
        This is a rough heuristic.
        """
        return self.a > 0 and self.b > 0 and self.c > 5.0

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

    def classify_regime(self) -> str:
        """Classify the current parameter regime.

        The Lu-Chen system unifies three classical attractors:
        - c ~ 12: Lu-like attractor
        - c ~ 20: Lu-Chen chaotic attractor (classic)
        - c ~ 28: Chen-like attractor
        """
        if self.c < 15:
            return "lu_like"
        elif self.c < 24:
            return "lu_chen"
        else:
            return "chen_like"
