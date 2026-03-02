"""Tigan (T-system) simulation -- a 3D chaotic system generalizing Lorenz.

The T-system was introduced by Tigan (2005) as a generalization of the Lorenz
attractor with different coupling structure:
    dx/dt = a*(y - x)
    dy/dt = (c - a)*x - a*x*z
    dz/dt = -b*z + x*y

Key properties:
- Horseshoe chaos for certain parameter regimes
- Fixed points at the origin and (+/-sqrt(b*c), +/-c, (c-a)/a)
- Divergence of the flow: -(2a + b) (always dissipative for a, b > 0)
- Structurally related to but distinct from Lorenz (different x*z coupling)

Target rediscoveries:
- SINDy recovery of Tigan ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin + two symmetric)
- Chaos transition as parameters vary
- Divergence = -(2a + b)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TiganSimulation(SimulationEnvironment):
    """Tigan (T-system): a 3D chaotic generalization of Lorenz.

    State vector: [x, y, z]

    ODEs:
        dx/dt = a*(y - x)
        dy/dt = (c - a)*x - a*x*z
        dz/dt = -b*z + x*y

    Parameters:
        a: coupling/diffusion parameter (default 2.1)
        b: damping parameter (default 0.6)
        c: driving parameter (default 30.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 2.1)
        self.b = p.get("b", 0.6)
        self.c = p.get("c", 30.0)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 1.0)
        self.z_0 = p.get("z_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Tigan state."""
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
        """Tigan equations: dx=a*(y-x), dy=(c-a)*x-a*x*z, dz=-b*z+x*y."""
        x, y, z = state
        dx = self.a * (y - x)
        dy = (self.c - self.a) * x - self.a * x * z
        dz = -self.b * z + x * y
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the three fixed points of the Tigan system.

        Setting derivatives to zero:
            a*(y - x) = 0            =>  y = x
            (c - a)*x - a*x*z = 0    =>  x*((c - a) - a*z) = 0
            -b*z + x*y = 0           with y=x  =>  x^2 = b*z

        Case 1: x = 0  =>  origin
        Case 2: z = (c - a)/a, then x^2 = b*(c - a)/a = b*c/a - b
                But we also get x^2 = b*z = b*(c-a)/a.
                More simply: from dy=0 => z = (c-a)/a
                from dz=0 => x^2 = b*z = b*(c-a)/a
                and y = x.

        Non-origin fixed points exist when b*(c - a)/a > 0.
        With the standard parameters (a=2.1, c=30), (c-a)/a = 27.9/2.1 > 0.
        """
        points = [np.array([0.0, 0.0, 0.0])]
        z_eq = (self.c - self.a) / self.a
        x_sq = self.b * z_eq
        if x_sq > 0:
            x_eq = np.sqrt(x_sq)
            points.append(np.array([x_eq, x_eq, z_eq]))
            points.append(np.array([-x_eq, -x_eq, z_eq]))
        return points

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[-a,    a,      0    ],
             [(c-a) - a*z, 0, -a*x],
             [y,     x,     -b   ]]
        """
        x, y, z = state
        return np.array([
            [-self.a, self.a, 0.0],
            [(self.c - self.a) - self.a * z, 0.0, -self.a * x],
            [y, x, -self.b],
        ])

    def compute_divergence(self) -> float:
        """Compute the divergence of the Tigan flow.

        div(F) = d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz
               = -a + 0 + (-b)
               = -(a + b)

        Wait, let us recompute more carefully:
            d(a*(y-x))/dx = -a
            d((c-a)*x - a*x*z)/dy = 0
            d(-b*z + x*y)/dz = -b

        So divergence = -a + 0 + (-b) = -(a + b).

        Note: The trace of the Jacobian is -a + 0 + (-b) = -(a + b).
        This is state-independent, confirming uniform contraction.
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
