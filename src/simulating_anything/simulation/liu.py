"""Liu chaotic system simulation -- 6-parameter 3D autonomous chaotic ODE.

The Liu system is a 3D chaotic system with quadratic nonlinearities:
    dx/dt = -a*x - e*y^2
    dy/dt = b*y - k*x*z
    dz/dt = -c*z + m*x*y

where a, b, c, e, k, m are positive parameters.

Key properties:
- Dissipative: divergence = -a + b - c (negative for default parameters)
- Quadratic nonlinearities: y^2, x*z, x*y
- Strange attractor with period-doubling route to chaos
- Origin is the only fixed point for typical parameter values

Reference:
    C. Liu, T. Liu, L. Liu, K. Liu, "A new chaotic attractor,"
    Chaos, Solitons & Fractals, 2004.

Target rediscoveries:
- SINDy recovery of Liu ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin only for typical parameters)
- Dissipation rate = a - b + c
- Parameter sweep mapping chaos transitions
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LiuSimulation(SimulationEnvironment):
    """Liu chaotic attractor: 6-parameter 3D autonomous system.

    State vector: [x, y, z]

    ODEs:
        dx/dt = -a*x - e*y^2
        dy/dt = b*y - k*x*z
        dz/dt = -c*z + m*x*y

    Parameters:
        a: linear damping in x (default 1.0)
        b: linear growth in y (default 2.5)
        c: linear damping in z (default 5.0)
        e: quadratic coupling y^2 -> x (default 1.0)
        k: coupling x*z -> y (default 4.0)
        m: coupling x*y -> z (default 4.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.0)
        self.b = p.get("b", 2.5)
        self.c = p.get("c", 5.0)
        self.e = p.get("e", 1.0)
        self.k = p.get("k", 4.0)
        self.m = p.get("m", 4.0)
        self.x_0 = p.get("x_0", 0.2)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Liu state."""
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
        """Liu equations: dx=-a*x-e*y^2, dy=b*y-k*x*z, dz=-c*z+m*x*y."""
        x, y, z = state
        dx = -self.a * x - self.e * y**2
        dy = self.b * y - self.k * x * z
        dz = -self.c * z + self.m * x * y
        return np.array([dx, dy, dz])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[-a,     -2*e*y,  0    ],
             [-k*z,   b,       -k*x ],
             [m*y,    m*x,     -c   ]]
        """
        x, y, z = state
        return np.array([
            [-self.a, -2.0 * self.e * y, 0.0],
            [-self.k * z, self.b, -self.k * x],
            [self.m * y, self.m * x, -self.c],
        ])

    def compute_divergence(self) -> float:
        """Compute the divergence of the Liu vector field.

        div(F) = d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz
               = -a + b + (-c) = -a + b - c

        The system is dissipative when divergence < 0, i.e., a - b + c > 0.
        """
        return -self.a + self.b - self.c

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Liu system.

        Setting derivatives to zero:
            -a*x - e*y^2 = 0        =>  x = -e*y^2/a
            b*y - k*x*z = 0
            -c*z + m*x*y = 0        =>  z = m*x*y/c

        Substituting x and z into the second equation:
            b*y - k*(-e*y^2/a)*(m*(-e*y^2/a)*y/c) = 0
            y*(b + k*m*e^2*y^4/(a^2*c)) = 0

        For typical parameters (all positive), the bracket is always positive,
        so the origin y=0 is the only solution.
        """
        points = [np.array([0.0, 0.0, 0.0])]

        # Check for non-origin fixed points analytically
        # The coefficient of y^4 is k*m*e^2/(a^2*c), always positive
        # Combined with b > 0, no real non-zero y solutions exist
        coeff = self.k * self.m * self.e**2 / (self.a**2 * self.c)
        if self.b > 0 and coeff > 0:
            # b + coeff*y^4 > 0 for all y, so only origin
            pass
        elif self.b < 0:
            # y^4 = -b*a^2*c / (k*m*e^2) could have real solutions
            val = -self.b * self.a**2 * self.c / (self.k * self.m * self.e**2)
            if val > 0:
                y_eq = val**0.25
                for y_val in [y_eq, -y_eq]:
                    x_val = -self.e * y_val**2 / self.a
                    z_val = self.m * x_val * y_val / self.c
                    points.append(np.array([x_val, y_val, z_val]))

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
