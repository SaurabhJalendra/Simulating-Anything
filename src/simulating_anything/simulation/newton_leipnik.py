"""Newton-Leipnik attractor simulation -- 3D chaotic system with multistability.

The Newton-Leipnik system is a 3D quadratic ODE discovered by Newton and
Leipnik (1986):
    dx/dt = -a*x + y + 10*y*z
    dy/dt = -x - 0.4*y + 5*x*z
    dz/dt = b*z - 5*x*y

It is notable for exhibiting two coexisting strange attractors
(multistability) for the classical parameter values a=0.4, b=0.175.
Different initial conditions lead to trajectories on different attractors.

The system features bilinear coupling terms (y*z, x*z, x*y) and is
dissipative with divergence -(a + 0.4 - b).

Target rediscoveries:
- SINDy recovery of Newton-Leipnik ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis
- Multistability demonstration (two coexisting attractors)
- Dissipation rate verification
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class NewtonLeipnikSimulation(SimulationEnvironment):
    """Newton-Leipnik attractor: 3D chaotic system with two strange attractors.

    State vector: [x, y, z]

    ODEs:
        dx/dt = -a*x + y + 10*y*z
        dy/dt = -x - 0.4*y + 5*x*z
        dz/dt = b*z - 5*x*y

    The system has a fixed coefficient of 0.4 in the y-damping term of the
    second equation. Parameters a and b control the overall dissipation and
    z-dynamics respectively.

    Parameters:
        a: x-damping parameter (default 0.4)
        b: z-growth parameter (default 0.175)
        x_0, y_0, z_0: initial conditions (default [0.349, 0.0, -0.16])
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.4)
        self.b = p.get("b", 0.175)
        self.x_0 = p.get("x_0", 0.349)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", -0.16)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Newton-Leipnik state."""
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
        """Newton-Leipnik equations.

        dx/dt = -a*x + y + 10*y*z
        dy/dt = -x - 0.4*y + 5*x*z
        dz/dt = b*z - 5*x*y
        """
        x, y, z = state
        dx = -self.a * x + y + 10.0 * y * z
        dy = -x - 0.4 * y + 5.0 * x * z
        dz = self.b * z - 5.0 * x * y
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute fixed points of the Newton-Leipnik system.

        Setting derivatives to zero:
            -a*x + y + 10*y*z = 0  => y*(1 + 10*z) = a*x   ... (1)
            -x - 0.4*y + 5*x*z = 0  => x*(-1 + 5*z) = 0.4*y  ... (2)
            b*z - 5*x*y = 0  => z = 5*x*y / b  ... (3)

        The origin is always a fixed point. Additional fixed points can be
        found numerically for general parameters.
        """
        points = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]

        # Find non-origin fixed points numerically via Newton-Raphson
        # from multiple initial guesses to capture both attractors
        guesses = [
            np.array([0.3, 0.3, -0.1]),
            np.array([-0.3, -0.3, -0.1]),
            np.array([0.3, -0.3, 0.1]),
            np.array([-0.3, 0.3, 0.1]),
            np.array([0.1, 0.1, 0.0]),
            np.array([-0.1, -0.1, 0.0]),
        ]

        found = []
        for guess in guesses:
            fp = self._newton_raphson(guess, max_iter=200, tol=1e-12)
            if fp is not None:
                # Check it is actually a fixed point
                deriv = self._derivatives(fp)
                if np.linalg.norm(deriv) < 1e-10:
                    # Check it is not a duplicate
                    is_dup = False
                    for existing in [points[0]] + found:
                        if np.linalg.norm(fp - existing) < 1e-6:
                            is_dup = True
                            break
                    if not is_dup:
                        found.append(fp)

        points.extend(found)
        return points

    def _newton_raphson(
        self,
        guess: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-12,
    ) -> np.ndarray | None:
        """Find a fixed point using Newton-Raphson iteration."""
        state = guess.copy()
        for _ in range(max_iter):
            f = self._derivatives(state)
            if np.linalg.norm(f) < tol:
                return state
            J = self.jacobian(state)
            try:
                delta = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None
            state = state + delta
            if np.any(np.abs(state) > 1e6):
                return None
        return None

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        For the Newton-Leipnik system:
            J = [[-a,        1 + 10*z,  10*y ],
                 [-1 + 5*z,  -0.4,      5*x  ],
                 [-5*y,      -5*x,      b    ]]

        Args:
            state: State vector [x, y, z].

        Returns:
            3x3 numpy array of partial derivatives.
        """
        x, y, z = state
        return np.array([
            [-self.a, 1.0 + 10.0 * z, 10.0 * y],
            [-1.0 + 5.0 * z, -0.4, 5.0 * x],
            [-5.0 * y, -5.0 * x, self.b],
        ])

    @property
    def divergence(self) -> float:
        """Phase space divergence (constant): -(a + 0.4 - b).

        The trace of the Jacobian is:
            trace(J) = -a + (-0.4) + b = -(a + 0.4 - b)

        For default parameters a=0.4, b=0.175:
            divergence = -(0.4 + 0.4 - 0.175) = -0.625

        Negative divergence confirms the system is dissipative.
        """
        return -(self.a + 0.4 - self.b)

    def compute_divergence(self, state: np.ndarray) -> float:
        """Compute the divergence of the vector field at a point.

        For the Newton-Leipnik system, the divergence is state-independent:
            div(F) = d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz
                   = -a + (-0.4) + b = -(a + 0.4 - b)

        Args:
            state: State vector [x, y, z] (unused, divergence is constant).

        Returns:
            The divergence value.
        """
        return self.divergence

    @property
    def is_chaotic(self) -> bool:
        """Heuristic for chaotic regime.

        The Newton-Leipnik system is chaotic for the standard parameter
        values a=0.4, b=0.175. Chaos generally requires weak dissipation,
        so the heuristic checks that divergence is small in magnitude.
        """
        return abs(self.divergence) < 2.0

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
