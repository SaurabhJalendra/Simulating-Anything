"""Dadras chaotic attractor simulation -- 3D quadratic ODE.

The Dadras system is a 3D chaotic ODE discovered by Sara Dadras and
Hamid Reza Momeni (2009) with simple quadratic nonlinearities:
    dx/dt = y - a*x + b*y*z
    dy/dt = c*y - x*z + z
    dz/dt = d*x*y - e*z

The system exhibits chaotic behavior with a strange attractor for the
classic parameter set (a=3, b=2.7, c=1.7, d=2, e=9). The attractor has
a double-scroll-like structure.

Target rediscoveries:
- SINDy recovery of Dadras ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin + possible non-trivial fixed points)
- Parameter sweep mapping chaos boundary
- Attractor boundedness and statistics
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DadrasSimulation(SimulationEnvironment):
    """Dadras chaotic attractor: a 3D ODE with quadratic nonlinearities.

    State vector: [x, y, z]

    ODEs:
        dx/dt = y - a*x + b*y*z
        dy/dt = c*y - x*z + z
        dz/dt = d*x*y - e*z

    The system has a dissipative structure with divergence = -a + c - e,
    which is negative for the classic parameters (-a + c - e = -3 + 1.7 - 9 = -10.3),
    ensuring volume contraction in phase space.

    Parameters:
        a: linear damping on x (default 3.0)
        b: quadratic coupling y*z -> x (default 2.7)
        c: linear growth on y (default 1.7)
        d: quadratic coupling x*y -> z (default 2.0)
        e: linear damping on z (default 9.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 3.0)
        self.b = p.get("b", 2.7)
        self.c = p.get("c", 1.7)
        self.d = p.get("d", 2.0)
        self.e = p.get("e", 9.0)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 1.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Dadras state."""
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
        """Dadras equations."""
        x, y, z = state
        dx = y - self.a * x + self.b * y * z
        dy = self.c * y - x * z + z
        dz = self.d * x * y - self.e * z
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Dadras system.

        Setting derivatives to zero:
            y - a*x + b*y*z = 0  ... (1)
            c*y - x*z + z = 0    ... (2)
            d*x*y - e*z = 0      ... (3)

        The origin (0, 0, 0) is always a fixed point.
        Non-origin fixed points are found numerically using multiple
        initial guesses to capture all branches.
        """
        points = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]

        def equations(xyz: np.ndarray) -> np.ndarray:
            x, y, z = xyz
            eq1 = y - self.a * x + self.b * y * z
            eq2 = self.c * y - x * z + z
            eq3 = self.d * x * y - self.e * z
            return np.array([eq1, eq2, eq3])

        # Search with multiple initial guesses to find all fixed points
        guesses = [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [2.0, 2.0, 2.0],
            [-2.0, -2.0, 2.0],
            [0.5, 0.5, 0.1],
            [-0.5, -0.5, 0.1],
        ]

        for guess in guesses:
            try:
                root, info, ier, _ = fsolve(
                    equations, guess, full_output=True
                )
                if ier == 1:
                    residual = np.linalg.norm(equations(root))
                    if residual < 1e-10:
                        # Check for duplicates
                        is_dup = False
                        for fp in points:
                            if np.linalg.norm(fp - root) < 1e-6:
                                is_dup = True
                                break
                        if not is_dup:
                            points.append(
                                np.array(root, dtype=np.float64)
                            )
            except (ValueError, RuntimeError):
                pass

        return points

    @property
    def divergence(self) -> float:
        """Phase space divergence: -a + c - e.

        Negative divergence means the system is dissipative and volumes
        contract in phase space, which is necessary for attractor formation.
        """
        return -self.a + self.c - self.e

    @property
    def is_chaotic(self) -> bool:
        """Heuristic for chaotic regime.

        The Dadras system is chaotic near the classic parameters
        (a=3, b=2.7, c=1.7, d=2, e=9). This heuristic checks that
        the dissipation is strong enough and the nonlinear coupling
        is sufficiently large.
        """
        return self.divergence < -5.0 and self.b > 1.0 and self.d > 1.0

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

    def parameter_sweep(
        self,
        param_name: str,
        param_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep a single parameter and record attractor statistics.

        For each parameter value, computes the Lyapunov exponent and
        trajectory statistics after skipping transients.

        Args:
            param_name: Name of the parameter to sweep (a, b, c, d, or e).
            param_values: Array of parameter values to sweep.
            n_transient: Steps to skip for transient.
            n_measure: Steps to use for Lyapunov estimation.

        Returns:
            Dict with parameter values, Lyapunov exponents, and attractor types.
        """
        lyapunov_exps = []
        attractor_types = []
        max_amplitudes = []

        base_params = {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "x_0": 1.0,
            "y_0": 1.0,
            "z_0": 0.0,
        }

        for val in param_values:
            params = base_params.copy()
            params[param_name] = val

            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_transient + n_measure,
                parameters=params,
            )
            sim = DadrasSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Estimate Lyapunov
            lam = sim.estimate_lyapunov(n_steps=n_measure, dt=self.config.dt)
            lyapunov_exps.append(lam)

            # Measure amplitude
            sim.reset()
            for _ in range(n_transient):
                sim.step()
            vals = []
            for _ in range(min(5000, n_measure)):
                state = sim.step()
                vals.append(np.linalg.norm(state))
            max_amplitudes.append(np.max(vals))

            # Classify
            if lam > 0.01:
                atype = "chaotic"
            elif lam < -0.01:
                atype = "fixed_point"
            else:
                atype = "marginal"
            attractor_types.append(atype)

        return {
            param_name: param_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "max_amplitude": np.array(max_amplitudes),
            "attractor_type": np.array(attractor_types),
        }
