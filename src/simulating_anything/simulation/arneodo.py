"""Arneodo attractor simulation -- minimal jerk-type chaotic ODE.

The Arneodo system is a 3D chaotic ODE with cubic nonlinearity:
    dx/dt = y
    dy/dt = z
    dz/dt = -a*x - b*y - z + d*x^3

Equivalently, the third-order scalar jerk equation:
    x''' + x'' + b*x' + a*x = d*x^3

Parameters:
    a: linear restoring coefficient (default 5.5)
    b: damping of the y term (default 3.5)
    d: cubic nonlinearity strength (default 1.0)

The z-equation has a fixed damping coefficient of -1 for the z term,
giving a constant divergence (trace of Jacobian) of -1. Compared to
the Genesio-Tesi system (quadratic x^2 nonlinearity), Arneodo uses a
cubic x^3 term, which produces a double-scroll attractor and
period-doubling cascade to chaos as d increases.

Target rediscoveries:
- SINDy recovery of Arneodo ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- d-parameter sweep mapping period-doubling cascade
- Fixed point analysis (origin + two symmetric for d > 0)
- Constant divergence = -1 verification
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ArneodoSimulation(SimulationEnvironment):
    """Arneodo attractor: a minimal jerk system with cubic nonlinearity.

    State vector: [x, y, z]

    ODEs:
        dx/dt = y
        dy/dt = z
        dz/dt = -a*x - b*y - z + d*x^3

    The jerk form is: x''' + x'' + b*x' + a*x = d*x^3

    The trace of the Jacobian is 0 + 0 + (-1) = -1, so the system is
    uniformly dissipative with divergence = -1.

    Parameters:
        a: linear restoring coefficient (default 5.5)
        b: damping of y term (default 3.5)
        d: cubic nonlinearity strength (default 1.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 5.5)
        self.b = p.get("b", 3.5)
        self.d = p.get("d", 1.0)
        self.x_0 = p.get("x_0", 0.2)
        self.y_0 = p.get("y_0", 0.2)
        self.z_0 = p.get("z_0", 0.2)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Arneodo state."""
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
        """Arneodo jerk equations: dx=y, dy=z, dz=-a*x-b*y-z+d*x^3."""
        x, y, z = state
        dx = y
        dy = z
        dz = -self.a * x - self.b * y - z + self.d * x**3
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Arneodo system.

        Setting derivatives to zero:
            y = 0
            z = 0
            -a*x - b*0 - 0 + d*x^3 = 0  =>  x*(-a + d*x^2) = 0

        Case 1: x = 0 => origin (always exists)
        Case 2: d*x^2 = a => x = +/-sqrt(a/d) (exists when a/d > 0)
        """
        points = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        if self.d != 0 and self.a / self.d > 0:
            x_eq = np.sqrt(self.a / self.d)
            points.append(np.array([x_eq, 0.0, 0.0], dtype=np.float64))
            points.append(np.array([-x_eq, 0.0, 0.0], dtype=np.float64))
        return points

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[0,    1,  0],
             [0,    0,  1],
             [-a + 3*d*x^2, -b, -1]]

        The trace is always 0 + 0 + (-1) = -1 (constant dissipation).
        """
        x = state[0]
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-self.a + 3.0 * self.d * x**2, -self.b, -1.0],
        ], dtype=np.float64)

    def compute_divergence(self, state: np.ndarray) -> float:
        """Compute the divergence (trace of Jacobian) at any state.

        For the Arneodo system:
            div = d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz
                = 0 + 0 + (-1) = -1

        The divergence is constant everywhere (independent of state),
        making the Arneodo system uniformly dissipative.
        """
        return -1.0

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

            # Bail out if trajectory diverges
            if not (
                np.all(np.isfinite(state1))
                and np.all(np.isfinite(state2))
            ):
                break

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

    def compute_jerk(self, state: np.ndarray) -> float:
        """Compute the jerk (third derivative of x) at a given state.

        From the jerk form: x''' = -x'' - b*x' - a*x + d*x^3
        With x' = y, x'' = z:
            jerk = -z - b*y - a*x + d*x^3
        """
        x, y, z = state
        return -z - self.b * y - self.a * x + self.d * x**3

    def bifurcation_sweep(
        self,
        d_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep parameter d and record attractor statistics.

        For each d value, computes the Lyapunov exponent after
        skipping transients. The Arneodo system undergoes a
        period-doubling cascade to chaos as d increases.

        Args:
            d_values: Array of 'd' parameter values to sweep.
            n_transient: Steps to skip for transient.
            n_measure: Steps to use for Lyapunov estimation.

        Returns:
            Dict with d values, Lyapunov exponents, and attractor types.
        """
        lyapunov_exps = []
        attractor_types = []

        for d_val in d_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_transient + n_measure,
                parameters={
                    "a": self.a,
                    "b": self.b,
                    "d": d_val,
                    "x_0": self.x_0,
                    "y_0": self.y_0,
                    "z_0": self.z_0,
                },
            )
            sim = ArneodoSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Estimate Lyapunov
            lam = sim.estimate_lyapunov(
                n_steps=n_measure, dt=self.config.dt
            )
            lyapunov_exps.append(lam)

            # Classify attractor
            if lam > 0.01:
                atype = "chaotic"
            elif lam < -0.01:
                atype = "stable"
            else:
                atype = "marginal"
            attractor_types.append(atype)

        return {
            "d": d_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "attractor_type": np.array(attractor_types),
        }
