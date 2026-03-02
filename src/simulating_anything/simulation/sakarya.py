"""Sakarya attractor simulation -- 3D chaotic ODE.

The Sakarya system is a 3D autonomous dissipative system that exhibits
chaotic behavior for certain parameter values:
    dx/dt = -x + y + y*z
    dy/dt = -x - y + a*x*z
    dz/dt = z - b*x*y

Target rediscoveries:
- SINDy recovery of Sakarya ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Chaos transition as parameters a, b vary
- Fixed point computation and verification
- Attractor boundedness verification
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SakaryaSimulation(SimulationEnvironment):
    """Sakarya attractor: a 3D chaotic ODE system.

    State vector: [x, y, z]

    ODEs:
        dx/dt = -x + y + y*z
        dy/dt = -x - y + a*x*z
        dz/dt = z - b*x*y

    Parameters:
        a: nonlinear coupling parameter (default 0.4)
        b: nonlinear coupling parameter (default 0.3)
        x_0, y_0, z_0: initial conditions (default 1, -1, 1)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.4)
        self.b = p.get("b", 0.3)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", -1.0)
        self.z_0 = p.get("z_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Sakarya state."""
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
        """Sakarya equations.

        dx/dt = -x + y + y*z
        dy/dt = -x - y + a*x*z
        dz/dt = z - b*x*y
        """
        x, y, z = state
        dx = -x + y + y * z
        dy = -x - y + self.a * x * z
        dz = z - self.b * x * y
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute fixed points of the Sakarya system.

        Setting derivatives to zero:
            -x + y + y*z = 0    =>  -x + y(1 + z) = 0
            -x - y + a*x*z = 0
            z - b*x*y = 0

        The origin (0, 0, 0) is always a fixed point.

        For non-trivial fixed points, from equation 3: z = b*x*y.
        Substituting into equation 1: -x + y(1 + b*x*y) = 0
        Substituting into equation 2: -x - y + a*x*b*x*y = 0
            => -x - y + a*b*x^2*y = 0

        From equation 1: x = y(1 + b*x*y) => x = y + b*x*y^2
        These are generally found numerically.
        """
        points = [np.array([0.0, 0.0, 0.0])]

        # Attempt to find non-trivial fixed points numerically by
        # solving the algebraic system. For generic a, b values,
        # we search on a grid and refine via Newton's method.
        try:
            from scipy.optimize import fsolve

            def equations(vars: np.ndarray) -> np.ndarray:
                x, y, z = vars
                eq1 = -x + y + y * z
                eq2 = -x - y + self.a * x * z
                eq3 = z - self.b * x * y
                return np.array([eq1, eq2, eq3])

            # Search from several initial guesses
            guesses = [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [2.0, -2.0, -1.0],
                [-2.0, 2.0, -1.0],
                [1.0, -1.0, 0.5],
                [-1.0, 1.0, 0.5],
                [3.0, 3.0, 2.0],
                [-3.0, -3.0, 2.0],
            ]
            for guess in guesses:
                sol, info, ier, _ = fsolve(
                    equations, guess, full_output=True
                )
                if ier == 1:
                    residual = np.linalg.norm(info["fvec"])
                    if residual < 1e-10:
                        # Check if this is a duplicate
                        is_dup = False
                        for fp in points:
                            if np.linalg.norm(sol - fp) < 1e-6:
                                is_dup = True
                                break
                        if not is_dup:
                            points.append(
                                np.array(sol, dtype=np.float64)
                            )
        except ImportError:
            pass

        return points

    @property
    def is_chaotic(self) -> bool:
        """Heuristic: system is chaotic for a=0.4, b=0.3.

        This is a rough heuristic based on the standard parameter choice.
        """
        return self.a > 0.1 and self.b > 0.1

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

            # Check for divergence (NaN/Inf)
            if not (np.all(np.isfinite(state1)) and np.all(np.isfinite(state2))):
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

    def bifurcation_sweep(
        self,
        a_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep parameter a and record attractor statistics.

        For each a value, computes the Lyapunov exponent and trajectory
        statistics after skipping transients.

        Args:
            a_values: Array of parameter a values to sweep.
            n_transient: Steps to skip for transient.
            n_measure: Steps to use for Lyapunov estimation.

        Returns:
            Dict with a values, Lyapunov exponents, and attractor types.
        """
        lyapunov_exps = []
        attractor_types = []
        amplitudes = []

        for a_val in a_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_transient + n_measure,
                parameters={
                    "a": a_val,
                    "b": self.b,
                    "x_0": self.x_0,
                    "y_0": self.y_0,
                    "z_0": self.z_0,
                },
            )
            sim = SakaryaSimulation(config)
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
                if np.all(np.isfinite(state)):
                    vals.append(np.linalg.norm(state))
            amplitudes.append(np.max(vals) if vals else np.nan)

            # Classify
            if lam > 0.01:
                atype = "chaotic"
            elif lam < -0.01:
                atype = "fixed_point"
            else:
                atype = "marginal"
            attractor_types.append(atype)

        return {
            "a": a_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "max_amplitude": np.array(amplitudes),
            "attractor_type": np.array(attractor_types),
        }
