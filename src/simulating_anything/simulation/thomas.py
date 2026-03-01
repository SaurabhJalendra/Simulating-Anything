"""Thomas cyclically symmetric attractor simulation.

The Thomas system is a 3D chaotic ODE with beautiful labyrinth-like structure:
    dx/dt = sin(y) - b*x
    dy/dt = sin(z) - b*y
    dz/dt = sin(x) - b*z

Target rediscoveries:
- SINDy recovery of Thomas ODEs
- Critical dissipation b_c ~ 0.208186 for chaos onset
- Cyclic symmetry (x,y,z) -> (y,z,x)
- Lyapunov exponent as function of b
- Fixed points at (x,x,x) where sin(x) = b*x
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ThomasSimulation(SimulationEnvironment):
    """Thomas cyclically symmetric attractor: labyrinth chaos in 3D.

    State vector: [x, y, z]

    ODEs:
        dx/dt = sin(y) - b*x
        dy/dt = sin(z) - b*y
        dz/dt = sin(x) - b*z

    The system has cyclic symmetry: applying (x,y,z) -> (y,z,x) maps
    solutions to solutions. For small dissipation b, the attractor has
    a complex labyrinth-like structure.

    Parameters:
        b: dissipation parameter (default 0.208186, critical value for chaos)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.b = p.get("b", 0.208186)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Thomas state."""
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
        """Thomas equations: dx/dt=sin(y)-b*x, dy/dt=sin(z)-b*y, dz/dt=sin(x)-b*z."""
        x, y, z = state
        dx = np.sin(y) - self.b * x
        dy = np.sin(z) - self.b * y
        dz = np.sin(x) - self.b * z
        return np.array([dx, dy, dz])

    def find_fixed_points(self, n_search: int = 50) -> list[np.ndarray]:
        """Find fixed points of the Thomas system.

        Fixed points satisfy sin(x) = b*x with x = y = z (by cyclic symmetry).
        The equation sin(x) = b*x has solutions that can be found numerically.
        For b < 1, there are multiple solutions in each period of sin.

        Args:
            n_search: Number of intervals to search for roots.

        Returns:
            List of fixed point arrays [x, x, x].
        """
        # The fixed point condition is f(x) = sin(x) - b*x = 0
        # Search over a wide range to find all roots
        fixed_points = []
        search_range = 20.0
        x_grid = np.linspace(-search_range, search_range, n_search * 10)

        def f(x: float) -> float:
            return np.sin(x) - self.b * x

        # Find sign changes
        f_vals = np.array([f(xi) for xi in x_grid])
        for i in range(len(f_vals) - 1):
            if f_vals[i] * f_vals[i + 1] < 0:
                try:
                    root = brentq(f, x_grid[i], x_grid[i + 1])
                    # Avoid duplicates
                    is_dup = False
                    for fp in fixed_points:
                        if abs(fp[0] - root) < 1e-8:
                            is_dup = True
                            break
                    if not is_dup:
                        fixed_points.append(
                            np.array([root, root, root], dtype=np.float64)
                        )
                except ValueError:
                    pass

        return fixed_points

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

    def bifurcation_sweep(
        self,
        b_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep dissipation parameter b and record attractor statistics.

        For each b value, computes the Lyapunov exponent and trajectory
        statistics after skipping transients.

        Args:
            b_values: Array of dissipation values to sweep.
            n_transient: Steps to skip for transient.
            n_measure: Steps to use for Lyapunov estimation.

        Returns:
            Dict with b values, Lyapunov exponents, and attractor types.
        """
        lyapunov_exps = []
        attractor_types = []
        amplitudes = []

        for b_val in b_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_transient + n_measure,
                parameters={"b": b_val, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
            )
            sim = ThomasSimulation(config)
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
            amplitudes.append(np.max(vals))

            # Classify
            if lam > 0.01:
                atype = "chaotic"
            elif lam < -0.01:
                atype = "fixed_point"
            else:
                atype = "marginal"
            attractor_types.append(atype)

        return {
            "b": b_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "max_amplitude": np.array(amplitudes),
            "attractor_type": np.array(attractor_types),
        }

    def verify_cyclic_symmetry(self, n_steps: int = 1000) -> dict[str, float]:
        """Verify the cyclic symmetry (x,y,z) -> (y,z,x) of the Thomas system.

        Runs two trajectories: one starting at (x0,y0,z0) and another starting
        at (y0,z0,x0). Due to cyclic symmetry, if the first evolves to (x,y,z),
        the second should evolve to (y,z,x).

        Args:
            n_steps: Number of steps to evolve.

        Returns:
            Dict with max and mean deviation between symmetry-related trajectories.
        """
        dt = self.config.dt

        # Trajectory 1: (x0, y0, z0)
        state1 = np.array([self.x_0, self.y_0, self.z_0], dtype=np.float64)
        # Trajectory 2: (y0, z0, x0) -- cyclic permutation of initial conditions
        state2 = np.array([self.y_0, self.z_0, self.x_0], dtype=np.float64)

        max_dev = 0.0
        sum_dev = 0.0

        for _ in range(n_steps):
            # Advance state1
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance state2
            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Check: state1 = (x,y,z), state2 should be (y,z,x)
            # So state2[0] ~ state1[1], state2[1] ~ state1[2], state2[2] ~ state1[0]
            expected = np.array([state1[1], state1[2], state1[0]])
            dev = np.linalg.norm(state2 - expected)
            max_dev = max(max_dev, dev)
            sum_dev += dev

        return {
            "max_deviation": max_dev,
            "mean_deviation": sum_dev / n_steps,
        }

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
