"""Halvorsen attractor simulation -- 3D chaotic ODE with cyclic symmetry.

The Halvorsen system is a 3D chaotic flow with beautiful cyclic (C3) symmetry:
    dx/dt = -a*x - 4*y - 4*z - y^2
    dy/dt = -a*y - 4*z - 4*x - z^2
    dz/dt = -a*z - 4*x - 4*y - x^2

The system is invariant under the cyclic permutation (x, y, z) -> (y, z, x),
which produces a trefoil-like attractor structure.

Classic parameter: a = 1.89 (chaos for a near 1.4 to 2.0).

Target rediscoveries:
- SINDy recovery of Halvorsen ODEs
- Positive Lyapunov exponent confirming chaos
- Cyclic symmetry (x, y, z) -> (y, z, x) verification
- a-parameter sweep mapping chaos boundary
- Attractor boundedness and statistics
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class HalvorsenSimulation(SimulationEnvironment):
    """Halvorsen attractor: a 3D chaotic ODE with cyclic symmetry.

    State vector: [x, y, z]

    ODEs:
        dx/dt = -a*x - 4*y - 4*z - y^2
        dy/dt = -a*y - 4*z - 4*x - z^2
        dz/dt = -a*z - 4*x - 4*y - x^2

    The system has cyclic (C3) symmetry: applying the permutation
    (x, y, z) -> (y, z, x) maps solutions to solutions. For the classic
    parameter a = 1.89, the attractor has a trefoil-like structure.

    Parameters:
        a: dissipation/coupling parameter (default 1.89, chaotic regime)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.89)
        self.x_0 = p.get("x_0", -5.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Halvorsen state."""
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
        """Halvorsen equations with cyclic symmetry.

        dx/dt = -a*x - 4*y - 4*z - y^2
        dy/dt = -a*y - 4*z - 4*x - z^2
        dz/dt = -a*z - 4*x - 4*y - x^2
        """
        x, y, z = state
        dx = -self.a * x - 4.0 * y - 4.0 * z - y**2
        dy = -self.a * y - 4.0 * z - 4.0 * x - z**2
        dz = -self.a * z - 4.0 * x - 4.0 * y - x**2
        return np.array([dx, dy, dz])

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

    def cyclic_symmetry_check(self, n_steps: int = 1000) -> dict[str, float]:
        """Verify the cyclic symmetry (x, y, z) -> (y, z, x) of the Halvorsen system.

        Runs two trajectories: one starting at (x0, y0, z0) and another starting
        at (y0, z0, x0). Due to cyclic symmetry, if the first evolves to (x, y, z),
        the second should evolve to (y, z, x).

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

            # Check: state1 = (x, y, z), state2 should be (y, z, x)
            # So state2[0] ~ state1[1], state2[1] ~ state1[2], state2[2] ~ state1[0]
            expected = np.array([state1[1], state1[2], state1[0]])
            dev = np.linalg.norm(state2 - expected)
            max_dev = max(max_dev, dev)
            sum_dev += dev

        return {
            "max_deviation": max_dev,
            "mean_deviation": sum_dev / n_steps,
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
            a_values: Array of a-parameter values to sweep.
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
                    "x_0": self.x_0,
                    "y_0": self.y_0,
                    "z_0": self.z_0,
                },
            )
            sim = HalvorsenSimulation(config)
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
            "a": a_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "max_amplitude": np.array(amplitudes),
            "attractor_type": np.array(attractor_types),
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
