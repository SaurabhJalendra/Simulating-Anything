"""Stochastic genetic toggle switch simulation (Gardner et al., Nature 2000).

Two mutually repressing genes with additive Gaussian white noise, forming a
noise-driven bistable switch. Integrated via Euler-Maruyama with non-negativity
clamp.

SDE system:
    du = (alpha1 / (1 + v^beta) - u) dt + sigma * dW1
    dv = (alpha2 / (1 + u^gamma) - v) dt + sigma * dW2

where u, v are protein concentrations of the two repressors and sigma controls
the intensity of intrinsic noise. The noise can induce stochastic switching
between the two stable fixed points of the deterministic system.

Target rediscoveries:
- Bistability: two stable fixed points (high-u/low-v) and (low-u/high-v)
- Noise-induced switching between stable states
- Switching rate increases with sigma
- Nullcline intersection yields fixed points
- Deterministic basins of attraction
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ToggleSwitchStochasticSimulation(SimulationEnvironment):
    """Stochastic genetic toggle switch with additive noise.

    du/dt = alpha1 / (1 + v^beta) - u + sigma * xi_1(t)
    dv/dt = alpha2 / (1 + u^gamma) - v + sigma * xi_2(t)

    where xi_1, xi_2 are independent Gaussian white noise processes.
    Integrated with Euler-Maruyama and non-negativity clamp (protein
    concentrations cannot be negative).

    State vector: [u, v] where u, v are protein concentrations.

    Parameters:
        alpha1: max production rate for gene 1 (default 5.0)
        alpha2: max production rate for gene 2 (default 5.0)
        beta_hill: Hill coefficient for repression of gene 1 by v (default 2.0)
        gamma_hill: Hill coefficient for repression of gene 2 by u (default 2.0)
        sigma: noise intensity (default 0.3)
        u_0: initial u concentration (default 0.5)
        v_0: initial v concentration (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha1 = p.get("alpha1", 5.0)
        self.alpha2 = p.get("alpha2", 5.0)
        self.beta_hill = p.get("beta_hill", 2.0)
        self.gamma_hill = p.get("gamma_hill", 2.0)
        self.sigma = p.get("sigma", 0.3)
        self.u_0 = p.get("u_0", 0.5)
        self.v_0 = p.get("v_0", 0.5)
        self._rng: np.random.Generator = np.random.default_rng(config.seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize protein concentrations."""
        self._rng = np.random.default_rng(seed)
        self._state = np.array([self.u_0, self.v_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using Euler-Maruyama for the SDE."""
        self._euler_maruyama_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [u, v]."""
        return self._state

    def _euler_maruyama_step(self) -> None:
        """Single Euler-Maruyama step with non-negativity clamp."""
        dt = self.config.dt
        u, v = self._state

        # Deterministic drift
        u_c = max(u, 0.0)
        v_c = max(v, 0.0)
        du_det = self.alpha1 / (1.0 + v_c ** self.beta_hill) - u
        dv_det = self.alpha2 / (1.0 + u_c ** self.gamma_hill) - v

        # Stochastic diffusion (independent noise on both equations)
        sqrt_dt = np.sqrt(dt)
        dW1 = self._rng.normal(0.0, sqrt_dt)
        dW2 = self._rng.normal(0.0, sqrt_dt)

        # Euler-Maruyama update with non-negativity clamp
        u_new = u + du_det * dt + self.sigma * dW1
        v_new = v + dv_det * dt + self.sigma * dW2

        self._state = np.array(
            [max(u_new, 0.0), max(v_new, 0.0)], dtype=np.float64
        )

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Deterministic part of the toggle switch dynamics (no noise).

        Args:
            y: State vector [u, v].

        Returns:
            Derivatives [du/dt, dv/dt].
        """
        u, v = y
        u_c = max(u, 0.0)
        v_c = max(v, 0.0)
        du = self.alpha1 / (1.0 + v_c ** self.beta_hill) - u
        dv = self.alpha2 / (1.0 + u_c ** self.gamma_hill) - v
        return np.array([du, dv])

    def find_fixed_points(self) -> list[tuple[float, float]]:
        """Find all fixed points of the deterministic system numerically.

        At a fixed point:
            u = alpha1 / (1 + v^beta)
            v = alpha2 / (1 + u^gamma)

        Returns:
            List of (u, v) tuples for each fixed point found.
        """
        def equations(uv: np.ndarray) -> np.ndarray:
            u, v = uv
            u_c = max(u, 0.0)
            v_c = max(v, 0.0)
            eq1 = self.alpha1 / (1.0 + v_c ** self.beta_hill) - u
            eq2 = self.alpha2 / (1.0 + u_c ** self.gamma_hill) - v
            return np.array([eq1, eq2])

        guesses = [
            (self.alpha1, 0.01),
            (0.01, self.alpha2),
            (self.alpha1 / 2, self.alpha2 / 2),
            (1.0, 1.0),
            (0.1, 4.0),
            (4.0, 0.1),
            (2.0, 2.0),
        ]

        found: list[tuple[float, float]] = []
        for guess in guesses:
            sol = fsolve(equations, guess, full_output=True)
            x_sol = sol[0]
            residual = np.linalg.norm(equations(x_sol))
            if residual < 1e-8 and x_sol[0] > -0.01 and x_sol[1] > -0.01:
                u_fp = float(max(x_sol[0], 0.0))
                v_fp = float(max(x_sol[1], 0.0))
                is_new = True
                for existing in found:
                    if abs(u_fp - existing[0]) < 1e-4 and abs(v_fp - existing[1]) < 1e-4:
                        is_new = False
                        break
                if is_new:
                    found.append((u_fp, v_fp))

        found.sort(key=lambda p: p[0])
        return found

    def classify_fixed_point(self, u_fp: float, v_fp: float) -> str:
        """Classify a fixed point as stable, unstable, or saddle.

        Computes eigenvalues of the Jacobian at the fixed point.

        Returns:
            'stable', 'unstable', or 'saddle'
        """
        jac = self._jacobian(u_fp, v_fp)
        eigenvalues = np.linalg.eigvals(jac)
        real_parts = np.real(eigenvalues)

        if all(r < 0 for r in real_parts):
            return "stable"
        elif all(r > 0 for r in real_parts):
            return "unstable"
        else:
            return "saddle"

    def _jacobian(self, u: float, v: float) -> np.ndarray:
        """Compute Jacobian matrix at point (u, v).

        J = [[-1,                                        -alpha1*beta*v^(beta-1)/(1+v^beta)^2 ],
             [-alpha2*gamma*u^(gamma-1)/(1+u^gamma)^2,   -1                                   ]]
        """
        u_c = max(u, 1e-12)
        v_c = max(v, 1e-12)

        j11 = -1.0
        j12 = (
            -self.alpha1 * self.beta_hill * v_c ** (self.beta_hill - 1)
            / (1.0 + v_c ** self.beta_hill) ** 2
        )
        j21 = (
            -self.alpha2 * self.gamma_hill * u_c ** (self.gamma_hill - 1)
            / (1.0 + u_c ** self.gamma_hill) ** 2
        )
        j22 = -1.0

        return np.array([[j11, j12], [j21, j22]])

    def compute_nullclines(
        self,
        u_range: tuple[float, float] = (0.0, 6.0),
        n_points: int = 200,
    ) -> dict[str, np.ndarray]:
        """Compute the u-nullcline and v-nullcline.

        u-nullcline: du/dt = 0  =>  u = alpha1 / (1 + v^beta)
        v-nullcline: dv/dt = 0  =>  v = alpha2 / (1 + u^gamma)

        Returns:
            Dict with 'u_null_v', 'u_null_u' (the u-nullcline as u(v))
            and 'v_null_u', 'v_null_v' (the v-nullcline as v(u)).
        """
        v_vals = np.linspace(u_range[0], u_range[1], n_points)
        u_vals = np.linspace(u_range[0], u_range[1], n_points)

        u_null_u = self.alpha1 / (1.0 + np.maximum(v_vals, 0.0) ** self.beta_hill)
        u_null_v = v_vals

        v_null_u = u_vals
        v_null_v = self.alpha2 / (1.0 + np.maximum(u_vals, 0.0) ** self.gamma_hill)

        return {
            "u_null_u": u_null_u,
            "u_null_v": u_null_v,
            "v_null_u": v_null_u,
            "v_null_v": v_null_v,
        }

    def count_switches(
        self,
        u_trace: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> int:
        """Count the number of switching events between bistable states.

        A switch is defined as crossing the midpoint between the two stable
        fixed points along the u axis.

        Args:
            u_trace: 1D array of u values. If None, runs a trajectory.
            threshold: u value to detect crossings. If None, uses the
                midpoint from fixed point analysis.

        Returns:
            Number of switching events (threshold crossings in either direction).
        """
        if u_trace is None:
            traj = self.run()
            u_trace = traj.states[:, 0]

        if threshold is None:
            fps = self.find_fixed_points()
            if len(fps) >= 2:
                threshold = (fps[0][0] + fps[-1][0]) / 2.0
            else:
                threshold = self.alpha1 / 2.0

        switches = 0
        above = u_trace[0] > threshold
        for i in range(1, len(u_trace)):
            now_above = u_trace[i] > threshold
            if now_above != above:
                switches += 1
                above = now_above
        return switches

    def dwell_times(
        self,
        u_trace: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute dwell times in each stable state.

        Args:
            u_trace: 1D array of u values.
            threshold: u threshold separating the two states.

        Returns:
            Dict with 'high_state_times' and 'low_state_times' arrays
            (in timestep units).
        """
        if u_trace is None:
            traj = self.run()
            u_trace = traj.states[:, 0]

        if threshold is None:
            fps = self.find_fixed_points()
            if len(fps) >= 2:
                threshold = (fps[0][0] + fps[-1][0]) / 2.0
            else:
                threshold = self.alpha1 / 2.0

        dt = self.config.dt
        high_times: list[float] = []
        low_times: list[float] = []

        above = u_trace[0] > threshold
        current_dwell = 0

        for i in range(1, len(u_trace)):
            now_above = u_trace[i] > threshold
            current_dwell += 1
            if now_above != above:
                dwell_time = current_dwell * dt
                if above:
                    high_times.append(dwell_time)
                else:
                    low_times.append(dwell_time)
                current_dwell = 0
                above = now_above

        return {
            "high_state_times": np.array(high_times),
            "low_state_times": np.array(low_times),
        }

    def which_state(self, state: np.ndarray | None = None) -> str:
        """Determine which stable state the system is currently in.

        Args:
            state: State [u, v]. If None, uses current state.

        Returns:
            'high_u' if u > v, 'high_v' if v > u, 'boundary' if near equal.
        """
        if state is None:
            state = self._state
        u, v = state
        if abs(u - v) < 0.1:
            return "boundary"
        return "high_u" if u > v else "high_v"
