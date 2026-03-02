"""Genetic toggle switch simulation (Gardner et al., Nature 2000).

Two mutually repressing genes forming a bistable circuit:

    du/dt = alpha1 / (1 + v^beta) - u
    dv/dt = alpha2 / (1 + u^gamma) - v

where u, v are protein concentrations of the two repressors.

Target rediscoveries:
- Bistability: two stable fixed points (high u, low v) and (low u, high v)
- One unstable saddle point on the separatrix
- Pitchfork-like bifurcation as alpha varies
- ODE recovery via SINDy
- Nullcline computation and intersection
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ToggleSwitchSimulation(SimulationEnvironment):
    """Genetic toggle switch: mutual repression of two genes.

    State vector: [u, v] where u, v are protein concentrations.

    du/dt = alpha1 / (1 + v^beta) - u
    dv/dt = alpha2 / (1 + u^gamma) - v

    For equal parameters (alpha1=alpha2, beta=gamma), the system is symmetric
    with two stable fixed points related by (u, v) <-> (v, u) and one unstable
    saddle on the diagonal u = v.

    Parameters:
        alpha1: max production rate for gene 1 (default 4.0)
        alpha2: max production rate for gene 2 (default 4.0)
        beta: Hill coefficient for repression of gene 1 by v (default 4.0)
        gamma: Hill coefficient for repression of gene 2 by u (default 4.0)
        u_0: initial u concentration (default 3.0)
        v_0: initial v concentration (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha1 = p.get("alpha1", 4.0)
        self.alpha2 = p.get("alpha2", 4.0)
        self.beta = p.get("beta", 4.0)
        self.gamma = p.get("gamma", 4.0)
        self.u_0 = p.get("u_0", 3.0)
        self.v_0 = p.get("v_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize protein concentrations."""
        if seed is not None:
            rng = np.random.RandomState(seed)
            perturbation = rng.randn(2) * 0.01
        else:
            perturbation = np.zeros(2)
        self._state = np.array(
            [self.u_0 + perturbation[0], self.v_0 + perturbation[1]],
            dtype=np.float64,
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [u, v]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute du/dt, dv/dt for the toggle switch system."""
        u, v = y
        # Clamp to avoid negative values in Hill function
        v_clamped = max(v, 0.0)
        u_clamped = max(u, 0.0)
        du = self.alpha1 / (1.0 + v_clamped**self.beta) - u
        dv = self.alpha2 / (1.0 + u_clamped**self.gamma) - v
        return np.array([du, dv])

    def find_fixed_points(self) -> list[tuple[float, float]]:
        """Find all fixed points numerically.

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
            eq1 = self.alpha1 / (1.0 + v_c**self.beta) - u
            eq2 = self.alpha2 / (1.0 + u_c**self.gamma) - v
            return np.array([eq1, eq2])

        # Use multiple initial guesses to find all fixed points
        guesses = [
            (self.alpha1, 0.01),          # high u, low v
            (0.01, self.alpha2),          # low u, high v
            (self.alpha1 / 2, self.alpha2 / 2),  # middle
            (1.0, 1.0),
            (0.1, 3.0),
            (3.0, 0.1),
            (2.0, 2.0),
        ]

        found: list[tuple[float, float]] = []
        for guess in guesses:
            sol = fsolve(equations, guess, full_output=True)
            x_sol = sol[0]
            # Check if solution is valid (converged and non-negative)
            residual = np.linalg.norm(equations(x_sol))
            if residual < 1e-8 and x_sol[0] > -0.01 and x_sol[1] > -0.01:
                u_fp, v_fp = float(max(x_sol[0], 0.0)), float(max(x_sol[1], 0.0))
                # Check if this is a new fixed point
                is_new = True
                for existing in found:
                    if abs(u_fp - existing[0]) < 1e-4 and abs(v_fp - existing[1]) < 1e-4:
                        is_new = False
                        break
                if is_new:
                    found.append((u_fp, v_fp))

        # Sort by u value for consistent ordering
        found.sort(key=lambda p: p[0])
        return found

    def classify_fixed_point(
        self, u_fp: float, v_fp: float
    ) -> str:
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

        J = [[-1,                          -alpha1 * beta * v^(beta-1) / (1+v^beta)^2],
             [-alpha2 * gamma * u^(gamma-1) / (1+u^gamma)^2,  -1                     ]]
        """
        u_c = max(u, 1e-12)
        v_c = max(v, 1e-12)

        j11 = -1.0
        j12 = -self.alpha1 * self.beta * v_c**(self.beta - 1) / (
            1.0 + v_c**self.beta
        )**2
        j21 = -self.alpha2 * self.gamma * u_c**(self.gamma - 1) / (
            1.0 + u_c**self.gamma
        )**2
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

        # u-nullcline: u = alpha1 / (1 + v^beta), parameterized by v
        u_null_u = self.alpha1 / (1.0 + np.maximum(v_vals, 0.0)**self.beta)
        u_null_v = v_vals

        # v-nullcline: v = alpha2 / (1 + u^gamma), parameterized by u
        v_null_u = u_vals
        v_null_v = self.alpha2 / (1.0 + np.maximum(u_vals, 0.0)**self.gamma)

        return {
            "u_null_u": u_null_u,
            "u_null_v": u_null_v,
            "v_null_u": v_null_u,
            "v_null_v": v_null_v,
        }

    def alpha_sweep(
        self,
        alpha_range: tuple[float, float] = (0.5, 8.0),
        n_alpha: int = 30,
        n_steps: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep alpha1=alpha2 and record final states from different ICs.

        This reveals the pitchfork-like bifurcation: for small alpha, only one
        stable fixed point exists; for large alpha, two stable states appear.

        Returns:
            Dict with 'alpha', 'final_u_high', 'final_v_high',
            'final_u_low', 'final_v_low'.
        """
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        final_u_high = []
        final_v_high = []
        final_u_low = []
        final_v_low = []

        for alpha in alpha_values:
            # IC1: high u, low v (biased toward gene 1 dominant)
            config_high = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "alpha1": float(alpha),
                    "alpha2": float(alpha),
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "u_0": float(alpha),
                    "v_0": 0.1,
                },
            )
            sim_high = ToggleSwitchSimulation(config_high)
            sim_high.reset()
            for _ in range(n_steps):
                sim_high.step()
            state_high = sim_high.observe()
            final_u_high.append(state_high[0])
            final_v_high.append(state_high[1])

            # IC2: low u, high v (biased toward gene 2 dominant)
            config_low = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "alpha1": float(alpha),
                    "alpha2": float(alpha),
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "u_0": 0.1,
                    "v_0": float(alpha),
                },
            )
            sim_low = ToggleSwitchSimulation(config_low)
            sim_low.reset()
            for _ in range(n_steps):
                sim_low.step()
            state_low = sim_low.observe()
            final_u_low.append(state_low[0])
            final_v_low.append(state_low[1])

        return {
            "alpha": alpha_values,
            "final_u_high": np.array(final_u_high),
            "final_v_high": np.array(final_v_high),
            "final_u_low": np.array(final_u_low),
            "final_v_low": np.array(final_v_low),
        }
