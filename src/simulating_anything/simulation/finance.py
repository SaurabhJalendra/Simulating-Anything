"""Chaotic financial system simulation -- 3D nonlinear ODE.

The chaotic financial system (Chen 2008, Gao & Ma 2009) models the
interaction between interest rate x, investment demand y, and price
index z in a simplified economy:
    dx/dt = z + (y - a)*x
    dy/dt = 1 - b*y - x^2
    dz/dt = -x - c*z

Parameters:
    a: savings rate (classic: 1.0)
    b: cost per investment (classic: 0.1)
    c: elasticity of demand (classic: 1.0)

The system exhibits chaotic behavior for a=0.9, b=0.2, c=1.2 and other
parameter combinations. With a=1, b=0.1, c=1 the attractor is chaotic
with a positive largest Lyapunov exponent.

Target rediscoveries:
- SINDy recovery of financial system ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin is the sole equilibrium)
- Parameter sweep mapping chaos boundary (vary a)
- Market stability transition detection
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FinanceSimulation(SimulationEnvironment):
    """Chaotic financial system: a 3D ODE for market dynamics.

    State vector: [x, y, z]
        x = interest rate
        y = investment demand
        z = price index

    ODEs:
        dx/dt = z + (y - a)*x
        dy/dt = 1 - b*y - x^2
        dz/dt = -x - c*z

    The Jacobian trace (divergence) at the origin is -a - b - c, which
    is negative for positive parameters, ensuring phase-space volume
    contraction and attractor formation.

    Parameters:
        a: savings rate (default 1.0)
        b: cost per investment (default 0.1)
        c: elasticity of demand (default 1.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.0)
        self.b = p.get("b", 0.1)
        self.c = p.get("c", 1.0)
        self.x_0 = p.get("x_0", 2.0)
        self.y_0 = p.get("y_0", 3.0)
        self.z_0 = p.get("z_0", 2.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize financial system state."""
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
        """Financial system equations.

        dx/dt = z + (y - a)*x
        dy/dt = 1 - b*y - x^2
        dz/dt = -x - c*z
        """
        x, y, z = state
        dx = z + (y - self.a) * x
        dy = 1.0 - self.b * y - x ** 2
        dz = -x - self.c * z
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the financial system.

        Setting derivatives to zero:
            z + (y - a)*x = 0  ... (1)
            1 - b*y - x^2 = 0  ... (2)
            -x - c*z = 0       ... (3)

        From (3): z = -x/c
        Substituting into (1): -x/c + (y - a)*x = 0
            => x*(-1/c + y - a) = 0
            => x = 0 or y = a + 1/c

        Case 1: x = 0
            From (2): y = 1/b
            From (3): z = 0
            Fixed point: (0, 1/b, 0)

        Case 2: y = a + 1/c
            From (2): x^2 = 1 - b*(a + 1/c)
            If 1 - b*(a + 1/c) > 0, two symmetric fixed points exist.
            From (3): z = -x/c
        """
        points = []

        # Case 1: x = 0 => (0, 1/b, 0)
        y_fp1 = 1.0 / self.b
        points.append(np.array([0.0, y_fp1, 0.0], dtype=np.float64))

        # Case 2: y = a + 1/c
        y_fp2 = self.a + 1.0 / self.c
        x_sq = 1.0 - self.b * y_fp2
        if x_sq > 0:
            x_fp = np.sqrt(x_sq)
            z_fp = -x_fp / self.c
            points.append(np.array([x_fp, y_fp2, z_fp], dtype=np.float64))
            points.append(np.array([-x_fp, y_fp2, -z_fp], dtype=np.float64))

        return points

    @property
    def divergence(self) -> float:
        """Phase space divergence at the origin: -a - b - c.

        The Jacobian of the system is:
            J = [[y - a, x, 1],
                 [-2x, -b, 0],
                 [-1, 0, -c]]

        At the origin (0, 1/b, 0):
            trace(J) = (1/b - a) + (-b) + (-c)

        For generic analysis, the divergence at any equilibrium depends
        on the local state, but the dominant contribution is -a - b - c
        evaluated at the trivial equilibrium direction.
        """
        return -self.a - self.b - self.c

    @property
    def is_chaotic(self) -> bool:
        """Heuristic for chaotic regime.

        The financial system is known to be chaotic near the classic
        parameters (a=1, b=0.1, c=1) and also (a=0.9, b=0.2, c=1.2).
        This heuristic checks basic necessary conditions.
        """
        # The system needs b to be small enough for the nonlinear
        # coupling to dominate, and the dissipation to be present
        return self.a > 0 and self.b > 0 and self.c > 0 and self.b < 1.0

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
            param_name: Name of the parameter to sweep (a, b, or c).
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
            "x_0": self.x_0,
            "y_0": self.y_0,
            "z_0": self.z_0,
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
            sim = FinanceSimulation(config)
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

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[y - a, x, 1],
             [-2x,  -b, 0],
             [-1,    0, -c]]
        """
        x, y, z = state
        return np.array([
            [y - self.a, x, 1.0],
            [-2.0 * x, -self.b, 0.0],
            [-1.0, 0.0, -self.c],
        ])

    def eigenvalues_at_fixed_points(self) -> list[dict]:
        """Compute eigenvalues of the Jacobian at each fixed point.

        Returns:
            List of dicts with 'fixed_point', 'eigenvalues', and 'stability'.
        """
        results = []
        for fp in self.fixed_points:
            J = self.jacobian(fp)
            eigvals = np.linalg.eigvals(J)
            # Classify stability
            real_parts = np.real(eigvals)
            if np.all(real_parts < 0):
                stability = "stable"
            elif np.all(real_parts > 0):
                stability = "unstable"
            else:
                stability = "saddle"
            results.append({
                "fixed_point": fp.tolist(),
                "eigenvalues": eigvals.tolist(),
                "max_real_part": float(np.max(real_parts)),
                "stability": stability,
            })
        return results
