"""Rucklidge attractor simulation -- double convection model.

The Rucklidge system is a 3D chaotic ODE introduced by A. M. Rucklidge (1992)
to model thermosolutal (double) convection:
    dx/dt = -kappa*x + lambda*y - y*z
    dy/dt = x
    dz/dt = -z + y^2

where kappa and lambda are positive parameters controlling the dynamics.

Target rediscoveries:
- SINDy recovery of Rucklidge ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin + two symmetric)
- Divergence = -(kappa + 1)
- Chaos transition as lambda varies
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RucklidgeSimulation(SimulationEnvironment):
    """Rucklidge attractor: double convection chaos.

    State vector: [x, y, z]

    ODEs:
        dx/dt = -kappa*x + lambda*y - y*z
        dy/dt = x
        dz/dt = -z + y^2

    Parameters:
        kappa: dissipation parameter (default 2.0)
        lambda_param: driving parameter (default 6.7)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.kappa = p.get("kappa", 2.0)
        self.lambda_param = p.get("lambda_param", 6.7)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 4.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Rucklidge state."""
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
        """Rucklidge equations.

        dx/dt = -kappa*x + lambda*y - y*z
        dy/dt = x
        dz/dt = -z + y^2
        """
        x, y, z = state
        dx = -self.kappa * x + self.lambda_param * y - y * z
        dy = x
        dz = -z + y**2
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Rucklidge system.

        Setting derivatives to zero:
            -kappa*x + lambda*y - y*z = 0
            x = 0
            -z + y^2 = 0

        From dy/dt=0: x=0. From dz/dt=0: z=y^2.
        Substituting into dx/dt=0: lambda*y - y*y^2 = y*(lambda - y^2) = 0.

        Case 1: y=0 => origin (0, 0, 0)
        Case 2: y^2 = lambda - kappa... wait, let's redo:
            -kappa*0 + lambda*y - y*z = 0 => y*(lambda - z) = 0
            If y != 0: z = lambda. But z = y^2, so y^2 = lambda.
            Actually from the first eq with x=0: lambda*y - y*z = 0 => y*(lambda - z) = 0
            If y != 0: z = lambda, and z = y^2, so y = +/-sqrt(lambda).
            But we need x=0 to hold, and from dy/dt=0 that gives x=0.
            Check: dx/dt = -kappa*0 + lambda*y - y*lambda = 0. Correct.

        Actually the standard Rucklidge fixed points depend on lambda vs kappa.
        With x=0, z=y^2, and y*(lambda - z)=0:
        - y=0: origin
        - y!=0: z=lambda, y=+-sqrt(lambda) (requires lambda > 0)

        But there's another branch when x != 0... let's check:
        From dy/dt=0: x=0 always. So x must be 0 at fixed points.

        Non-origin fixed points exist when lambda > 0 (always true for physical params).
        """
        points = [np.array([0.0, 0.0, 0.0])]
        if self.lambda_param > 0:
            y_eq = np.sqrt(self.lambda_param)
            z_eq = self.lambda_param
            points.append(np.array([0.0, y_eq, z_eq]))
            points.append(np.array([0.0, -y_eq, z_eq]))
        return points

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[-kappa,  lambda - z,  -y],
             [  1,       0,          0],
             [  0,      2*y,        -1]]
        """
        x, y, z = state
        return np.array([
            [-self.kappa, self.lambda_param - z, -y],
            [1.0, 0.0, 0.0],
            [0.0, 2.0 * y, -1.0],
        ])

    def compute_divergence(self) -> float:
        """Compute the divergence of the Rucklidge vector field.

        div(F) = d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz
               = -kappa + 0 + (-1) = -(kappa + 1)

        This is constant (independent of state), confirming the system is
        dissipative for kappa > -1 (always true for physical parameters).
        """
        return -(self.kappa + 1.0)

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
