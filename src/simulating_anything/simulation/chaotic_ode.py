"""Double pendulum simulation -- chaotic ODE system.

Target rediscoveries:
- Energy conservation: E = const (within numerical tolerance)
- Lyapunov exponent estimation from trajectory divergence
- Small-angle linearization: omega^2 = g/L for each pendulum
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DoublePendulumSimulation(SimulationEnvironment):
    """Double pendulum with point masses.

    State vector: [theta1, theta2, omega1, omega2]
    where theta = angle from vertical, omega = angular velocity.

    Parameters: m1, m2 (masses), L1, L2 (lengths), g (gravity).

    The equations of motion are derived from the Lagrangian and are
    highly nonlinear, leading to chaotic dynamics for large amplitudes.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.m1 = p.get("m1", 1.0)
        self.m2 = p.get("m2", 1.0)
        self.L1 = p.get("L1", 1.0)
        self.L2 = p.get("L2", 1.0)
        self.g = p.get("g", 9.81)
        self.theta1_0 = p.get("theta1_0", np.pi / 2)
        self.theta2_0 = p.get("theta2_0", np.pi / 2)
        self.omega1_0 = p.get("omega1_0", 0.0)
        self.omega2_0 = p.get("omega2_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize pendulum angles and angular velocities."""
        self._state = np.array(
            [self.theta1_0, self.theta2_0, self.omega1_0, self.omega2_0],
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
        """Return current state [theta1, theta2, omega1, omega2]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Double pendulum equations of motion.

        Derived from Lagrangian mechanics. Uses the standard formulation
        with mass matrix inversion.
        """
        th1, th2, w1, w2 = y
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g

        delta = th1 - th2
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        den = 2 * m1 + m2 - m2 * np.cos(2 * delta)

        # Angular acceleration of pendulum 1
        alpha1_num = (
            -g * (2 * m1 + m2) * np.sin(th1)
            - m2 * g * np.sin(th1 - 2 * th2)
            - 2 * sin_d * m2 * (w2**2 * L2 + w1**2 * L1 * cos_d)
        )
        alpha1 = alpha1_num / (L1 * den)

        # Angular acceleration of pendulum 2
        alpha2_num = (
            2 * sin_d * (
                w1**2 * L1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(th1)
                + w2**2 * L2 * m2 * cos_d
            )
        )
        alpha2 = alpha2_num / (L2 * den)

        return np.array([w1, w2, alpha1, alpha2])

    def total_energy(self, state: np.ndarray | None = None) -> float:
        """Compute total mechanical energy (kinetic + potential).

        E = T + V where:
        T = 0.5*m1*v1^2 + 0.5*m2*v2^2
        V = -m1*g*L1*cos(th1) - m2*g*(L1*cos(th1) + L2*cos(th2))
        """
        if state is None:
            state = self._state
        th1, th2, w1, w2 = state
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g

        # Kinetic energy
        v1_sq = (L1 * w1) ** 2
        v2_sq = (L1 * w1) ** 2 + (L2 * w2) ** 2 + 2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)
        T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq

        # Potential energy (zero at pivot)
        V = -m1 * g * L1 * np.cos(th1) - m2 * g * (L1 * np.cos(th1) + L2 * np.cos(th2))

        return float(T + V)

    def cartesian_positions(self, state: np.ndarray | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute (x, y) positions of both pendulum bobs."""
        if state is None:
            state = self._state
        th1, th2 = state[0], state[1]
        x1 = self.L1 * np.sin(th1)
        y1 = -self.L1 * np.cos(th1)
        x2 = x1 + self.L2 * np.sin(th2)
        y2 = y1 - self.L2 * np.cos(th2)
        return (float(x1), float(y1)), (float(x2), float(y2))
