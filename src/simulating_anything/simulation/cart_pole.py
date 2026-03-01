"""Cart-pole (inverted pendulum on cart) simulation.

The classic control theory benchmark: a pendulum mounted on a cart that
can slide horizontally along a frictionless track.

Target rediscoveries:
- Small-angle frequency: omega = sqrt(g*(M+m) / (M*L))
- Energy conservation: E = const when friction = 0 and F = 0
- Linearized equations of motion recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CartPole(SimulationEnvironment):
    """Cart-pole system with a pendulum on a sliding cart.

    State vector: [x, x_dot, theta, theta_dot]
    where x = cart position, theta = pendulum angle from upward vertical.

    Equations of motion (Lagrangian mechanics):
        (M + m) * x_ddot + m * L * theta_ddot * cos(theta)
            - m * L * theta_dot^2 * sin(theta) = F - mu_c * x_dot
        m * L * x_ddot * cos(theta) + m * L^2 * theta_ddot
            - m * g * L * sin(theta) = -mu_p * theta_dot

    Parameters:
        M: cart mass (kg)
        m: pendulum bob mass (kg)
        L: pendulum length (m)
        g: gravitational acceleration (m/s^2)
        mu_c: cart friction coefficient
        mu_p: pendulum friction coefficient
        F: external force on cart (N)
        x_0: initial cart position
        x_dot_0: initial cart velocity
        theta_0: initial pendulum angle from vertical
        theta_dot_0: initial angular velocity
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.M = p.get("M", 1.0)
        self.m = p.get("m", 0.1)
        self.L = p.get("L", 0.5)
        self.g = p.get("g", 9.81)
        self.mu_c = p.get("mu_c", 0.0)
        self.mu_p = p.get("mu_p", 0.0)
        self.F = p.get("F", 0.0)
        self.x_0 = p.get("x_0", 0.0)
        self.x_dot_0 = p.get("x_dot_0", 0.0)
        self.theta_0 = p.get("theta_0", 0.1)
        self.theta_dot_0 = p.get("theta_dot_0", 0.0)

    @property
    def small_angle_frequency(self) -> float:
        """Linearized small-angle frequency about the upward vertical.

        For theta near 0 (upright position), the linearized system gives
        omega = sqrt(g * (M + m) / (M * L)).
        """
        return np.sqrt(self.g * (self.M + self.m) / (self.M * self.L))

    @property
    def total_energy(self) -> float:
        """Compute total mechanical energy of the cart-pole system.

        Kinetic energy:
            T = 0.5 * M * x_dot^2
              + 0.5 * m * (x_dot^2 + 2*L*x_dot*theta_dot*cos(theta) + L^2*theta_dot^2)

        Potential energy (zero at pivot height, positive upward):
            V = m * g * L * cos(theta)

        Note: theta=0 is upward vertical, so V = m*g*L when upright.
        """
        return self._compute_energy(self._state)

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy of the cart-pole system."""
        x, x_dot, theta, theta_dot = self._state
        M, m, L = self.M, self.m, self.L

        T_cart = 0.5 * M * x_dot**2
        # Pendulum tip: (x + L*sin(theta), L*cos(theta))
        # Velocity: (x_dot + L*theta_dot*cos(theta), -L*theta_dot*sin(theta))
        vx_pend = x_dot + L * theta_dot * np.cos(theta)
        vy_pend = -L * theta_dot * np.sin(theta)
        T_pend = 0.5 * m * (vx_pend**2 + vy_pend**2)
        return float(T_cart + T_pend)

    @property
    def potential_energy(self) -> float:
        """Potential energy with zero at pivot height."""
        theta = self._state[2]
        return float(self.m * self.g * self.L * np.cos(theta))

    def _compute_energy(self, state: np.ndarray) -> float:
        """Compute total energy from a state vector."""
        x, x_dot, theta, theta_dot = state
        M, m, L, g = self.M, self.m, self.L, self.g

        # Kinetic energy
        # Pendulum position: (x + L*sin(theta), L*cos(theta))
        # Pendulum velocity: (x_dot + L*theta_dot*cos(theta), -L*theta_dot*sin(theta))
        vx_pend = x_dot + L * theta_dot * np.cos(theta)
        vy_pend = -L * theta_dot * np.sin(theta)
        T_cart = 0.5 * M * x_dot**2
        T_pend = 0.5 * m * (vx_pend**2 + vy_pend**2)
        T = T_cart + T_pend

        # Potential energy (zero at pivot)
        V = m * g * L * np.cos(theta)

        return float(T + V)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize cart position, velocity, pendulum angle, angular velocity."""
        self._state = np.array(
            [self.x_0, self.x_dot_0, self.theta_0, self.theta_dot_0],
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
        """Return current state [x, x_dot, theta, theta_dot]."""
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
        """Cart-pole equations of motion.

        Derived from the Lagrangian with mass matrix inversion.

        The coupled equations:
            (M + m) * x_ddot + m*L*theta_ddot*cos(theta)
                = m*L*theta_dot^2*sin(theta) + F - mu_c*x_dot
            m*L*x_ddot*cos(theta) + m*L^2*theta_ddot
                = m*g*L*sin(theta) - mu_p*theta_dot

        Solve by inverting the 2x2 mass matrix.
        """
        x, x_dot, theta, theta_dot = y
        M, m, L, g = self.M, self.m, self.L, self.g
        mu_c, mu_p, F = self.mu_c, self.mu_p, self.F

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        # Mass matrix: [[M+m, m*L*cos(theta)], [m*L*cos(theta), m*L^2]]
        # RHS: [m*L*theta_dot^2*sin(theta) + F - mu_c*x_dot,
        #        m*g*L*sin(theta) - mu_p*theta_dot]
        a11 = M + m
        a12 = m * L * cos_th
        a21 = m * L * cos_th
        a22 = m * L**2

        b1 = m * L * theta_dot**2 * sin_th + F - mu_c * x_dot
        b2 = m * g * L * sin_th - mu_p * theta_dot

        # Determinant of mass matrix
        det = a11 * a22 - a12 * a21

        # Solve: [x_ddot, theta_ddot] = M^{-1} * [b1, b2]
        x_ddot = (a22 * b1 - a12 * b2) / det
        theta_ddot = (a11 * b2 - a21 * b1) / det

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def pendulum_position(self, state: np.ndarray | None = None) -> tuple[float, float]:
        """Compute (x_pend, y_pend) position of the pendulum bob.

        The pendulum tip is at:
            x_pend = x + L * sin(theta)
            y_pend = L * cos(theta)
        where y is measured upward from the cart.
        """
        if state is None:
            state = self._state
        x, _, theta, _ = state
        x_pend = x + self.L * np.sin(theta)
        y_pend = self.L * np.cos(theta)
        return float(x_pend), float(y_pend)
