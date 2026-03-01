"""Elastic pendulum (spring-pendulum) simulation.

A mass m on a spring (natural length L0, spring constant k) that can also
swing like a pendulum under gravity g.  This is a classic nonlinear coupled
oscillator with two degrees of freedom.

State vector: [r, r_dot, theta, theta_dot]
where r = spring length, theta = angle from vertical.

Equations of motion (Lagrangian):
    r_ddot = r * theta_dot^2 + g * cos(theta) - (k/m) * (r - L0)
    theta_ddot = -(2 * r_dot * theta_dot + g * sin(theta)) / r

Target rediscoveries:
- Energy conservation: E = const (within numerical tolerance)
- Radial frequency: omega_r = sqrt(k/m)
- Angular frequency (small angle): omega_theta = sqrt(g/L0)
- 1:2 autoparametric resonance when k/m = 4*g/L0
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ElasticPendulum(SimulationEnvironment):
    """Elastic pendulum with a mass on a spring that can also swing.

    State vector: [r, r_dot, theta, theta_dot]
    where r = spring length, theta = angle from vertical (downward).

    Parameters:
        m: mass (kg)
        k: spring constant (N/m)
        L0: natural (unstretched) spring length (m)
        g: gravitational acceleration (m/s^2)
        r_0: initial spring length
        r_dot_0: initial radial velocity
        theta_0: initial angle from vertical
        theta_dot_0: initial angular velocity
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.m = p.get("m", 1.0)
        self.k = p.get("k", 10.0)
        self.L0 = p.get("L0", 1.0)
        self.g = p.get("g", 9.81)
        self.r_0 = p.get("r_0", 1.0 + 9.81 / 10.0)  # Equilibrium: L0 + mg/k
        self.r_dot_0 = p.get("r_dot_0", 0.0)
        self.theta_0 = p.get("theta_0", 0.0)
        self.theta_dot_0 = p.get("theta_dot_0", 0.0)

    @property
    def radial_frequency(self) -> float:
        """Small-oscillation radial (spring) frequency: omega_r = sqrt(k/m)."""
        return np.sqrt(self.k / self.m)

    @property
    def angular_frequency(self) -> float:
        """Small-oscillation angular (pendulum) frequency: omega_theta = sqrt(g/L0)."""
        return np.sqrt(self.g / self.L0)

    @property
    def equilibrium_length(self) -> float:
        """Equilibrium spring length under gravity: r_eq = L0 + mg/k."""
        return self.L0 + self.m * self.g / self.k

    @property
    def total_energy(self) -> float:
        """Compute total mechanical energy from current state."""
        return self._compute_energy(self._state)

    @property
    def kinetic_energy(self) -> float:
        """Compute kinetic energy from current state.

        T = 0.5 * m * (r_dot^2 + r^2 * theta_dot^2)
        """
        r, r_dot, theta, theta_dot = self._state
        return float(0.5 * self.m * (r_dot**2 + r**2 * theta_dot**2))

    @property
    def potential_energy(self) -> float:
        """Compute potential energy from current state.

        V = 0.5 * k * (r - L0)^2 - m * g * r * cos(theta)
        """
        r, r_dot, theta, theta_dot = self._state
        V_spring = 0.5 * self.k * (r - self.L0) ** 2
        V_gravity = -self.m * self.g * r * np.cos(theta)
        return float(V_spring + V_gravity)

    def _compute_energy(self, state: np.ndarray) -> float:
        """Compute total energy from a state vector.

        E = 0.5*m*(r_dot^2 + r^2*theta_dot^2)
            + 0.5*k*(r - L0)^2
            - m*g*r*cos(theta)
        """
        r, r_dot, theta, theta_dot = state
        T = 0.5 * self.m * (r_dot**2 + r**2 * theta_dot**2)
        V_spring = 0.5 * self.k * (r - self.L0) ** 2
        V_gravity = -self.m * self.g * r * np.cos(theta)
        return float(T + V_spring + V_gravity)

    def pendulum_position(self, state: np.ndarray | None = None) -> tuple[float, float]:
        """Compute (x, y) Cartesian position of the mass.

        Origin is at the pivot point.  Positive x is rightward, positive y is
        downward (so that theta=0 means hanging straight down).

        x = r * sin(theta)
        y = r * cos(theta)   (positive downward)
        """
        if state is None:
            state = self._state
        r, _, theta, _ = state
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        return float(x), float(y)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize spring length, radial velocity, angle, angular velocity."""
        self._state = np.array(
            [self.r_0, self.r_dot_0, self.theta_0, self.theta_dot_0],
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
        """Return current state [r, r_dot, theta, theta_dot]."""
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
        """Elastic pendulum equations of motion.

        Derived from the Lagrangian:
            L = 0.5*m*(r_dot^2 + r^2*theta_dot^2) - 0.5*k*(r-L0)^2 + m*g*r*cos(theta)

        Euler-Lagrange equations give:
            r_ddot = r * theta_dot^2 + g * cos(theta) - (k/m) * (r - L0)
            theta_ddot = -(2 * r_dot * theta_dot + g * sin(theta)) / r
        """
        r, r_dot, theta, theta_dot = y
        m, k, L0, g = self.m, self.k, self.L0, self.g

        r_ddot = r * theta_dot**2 + g * np.cos(theta) - (k / m) * (r - L0)
        theta_ddot = -(2.0 * r_dot * theta_dot + g * np.sin(theta)) / r

        return np.array([r_dot, r_ddot, theta_dot, theta_ddot])
