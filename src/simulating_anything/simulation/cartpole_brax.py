"""Brax-style robotics cart-pole simulation.

A pure-numpy implementation of the cart-pole balancing problem
matching the Brax/MuJoCo physics convention. Serves as the Brax
equivalent domain -- same physics, no external dependency.

This differs from cart_pole.py (Lagrangian formulation) by using
the control-oriented formulation with actions (applied forces).

Discovery targets:
- Linearized dynamics near upright: omega = sqrt(g/L) for small angles
- Energy-based swing-up strategy
- LQR-optimal control gains
- Region of attraction for stabilization
- Action-value function approximation
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CartPoleBraxSimulation(SimulationEnvironment):
    """Cart-pole with applied force control (Brax/MuJoCo-style physics).

    State: [x, x_dot, theta, theta_dot] where theta=0 is upright.
    Action: horizontal force F applied to cart.

    Uses standard Newton-Euler equations:
        theta_ddot = (g*sin(theta) + cos(theta)*(-F - m_p*L*theta_dot^2*sin(theta))/(m_c+m_p))
                     / (L * (4/3 - m_p*cos(theta)^2/(m_c+m_p)))
        x_ddot = (F + m_p*L*(theta_dot^2*sin(theta) - theta_ddot*cos(theta))) / (m_c + m_p)

    Parameters:
        m_c: cart mass (default: 1.0)
        m_p: pole mass (default: 0.1)
        L: half-pole length (default: 0.5)
        g: gravity (default: 9.81)
        force_mag: maximum applied force (default: 10.0)
        mu_c: cart friction (default: 0.0)
        mu_p: pole friction (default: 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.m_c = p.get("m_c", 1.0)
        self.m_p = p.get("m_p", 0.1)
        self.L = p.get("L", 0.5)
        self.g = p.get("g", 9.81)
        self.force_mag = p.get("force_mag", 10.0)
        self.mu_c = p.get("mu_c", 0.0)
        self.mu_p = p.get("mu_p", 0.0)
        self.dt = config.dt

        # Action (normalized: -1 to 1)
        self._action = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._step_count = 0
        # Start near upright (theta=0) with small perturbation
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState(42)
        self._state = np.array([
            rng.uniform(-0.05, 0.05),   # x
            rng.uniform(-0.05, 0.05),   # x_dot
            rng.uniform(-0.05, 0.05),   # theta (rad from upright)
            rng.uniform(-0.05, 0.05),   # theta_dot
        ])
        self._action = 0.0
        return self._state.copy()

    def set_action(self, action: float) -> None:
        """Set the control action (force normalized to [-1, 1])."""
        self._action = float(np.clip(action, -1.0, 1.0))

    def step(self) -> np.ndarray:
        """RK4 integration step."""
        dt = self.dt
        s = self._state

        k1 = self._derivatives(s)
        k2 = self._derivatives(s + 0.5 * dt * k1)
        k3 = self._derivatives(s + 0.5 * dt * k2)
        k4 = self._derivatives(s + dt * k3)

        self._state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._step_count += 1
        return self._state.copy()

    def observe(self) -> np.ndarray:
        return self._state.copy()

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        x, x_dot, theta, theta_dot = state
        F = self._action * self.force_mag

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        total_m = self.m_c + self.m_p

        # Friction terms
        friction_cart = self.mu_c * np.sign(x_dot)
        friction_pole = self.mu_p * theta_dot

        # Pole angular acceleration
        temp = (F + self.m_p * self.L * theta_dot ** 2 * sin_t - friction_cart) / total_m
        theta_ddot = (self.g * sin_t - cos_t * temp - friction_pole / (self.m_p * self.L))
        theta_ddot /= self.L * (4.0 / 3.0 - self.m_p * cos_t ** 2 / total_m)

        # Cart acceleration
        x_ddot = temp - self.m_p * self.L * theta_ddot * cos_t / total_m

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def total_energy(self) -> float:
        """Total mechanical energy."""
        x, x_dot, theta, theta_dot = self._state
        # Kinetic energy
        ke_cart = 0.5 * self.m_c * x_dot ** 2
        # Pole tip velocity
        vx_tip = x_dot + self.L * theta_dot * np.cos(theta)
        vy_tip = self.L * theta_dot * np.sin(theta)
        ke_pole = 0.5 * self.m_p * (vx_tip ** 2 + vy_tip ** 2)
        # Potential energy (reference: pivot point)
        pe = self.m_p * self.g * self.L * np.cos(theta)
        return ke_cart + ke_pole + pe

    def is_balanced(self, angle_threshold: float = 0.2) -> bool:
        """Check if pole is near upright."""
        return abs(self._state[2]) < angle_threshold

    def linearized_frequency(self) -> float:
        """Natural frequency of small oscillations near upright: omega = sqrt(g/L)."""
        return np.sqrt(self.g / self.L)
