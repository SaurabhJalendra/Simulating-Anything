"""Swinging Atwood machine simulation -- Lagrangian ODE system.

Two masses connected by an inextensible string over a pulley.  Mass M hangs
vertically; mass m swings as a pendulum.  The motion depends on the mass
ratio mu = M/m.

State vector: [r, theta, r_dot, theta_dot]
where r = string length on the swinging side, theta = angle from vertical.

Equations of motion (Lagrangian mechanics):
    r'' = (m*r*theta'^2 + m*g*cos(theta) - M*g) / (M + m)
    theta'' = (-2*r'*theta' - g*sin(theta)) / r

Target rediscoveries:
- Energy conservation: E = const (within numerical tolerance)
- Lyapunov exponent as a function of mass ratio mu = M/m
- Integrable case at mu = 1, chaotic for mu != 1
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SwingingAtwoodSimulation(SimulationEnvironment):
    """Swinging Atwood machine with two masses over a pulley.

    State vector: [r, theta, r_dot, theta_dot]
    where r = string length on the swinging side (m > 0),
    theta = angle of swinging mass from vertical downward.

    Parameters:
        M: hanging mass (kg)
        m: swinging mass (kg)
        g: gravitational acceleration (m/s^2)
        r_min: minimum allowed r to prevent pulley collision
        r_0: initial string length on swinging side
        theta_0: initial angle from vertical
        r_dot_0: initial radial velocity
        theta_dot_0: initial angular velocity
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.M = p.get("M", 3.0)
        self.m = p.get("m", 1.0)
        self.g = p.get("g", 9.81)
        self.r_min = p.get("r_min", 0.1)
        self.r_0 = p.get("r_0", 1.0)
        self.theta_0 = p.get("theta_0", 0.5)
        self.r_dot_0 = p.get("r_dot_0", 0.0)
        self.theta_dot_0 = p.get("theta_dot_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state to [r_0, theta_0, r_dot_0, theta_dot_0]."""
        self._state = np.array(
            [self.r_0, self.theta_0, self.r_dot_0, self.theta_dot_0],
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
        """Return current state [r, theta, r_dot, theta_dot]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step with r clamping."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce r > r_min constraint (elastic wall at pulley)
        if self._state[0] < self.r_min:
            self._state[0] = self.r_min
            # Reverse radial velocity (elastic bounce)
            if self._state[2] < 0:
                self._state[2] = -self._state[2]

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Swinging Atwood machine equations of motion.

        Derived from the Lagrangian:
            L = 0.5*(M+m)*r_dot^2 + 0.5*m*r^2*theta_dot^2
                - M*g*r + m*g*r*cos(theta)
            (choosing potential zero at the pulley, with the M side
             having length that increases as r decreases)

        Euler-Lagrange equations:
            (M+m)*r'' = m*r*theta'^2 + m*g*cos(theta) - M*g
            m*r^2*theta'' = -2*m*r*r'*theta' - m*g*r*sin(theta)

        Simplified:
            r'' = (m*r*theta'^2 + m*g*cos(theta) - M*g) / (M+m)
            theta'' = (-2*r'*theta' - g*sin(theta)) / r
        """
        r, theta, r_dot, theta_dot = y
        M, m, g = self.M, self.m, self.g

        # Protect against r = 0 division
        r_safe = max(r, self.r_min)

        r_ddot = (
            m * r_safe * theta_dot**2
            + m * g * np.cos(theta)
            - M * g
        ) / (M + m)

        theta_ddot = (
            -2.0 * r_dot * theta_dot - g * np.sin(theta)
        ) / r_safe

        return np.array([r_dot, theta_dot, r_ddot, theta_ddot])

    def total_energy(self, state: np.ndarray | None = None) -> float:
        """Compute total mechanical energy.

        E = 0.5*(M+m)*r_dot^2 + 0.5*m*r^2*theta_dot^2
            + M*g*r - m*g*r*cos(theta)

        The potential energy convention: increasing r lifts M (costs energy)
        and lowers m*cos(theta) (gains energy when cos(theta) > 0).
        """
        if state is None:
            state = self._state
        r, theta, r_dot, theta_dot = state
        M, m, g = self.M, self.m, self.g

        T = 0.5 * (M + m) * r_dot**2 + 0.5 * m * r**2 * theta_dot**2
        V = M * g * r - m * g * r * np.cos(theta)

        return float(T + V)

    def angular_momentum(self, state: np.ndarray | None = None) -> float:
        """Compute angular momentum of the swinging mass: L = m*r^2*theta_dot.

        This is NOT conserved in general because gravity provides a torque.
        """
        if state is None:
            state = self._state
        r, theta, r_dot, theta_dot = state
        return float(self.m * r**2 * theta_dot)

    def mass_ratio(self) -> float:
        """Return mass ratio mu = M/m."""
        return self.M / self.m

    def cartesian_position(
        self, state: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Compute (x, y) position of the swinging mass.

        Origin at the pulley.  x = r*sin(theta), y = -r*cos(theta).
        Positive y is downward.
        """
        if state is None:
            state = self._state
        r, theta = state[0], state[1]
        x = r * np.sin(theta)
        y = -r * np.cos(theta)
        return float(x), float(y)

    def estimate_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None,
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the Wolf et al. (1985) method: track two nearby trajectories,
        renormalize when they diverge.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1.copy()
        state2[1] += eps  # Perturb theta

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            state1 = self._rk4_advance(state1, dt)
            state2 = self._rk4_advance(state2, dt)

            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def _rk4_advance(self, y: np.ndarray, dt: float) -> np.ndarray:
        """Advance a state vector by one RK4 step (standalone, no side effects)."""
        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)
        result = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce r > r_min
        if result[0] < self.r_min:
            result[0] = self.r_min
            if result[2] < 0:
                result[2] = -result[2]
        return result

    def chaos_sweep(
        self,
        mu_values: np.ndarray | list[float],
        n_steps: int = 50000,
        dt: float = 0.001,
    ) -> dict[str, np.ndarray]:
        """Sweep mass ratio mu = M/m and measure Lyapunov exponent at each.

        Returns dict with 'mu', 'lyapunov_exponent' arrays.
        """
        mu_values = np.asarray(mu_values)
        lyapunov_exps = []

        for mu in mu_values:
            M_val = mu * self.m
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    "M": M_val,
                    "m": self.m,
                    "g": self.g,
                    "r_min": self.r_min,
                    "r_0": self.r_0,
                    "theta_0": self.theta_0,
                    "r_dot_0": self.r_dot_0,
                    "theta_dot_0": self.theta_dot_0,
                },
            )
            sim = SwingingAtwoodSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(2000):
                sim.step()

            lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
            lyapunov_exps.append(lam)

        return {
            "mu": mu_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
        }
