"""Damped driven pendulum simulation -- period-doubling route to chaos.

Target rediscoveries:
- ODE: theta'' + gamma*theta' + omega0^2*sin(theta) = A*cos(omega_d*t)
- Period-doubling bifurcation as driving amplitude A increases
- Chaos onset for A ~ 1.5 (with gamma=0.5, omega0=1.5, omega_d=2/3)
- Resonance curve: amplitude vs driving frequency
- Lyapunov exponent as function of A
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DrivenPendulum(SimulationEnvironment):
    """Damped driven pendulum: theta'' + gamma*theta' + omega0^2*sin(theta) = A*cos(omega_d*t).

    State vector: [theta, omega] where theta = angle, omega = angular velocity.
    Observation: [theta, omega, t] (time included for Poincare section sampling).

    The dynamics exhibit:
    - Simple periodic motion for small driving amplitude A
    - Period-doubling cascade as A increases (~1.2)
    - Chaotic motion for large A (~1.5)
    - Poincare section: sample state at drive period intervals T_d = 2*pi/omega_d

    Parameters:
        gamma: damping coefficient (default 0.5)
        omega0: natural frequency (default 1.5)
        A_drive: driving amplitude (default 1.2)
        omega_d: driving frequency (default 2/3)
        theta_0: initial angle (default 0.1)
        omega_init: initial angular velocity (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.gamma = p.get("gamma", 0.5)
        self.omega0 = p.get("omega0", 1.5)
        self.A_drive = p.get("A_drive", 1.2)
        self.omega_d = p.get("omega_d", 2.0 / 3.0)
        self.theta_0 = p.get("theta_0", 0.1)
        self.omega_init = p.get("omega_init", 0.0)
        self._t = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize angle and angular velocity."""
        self._state = np.array(
            [self.theta_0, self.omega_init], dtype=np.float64
        )
        self._step_count = 0
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with time-dependent forcing."""
        self._rk4_step()
        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [theta, omega, t]."""
        return np.array(
            [self._state[0], self._state[1], self._t], dtype=np.float64
        )

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta with explicit time dependence."""
        dt = self.config.dt
        y = self._state
        t = self._t

        k1 = self._derivatives(y, t)
        k2 = self._derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._derivatives(y + dt * k3, t + dt)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._t += dt

    def _derivatives(self, y: np.ndarray, t: float) -> np.ndarray:
        """Compute dy/dt for the driven pendulum.

        theta'' = -gamma*theta' - omega0^2*sin(theta) + A*cos(omega_d*t)

        Args:
            y: State vector [theta, omega].
            t: Current time (needed for driving term).

        Returns:
            Derivatives [dtheta/dt, domega/dt].
        """
        theta, omega = y
        dtheta_dt = omega
        domega_dt = (
            -self.gamma * omega
            - self.omega0**2 * np.sin(theta)
            + self.A_drive * np.cos(self.omega_d * t)
        )
        return np.array([dtheta_dt, domega_dt])

    @property
    def energy(self) -> float:
        """Pendulum energy without drive: E = 0.5*omega^2 - omega0^2*cos(theta).

        This is the mechanical energy of the unforced, undamped pendulum.
        Not conserved with damping and driving, but bounded.
        """
        theta, omega = self._state
        return 0.5 * omega**2 - self.omega0**2 * np.cos(theta)

    @property
    def drive_period(self) -> float:
        """Period of the driving force: T_d = 2*pi/omega_d."""
        if self.omega_d == 0:
            return float("inf")
        return 2 * np.pi / self.omega_d

    @property
    def is_chaotic(self) -> bool:
        """Estimate whether the current parameters produce chaos.

        Uses a quick Lyapunov exponent computation. Positive => chaotic.
        """
        lam = self.compute_lyapunov(n_steps=10000)
        return lam > 0.01

    def compute_lyapunov(self, n_steps: int = 20000) -> float:
        """Compute the maximum Lyapunov exponent via tangent dynamics.

        Uses the Wolf et al. (1985) method: track two nearby trajectories
        and renormalize when they diverge.

        Args:
            n_steps: Number of integration steps for the estimate.

        Returns:
            Estimated maximum Lyapunov exponent (positive => chaos).
        """
        dt = self.config.dt
        eps = 1e-8

        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0.0])
        t = self._t

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance state1
            k1 = self._derivatives(state1, t)
            k2 = self._derivatives(state1 + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self._derivatives(state1 + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self._derivatives(state1 + dt * k3, t + dt)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance state2
            k1 = self._derivatives(state2, t)
            k2 = self._derivatives(state2 + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self._derivatives(state2 + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self._derivatives(state2 + dt * k3, t + dt)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            t += dt

            # Compute distance and renormalize
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def poincare_section(self, n_periods: int = 200) -> np.ndarray:
        """Sample state at integer multiples of the drive period.

        Useful for visualizing period-doubling: a period-1 orbit gives 1 point,
        period-2 gives 2 points, chaos gives a fractal set.

        Args:
            n_periods: Number of drive periods to sample.

        Returns:
            Array of shape (n_periods, 2) with [theta, omega] at each sample.
        """
        dt = self.config.dt
        T_d = self.drive_period
        steps_per_period = int(round(T_d / dt))

        points = []
        for _ in range(n_periods):
            for _ in range(steps_per_period):
                self._rk4_step()
                self._step_count += 1
            # Normalize theta to [-pi, pi] for cleaner visualization
            theta = self._state[0]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            points.append([theta, self._state[1]])

        return np.array(points)

    def measure_steady_amplitude(self, n_periods: int = 20) -> float:
        """Measure the steady-state oscillation amplitude after transient.

        Args:
            n_periods: Number of drive periods to measure over.

        Returns:
            Maximum absolute angle observed.
        """
        dt = self.config.dt
        T_d = self.drive_period
        # Skip transient: 100 drive periods
        transient_steps = int(100 * T_d / dt)
        for _ in range(transient_steps):
            self._rk4_step()
            self._step_count += 1

        # Measure amplitude
        theta_max = 0.0
        measure_steps = int(n_periods * T_d / dt)
        for _ in range(measure_steps):
            self._rk4_step()
            self._step_count += 1
            theta_max = max(theta_max, abs(self._state[0]))

        return float(theta_max)
