"""Lienard oscillator simulation.

The Lienard equation is a generalization of Van der Pol:
    x'' + f(x)*x' + g(x) = 0

We use f(x) = mu*(x^2 - 1) (VdP-like damping) and g(x) = x + alpha*x^3
(Duffing-like restoring force), giving a VdP-Duffing-Lienard hybrid:

    dx/dt = v
    dv/dt = -mu*(x^2 - 1)*v - x - alpha*x^3

Target rediscoveries:
- Limit cycle existence for mu > 0 (Lienard theorem)
- Period and amplitude dependence on mu and alpha
- Energy: E = v^2/2 + x^2/2 + alpha*x^4/4
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LienardSimulation(SimulationEnvironment):
    """Lienard oscillator: x'' + mu*(x^2 - 1)*x' + x + alpha*x^3 = 0.

    State vector: [x, v] where x = displacement, v = velocity (dx/dt).

    The nonlinear damping f(x) = mu*(x^2 - 1) provides:
    - Negative damping for |x| < 1 (energy injection)
    - Positive damping for |x| > 1 (energy dissipation)
    - Stable limit cycle for mu > 0 (by Lienard theorem)

    The restoring force g(x) = x + alpha*x^3 adds Duffing-type nonlinearity:
    - alpha > 0: hardening spring (increased restoring force at large x)
    - alpha = 0: reduces to Van der Pol oscillator

    Parameters:
        mu: nonlinear damping parameter (controls damping strength)
        alpha: cubic stiffness coefficient (Duffing restoring force)
        x_0: initial displacement
        v_0: initial velocity
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.mu = p.get("mu", 1.0)
        self.alpha = p.get("alpha", 0.1)
        self.x_0 = p.get("x_0", 2.0)
        self.v_0 = p.get("v_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize position and velocity."""
        self._state = np.array([self.x_0, self.v_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, v]."""
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

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dy/dt for the Lienard equation.

        Args:
            y: State vector [x, v].

        Returns:
            Derivatives [dx/dt, dv/dt].
        """
        x, v = y
        dx_dt = v
        # dv/dt = -f(x)*v - g(x) = -mu*(x^2 - 1)*v - x - alpha*x^3
        dv_dt = -self.mu * (x**2 - 1) * v - x - self.alpha * x**3
        return np.array([dx_dt, dv_dt])

    def compute_energy(self, state: np.ndarray | None = None) -> float:
        """Compute instantaneous energy: E = v^2/2 + x^2/2 + alpha*x^4/4.

        This is the energy of the conservative subsystem (g(x) = x + alpha*x^3
        derives from potential V(x) = x^2/2 + alpha*x^4/4). For mu=0 this
        energy is exactly conserved.

        Args:
            state: State [x, v] to evaluate. Uses current state if None.

        Returns:
            Energy value.
        """
        if state is None:
            state = self._state
        x, v = state
        return 0.5 * v**2 + 0.5 * x**2 + 0.25 * self.alpha * x**4

    def amplitude_from_trajectory(self, states: np.ndarray) -> float:
        """Compute peak displacement amplitude from a trajectory.

        Args:
            states: Array of shape (n_steps, 2) with columns [x, v].

        Returns:
            Maximum absolute displacement observed.
        """
        return float(np.max(np.abs(states[:, 0])))

    def period_from_trajectory(
        self, states: np.ndarray, dt: float
    ) -> float:
        """Measure oscillation period from upward zero crossings.

        Args:
            states: Array of shape (n_steps, 2) with columns [x, v].
            dt: Timestep between consecutive states.

        Returns:
            Mean period, or inf if fewer than 2 crossings detected.
        """
        x = states[:, 0]
        crossings: list[float] = []
        for i in range(1, len(x)):
            if x[i - 1] < 0 and x[i] >= 0:
                # Linear interpolation for crossing time
                frac = -x[i - 1] / (x[i] - x[i - 1])
                t_cross = (i - 1 + frac) * dt
                crossings.append(t_cross)

        if len(crossings) < 2:
            return float("inf")

        periods = np.diff(crossings)
        return float(np.mean(periods))

    @property
    def approximate_period(self) -> float:
        """Approximate period estimate for small alpha.

        For alpha=0 this reduces to VdP:
        - mu << 1: T ~ 2*pi
        - mu >> 1: T ~ (3 - 2*ln(2))*mu

        For alpha > 0, the period is shorter due to the hardening spring.
        """
        if self.mu < 0.1:
            T_vdp = 2 * np.pi
        elif self.mu > 5.0:
            T_vdp = (3 - 2 * np.log(2)) * self.mu
        else:
            T_vdp = 2 * np.pi * (1 + self.mu**2 / 16)

        # Correction factor for hardening spring (heuristic)
        correction = 1.0 / (1.0 + 0.3 * self.alpha)
        return T_vdp * correction

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the period by detecting zero crossings after transient.

        Args:
            n_periods: Number of periods to average over.

        Returns:
            Mean period, or inf if no oscillation detected.
        """
        dt = self.config.dt
        T_approx = self.approximate_period
        if not np.isfinite(T_approx):
            T_approx = 2 * np.pi

        # Let transient die out
        transient_steps = max(int(50 / dt), int(20 * T_approx / dt))
        for _ in range(transient_steps):
            self.step()

        # Detect upward zero crossings
        crossings: list[float] = []
        prev_x = self._state[0]
        measure_steps = int(n_periods * T_approx / dt * 2)
        for _ in range(measure_steps):
            self.step()
            x = self._state[0]
            if prev_x < 0 and x >= 0:
                t_cross = (
                    (self._step_count - 1) * dt
                    + dt * (-prev_x) / (x - prev_x)
                )
                crossings.append(t_cross)
            prev_x = x

        if len(crossings) < 2:
            return float("inf")

        periods = np.diff(crossings)
        return float(np.mean(periods))

    def measure_amplitude(self, n_periods: int = 3) -> float:
        """Measure the limit cycle amplitude after transient.

        Args:
            n_periods: Number of periods to measure over.

        Returns:
            Maximum absolute displacement observed.
        """
        dt = self.config.dt
        T_approx = self.approximate_period
        if not np.isfinite(T_approx):
            T_approx = 2 * np.pi

        transient_steps = max(int(50 / dt), int(20 * T_approx / dt))
        for _ in range(transient_steps):
            self.step()

        x_max = 0.0
        measure_steps = int(n_periods * T_approx / dt * 2)
        for _ in range(measure_steps):
            self.step()
            x_max = max(x_max, abs(self._state[0]))

        return float(x_max)
