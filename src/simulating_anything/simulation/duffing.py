"""Duffing oscillator simulation.

Target rediscoveries:
- ODE: x'' + delta*x' + alpha*x + beta*x^3 = gamma_f*cos(omega*t)
- Chaos onset as gamma_f increases
- ODE coefficient recovery via SINDy
- Amplitude-frequency response curve
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DuffingOscillator(SimulationEnvironment):
    """Duffing oscillator: x'' + delta*x' + alpha*x + beta*x^3 = gamma_f*cos(omega*t).

    A nonlinear oscillator that exhibits chaos, bifurcations, and strange
    attractors depending on forcing amplitude and frequency.

    State vector: [x, v] where x = displacement, v = velocity (dx/dt).

    The potential energy is V(x) = (alpha/2)*x^2 + (beta/4)*x^4, giving a
    double-well (alpha < 0, beta > 0) or single hardening-spring (alpha > 0,
    beta > 0) potential.

    Parameters:
        alpha: linear stiffness coefficient (default 1.0)
        beta: cubic nonlinearity coefficient (default 1.0)
        delta: damping coefficient (default 0.2)
        gamma_f: forcing amplitude (default 0.3)
        omega: forcing angular frequency (default 1.0)
        x_0: initial displacement (default 0.5)
        v_0: initial velocity (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 1.0)
        self.beta = p.get("beta", 1.0)
        self.delta = p.get("delta", 0.2)
        self.gamma_f = p.get("gamma_f", 0.3)
        self.omega = p.get("omega", 1.0)
        self.x_0 = p.get("x_0", 0.5)
        self.v_0 = p.get("v_0", 0.0)
        self._t = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize position and velocity."""
        self._state = np.array([self.x_0, self.v_0], dtype=np.float64)
        self._step_count = 0
        self._t = 0.0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with time-dependent forcing."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, v]."""
        return self._state

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
        """Compute dy/dt for the Duffing equation.

        Args:
            y: State vector [x, v].
            t: Current time (needed for forcing term).

        Returns:
            Derivatives [dx/dt, dv/dt].
        """
        x, v = y
        dx_dt = v
        dv_dt = (
            self.gamma_f * np.cos(self.omega * t)
            - self.delta * v
            - self.alpha * x
            - self.beta * x**3
        )
        return np.array([dx_dt, dv_dt])

    @property
    def total_energy(self) -> float:
        """Instantaneous mechanical energy: E = 0.5*v^2 + 0.5*alpha*x^2 + 0.25*beta*x^4.

        This is the energy of the conservative part of the system (ignoring
        damping and forcing). For delta=0 and gamma_f=0, this is conserved.
        """
        x, v = self._state
        return 0.5 * v**2 + 0.5 * self.alpha * x**2 + 0.25 * self.beta * x**4

    @property
    def natural_frequency(self) -> float:
        """Small-amplitude natural frequency: omega_0 = sqrt(alpha).

        Valid only for alpha > 0 (single-well potential).
        """
        if self.alpha <= 0:
            return 0.0
        return np.sqrt(self.alpha)

    @property
    def approximate_period(self) -> float:
        """Approximate small-amplitude period: T = 2*pi/sqrt(alpha).

        Valid for alpha > 0 and small oscillations where beta*x^2 << alpha.
        """
        w0 = self.natural_frequency
        if w0 == 0:
            return float("inf")
        return 2 * np.pi / w0

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period via zero crossings after transient.

        Args:
            n_periods: Number of periods to average over.

        Returns:
            Mean period, or inf if no oscillation detected.
        """
        dt = self.config.dt
        T_approx = self.approximate_period
        if not np.isfinite(T_approx):
            T_approx = 2 * np.pi  # fallback estimate

        # Let transient die out
        transient_steps = max(int(50 / dt), int(20 * T_approx / dt))
        for _ in range(transient_steps):
            self.step()

        # Detect upward zero crossings of x
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
        """Measure the peak displacement amplitude after transient.

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
