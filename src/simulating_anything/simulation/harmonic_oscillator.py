"""Damped harmonic oscillator simulation.

Target rediscoveries:
- Natural frequency: omega_0 = sqrt(k/m)
- Damping ratio: zeta = c / (2*sqrt(k*m))
- Damped frequency: omega_d = omega_0 * sqrt(1 - zeta^2)
- Amplitude decay: A(t) ~ exp(-zeta*omega_0*t)
- Energy decay: E(t) ~ exp(-2*zeta*omega_0*t)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DampedHarmonicOscillator(SimulationEnvironment):
    """Damped harmonic oscillator: m*x'' + c*x' + k*x = F(t).

    State vector: [x, v] where x = displacement, v = velocity.

    Parameters:
        m: mass (kg)
        k: spring constant (N/m)
        c: damping coefficient (N*s/m)
        x_0: initial displacement
        v_0: initial velocity
        F_amplitude: external forcing amplitude (default 0)
        F_frequency: external forcing frequency (default 0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.m = p.get("m", 1.0)
        self.k = p.get("k", 1.0)
        self.c = p.get("c", 0.0)
        self.x_0 = p.get("x_0", 1.0)
        self.v_0 = p.get("v_0", 0.0)
        self.F_amplitude = p.get("F_amplitude", 0.0)
        self.F_frequency = p.get("F_frequency", 0.0)

    @property
    def omega_0(self) -> float:
        """Natural frequency."""
        return np.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        """Damping ratio."""
        return self.c / (2 * np.sqrt(self.k * self.m))

    @property
    def omega_d(self) -> float:
        """Damped natural frequency."""
        z = self.zeta
        if z >= 1.0:
            return 0.0  # Overdamped
        return self.omega_0 * np.sqrt(1 - z**2)

    @property
    def period(self) -> float:
        """Period of damped oscillation."""
        wd = self.omega_d
        if wd == 0:
            return float("inf")
        return 2 * np.pi / wd

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
        dt = self.config.dt
        t = self._step_count * dt
        y = self._state

        k1 = self._derivatives(y, t)
        k2 = self._derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._derivatives(y + dt * k3, t + dt)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray, t: float) -> np.ndarray:
        x, v = y
        F = self.F_amplitude * np.sin(self.F_frequency * t) if self.F_amplitude else 0.0
        a = (F - self.c * v - self.k * x) / self.m
        return np.array([v, a])

    def total_energy(self, state: np.ndarray | None = None) -> float:
        """Compute total mechanical energy: E = 0.5*k*x^2 + 0.5*m*v^2."""
        if state is None:
            state = self._state
        x, v = state
        return 0.5 * self.k * x**2 + 0.5 * self.m * v**2

    def analytical_solution(self, t: float) -> tuple[float, float]:
        """Analytical solution for underdamped free oscillation.

        Returns (x(t), v(t)) for zeta < 1, F=0.
        """
        z = self.zeta
        w0 = self.omega_0
        wd = self.omega_d

        if z >= 1.0 or wd == 0:
            raise ValueError("Analytical solution only for underdamped (zeta < 1)")

        # x(t) = exp(-z*w0*t) * [x0*cos(wd*t) + (v0 + z*w0*x0)/wd * sin(wd*t)]
        exp_term = np.exp(-z * w0 * t)
        x = exp_term * (
            self.x_0 * np.cos(wd * t)
            + (self.v_0 + z * w0 * self.x_0) / wd * np.sin(wd * t)
        )
        v = exp_term * (
            -z * w0 * self.x_0 * np.cos(wd * t)
            - z * w0 * (self.v_0 + z * w0 * self.x_0) / wd * np.sin(wd * t)
            - self.x_0 * wd * np.sin(wd * t)
            + (self.v_0 + z * w0 * self.x_0) * np.cos(wd * t)
        )
        return float(x), float(v)
