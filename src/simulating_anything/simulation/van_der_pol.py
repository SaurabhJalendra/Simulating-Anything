"""Van der Pol oscillator simulation.

Target rediscoveries:
- The Van der Pol ODE: x'' - mu*(1-x^2)*x' + x = 0
- Limit cycle amplitude: A ~ 2 for any mu > 0 (exact for relaxation oscillations)
- Period scaling: T ~ mu for large mu (relaxation regime)
- Small mu: T ~ 2*pi (near-harmonic), with amplitude correction
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class VanDerPolSimulation(SimulationEnvironment):
    """Van der Pol oscillator: x'' - mu*(1-x^2)*x' + x = 0.

    State vector: [x, v] where x = displacement, v = velocity (dx/dt).

    The nonlinear damping term -mu*(1-x^2)*x' provides:
    - Negative damping for |x| < 1 (energy injection)
    - Positive damping for |x| > 1 (energy dissipation)
    - Stable limit cycle for any mu > 0

    Parameters:
        mu: nonlinearity parameter (controls damping strength)
        x_0: initial displacement
        v_0: initial velocity
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.mu = p.get("mu", 1.0)
        self.x_0 = p.get("x_0", 0.1)
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
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        x, v = y
        a = self.mu * (1 - x**2) * v - x
        return np.array([v, a])

    @property
    def limit_cycle_amplitude(self) -> float:
        """Theoretical limit cycle amplitude (exact: 2 for relaxation oscillations)."""
        return 2.0

    @property
    def approximate_period(self) -> float:
        """Approximate period based on mu regime.

        - mu << 1: T ~ 2*pi (harmonic limit)
        - mu >> 1: T ~ (3 - 2*ln(2))*mu ~ 1.614*mu (relaxation regime)
        """
        if self.mu < 0.1:
            return 2 * np.pi
        elif self.mu > 5.0:
            return (3 - 2 * np.log(2)) * self.mu
        else:
            # Interpolation for intermediate mu
            return 2 * np.pi * (1 + self.mu**2 / 16)

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the period by detecting zero crossings after transient."""
        dt = self.config.dt
        # Let transient die out
        transient_steps = max(int(50 / dt), int(20 * self.approximate_period / dt))
        for _ in range(transient_steps):
            self.step()

        # Detect upward zero crossings
        crossings = []
        prev_x = self._state[0]
        for i in range(int(n_periods * self.approximate_period / dt * 2)):
            self.step()
            x = self._state[0]
            if prev_x < 0 and x >= 0:
                # Linear interpolation for crossing time
                t_cross = (self._step_count - 1) * dt + dt * (-prev_x) / (x - prev_x)
                crossings.append(t_cross)
            prev_x = x

        if len(crossings) < 2:
            return float("inf")

        periods = np.diff(crossings)
        return float(np.mean(periods))

    def measure_amplitude(self, n_periods: int = 3) -> float:
        """Measure the limit cycle amplitude after transient."""
        dt = self.config.dt
        transient_steps = max(int(50 / dt), int(20 * self.approximate_period / dt))
        for _ in range(transient_steps):
            self.step()

        # Record peaks
        x_max = -np.inf
        measure_steps = int(n_periods * self.approximate_period / dt * 2)
        for _ in range(measure_steps):
            self.step()
            x_max = max(x_max, abs(self._state[0]))

        return float(x_max)

    def total_energy(self, state: np.ndarray | None = None) -> float:
        """Compute instantaneous energy (not conserved): E = 0.5*(x^2 + v^2)."""
        if state is None:
            state = self._state
        x, v = state
        return 0.5 * (x**2 + v**2)
