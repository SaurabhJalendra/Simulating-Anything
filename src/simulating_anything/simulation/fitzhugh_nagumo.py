"""FitzHugh-Nagumo neuron model simulation.

Target rediscoveries:
- Excitation threshold for spiking
- f-I curve: firing frequency vs input current
- Hopf bifurcation at critical current I_c
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FitzHughNagumoSimulation(SimulationEnvironment):
    """FitzHugh-Nagumo: dv/dt = v - v^3/3 - w + I, dw/dt = eps*(v + a - b*w).

    Simplified model of neural excitability. State: [v, w] where
    v = membrane voltage (fast), w = recovery variable (slow).

    Parameters:
        a: recovery parameter (default 0.7)
        b: recovery parameter (default 0.8)
        eps: timescale separation (default 0.08)
        I: external current (default 0.0)
        v_0: initial voltage
        w_0: initial recovery
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.7)
        self.b_param = p.get("b", 0.8)
        self.eps = p.get("eps", 0.08)
        self.I = p.get("I", 0.0)
        self.v_0 = p.get("v_0", -1.0)
        self.w_0 = p.get("w_0", -0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize voltage and recovery."""
        self._state = np.array([self.v_0, self.w_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [v, w]."""
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
        v, w = y
        dv = v - v**3 / 3 - w + self.I
        dw = self.eps * (v + self.a - self.b_param * w)
        return np.array([dv, dw])

    @property
    def nullcline_v(self):
        """V-nullcline: w = v - v^3/3 + I."""
        return lambda v: v - v**3 / 3 + self.I

    @property
    def nullcline_w(self):
        """W-nullcline: w = (v + a) / b."""
        return lambda v: (v + self.a) / self.b_param

    def is_oscillatory(self, n_test_steps: int = 5000) -> bool:
        """Check if current parameters produce sustained oscillations."""
        self.reset()
        # Skip transient
        for _ in range(n_test_steps):
            self.step()
        # Check for oscillation over next period
        v_values = []
        for _ in range(n_test_steps):
            self.step()
            v_values.append(self._state[0])
        v_range = max(v_values) - min(v_values)
        return v_range > 0.5  # Threshold for oscillation

    def measure_firing_frequency(self, n_spikes: int = 5) -> float:
        """Measure firing frequency by counting upward threshold crossings."""
        dt = self.config.dt
        threshold = 0.0

        # Transient
        for _ in range(int(500 / dt)):
            self.step()

        crossings = []
        prev_v = self._state[0]
        max_steps = int(n_spikes * 500 / dt)
        for _ in range(max_steps):
            self.step()
            v = self._state[0]
            if prev_v < threshold and v >= threshold:
                crossings.append(self._step_count * dt)
            prev_v = v

        if len(crossings) < 2:
            return 0.0

        periods = np.diff(crossings)
        return float(1.0 / np.mean(periods))
