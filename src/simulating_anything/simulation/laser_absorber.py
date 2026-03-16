"""Novel coupled Laser-Saturable-Absorber simulation (4D).

Couples laser rate equations with a saturable absorber medium.
The absorber bleaches under high intensity, creating Q-switched
pulses. Gain medium recovery and absorber relaxation compete on
different timescales, producing complex pulsing dynamics.

State: [G, I_l, Q, N_a] where:
  G = gain medium inversion (population difference)
  I_l = intracavity laser intensity
  Q = cavity quality factor (modulated by absorber)
  N_a = absorber ground state population
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LaserAbsorberSimulation(SimulationEnvironment):
    """Coupled laser rate equations + saturable absorber."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.W_p = p.get("W_p", 1.5)
        self.tau_g = p.get("tau_g", 1.0)
        self.sigma_g = p.get("sigma_g", 1.0)
        self.tau_c = p.get("tau_c", 0.01)
        self.sigma_a = p.get("sigma_a", 3.0)
        self.tau_a = p.get("tau_a", 0.5)
        self.N_a0 = p.get("N_a0", 1.0)
        self.alpha_loss = p.get("alpha_loss", 0.1)

        self.G_0 = p.get("G_0", 0.5)
        self.Il_0 = p.get("Il_0", 0.01)
        self.Q_0 = p.get("Q_0", 0.5)
        self.Na_0 = p.get("Na_0_init", 1.0)

        self.dt = config.dt
        self._state = np.array([self.G_0, self.Il_0, self.Q_0, self.Na_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        G, I_l, Q, N_a = state
        gain = self.sigma_g * G * I_l
        absorption = self.sigma_a * N_a * I_l

        dG = self.W_p - G / self.tau_g - gain
        dI = (gain - absorption - self.alpha_loss * I_l) / self.tau_c
        dQ = (1.0 - self.sigma_a * N_a) - Q
        dNa = (self.N_a0 - N_a) / self.tau_a - self.sigma_a * N_a * I_l

        return np.array([dG, dI, dQ, dNa])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.G_0 + rng.normal(0, 0.1), 0.01),
                max(self.Il_0 + rng.normal(0, 0.005), 0.001),
                max(self.Q_0 + rng.normal(0, 0.05), 0.01),
                max(self.Na_0 + rng.normal(0, 0.1), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.G_0, self.Il_0, self.Q_0, self.Na_0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
