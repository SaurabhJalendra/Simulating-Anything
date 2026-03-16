"""Novel coupled Hormone-Glucose simulation (4D).

Couples insulin-glucose regulation with glucagon counter-regulation.
Pancreatic beta cells secrete insulin in response to glucose,
lowering blood sugar. Alpha cells secrete glucagon when glucose
drops, raising it. Creates homeostatic oscillations.

State: [G, I, Gl, B] where:
  G = blood glucose concentration
  I = plasma insulin concentration
  Gl = plasma glucagon concentration
  B = beta cell function (insulin secretion capacity)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class HormoneGlucoseSimulation(SimulationEnvironment):
    """Coupled glucose-insulin-glucagon regulation."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.G_in = p.get("G_in", 1.0)
        self.S_i = p.get("S_i", 0.5)
        self.K_g = p.get("K_g", 5.0)
        self.d_g = p.get("d_g", 0.1)
        self.d_i = p.get("d_i", 0.2)
        self.d_gl = p.get("d_gl", 0.3)
        self.alpha_gl = p.get("alpha_gl", 0.5)
        self.K_gl = p.get("K_gl", 4.0)
        self.v_glu = p.get("v_glu", 0.3)
        self.tau_b = p.get("tau_b", 10.0)
        self.G_target = p.get("G_target", 5.0)

        self.G_0 = p.get("G_0", 5.0)
        self.I_0 = p.get("I_0", 1.0)
        self.Gl_0 = p.get("Gl_0", 0.5)
        self.B_0 = p.get("B_0", 1.0)

        self.dt = config.dt
        self._state = np.array([self.G_0, self.I_0, self.Gl_0, self.B_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        G, I, Gl, B = state

        insulin_effect = self.S_i * I * G / (self.K_g + G)
        glucagon_effect = self.v_glu * Gl

        dG = self.G_in - insulin_effect + glucagon_effect - self.d_g * G
        dI = B * G ** 2 / (self.K_g ** 2 + G ** 2) - self.d_i * I
        dGl = self.alpha_gl * self.K_gl ** 2 / (self.K_gl ** 2 + G ** 2) - self.d_gl * Gl
        dB = (self.G_target - G) / self.tau_b * (1.0 - B) if G < self.G_target else -(G - self.G_target) / self.tau_b * B * 0.1

        return np.array([dG, dI, dGl, dB])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.G_0 + rng.normal(0, 0.5), 0.5),
                max(self.I_0 + rng.normal(0, 0.2), 0.01),
                max(self.Gl_0 + rng.normal(0, 0.1), 0.01),
                np.clip(self.B_0 + rng.normal(0, 0.1), 0.1, 2.0),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.G_0, self.I_0, self.Gl_0, self.B_0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0.0, None)
        self._state[3] = np.clip(self._state[3], 0.0, 2.0)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
