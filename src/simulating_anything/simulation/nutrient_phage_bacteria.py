"""Novel coupled Nutrient-Phage-Bacteria simulation (4D).

Couples bacterial growth on nutrients with bacteriophage (virus)
infection dynamics. Phages infect bacteria, lysing them to release
more phages. Nutrient depletion limits bacterial growth, while
lysis releases nutrients back. Creates "kill the winner" dynamics
important in marine microbiology.

State: [N, B, I, V] where:
  N = nutrient concentration
  B = uninfected bacteria density
  I = infected bacteria density
  V = free phage (virus) density
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class NutrientPhageBacteriaSimulation(SimulationEnvironment):
    """Nutrient-bacteria-phage-infected dynamics."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.N_in = p.get("N_in", 5.0)
        self.D = p.get("D_dilution", 0.1)
        self.mu_max = p.get("mu_max", 1.0)
        self.K_n = p.get("K_n", 0.5)
        self.Y = p.get("Y_yield", 0.5)
        self.phi = p.get("phi", 0.01)
        self.eta = p.get("eta", 0.5)
        self.burst = p.get("burst", 50.0)
        self.d_v = p.get("d_v", 0.1)
        self.recycle = p.get("recycle", 0.3)

        self.N_0 = p.get("N_0", 3.0)
        self.B_0 = p.get("B_0", 1.0)
        self.I_0 = p.get("I_0", 0.1)
        self.V_0 = p.get("V_0", 5.0)

        self.dt = config.dt
        self._state = np.array([self.N_0, self.B_0, self.I_0, self.V_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        N, B, I, V = state
        growth = self.mu_max * N / (self.K_n + N)
        adsorption = self.phi * B * V

        dN = self.D * (self.N_in - N) - growth * B / self.Y + self.recycle * self.eta * I
        dB = growth * B - adsorption - self.D * B
        dI = adsorption - self.eta * I - self.D * I
        dV = self.burst * self.eta * I - adsorption - self.d_v * V - self.D * V

        return np.array([dN, dB, dI, dV])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.N_0 + rng.normal(0, 0.5), 0.01),
                max(self.B_0 + rng.normal(0, 0.2), 0.01),
                max(self.I_0 + rng.normal(0, 0.02), 0.001),
                max(self.V_0 + rng.normal(0, 1.0), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.N_0, self.B_0, self.I_0, self.V_0], dtype=np.float64)
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
