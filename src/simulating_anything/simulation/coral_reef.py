"""Novel Coral Reef Ecosystem simulation (4D).

Couples coral-algae competition with herbivore grazing and
water quality. Coral and macroalgae compete for space on the reef.
Herbivorous fish graze algae, maintaining coral dominance.
Nutrient pollution favors algae, leading to phase shifts.

State: [C, A, H, N] where:
  C = coral cover fraction
  A = macroalgae cover fraction
  H = herbivore biomass
  N = dissolved nutrient concentration
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CoralReefSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r_c = p.get("r_c", 0.1)
        self.r_a = p.get("r_a", 0.3)
        self.d_c = p.get("d_c", 0.05)
        self.gamma_ac = p.get("gamma_ac", 0.1)
        self.g_max = p.get("g_max", 0.5)
        self.K_a = p.get("K_a", 0.3)
        self.e_h = p.get("e_h", 0.3)
        self.d_h = p.get("d_h", 0.1)
        self.N_in = p.get("N_in", 0.5)
        self.d_n = p.get("d_n", 0.2)
        self.alpha_na = p.get("alpha_na", 0.5)
        self.dt = config.dt
        self._state = np.array([p.get("C_0", 0.4), p.get("A_0", 0.2),
                                p.get("H_0", 1.0), p.get("N_0", 1.0)], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        C, A, H, N = state
        S = max(1.0 - C - A, 0.0)
        r_a_eff = self.r_a * (1.0 + self.alpha_na * N)
        grazing = self.g_max * A / (self.K_a + A) * H

        dC = self.r_c * C * S - self.d_c * C - self.gamma_ac * A * C
        dA = r_a_eff * A * S + self.gamma_ac * A * C - grazing
        dH = self.e_h * grazing - self.d_h * H
        dN = self.N_in - self.d_n * N - 0.1 * r_a_eff * A

        return np.array([dC, dA, dH, dN])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            C = np.clip(0.4 + rng.normal(0, 0.05), 0.01, 0.8)
            A = np.clip(0.2 + rng.normal(0, 0.05), 0.01, 0.8 - C)
            self._state = np.array([C, A, max(1.0 + rng.normal(0, 0.2), 0.1),
                                    max(1.0 + rng.normal(0, 0.2), 0.1)], dtype=np.float64)
        else:
            self._state = np.array([0.4, 0.2, 1.0, 1.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[:2] = np.clip(self._state[:2], 0.0, 1.0)
        self._state[2:] = np.clip(self._state[2:], 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
