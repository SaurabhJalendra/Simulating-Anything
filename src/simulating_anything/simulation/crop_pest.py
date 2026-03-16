"""Novel Crop-Pest Dynamics simulation (4D).

Coupled crop growth with pest population and natural enemy
(biocontrol) dynamics. Pests reduce crop yield, natural enemies
control pests. Pesticide application kills pests but also harms
natural enemies, creating a pest resurgence problem.

State: [C, P_pest, E_nat, Y] where:
  C = crop biomass
  P_pest = pest population
  E_nat = natural enemy population
  Y = cumulative yield
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CropPestSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r_c = p.get("r_c", 0.5)
        self.K_c = p.get("K_c", 100.0)
        self.a_pc = p.get("a_pc", 0.3)
        self.r_p = p.get("r_p", 0.8)
        self.K_p = p.get("K_p", 50.0)
        self.a_ep = p.get("a_ep", 0.4)
        self.e_ep = p.get("e_ep", 0.3)
        self.d_e = p.get("d_e", 0.15)
        self.harvest = p.get("harvest", 0.01)
        self.dt = config.dt
        self._state = np.array([p.get("C_0", 50.0), p.get("P_0", 5.0),
                                p.get("E_0", 2.0), 0.0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        C, Pp, En, Y = state
        dC = self.r_c * C * (1 - C / self.K_c) - self.a_pc * Pp * C / (C + 10)
        dPp = self.r_p * Pp * (1 - Pp / self.K_p) - self.a_ep * En * Pp / (Pp + 5)
        dEn = self.e_ep * self.a_ep * En * Pp / (Pp + 5) - self.d_e * En
        dY = self.harvest * C
        return np.array([dC, dPp, dEn, dY])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(50 + rng.normal(0, 5), 1), max(5 + rng.normal(0, 1), 0.1),
                max(2 + rng.normal(0, 0.5), 0.1), 0.0], dtype=np.float64)
        else:
            self._state = np.array([50.0, 5.0, 2.0, 0.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[:3] = np.clip(self._state[:3], 0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
