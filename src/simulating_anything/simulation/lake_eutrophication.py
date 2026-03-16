"""Novel Lake Eutrophication simulation (4D).

Coupled phosphorus loading with algal bloom dynamics, dissolved
oxygen depletion, and fish population response. Excessive nutrients
trigger algal blooms, which decompose and deplete oxygen, causing
fish kills. Classic environmental catastrophe model.

State: [P_lake, A_algae, DO, F_fish] where:
  P_lake = dissolved phosphorus
  A_algae = algal biomass
  DO = dissolved oxygen
  F_fish = fish population
"""
from __future__ import annotations
import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

class LakeEutrophicationSimulation(SimulationEnvironment):
    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.P_in = p.get("P_in", 0.1)
        self.k_sed = p.get("k_sed", 0.05)
        self.mu_a = p.get("mu_a", 0.5)
        self.K_p = p.get("K_p", 0.1)
        self.d_a = p.get("d_a", 0.1)
        self.k_reaer = p.get("k_reaer", 0.3)
        self.DO_sat = p.get("DO_sat", 8.0)
        self.k_bod = p.get("k_bod", 0.2)
        self.r_f = p.get("r_f", 0.1)
        self.K_f = p.get("K_f", 100.0)
        self.DO_crit = p.get("DO_crit", 2.0)
        self.dt = config.dt
        self._state = np.array([p.get("P_0", 0.5), p.get("A_0", 5.0),
                                p.get("DO_0", 7.0), p.get("F_0", 50.0)], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        P, A, DO, F = state
        growth = self.mu_a * P / (self.K_p + P) * A
        dP = self.P_in - self.k_sed * P - growth * 0.01
        dA = growth - self.d_a * A
        dDO = self.k_reaer * (self.DO_sat - DO) - self.k_bod * self.d_a * A
        fish_stress = 1.0 / (1.0 + np.exp(-(DO - self.DO_crit)))
        dF = self.r_f * F * (1 - F / self.K_f) * fish_stress - 0.05 * F * (1 - fish_stress)
        return np.array([dP, dA, dDO, dF])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(0.5 + rng.normal(0, 0.1), 0.01), max(5 + rng.normal(0, 1), 0.1),
                max(7 + rng.normal(0, 0.5), 0.1), max(50 + rng.normal(0, 5), 1)
            ], dtype=np.float64)
        else:
            self._state = np.array([0.5, 5.0, 7.0, 50.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
