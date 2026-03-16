"""Novel coupled Soil Carbon-Nitrogen simulation (4D).

Couples soil organic carbon decomposition with nitrogen
mineralization-immobilization dynamics. Microbial decomposition
of organic matter releases CO2 and mineralizes nitrogen, but
microbes also immobilize mineral nitrogen for growth. The C:N
ratio of substrate determines net mineralization vs immobilization.

State: [C_s, C_m, N_min, N_org] where:
  C_s = soil organic carbon
  C_m = microbial biomass carbon
  N_min = mineral (inorganic) nitrogen
  N_org = organic nitrogen in substrate
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SoilCarbonNitrogenSimulation(SimulationEnvironment):
    """Coupled soil C decomposition + N mineralization-immobilization."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.k_dec = p.get("k_dec", 0.05)
        self.CUE = p.get("CUE", 0.3)
        self.k_death = p.get("k_death", 0.02)
        self.CN_mic = p.get("CN_mic", 8.0)
        self.k_nit = p.get("k_nit", 0.01)
        self.N_dep = p.get("N_dep", 0.005)
        self.C_input = p.get("C_input", 0.1)
        self.CN_input = p.get("CN_input", 20.0)

        self.Cs_0 = p.get("Cs_0", 50.0)
        self.Cm_0 = p.get("Cm_0", 5.0)
        self.Nmin_0 = p.get("Nmin_0", 2.0)
        self.Norg_0 = p.get("Norg_0", 3.0)

        self.dt = config.dt
        self._state = np.array([self.Cs_0, self.Cm_0, self.Nmin_0, self.Norg_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        Cs, Cm, Nmin, Norg = state

        decomp = self.k_dec * Cs * Cm / (Cs + 10.0)
        growth = self.CUE * decomp
        respiration = (1.0 - self.CUE) * decomp
        death = self.k_death * Cm

        CN_sub = (Cs + 1e-10) / (Norg + 1e-10)
        N_demand = growth / self.CN_mic
        N_supply = decomp / CN_sub
        net_mineralization = N_supply - N_demand

        dCs = self.C_input - decomp + death
        dCm = growth - death
        dNmin = net_mineralization + self.N_dep - self.k_nit * Nmin
        dNorg = self.C_input / self.CN_input - N_supply + death / self.CN_mic

        return np.array([dCs, dCm, dNmin, dNorg])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.Cs_0 + rng.normal(0, 5.0), 1.0),
                max(self.Cm_0 + rng.normal(0, 0.5), 0.1),
                max(self.Nmin_0 + rng.normal(0, 0.3), 0.01),
                max(self.Norg_0 + rng.normal(0, 0.3), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.Cs_0, self.Cm_0, self.Nmin_0, self.Norg_0], dtype=np.float64)
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
