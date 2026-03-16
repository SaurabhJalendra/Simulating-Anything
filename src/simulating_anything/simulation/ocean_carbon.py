"""Novel coupled Ocean Carbon Cycle simulation (4D).

Couples ocean carbonate chemistry with biological carbon pump.
Phytoplankton fix CO2, sink as detritus, remineralize at depth.
Surface-deep ocean exchange modulates atmospheric CO2. Critical
for understanding climate change feedbacks.

State: [DIC_s, ALK_s, P, DIC_d] where:
  DIC_s = surface dissolved inorganic carbon
  ALK_s = surface alkalinity
  P = phytoplankton biomass
  DIC_d = deep ocean dissolved inorganic carbon
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class OceanCarbonSimulation(SimulationEnvironment):
    """Coupled ocean carbonate chemistry + biological pump."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.mu_p = p.get("mu_p", 0.5)
        self.K_dic = p.get("K_dic", 1.0)
        self.m_p = p.get("m_p", 0.1)
        self.r_remin = p.get("r_remin", 0.05)
        self.w_sink = p.get("w_sink", 0.02)
        self.k_mix = p.get("k_mix", 0.01)
        self.F_atm = p.get("F_atm", 0.01)
        self.DIC_atm_eq = p.get("DIC_atm_eq", 2.1)
        self.c_ratio = p.get("c_ratio", 6.625)

        self.DIC_s0 = p.get("DIC_s0", 2.0)
        self.ALK_s0 = p.get("ALK_s0", 2.3)
        self.P_0 = p.get("P_0", 0.5)
        self.DIC_d0 = p.get("DIC_d0", 2.2)

        self.dt = config.dt
        self._state = np.array([self.DIC_s0, self.ALK_s0, self.P_0, self.DIC_d0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        DIC_s, ALK_s, P, DIC_d = state

        growth = self.mu_p * DIC_s / (self.K_dic + DIC_s) * P
        mortality = self.m_p * P
        export = self.w_sink * mortality
        remin = self.r_remin * export
        air_sea = self.F_atm * (self.DIC_atm_eq - DIC_s)
        mixing = self.k_mix * (DIC_d - DIC_s)

        dDIC_s = -growth / self.c_ratio + air_sea + mixing
        dALK_s = -0.15 * growth / self.c_ratio + self.k_mix * (2.35 - ALK_s)
        dP = growth - mortality
        dDIC_d = remin / self.c_ratio - mixing

        return np.array([dDIC_s, dALK_s, dP, dDIC_d])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.DIC_s0 + rng.normal(0, 0.1), 0.1),
                max(self.ALK_s0 + rng.normal(0, 0.05), 0.1),
                max(self.P_0 + rng.normal(0, 0.1), 0.01),
                max(self.DIC_d0 + rng.normal(0, 0.1), 0.1),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.DIC_s0, self.ALK_s0, self.P_0, self.DIC_d0], dtype=np.float64)
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
