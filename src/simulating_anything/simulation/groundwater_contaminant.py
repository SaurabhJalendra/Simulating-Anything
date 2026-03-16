"""Novel Groundwater Contaminant simulation (4D).

Coupled groundwater flow with contaminant transport and
biodegradation. Contaminant plume moves with groundwater flow
and disperses. Indigenous bacteria degrade the contaminant,
growing on it as a carbon source. Models natural attenuation
of groundwater pollution.

State: [h, C_cont, B_bio, O2] where:
  h = hydraulic head (drives flow)
  C_cont = contaminant concentration
  B_bio = biodegrading bacteria density
  O2 = dissolved oxygen
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GroundwaterContaminantSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.K_hyd = p.get("K_hyd", 0.01)
        self.S_s = p.get("S_s", 0.001)
        self.recharge = p.get("recharge", 0.001)
        self.D_disp = p.get("D_disp", 0.1)
        self.v_max = p.get("v_max", 0.5)
        self.K_c = p.get("K_c", 1.0)
        self.K_o = p.get("K_o", 0.5)
        self.Y_b = p.get("Y_b", 0.3)
        self.d_b = p.get("d_b", 0.05)
        self.O2_sat = p.get("O2_sat", 8.0)
        self.k_reaer = p.get("k_reaer", 0.02)
        self.dt = config.dt
        self._state = np.array([p.get("h_0", 10.0), p.get("Cc_0", 5.0),
                                p.get("Bb_0", 0.5), p.get("O2_0", 6.0)], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        h, Cc, Bb, O2 = state
        biodeg = self.v_max * Cc / (self.K_c + Cc) * O2 / (self.K_o + O2) * Bb
        dh = (self.recharge - self.K_hyd * (h - 5.0)) / self.S_s
        dCc = -biodeg + self.D_disp * (5.0 - Cc) * 0.01
        dBb = self.Y_b * biodeg - self.d_b * Bb
        dO2 = self.k_reaer * (self.O2_sat - O2) - 2.0 * biodeg
        return np.array([dh, dCc, dBb, dO2])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                10 + rng.normal(0, 0.5), max(5 + rng.normal(0, 0.5), 0.1),
                max(0.5 + rng.normal(0, 0.1), 0.01), max(6 + rng.normal(0, 0.5), 0.1)
            ], dtype=np.float64)
        else:
            self._state = np.array([10.0, 5.0, 0.5, 6.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[1:] = np.clip(self._state[1:], 0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
