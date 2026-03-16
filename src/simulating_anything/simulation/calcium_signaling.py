"""Novel coupled Calcium Signaling simulation (4D).

Couples IP3-dependent calcium release from ER with mitochondrial
calcium uptake and cytoplasmic buffering. Models intracellular
calcium oscillations critical for cell signaling, muscle contraction,
and neurotransmitter release.

State: [Ca_c, Ca_ER, Ca_m, IP3] where:
  Ca_c = cytoplasmic calcium concentration
  Ca_ER = endoplasmic reticulum calcium store
  Ca_m = mitochondrial calcium
  IP3 = inositol trisphosphate concentration
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CalciumSignalingSimulation(SimulationEnvironment):
    """IP3-dependent calcium oscillation with ER and mitochondria."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.v_ip3r = p.get("v_ip3r", 0.5)
        self.K_ip3 = p.get("K_ip3", 0.3)
        self.K_act = p.get("K_act", 0.3)
        self.v_serca = p.get("v_serca", 0.4)
        self.K_serca = p.get("K_serca", 0.2)
        self.v_leak = p.get("v_leak", 0.002)
        self.v_mcu = p.get("v_mcu", 0.01)
        self.K_mcu = p.get("K_mcu", 0.5)
        self.v_nclx = p.get("v_nclx", 0.005)
        self.v_plc = p.get("v_plc", 0.02)
        self.d_ip3 = p.get("d_ip3", 0.1)
        self.v_stim = p.get("v_stim", 0.05)

        self.Ca_c0 = p.get("Ca_c0", 0.1)
        self.Ca_ER0 = p.get("Ca_ER0", 5.0)
        self.Ca_m0 = p.get("Ca_m0", 0.2)
        self.IP3_0 = p.get("IP3_0", 0.3)

        self.dt = config.dt
        self._state = np.array([self.Ca_c0, self.Ca_ER0, self.Ca_m0, self.IP3_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        Ca_c, Ca_ER, Ca_m, IP3 = state

        ip3r_open = IP3 ** 2 / (self.K_ip3 ** 2 + IP3 ** 2)
        ca_act = Ca_c ** 2 / (self.K_act ** 2 + Ca_c ** 2)
        J_ip3r = self.v_ip3r * ip3r_open * ca_act * (Ca_ER - Ca_c)
        J_serca = self.v_serca * Ca_c ** 2 / (self.K_serca ** 2 + Ca_c ** 2)
        J_leak = self.v_leak * (Ca_ER - Ca_c)
        J_mcu = self.v_mcu * Ca_c ** 2 / (self.K_mcu ** 2 + Ca_c ** 2)
        J_nclx = self.v_nclx * Ca_m

        dCa_c = J_ip3r - J_serca + J_leak - J_mcu + J_nclx
        dCa_ER = -J_ip3r + J_serca - J_leak
        dCa_m = J_mcu - J_nclx
        dIP3 = self.v_stim + self.v_plc * Ca_c - self.d_ip3 * IP3

        return np.array([dCa_c, dCa_ER, dCa_m, dIP3])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.Ca_c0 + rng.normal(0, 0.02), 0.01),
                max(self.Ca_ER0 + rng.normal(0, 0.5), 0.1),
                max(self.Ca_m0 + rng.normal(0, 0.05), 0.01),
                max(self.IP3_0 + rng.normal(0, 0.05), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.Ca_c0, self.Ca_ER0, self.Ca_m0, self.IP3_0], dtype=np.float64)
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
