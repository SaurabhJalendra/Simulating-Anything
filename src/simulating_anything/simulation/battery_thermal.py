"""Novel coupled Battery-Thermal simulation (4D).

Couples lithium-ion battery equivalent circuit dynamics with
thermal runaway modeling. Internal resistance increases with
temperature, generating more heat, which further raises temperature
in a positive feedback loop. Models the critical safety concern
of thermal runaway in lithium-ion batteries.

State: [V_oc, V_rc, T, SOC] where:
  V_oc = open circuit voltage (depends on SOC)
  V_rc = RC circuit voltage (transient response)
  T = cell temperature
  SOC = state of charge (0-1)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BatteryThermalSimulation(SimulationEnvironment):
    """Coupled battery equivalent circuit + thermal dynamics."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.I_load = p.get("I_load", 2.0)
        self.R_0 = p.get("R_0", 0.05)
        self.R_1 = p.get("R_1", 0.03)
        self.C_1 = p.get("C_1", 1000.0)
        self.Q_cap = p.get("Q_cap", 3.0)

        self.C_th = p.get("C_th", 50.0)
        self.h_conv = p.get("h_conv", 5.0)
        self.T_amb = p.get("T_amb", 25.0)
        self.alpha_R = p.get("alpha_R", 0.005)

        self.V_max = p.get("V_max", 4.2)
        self.V_min = p.get("V_min", 3.0)

        self.SOC_0 = p.get("SOC_0", 0.8)
        self.T_0 = p.get("T_0", 25.0)

        self.dt = config.dt
        V_oc = self.V_min + (self.V_max - self.V_min) * self.SOC_0
        self._state = np.array([V_oc, 0.0, self.T_0, self.SOC_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        V_oc, V_rc, T, SOC = state

        R_eff = self.R_0 * (1.0 + self.alpha_R * (T - self.T_amb))
        tau_rc = self.R_1 * self.C_1

        dV_oc = -(self.V_max - self.V_min) * self.I_load / (3600.0 * self.Q_cap)
        dV_rc = (self.I_load * self.R_1 - V_rc) / tau_rc

        Q_heat = self.I_load ** 2 * R_eff + self.I_load * V_rc
        Q_cool = self.h_conv * (T - self.T_amb)
        dT = (Q_heat - Q_cool) / self.C_th

        dSOC = -self.I_load / (3600.0 * self.Q_cap)

        return np.array([dV_oc, dV_rc, dT, dSOC])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            SOC = np.clip(self.SOC_0 + rng.normal(0, 0.05), 0.1, 1.0)
            T = max(self.T_0 + rng.normal(0, 2.0), 15.0)
            V_oc = self.V_min + (self.V_max - self.V_min) * SOC
            self._state = np.array([V_oc, 0.0, T, SOC], dtype=np.float64)
        else:
            V_oc = self.V_min + (self.V_max - self.V_min) * self.SOC_0
            self._state = np.array([V_oc, 0.0, self.T_0, self.SOC_0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[3] = np.clip(self._state[3], 0.0, 1.0)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
