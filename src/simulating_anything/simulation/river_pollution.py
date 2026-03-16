"""Novel River-Pollution simulation (4D).

Streeter-Phelps dissolved oxygen model coupled with pollutant
biodegradation. Organic waste depletes oxygen; reaeration from
atmosphere restores it. Creates the classic DO sag curve.

State: [DO, BOD, T_w, Q] where:
  DO = dissolved oxygen concentration
  BOD = biochemical oxygen demand
  T_w = water temperature
  Q = flow rate
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RiverPollutionSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.k_d = p.get("k_d", 0.2)
        self.k_r = p.get("k_r", 0.4)
        self.DO_sat = p.get("DO_sat", 9.0)
        self.BOD_in = p.get("BOD_in", 0.1)
        self.k_temp = p.get("k_temp", 0.02)
        self.T_eq = p.get("T_eq", 20.0)
        self.Q_base = p.get("Q_base", 10.0)
        self.Q_var = p.get("Q_var", 0.1)
        self.dt = config.dt
        self._state = np.array([p.get("DO_0", 8.0), p.get("BOD_0", 5.0),
                                p.get("T_0", 20.0), self.Q_base], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        DO, BOD, T, Q = state
        k_d_eff = self.k_d * 1.047 ** (T - 20.0)
        k_r_eff = self.k_r * 1.024 ** (T - 20.0)
        dDO = k_r_eff * (self.DO_sat - DO) - k_d_eff * BOD
        dBOD = self.BOD_in - k_d_eff * BOD
        dT = self.k_temp * (self.T_eq - T)
        dQ = self.Q_var * np.sin(2 * np.pi * 0.01 * t)
        return np.array([dDO, dBOD, dT, dQ])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(8.0 + rng.normal(0, 0.5), 0.1),
                max(5.0 + rng.normal(0, 1.0), 0.1),
                20.0 + rng.normal(0, 2.0),
                max(self.Q_base + rng.normal(0, 1.0), 1.0),
            ], dtype=np.float64)
        else:
            self._state = np.array([8.0, 5.0, 20.0, self.Q_base], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[:2] = np.clip(self._state[:2], 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
