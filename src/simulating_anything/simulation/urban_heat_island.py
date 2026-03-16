"""Novel Urban Heat Island simulation (4D).

Couples urban surface energy balance with vegetation cooling
and anthropogenic heat. Impervious surfaces absorb solar radiation,
raising air temperature. Urban vegetation provides evaporative
cooling. Anthropogenic heat from buildings and traffic adds to
the heat island effect.

State: [T_u, T_r, V_u, Q_h] where:
  T_u = urban air temperature
  T_r = rural reference temperature
  V_u = urban vegetation fraction
  Q_h = anthropogenic heat flux
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class UrbanHeatIslandSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha_s = p.get("alpha_s", 0.3)
        self.sigma_sb = p.get("sigma_sb", 0.1)
        self.h_conv = p.get("h_conv", 5.0)
        self.Q_solar = p.get("Q_solar", 200.0)
        self.lambda_et = p.get("lambda_et", 50.0)
        self.d_v = p.get("d_v", 0.01)
        self.r_v = p.get("r_v", 0.02)
        self.T_opt = p.get("T_opt", 25.0)
        self.Q_base = p.get("Q_base", 50.0)
        self.Q_amp = p.get("Q_amp", 20.0)
        self.C_th = p.get("C_th", 100.0)
        self.dt = config.dt
        self._state = np.array([p.get("Tu_0", 30.0), p.get("Tr_0", 25.0),
                                p.get("Vu_0", 0.2), self.Q_base], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        Tu, Tr, Vu, Qh = state
        Q_absorbed = (1 - self.alpha_s) * self.Q_solar
        Q_lw = self.sigma_sb * Tu
        Q_conv = self.h_conv * (Tu - Tr)
        Q_et = self.lambda_et * Vu * max(Tu - 15, 0) / (10 + max(Tu - 15, 0))
        Q_anthro = self.Q_base + self.Q_amp * (0.5 + 0.5 * np.sin(2 * np.pi * 0.04 * t))

        dTu = (Q_absorbed - Q_lw - Q_conv - Q_et + Q_anthro) / self.C_th
        dTr = -0.1 * (Tr - 25.0)
        dVu = self.r_v * Vu * (1 - Vu) * np.exp(-((Tu - self.T_opt) / 10) ** 2) - self.d_v * Vu
        dQh = Q_anthro - Qh

        return np.array([dTu, dTr, dVu, dQh])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                30.0 + rng.normal(0, 2.0),
                25.0 + rng.normal(0, 1.0),
                np.clip(0.2 + rng.normal(0, 0.05), 0.01, 0.8),
                self.Q_base,
            ], dtype=np.float64)
        else:
            self._state = np.array([30.0, 25.0, 0.2, self.Q_base], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[2] = np.clip(self._state[2], 0.0, 1.0)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
