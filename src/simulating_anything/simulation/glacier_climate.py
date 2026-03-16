"""Novel Glacier-Climate Feedback simulation (4D).

Coupled glacier mass balance with surface albedo feedback.
Glaciers reflect sunlight (high albedo), cooling the climate.
As glaciers melt, albedo decreases, warming accelerates,
creating a positive feedback (ice-albedo feedback).

State: [V_ice, T_s, A_s, P_snow] where:
  V_ice = glacier volume
  T_s = surface temperature
  A_s = surface albedo
  P_snow = snowfall accumulation rate
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GlacierClimateSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.S_solar = p.get("S_solar", 340.0)
        self.eps_lw = p.get("eps_lw", 0.6)
        self.sigma = p.get("sigma_sb", 5.67e-8)
        self.C_th = p.get("C_th", 5e8)
        self.A_ice = p.get("A_ice", 0.7)
        self.A_land = p.get("A_land", 0.3)
        self.melt_rate = p.get("melt_rate", 0.1)
        self.T_melt = p.get("T_melt", 0.0)
        self.P_base = p.get("P_base", 0.5)
        self.P_temp = p.get("P_temp", 0.02)
        self.dt = config.dt
        self._state = np.array([p.get("Vice_0", 10.0), p.get("Ts_0", -5.0),
                                0.5, self.P_base], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        V, T, A, P = state
        f_ice = np.clip(V / 20.0, 0, 1)
        A_eff = f_ice * self.A_ice + (1 - f_ice) * self.A_land
        Q_in = self.S_solar * (1 - A_eff)
        Q_out = self.eps_lw * self.sigma * (T + 273.15) ** 4
        dT = (Q_in - Q_out) / self.C_th * 1e6
        snow = self.P_base * max(1 - self.P_temp * T, 0)
        melt = self.melt_rate * max(T - self.T_melt, 0) * f_ice
        dV = snow - melt
        dA = A_eff - A
        dP = snow - P
        return np.array([dV, dT, dA, dP])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            V = max(10.0 + rng.normal(0, 2.0), 0.1)
            self._state = np.array([V, -5.0 + rng.normal(0, 2.0), 0.5, self.P_base], dtype=np.float64)
        else:
            self._state = np.array([10.0, -5.0, 0.5, self.P_base], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[0] = max(self._state[0], 0)
        self._state[2] = np.clip(self._state[2], 0, 1)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
