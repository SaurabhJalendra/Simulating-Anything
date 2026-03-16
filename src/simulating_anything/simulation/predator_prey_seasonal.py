"""Novel Predator-Prey-Seasonal simulation (4D).

LV predator-prey with seasonal forcing on prey growth and
temperature-dependent predator metabolism. Creates complex
dynamics where seasonality interacts with population cycles.

State: [N, P, T_env, season] where:
  N = prey density
  P = predator density
  T_env = environmental temperature
  season = seasonal phase (sin cycle)
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorPreySeasonalSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 15.0)
        self.a = p.get("a_pred", 0.4)
        self.e = p.get("e_pred", 0.3)
        self.d = p.get("d_pred", 0.2)
        self.season_amp = p.get("season_amp", 0.5)
        self.season_freq = p.get("season_freq", 0.017)
        self.T_mean = p.get("T_mean", 15.0)
        self.T_amp = p.get("T_amp", 10.0)
        self.Q10 = p.get("Q10", 2.0)
        self.dt = config.dt
        self._state = np.array([p.get("N_0", 8.0), p.get("P_0", 3.0),
                                self.T_mean, 0.0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        N, P, T, _ = state
        season = np.sin(2 * np.pi * self.season_freq * t)
        r_eff = self.r * (1.0 + self.season_amp * season)
        d_eff = self.d * self.Q10 ** ((T - self.T_mean) / 10.0)
        dN = r_eff * N * (1.0 - N / self.K) - self.a * N * P
        dP = self.e * self.a * N * P - d_eff * P
        dT = self.T_amp * 2 * np.pi * self.season_freq * np.cos(2 * np.pi * self.season_freq * t)
        ds = season - state[3]
        return np.array([dN, dP, dT, ds])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(8.0 + rng.normal(0, 1.0), 0.1),
                max(3.0 + rng.normal(0, 0.5), 0.1),
                self.T_mean, 0.0], dtype=np.float64)
        else:
            self._state = np.array([8.0, 3.0, self.T_mean, 0.0], dtype=np.float64)
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
        self._state[3] = np.sin(2 * np.pi * self.season_freq * (self._t + self.dt))
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
