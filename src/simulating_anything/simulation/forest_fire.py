"""Novel Forest Fire Spread simulation (4D).

Simplified forest fire dynamics coupling fuel load, fire intensity,
moisture content, and wind-driven spread rate. Dry conditions and
high fuel loads create explosive fire growth, while rain and
fuel depletion create natural firebreaks.

State: [F, I_f, M, W] where:
  F = fuel load (biomass available to burn)
  I_f = fire intensity
  M = fuel moisture content
  W = wind-driven spread factor
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ForestFireSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r_grow = p.get("r_grow", 0.05)
        self.K_fuel = p.get("K_fuel", 10.0)
        self.alpha_burn = p.get("alpha_burn", 0.3)
        self.beta_moist = p.get("beta_moist", 2.0)
        self.P_rain = p.get("P_rain", 0.1)
        self.E_dry = p.get("E_dry", 0.05)
        self.W_base = p.get("W_base", 1.0)
        self.W_var = p.get("W_var", 0.3)
        self.ignition = p.get("ignition", 0.01)
        self.d_fire = p.get("d_fire", 0.5)
        self.dt = config.dt
        self._state = np.array([p.get("F_0", 8.0), p.get("If_0", 0.1),
                                p.get("M_0", 0.5), self.W_base], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        F, If, M, W = state
        moisture_factor = np.exp(-self.beta_moist * M)
        burn_rate = self.alpha_burn * F * moisture_factor * W

        dF = self.r_grow * F * (1.0 - F / self.K_fuel) - burn_rate * If / (1.0 + If)
        dIf = burn_rate * If / (1.0 + If) + self.ignition - self.d_fire * If
        dM = self.P_rain - self.E_dry * M - 0.1 * If
        dW = self.W_var * np.cos(2 * np.pi * 0.02 * t)

        return np.array([dF, dIf, dM, dW])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(8.0 + rng.normal(0, 1.0), 0.1),
                max(0.1 + rng.normal(0, 0.05), 0.001),
                np.clip(0.5 + rng.normal(0, 0.1), 0.01, 1.0),
                max(self.W_base + rng.normal(0, 0.1), 0.1),
            ], dtype=np.float64)
        else:
            self._state = np.array([8.0, 0.1, 0.5, self.W_base], dtype=np.float64)
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
        self._state[2] = np.clip(self._state[2], 0.0, 1.0)
        self._state[3] = max(self._state[3], 0.1)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
