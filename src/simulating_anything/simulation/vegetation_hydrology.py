"""Novel coupled Vegetation-Hydrology simulation (4D).

Couples vegetation-soil moisture dynamics (Rietkerk-type) with
a simplified groundwater model. Vegetation increases infiltration
but also consumes water, creating positive feedback loops that
drive dryland pattern formation.

State: [V, W, G, R] where:
  V = vegetation biomass density
  W = soil moisture (root zone)
  G = groundwater level
  R = surface runoff/overland flow
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class VegetationHydrologySimulation(SimulationEnvironment):
    """Coupled vegetation-soil moisture-groundwater simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.c_v = p.get("c_v", 0.5)
        self.d_v = p.get("d_v", 0.1)
        self.alpha_vw = p.get("alpha_vw", 0.8)
        self.K_w = p.get("K_w", 1.0)

        self.P_rain = p.get("P_rain", 0.5)
        self.d_w = p.get("d_w", 0.2)
        self.eta_wg = p.get("eta_wg", 0.1)
        self.infilt = p.get("infilt", 0.3)

        self.d_g = p.get("d_g", 0.05)
        self.eta_gw = p.get("eta_gw", 0.05)
        self.d_r = p.get("d_r", 0.3)

        self.coupling_vw = p.get("coupling_vw", 0.5)

        self.V_0 = p.get("V_0", 1.0)
        self.W_0 = p.get("W_0", 1.0)
        self.G_0 = p.get("G_0", 0.5)
        self.R_0 = p.get("R_0", 0.2)

        self.dt = config.dt
        self._state = np.array([self.V_0, self.W_0, self.G_0, self.R_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        V, W, G, R = state
        infilt_eff = self.infilt * (1.0 + self.coupling_vw * V)
        uptake = self.alpha_vw * V * W / (self.K_w + W)

        dV = self.c_v * uptake - self.d_v * V
        dW = infilt_eff * R - uptake - self.d_w * W - self.eta_wg * W + self.eta_gw * G
        dG = self.eta_wg * W - self.eta_gw * G - self.d_g * G
        dR = self.P_rain - infilt_eff * R - self.d_r * R

        return np.array([dV, dW, dG, dR])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.V_0 + rng.normal(0, 0.2), 0.01),
                max(self.W_0 + rng.normal(0, 0.2), 0.01),
                max(self.G_0 + rng.normal(0, 0.1), 0.01),
                max(self.R_0 + rng.normal(0, 0.05), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.V_0, self.W_0, self.G_0, self.R_0], dtype=np.float64)
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
