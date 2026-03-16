"""Novel coupled Atmosphere-Vegetation simulation (4D).

Couples simplified atmospheric convection (Lorenz-like 2D) with
vegetation dynamics. Vegetation transpiration cools the surface
and adds moisture, modulating convective instability. Rainfall
from convection waters vegetation. Creates land-atmosphere feedback
loops central to climate science.

State: [X, Z, V, M] where:
  X = atmospheric temperature anomaly (convective mode)
  Z = vertical temperature gradient
  V = vegetation biomass
  M = soil moisture
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AtmosphereVegetationSimulation(SimulationEnvironment):
    """Coupled atmospheric convection + vegetation transpiration."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.sigma_a = p.get("sigma_a", 5.0)
        self.rho_a = p.get("rho_a", 15.0)
        self.beta_a = p.get("beta_a", 1.0)

        self.r_v = p.get("r_v", 0.3)
        self.K_v = p.get("K_v", 5.0)
        self.d_v = p.get("d_v", 0.05)
        self.alpha_vm = p.get("alpha_vm", 0.5)

        self.P_base = p.get("P_base", 0.3)
        self.evap = p.get("evap", 0.1)
        self.d_m = p.get("d_m", 0.05)

        self.coupling_VX = p.get("coupling_VX", 0.2)
        self.coupling_XM = p.get("coupling_XM", 0.1)

        self.X_0 = p.get("X_0", 1.0)
        self.Z_0 = p.get("Z_0", 1.0)
        self.V_0 = p.get("V_0", 2.0)
        self.M_0 = p.get("M_0", 1.0)

        self.dt = config.dt
        self._state = np.array([self.X_0, self.Z_0, self.V_0, self.M_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        X, Z, V, M = state
        rho_eff = self.rho_a - self.coupling_VX * V
        rain = self.P_base + self.coupling_XM * max(X, 0)

        dX = self.sigma_a * (-X + rho_eff * Z / (1.0 + Z ** 2))
        dZ = -self.beta_a * Z + X
        dV = self.r_v * V * M / (self.alpha_vm + M) * (1.0 - V / self.K_v) - self.d_v * V
        dM = rain - self.evap * V * M / (self.alpha_vm + M) - self.d_m * M

        return np.array([dX, dZ, dV, dM])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                self.X_0 + rng.normal(0, 0.3),
                self.Z_0 + rng.normal(0, 0.2),
                max(self.V_0 + rng.normal(0, 0.3), 0.1),
                max(self.M_0 + rng.normal(0, 0.2), 0.1),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.X_0, self.Z_0, self.V_0, self.M_0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[2:] = np.clip(self._state[2:], 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
