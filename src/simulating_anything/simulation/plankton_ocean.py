"""Novel coupled Plankton-Ocean simulation (4D).

Couples phytoplankton-zooplankton (NPZ-type) dynamics with nutrient
cycling driven by ocean mixing. Nutrient upwelling fuels phytoplankton
blooms that are grazed by zooplankton, while nutrient depletion and
sinking create bloom-bust cycles modulated by ocean physics.

State: [P, Z, N, D] where:
  P = phytoplankton concentration
  Z = zooplankton concentration
  N = dissolved nutrient concentration
  D = detritus (dead organic matter) concentration

Coupling:
  - Nutrients N fuel phytoplankton growth (Monod kinetics)
  - Phytoplankton P is grazed by zooplankton Z (Holling III)
  - Dead matter D remineralizes back to nutrients N
  - Ocean mixing supplies deep nutrients (upwelling term)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PlanktonOceanSimulation(SimulationEnvironment):
    """Coupled NPZ-D plankton + ocean nutrient cycling simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Phytoplankton parameters
        self.mu_p = p.get("mu_p", 1.5)       # max phyto growth rate
        self.K_n = p.get("K_n", 0.5)         # nutrient half-saturation
        self.m_p = p.get("m_p", 0.1)         # phyto natural mortality

        # Zooplankton parameters
        self.g_max = p.get("g_max", 0.8)     # max grazing rate
        self.K_p = p.get("K_p", 1.0)         # grazing half-saturation
        self.beta_z = p.get("beta_z", 0.6)   # grazing efficiency
        self.m_z = p.get("m_z", 0.15)        # zoo mortality

        # Nutrient/detritus parameters
        self.N_deep = p.get("N_deep", 5.0)   # deep ocean nutrient concentration
        self.w_mix = p.get("w_mix", 0.05)    # mixing/upwelling rate
        self.r_remin = p.get("r_remin", 0.1)  # remineralization rate
        self.w_sink = p.get("w_sink", 0.05)  # sinking rate

        # Coupling: seasonal light modulation
        self.light_amp = p.get("light_amp", 0.3)   # seasonal light amplitude
        self.light_freq = p.get("light_freq", 0.017)  # ~annual frequency

        # Initial conditions
        self.P_0 = p.get("P_0", 1.0)
        self.Z_0 = p.get("Z_0", 0.5)
        self.N_0 = p.get("N_0", 3.0)
        self.D_0 = p.get("D_0", 0.5)

        self.dt = config.dt
        self._state = np.array(
            [self.P_0, self.Z_0, self.N_0, self.D_0], dtype=np.float64,
        )
        self._t = 0.0

    def _rhs(self, state: np.ndarray, t: float) -> np.ndarray:
        P, Z, N, D = state

        # Seasonal light modulation
        light = 1.0 + self.light_amp * np.sin(2 * np.pi * self.light_freq * t)

        # Monod nutrient limitation
        nutrient_lim = N / (self.K_n + N)

        # Phytoplankton growth
        growth = self.mu_p * light * nutrient_lim * P

        # Holling Type III grazing
        grazing = self.g_max * P ** 2 / (self.K_p ** 2 + P ** 2) * Z

        # Equations
        dP = growth - grazing - self.m_p * P
        dZ = self.beta_z * grazing - self.m_z * Z
        dN = (-growth + self.r_remin * D
              + self.w_mix * (self.N_deep - N))
        dD = ((1.0 - self.beta_z) * grazing + self.m_p * P + self.m_z * Z
              - self.r_remin * D - self.w_sink * D)

        return np.array([dP, dZ, dN, dD])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.P_0 + rng.normal(0, 0.2), 0.01),
                max(self.Z_0 + rng.normal(0, 0.1), 0.01),
                max(self.N_0 + rng.normal(0, 0.5), 0.01),
                max(self.D_0 + rng.normal(0, 0.1), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array(
                [self.P_0, self.Z_0, self.N_0, self.D_0], dtype=np.float64,
            )
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state = self._state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [P, Z, N, D]."""
        return self._state.copy()
