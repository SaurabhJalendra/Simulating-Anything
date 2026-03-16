"""Novel Supply Chain Dynamics simulation (4D).

Coupled inventory-demand-production-shipping dynamics modeled as
a simplified bullwhip effect system. Small demand fluctuations
amplify upstream through the supply chain, creating inventory
oscillations. Key problem in operations research.

State: [I, D, P, B] where:
  I = inventory level
  D = demand rate (stochastic + trend)
  P = production rate
  B = backlog (unfulfilled orders)
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SupplyChainSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.D_base = p.get("D_base", 10.0)
        self.D_amp = p.get("D_amp", 2.0)
        self.D_freq = p.get("D_freq", 0.02)
        self.tau_p = p.get("tau_p", 5.0)
        self.tau_i = p.get("tau_i", 2.0)
        self.I_target = p.get("I_target", 20.0)
        self.alpha_bull = p.get("alpha_bull", 2.0)
        self.max_prod = p.get("max_prod", 20.0)
        self.dt = config.dt
        self._state = np.array([self.I_target, self.D_base, self.D_base, 0.0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        I, D, P, B = state
        D_actual = self.D_base + self.D_amp * np.sin(2 * np.pi * self.D_freq * t)
        gap = self.I_target - I + self.alpha_bull * B
        P_target = D_actual + gap / self.tau_i
        P_target = np.clip(P_target, 0, self.max_prod)

        dI = P - D_actual
        dD = D_actual - D
        dP = (P_target - P) / self.tau_p
        dB = max(D_actual - P, 0) - 0.5 * B

        return np.array([dI, dD, dP, dB])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.I_target + rng.normal(0, 2.0), 1.0),
                self.D_base + rng.normal(0, 0.5),
                self.D_base + rng.normal(0, 0.5),
                max(rng.normal(0, 0.5), 0),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.I_target, self.D_base, self.D_base, 0.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[3] = max(self._state[3], 0)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
