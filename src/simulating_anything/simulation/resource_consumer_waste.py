"""Novel coupled Resource-Consumer-Waste simulation (4D).

Chemostat-like resource-consumer dynamics with waste product
inhibition. Consumer grows on resource via Monod kinetics,
producing waste that inhibits its own growth (product inhibition).
Resource is continuously supplied and diluted.

State: [R, C, W, P] where:
  R = resource concentration
  C = consumer (biomass) concentration
  W = waste product concentration
  P = product inhibition factor (1 / (1 + k_w * W))
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ResourceConsumerWasteSimulation(SimulationEnvironment):
    """Chemostat + waste product inhibition dynamics."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.R_in = p.get("R_in", 5.0)
        self.D = p.get("D_dilution", 0.2)
        self.mu_max = p.get("mu_max", 1.0)
        self.K_s = p.get("K_s", 0.5)
        self.Y = p.get("Y_yield", 0.5)
        self.k_w = p.get("k_w", 0.5)
        self.alpha_w = p.get("alpha_w", 0.3)
        self.d_w = p.get("d_w", 0.1)

        self.R_0 = p.get("R_0", 3.0)
        self.C_0 = p.get("C_0", 1.0)
        self.W_0 = p.get("W_0", 0.5)

        self.dt = config.dt
        P_0 = 1.0 / (1.0 + self.k_w * self.W_0)
        self._state = np.array([self.R_0, self.C_0, self.W_0, P_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        R, C, W, _ = state
        P = 1.0 / (1.0 + self.k_w * W)
        mu = self.mu_max * R / (self.K_s + R) * P

        dR = self.D * (self.R_in - R) - mu * C / self.Y
        dC = (mu - self.D) * C
        dW = self.alpha_w * mu * C - self.d_w * W - self.D * W
        dP = P - state[3]

        return np.array([dR, dC, dW, dP])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            W = max(self.W_0 + rng.normal(0, 0.1), 0.01)
            self._state = np.array([
                max(self.R_0 + rng.normal(0, 0.5), 0.01),
                max(self.C_0 + rng.normal(0, 0.2), 0.01),
                W,
                1.0 / (1.0 + self.k_w * W),
            ], dtype=np.float64)
        else:
            P = 1.0 / (1.0 + self.k_w * self.W_0)
            self._state = np.array([self.R_0, self.C_0, self.W_0, P], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[:3] = np.clip(self._state[:3], 0.0, None)
        self._state[3] = 1.0 / (1.0 + self.k_w * self._state[2])
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
