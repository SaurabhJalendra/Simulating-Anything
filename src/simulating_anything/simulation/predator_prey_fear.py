"""Novel coupled Predator-Prey with Fear Effect simulation (4D).

Predator-prey dynamics where prey reduce foraging activity in the
presence of predators (landscape of fear). Fear reduces prey growth
rate but also reduces predation risk. Creates a trade-off between
growth and safety.

State: [N, P, F, E] where:
  N = prey density
  P = predator density
  F = fear level (prey behavioral response, 0-1)
  E = ecosystem energy (total biomass proxy)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorPreyFearSimulation(SimulationEnvironment):
    """LV with fear-mediated behavioral modification."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.r = p.get("r", 1.0)
        self.K = p.get("K", 15.0)
        self.a = p.get("a_pred", 0.5)
        self.e = p.get("e_pred", 0.3)
        self.d = p.get("d_pred", 0.2)
        self.k_fear = p.get("k_fear", 0.5)
        self.tau_f = p.get("tau_f", 1.0)
        self.fear_growth = p.get("fear_growth", 0.3)
        self.fear_pred = p.get("fear_pred", 0.4)

        self.N_0 = p.get("N_0", 8.0)
        self.P_0 = p.get("P_0", 3.0)
        self.F_0 = p.get("F_0", 0.3)

        self.dt = config.dt
        E_0 = self.N_0 + self.P_0
        self._state = np.array([self.N_0, self.P_0, self.F_0, E_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        N, P, F, _ = state
        r_eff = self.r * (1.0 - self.fear_growth * F)
        a_eff = self.a * (1.0 - self.fear_pred * F)

        dN = r_eff * N * (1.0 - N / self.K) - a_eff * N * P
        dP = self.e * a_eff * N * P - self.d * P
        dF = (self.k_fear * P / (1.0 + P) - F) / self.tau_f
        dE = (N + P) - state[3]

        return np.array([dN, dP, dF, dE])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            N = max(self.N_0 + rng.normal(0, 1.0), 0.1)
            P = max(self.P_0 + rng.normal(0, 0.5), 0.1)
            F = np.clip(self.F_0 + rng.normal(0, 0.05), 0.0, 1.0)
            self._state = np.array([N, P, F, N + P], dtype=np.float64)
        else:
            self._state = np.array([self.N_0, self.P_0, self.F_0, self.N_0 + self.P_0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[:2] = np.clip(self._state[:2], 0.0, None)
        self._state[2] = np.clip(self._state[2], 0.0, 1.0)
        self._state[3] = self._state[0] + self._state[1]
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
