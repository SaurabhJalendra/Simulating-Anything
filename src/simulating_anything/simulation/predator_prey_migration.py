"""Novel coupled Predator-Prey-Migration simulation (4D).

Two-patch Lotka-Volterra predator-prey with density-dependent
migration. Prey migrate from high-density to low-density patches,
creating metapopulation dynamics with asynchronous oscillations
between patches.

State: [N1, P1, N2, P2] where:
  N1, P1 = prey and predator in patch 1
  N2, P2 = prey and predator in patch 2
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorPreyMigrationSimulation(SimulationEnvironment):
    """Two-patch LV with density-dependent migration."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.r = p.get("r", 1.0)
        self.K = p.get("K", 15.0)
        self.a = p.get("a_pred", 0.4)
        self.e = p.get("e_pred", 0.3)
        self.d = p.get("d_pred", 0.2)

        self.m_n = p.get("m_n", 0.1)   # prey migration rate
        self.m_p = p.get("m_p", 0.05)  # predator migration rate

        self.N1_0 = p.get("N1_0", 10.0)
        self.P1_0 = p.get("P1_0", 3.0)
        self.N2_0 = p.get("N2_0", 5.0)
        self.P2_0 = p.get("P2_0", 1.0)

        self.dt = config.dt
        self._state = np.array([self.N1_0, self.P1_0, self.N2_0, self.P2_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        N1, P1, N2, P2 = state

        # Density-dependent migration (move from high to low density)
        mig_N = self.m_n * (N1 - N2)
        mig_P = self.m_p * (P1 - P2)

        # Patch 1
        dN1 = self.r * N1 * (1.0 - N1 / self.K) - self.a * N1 * P1 - mig_N
        dP1 = self.e * self.a * N1 * P1 - self.d * P1 - mig_P

        # Patch 2
        dN2 = self.r * N2 * (1.0 - N2 / self.K) - self.a * N2 * P2 + mig_N
        dP2 = self.e * self.a * N2 * P2 - self.d * P2 + mig_P

        return np.array([dN1, dP1, dN2, dP2])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.N1_0 + rng.normal(0, 1.0), 0.1),
                max(self.P1_0 + rng.normal(0, 0.5), 0.1),
                max(self.N2_0 + rng.normal(0, 1.0), 0.1),
                max(self.P2_0 + rng.normal(0, 0.3), 0.1),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.N1_0, self.P1_0, self.N2_0, self.P2_0], dtype=np.float64)
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
