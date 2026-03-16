"""Novel Immune-Virus dynamics simulation (4D).

Coupled viral replication with adaptive immune response including
antibody production. Virus infects target cells, immune system
produces antibodies that neutralize virus. Creates acute infection
kinetics with potential for chronic equilibrium.

State: [V, T_c, A, I_c] where:
  V = free virus particles
  T_c = target (uninfected) cells
  A = antibody concentration
  I_c = infected cells
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ImmuneVirusSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta_v = p.get("beta_v", 0.01)
        self.delta_i = p.get("delta_i", 0.5)
        self.p_v = p.get("p_v", 100.0)
        self.c_v = p.get("c_v", 3.0)
        self.s_t = p.get("s_t", 10.0)
        self.d_t = p.get("d_t", 0.01)
        self.s_a = p.get("s_a", 0.1)
        self.d_a = p.get("d_a", 0.05)
        self.k_neut = p.get("k_neut", 0.5)
        self.k_stim = p.get("k_stim", 0.01)
        self.dt = config.dt
        self._state = np.array([p.get("V_0", 1.0), p.get("Tc_0", 1000.0),
                                p.get("A_0", 0.1), p.get("Ic_0", 0.0)], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        V, Tc, A, Ic = state
        infection = self.beta_v * V * Tc
        dV = self.p_v * Ic - self.c_v * V - self.k_neut * A * V
        dTc = self.s_t - self.d_t * Tc - infection
        dA = self.s_a + self.k_stim * V * A / (1.0 + V) - self.d_a * A
        dIc = infection - self.delta_i * Ic
        return np.array([dV, dTc, dA, dIc])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(1.0 + rng.normal(0, 0.5), 0.01),
                max(1000.0 + rng.normal(0, 50.0), 100.0),
                max(0.1 + rng.normal(0, 0.02), 0.01),
                max(rng.normal(0, 0.1), 0.0),
            ], dtype=np.float64)
        else:
            self._state = np.array([1.0, 1000.0, 0.1, 0.0], dtype=np.float64)
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
