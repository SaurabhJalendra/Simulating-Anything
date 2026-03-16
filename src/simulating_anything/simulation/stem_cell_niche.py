"""Novel Stem Cell Niche simulation (4D).

Coupled stem cell self-renewal with differentiation and niche
signaling. Stem cells divide symmetrically (self-renew) or
asymmetrically (differentiate). Niche signals regulate the
balance. Differentiated cells provide negative feedback.

State: [S, D, N_sig, F] where:
  S = stem cell count
  D = differentiated cell count
  N_sig = niche signal strength
  F = feedback signal from differentiated cells
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class StemCellNicheSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r_s = p.get("r_s", 0.1)
        self.p_self = p.get("p_self", 0.5)
        self.d_s = p.get("d_s", 0.01)
        self.d_d = p.get("d_d", 0.05)
        self.alpha_n = p.get("alpha_n", 0.3)
        self.K_n = p.get("K_n", 1.0)
        self.beta_f = p.get("beta_f", 0.1)
        self.d_n = p.get("d_n", 0.2)
        self.d_f = p.get("d_f", 0.1)
        self.dt = config.dt
        self._state = np.array([p.get("S_0", 10.0), p.get("D_0", 50.0),
                                p.get("Nsig_0", 0.5), p.get("F_0", 0.3)], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        S, D, N, F = state
        p_eff = self.p_self * N / (self.K_n + N) * 1.0 / (1.0 + self.beta_f * F)
        dS = self.r_s * S * (2 * p_eff - 1) - self.d_s * S
        dD = self.r_s * S * 2 * (1 - p_eff) - self.d_d * D
        dN = self.alpha_n - self.d_n * N - 0.01 * N * S
        dF = 0.05 * D - self.d_f * F
        return np.array([dS, dD, dN, dF])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(10 + rng.normal(0, 2), 0.1), max(50 + rng.normal(0, 5), 1),
                max(0.5 + rng.normal(0, 0.1), 0.01), max(0.3 + rng.normal(0, 0.05), 0.01)
            ], dtype=np.float64)
        else:
            self._state = np.array([10.0, 50.0, 0.5, 0.3], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
