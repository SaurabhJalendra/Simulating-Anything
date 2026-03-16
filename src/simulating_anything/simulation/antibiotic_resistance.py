"""Novel Antibiotic Resistance simulation (4D).

Coupled sensitive-resistant bacterial competition under antibiotic
treatment. Sensitive bacteria grow faster without antibiotics but
are killed by treatment. Resistant bacteria have a fitness cost but
survive treatment. Antibiotic concentration decays pharmacokinetically.

State: [S_b, R_b, A, N_tot] where:
  S_b = sensitive bacteria density
  R_b = resistant bacteria density
  A = antibiotic concentration
  N_tot = total nutrient (shared resource)
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AntibioticResistanceSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r_s = p.get("r_s", 1.0)
        self.r_r = p.get("r_r", 0.7)
        self.K = p.get("K", 100.0)
        self.k_kill = p.get("k_kill", 0.5)
        self.MIC = p.get("MIC", 1.0)
        self.mu_sr = p.get("mu_sr", 0.001)
        self.d_a = p.get("d_a", 0.1)
        self.A_dose = p.get("A_dose", 0.0)
        self.dose_freq = p.get("dose_freq", 0.01)
        self.N_supply = p.get("N_supply", 5.0)
        self.d_n = p.get("d_n", 0.1)
        self.dt = config.dt
        self._state = np.array([p.get("Sb_0", 50.0), p.get("Rb_0", 1.0),
                                p.get("A_0", 0.0), self.N_supply / self.d_n], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        Sb, Rb, A, N = state
        total = Sb + Rb + 1e-10
        kill = self.k_kill * A / (self.MIC + A)
        dose = self.A_dose if np.sin(2 * np.pi * self.dose_freq * t) > 0.9 else 0.0

        dSb = self.r_s * Sb * (1.0 - total / self.K) * N / (1.0 + N) - kill * Sb - self.mu_sr * Sb
        dRb = self.r_r * Rb * (1.0 - total / self.K) * N / (1.0 + N) + self.mu_sr * Sb
        dA = dose - self.d_a * A
        dN = self.N_supply - self.d_n * N - 0.01 * (self.r_s * Sb + self.r_r * Rb) * N / (1.0 + N)

        return np.array([dSb, dRb, dA, dN])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(50.0 + rng.normal(0, 5.0), 0.1),
                max(1.0 + rng.normal(0, 0.3), 0.01),
                0.0,
                self.N_supply / self.d_n + rng.normal(0, 1.0),
            ], dtype=np.float64)
        else:
            self._state = np.array([50.0, 1.0, 0.0, self.N_supply / self.d_n], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
