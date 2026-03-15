"""Novel coupled Tumor-Immune simulation (4D).

Couples tumor growth dynamics with immune response, modeling the
complex interplay between cancer proliferation and adaptive immunity.
Based on de Pillis-Radunskaya framework with novel coupling terms.

State: [T, N, I, C] where:
  T = tumor cell population
  N = natural killer (NK) cell population
  I = CD8+ T cell (adaptive immune) population
  C = circulating cytokine concentration

Coupling:
  - Tumor growth stimulates immune activation (antigen presentation)
  - Immune cells kill tumor cells (cytotoxic response)
  - Tumor secretes immunosuppressive cytokines (immune evasion)
  - Cytokines modulate both immune activation and tumor growth
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TumorImmuneSimulation(SimulationEnvironment):
    """Coupled tumor growth + immune response simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Tumor parameters
        self.a_t = p.get("a_t", 0.2)       # tumor growth rate
        self.K_t = p.get("K_t", 100.0)     # tumor carrying capacity
        self.d_tn = p.get("d_tn", 0.02)    # NK killing rate
        self.d_ti = p.get("d_ti", 0.03)    # CD8+ killing rate

        # NK cell parameters
        self.s_n = p.get("s_n", 0.3)       # NK production rate
        self.d_n = p.get("d_n", 0.1)       # NK death rate
        self.p_n = p.get("p_n", 0.01)      # NK stimulation by tumor

        # CD8+ T cell parameters
        self.s_i = p.get("s_i", 0.1)       # CD8+ baseline production
        self.d_i = p.get("d_i", 0.05)      # CD8+ death rate
        self.p_i = p.get("p_i", 0.005)     # CD8+ activation by tumor antigen
        self.g_i = p.get("g_i", 2.0)       # CD8+ half-saturation constant

        # Cytokine parameters
        self.s_c = p.get("s_c", 0.05)      # cytokine production by immune cells
        self.d_c = p.get("d_c", 0.5)       # cytokine decay rate
        self.alpha_c = p.get("alpha_c", 0.1)  # immunosuppressive effect of cytokine

        # Coupling: cytokine-mediated immune evasion
        self.coupling_ct = p.get("coupling_ct", 0.05)  # cytokine suppresses immune

        # Initial conditions
        self.T_0 = p.get("T_0", 1.0)
        self.N_0 = p.get("N_0", 3.0)
        self.I_0 = p.get("I_0", 1.0)
        self.C_0 = p.get("C_0", 0.5)

        self.dt = config.dt
        self._state = np.array(
            [self.T_0, self.N_0, self.I_0, self.C_0], dtype=np.float64,
        )
        self._t = 0.0

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        T, N, I, C = state

        # Immune suppression by cytokines (tumor-produced)
        immune_suppression = 1.0 / (1.0 + self.coupling_ct * C)

        # Tumor dynamics: logistic growth - immune killing
        dT = (self.a_t * T * (1.0 - T / self.K_t)
              - self.d_tn * N * T * immune_suppression
              - self.d_ti * I * T * immune_suppression)

        # NK cell dynamics: production + tumor stimulation - death
        dN = self.s_n + self.p_n * T * N / (1.0 + T) - self.d_n * N

        # CD8+ T cell dynamics: antigen-activated expansion
        dI = (self.s_i + self.p_i * T * I / (self.g_i + T)
              - self.d_i * I
              - self.alpha_c * C * I)

        # Cytokine dynamics: produced by immune, enhanced by tumor, decays
        dC = self.s_c * (N + I) * T / (1.0 + T) - self.d_c * C

        return np.array([dT, dN, dI, dC])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.T_0 + rng.normal(0, 0.3), 0.01),
                max(self.N_0 + rng.normal(0, 0.5), 0.1),
                max(self.I_0 + rng.normal(0, 0.2), 0.1),
                max(self.C_0 + rng.normal(0, 0.1), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array(
                [self.T_0, self.N_0, self.I_0, self.C_0], dtype=np.float64,
            )
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state = self._state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [T, N, I, C]."""
        return self._state.copy()
