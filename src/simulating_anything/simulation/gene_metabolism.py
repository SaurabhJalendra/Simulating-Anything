"""Novel coupled Gene-Regulation-Metabolism simulation (4D).

Couples a simplified gene regulatory toggle switch with a metabolic
flux oscillator. Gene expression controls enzyme levels that modulate
metabolic rates, while metabolite concentrations feed back to regulate
gene expression.

State: [g1, g2, m1, m2] where:
  g1 = gene 1 expression level (represses g2)
  g2 = gene 2 expression level (represses g1)
  m1 = metabolite 1 concentration (produced by enzyme from g1)
  m2 = metabolite 2 concentration (produced by enzyme from g2)

Coupling:
  - Gene expression g1/g2 produces enzymes that convert metabolites
  - Metabolite m2 activates g1 transcription (positive feedback loop)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GeneMetabolismSimulation(SimulationEnvironment):
    """Coupled gene toggle switch + metabolic flux simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Gene regulation parameters (toggle switch)
        self.alpha1 = p.get("alpha1", 3.0)   # max transcription rate g1
        self.alpha2 = p.get("alpha2", 3.0)   # max transcription rate g2
        self.n = p.get("hill_n", 2.0)        # Hill coefficient
        self.K = p.get("hill_K", 1.0)        # Hill half-saturation
        self.delta_g = p.get("delta_g", 0.5)  # gene product degradation

        # Metabolic parameters
        self.v_max1 = p.get("v_max1", 1.0)   # max metabolic rate (g1 enzyme)
        self.v_max2 = p.get("v_max2", 1.0)   # max metabolic rate (g2 enzyme)
        self.Km = p.get("Km", 0.5)           # Michaelis-Menten constant
        self.delta_m = p.get("delta_m", 0.3)  # metabolite degradation
        self.m_supply = p.get("m_supply", 0.5)  # metabolite supply rate

        # Coupling parameters
        self.coupling_gm = p.get("coupling_gm", 0.5)  # gene -> metabolite
        self.coupling_mg = p.get("coupling_mg", 0.3)   # metabolite -> gene

        # Initial conditions
        self.g1_0 = p.get("g1_0", 2.0)
        self.g2_0 = p.get("g2_0", 0.5)
        self.m1_0 = p.get("m1_0", 1.0)
        self.m2_0 = p.get("m2_0", 0.5)

        self.dt = config.dt
        self._state = np.array(
            [self.g1_0, self.g2_0, self.m1_0, self.m2_0], dtype=np.float64,
        )
        self._t = 0.0

    def _hill_inhibition(self, x: float) -> float:
        """Hill function for transcriptional repression."""
        return 1.0 / (1.0 + (x / self.K) ** self.n)

    def _michaelis_menten(self, s: float) -> float:
        """Michaelis-Menten enzyme kinetics."""
        return s / (self.Km + s)

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        g1, g2, m1, m2 = state

        # Metabolite m2 activates g1 transcription (coupling: met -> gene)
        activation = 1.0 + self.coupling_mg * m2

        # Gene regulation: mutual repression (toggle switch) + metabolic feedback
        dg1 = (self.alpha1 * self._hill_inhibition(g2) * activation
               - self.delta_g * g1)
        dg2 = (self.alpha2 * self._hill_inhibition(g1)
               - self.delta_g * g2)

        # Metabolic flux: enzyme (from gene) converts substrate
        flux1 = self.coupling_gm * g1 * self.v_max1 * self._michaelis_menten(m1)
        flux2 = self.coupling_gm * g2 * self.v_max2 * self._michaelis_menten(m2)

        dm1 = self.m_supply - flux1 - self.delta_m * m1
        dm2 = flux1 - flux2 - self.delta_m * m2  # m1 -> m2 -> degraded

        return np.array([dg1, dg2, dm1, dm2])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.g1_0 + rng.normal(0, 0.3), 0.01),
                max(self.g2_0 + rng.normal(0, 0.1), 0.01),
                max(self.m1_0 + rng.normal(0, 0.2), 0.01),
                max(self.m2_0 + rng.normal(0, 0.1), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array(
                [self.g1_0, self.g2_0, self.m1_0, self.m2_0], dtype=np.float64,
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
        """Return current state [g1, g2, m1, m2]."""
        return self._state.copy()
