"""Novel coupled Circadian-Metabolism simulation (4D).

Couples a Goodwin-type circadian clock oscillator with a simple
metabolic flux model. Clock proteins regulate metabolic enzyme
expression on a ~24h cycle, while metabolite levels feed back
to modulate clock gene transcription.

State: [M, P_c, E, S] where:
  M = clock mRNA concentration
  P_c = clock protein concentration
  E = metabolic enzyme concentration (regulated by clock)
  S = substrate (metabolite) concentration

Coupling:
  - Clock protein P_c activates enzyme E transcription
  - Substrate S inhibits clock mRNA M transcription (metabolic feedback)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CircadianMetabolismSimulation(SimulationEnvironment):
    """Coupled Goodwin circadian clock + metabolic flux."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Goodwin oscillator (circadian clock)
        self.v_m = p.get("v_m", 1.0)         # max mRNA synthesis rate
        self.K_m = p.get("K_m", 0.5)         # mRNA degradation rate
        self.v_p = p.get("v_p", 0.5)         # protein synthesis rate
        self.K_p = p.get("K_p", 0.3)         # protein degradation rate
        self.n_hill = p.get("n_hill", 4.0)   # Hill coefficient (oscillation)
        self.K_hill = p.get("K_hill", 1.0)   # Hill half-saturation

        # Metabolic flux
        self.v_enz = p.get("v_enz", 0.8)     # enzyme catalytic rate
        self.K_enz = p.get("K_enz", 0.3)     # enzyme degradation rate
        self.S_input = p.get("S_input", 0.5)  # substrate supply rate
        self.K_sub = p.get("K_sub", 0.5)     # substrate Michaelis constant
        self.K_sdeg = p.get("K_sdeg", 0.1)   # substrate degradation rate

        # Coupling
        self.coupling_PE = p.get("coupling_PE", 0.3)   # clock -> enzyme
        self.coupling_SM = p.get("coupling_SM", 0.2)   # metabolite -> clock

        # Initial conditions
        self.M_0 = p.get("M_0", 1.0)
        self.Pc_0 = p.get("Pc_0", 0.5)
        self.E_0 = p.get("E_0", 0.5)
        self.S_0 = p.get("S_0", 1.0)

        self.dt = config.dt
        self._state = np.array([self.M_0, self.Pc_0, self.E_0, self.S_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        M, Pc, E, S = state

        # Hill inhibition by clock protein (negative feedback loop)
        hill_inhib = 1.0 / (1.0 + (Pc / self.K_hill) ** self.n_hill)

        # Metabolite inhibition of clock transcription
        met_inhib = 1.0 / (1.0 + self.coupling_SM * S)

        # Clock dynamics (Goodwin with metabolic feedback)
        dM = self.v_m * hill_inhib * met_inhib - self.K_m * M
        dPc = self.v_p * M - self.K_p * Pc

        # Enzyme dynamics (clock-regulated)
        dE = self.coupling_PE * Pc - self.K_enz * E

        # Substrate dynamics (enzyme-catalyzed conversion)
        flux = self.v_enz * E * S / (self.K_sub + S)
        dS = self.S_input - flux - self.K_sdeg * S

        return np.array([dM, dPc, dE, dS])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.M_0 + rng.normal(0, 0.2), 0.01),
                max(self.Pc_0 + rng.normal(0, 0.1), 0.01),
                max(self.E_0 + rng.normal(0, 0.1), 0.01),
                max(self.S_0 + rng.normal(0, 0.2), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.M_0, self.Pc_0, self.E_0, self.S_0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        return self._state.copy()
