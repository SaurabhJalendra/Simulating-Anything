"""Novel coupled Infection-Immunity simulation (4D).

SIR epidemic with waning immunity and immune memory dynamics.
Recovered individuals gradually lose immunity, becoming susceptible
again. The rate of waning depends on immune memory strength, which
is boosted by re-exposure. Creates endemic oscillations and
reinfection waves.

State: [S, I, R, M] where:
  S = susceptible fraction
  I = infected fraction
  R = recovered (immune) fraction
  M = immune memory strength (0 = naive, 1 = fully boosted)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class InfectionImmunitySimulation(SimulationEnvironment):
    """SIR with waning immunity and immune memory."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.beta = p.get("beta", 0.4)
        self.gamma_epi = p.get("gamma_epi", 0.1)
        self.mu = p.get("mu_pop", 0.01)
        self.omega_base = p.get("omega_base", 0.02)  # base waning rate
        self.boost_rate = p.get("boost_rate", 0.5)    # immune boost on re-exposure
        self.memory_decay = p.get("memory_decay", 0.01)  # memory loss rate

        self.coupling_MR = p.get("coupling_MR", 2.0)  # memory slows waning

        self.S_0 = p.get("S_0", 0.6)
        self.I_0 = p.get("I_0", 0.05)
        self.R_0_init = p.get("R_0_init", 0.34)
        self.M_0 = p.get("M_0", 0.3)

        self.dt = config.dt
        self._state = np.array([self.S_0, self.I_0, self.R_0_init, self.M_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        S, I, R, M = state
        # Waning rate decreases with immune memory
        omega = self.omega_base / (1.0 + self.coupling_MR * M)

        dS = -self.beta * S * I + omega * R + self.mu * (1.0 - S)
        dI = self.beta * S * I - self.gamma_epi * I - self.mu * I
        dR = self.gamma_epi * I - omega * R - self.mu * R
        # Memory boosted by infection exposure, decays otherwise
        dM = self.boost_rate * I * (1.0 - M) - self.memory_decay * M

        return np.array([dS, dI, dR, dM])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                np.clip(self.S_0 + rng.normal(0, 0.03), 0.01, 0.99),
                np.clip(self.I_0 + rng.normal(0, 0.01), 0.001, 0.2),
                np.clip(self.R_0_init + rng.normal(0, 0.03), 0.01, 0.99),
                np.clip(self.M_0 + rng.normal(0, 0.05), 0.0, 1.0),
            ], dtype=np.float64)
            total = self._state[0] + self._state[1] + self._state[2]
            self._state[:3] /= total
        else:
            self._state = np.array([self.S_0, self.I_0, self.R_0_init, self.M_0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[:3] = np.clip(self._state[:3], 0.0, 1.0)
        self._state[3] = np.clip(self._state[3], 0.0, 1.0)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
