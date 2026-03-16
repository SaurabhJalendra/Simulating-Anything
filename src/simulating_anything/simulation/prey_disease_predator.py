"""Novel coupled Prey-Disease-Predator simulation (4D).

Classic eco-epidemiological model where disease in the prey makes
infected individuals easier to catch by predators. Susceptible and
infected prey have different predation rates, creating complex
dynamics where disease can either stabilize or destabilize the
predator-prey oscillation.

State: [Xs, Xi, Y, Z] where:
  Xs = susceptible prey density
  Xi = infected prey density
  Y = predator density
  Z = disease prevalence indicator (Xi / (Xs + Xi))

Coupling:
  - Infected prey Xi are caught at higher rate (reduced escape)
  - Predators preferentially eat infected prey (selective predation)
  - Disease spreads via density-dependent contact within prey
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PreyDiseasePredatorSimulation(SimulationEnvironment):
    """Eco-epidemiological predator-prey with disease in prey."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.r = p.get("r", 1.0)
        self.K = p.get("K", 20.0)
        self.beta_d = p.get("beta_d", 0.3)
        self.gamma_d = p.get("gamma_d", 0.1)
        self.alpha_d = p.get("alpha_d", 0.05)

        self.a_s = p.get("a_s", 0.3)
        self.a_i = p.get("a_i", 0.6)
        self.e = p.get("e_pred", 0.3)
        self.d = p.get("d_pred", 0.2)

        self.Xs_0 = p.get("Xs_0", 10.0)
        self.Xi_0 = p.get("Xi_0", 1.0)
        self.Y_0 = p.get("Y_0", 3.0)

        self.dt = config.dt
        self._state = np.array([self.Xs_0, self.Xi_0, self.Y_0, 0.0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        Xs, Xi, Y, _ = state
        N = Xs + Xi + 1e-10

        dXs = self.r * Xs * (1.0 - N / self.K) - self.beta_d * Xs * Xi / N - self.a_s * Xs * Y
        dXi = self.beta_d * Xs * Xi / N - self.gamma_d * Xi - self.alpha_d * Xi - self.a_i * Xi * Y
        dY = self.e * (self.a_s * Xs + self.a_i * Xi) * Y - self.d * Y
        prev = Xi / N
        dZ = prev - state[3]

        return np.array([dXs, dXi, dY, dZ])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.Xs_0 + rng.normal(0, 1.0), 0.1),
                max(self.Xi_0 + rng.normal(0, 0.2), 0.01),
                max(self.Y_0 + rng.normal(0, 0.5), 0.1),
                0.0,
            ], dtype=np.float64)
        else:
            self._state = np.array([self.Xs_0, self.Xi_0, self.Y_0, 0.0], dtype=np.float64)
        N = self._state[0] + self._state[1] + 1e-10
        self._state[3] = self._state[1] / N
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[:3] = np.clip(self._state[:3], 0.0, None)
        N = self._state[0] + self._state[1] + 1e-10
        self._state[3] = self._state[1] / N
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
