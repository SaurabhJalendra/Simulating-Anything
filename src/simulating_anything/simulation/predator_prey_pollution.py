"""Novel coupled Predator-Prey-Pollution simulation (4D).

Couples Lotka-Volterra predator-prey dynamics with environmental
pollutant dynamics. Pollution reduces prey carrying capacity and
bioaccumulates up the food chain, creating eco-toxicological
dynamics with no known analytical solution.

State: [N, P, C, B] where:
  N = prey density
  P = predator density
  C = environmental pollutant concentration
  B = bioaccumulated toxin in predators

Coupling:
  - Pollutant C reduces prey carrying capacity: K_eff = K / (1 + c1*C)
  - Prey uptake transfers pollutant to predators (bioaccumulation)
  - High toxin load B increases predator mortality
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorPreyPollutionSimulation(SimulationEnvironment):
    """Coupled LV predator-prey + pollution bioaccumulation."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.r = p.get("r", 1.0)
        self.K = p.get("K", 20.0)
        self.a = p.get("a_pred", 0.5)
        self.e = p.get("e_pred", 0.3)
        self.d = p.get("d_pred", 0.2)

        self.C_input = p.get("C_input", 0.1)
        self.lambda_c = p.get("lambda_c", 0.05)
        self.uptake_rate = p.get("uptake_rate", 0.02)

        self.coupling_CK = p.get("coupling_CK", 0.5)
        self.coupling_Bd = p.get("coupling_Bd", 0.1)
        self.bioaccum = p.get("bioaccum", 0.01)
        self.lambda_b = p.get("lambda_b", 0.03)

        self.N_0 = p.get("N_0", 10.0)
        self.P_0 = p.get("P_0", 5.0)
        self.C_0 = p.get("C_0", 1.0)
        self.B_0 = p.get("B_0", 0.1)

        self.dt = config.dt
        self._state = np.array([self.N_0, self.P_0, self.C_0, self.B_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        N, P, C, B = state
        K_eff = self.K / (1.0 + self.coupling_CK * C)
        d_eff = self.d + self.coupling_Bd * B

        dN = self.r * N * (1.0 - N / K_eff) - self.a * N * P
        dP = self.e * self.a * N * P - d_eff * P
        dC = self.C_input - self.lambda_c * C - self.uptake_rate * C * N
        dB = self.bioaccum * self.a * N * P * C - self.lambda_b * B * P

        return np.array([dN, dP, dC, dB])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.N_0 + rng.normal(0, 1.0), 0.1),
                max(self.P_0 + rng.normal(0, 0.5), 0.1),
                max(self.C_0 + rng.normal(0, 0.2), 0.01),
                max(self.B_0 + rng.normal(0, 0.02), 0.0),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.N_0, self.P_0, self.C_0, self.B_0], dtype=np.float64)
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
