"""Novel coupled Predator-Prey-Climate simulation (4D).

Couples Rosenzweig-MacArthur predator-prey dynamics with Stommel
thermohaline ocean circulation. Temperature oscillations modulate
the prey carrying capacity, creating climate-driven population crashes
and recoveries that have no known analytical solution.

State: [N, P, T, S] where:
  N = prey density
  P = predator density
  T = ocean temperature difference
  S = ocean salinity difference

Coupling:
  - Ocean temperature T modulates prey carrying capacity: K_eff = K * (1 + c1 * T)
  - Predator density P modulates ocean mixing: eta1_eff = eta1 + c2 * P
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorPreyClimateSimulation(SimulationEnvironment):
    """Coupled Rosenzweig-MacArthur + Stommel ocean simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Prey-predator (Rosenzweig-MacArthur) parameters
        self.r = p.get("r", 1.0)          # prey intrinsic growth rate
        self.K = p.get("K", 10.0)         # prey carrying capacity
        self.a = p.get("a_pred", 1.0)     # predation rate
        self.h = p.get("h_pred", 0.5)     # handling time (Holling Type II)
        self.e = p.get("e_pred", 0.5)     # conversion efficiency
        self.d = p.get("d_pred", 0.3)     # predator death rate

        # Stommel ocean parameters
        self.eta1 = p.get("eta1", 3.0)    # thermal forcing
        self.eta2 = p.get("eta2", 1.0)    # haline forcing
        self.delta = p.get("delta_s", 0.3)  # salinity relaxation

        # Coupling parameters
        self.coupling_TK = p.get("coupling_TK", 0.2)   # T -> carrying capacity
        self.coupling_Peta = p.get("coupling_Peta", 0.01)  # P -> ocean mixing

        # Initial conditions
        self.N_0 = p.get("N_0", 5.0)
        self.P_0 = p.get("P_0", 2.0)
        self.T_0 = p.get("T_0", 2.0)
        self.S_0 = p.get("S_0", 1.0)

        self.dt = config.dt
        self._state = np.array(
            [self.N_0, self.P_0, self.T_0, self.S_0], dtype=np.float64,
        )
        self._t = 0.0

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        N, P, T, S = state

        # Stommel overturning flow
        q = abs(T - S)

        # Effective carrying capacity (climate modulation)
        K_eff = self.K * (1.0 + self.coupling_TK * T)
        K_eff = max(K_eff, 0.1)  # prevent negative capacity

        # Effective thermal forcing (predator biomass heats/cools)
        eta1_eff = self.eta1 + self.coupling_Peta * P

        # Holling Type II functional response
        functional_response = self.a * N / (1.0 + self.a * self.h * N)

        # Prey dynamics (logistic + predation)
        dN = self.r * N * (1.0 - N / K_eff) - functional_response * P

        # Predator dynamics (conversion - death)
        dP = self.e * functional_response * P - self.d * P

        # Stommel ocean dynamics
        dT = eta1_eff - T - q * T
        dS = self.eta2 - self.delta * S - q * S

        return np.array([dN, dP, dT, dS])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                self.N_0 + rng.normal(0, 0.5),
                self.P_0 + rng.normal(0, 0.3),
                self.T_0 + rng.normal(0, 0.2),
                self.S_0 + rng.normal(0, 0.1),
            ], dtype=np.float64)
            self._state[:2] = np.clip(self._state[:2], 0.1, None)
        else:
            self._state = np.array(
                [self.N_0, self.P_0, self.T_0, self.S_0], dtype=np.float64,
            )
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state = self._state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce positivity for populations
        self._state[:2] = np.clip(self._state[:2], 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [N, P, T, S]."""
        return self._state.copy()
