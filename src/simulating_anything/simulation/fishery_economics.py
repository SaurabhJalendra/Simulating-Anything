"""Novel Fishery Economics simulation (4D).

Coupled fish stock dynamics with economic fishing effort.
Open-access fishery where profit drives entry/exit of fishing
vessels. Overexploitation leads to stock collapse, demonstrating
the tragedy of the commons.

State: [X, E, P, C] where:
  X = fish stock biomass
  E = fishing effort (number of boats)
  P = market price (inversely related to catch)
  C = cumulative catch
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FisheryEconomicsSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 0.5)
        self.K = p.get("K", 1000.0)
        self.q = p.get("q", 0.001)
        self.c_cost = p.get("c_cost", 1.0)
        self.P_base = p.get("P_base", 10.0)
        self.alpha_p = p.get("alpha_p", 0.01)
        self.phi = p.get("phi", 0.5)
        self.dt = config.dt
        self._state = np.array([p.get("X_0", 500.0), p.get("E_0", 50.0),
                                self.P_base, 0.0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        X, E, P, C = state
        catch = self.q * E * X
        profit = P * catch - self.c_cost * E
        dX = self.r * X * (1 - X / self.K) - catch
        dE = self.phi * profit
        dP = -self.alpha_p * (catch - 5.0)
        dC = catch
        return np.array([dX, dE, dP, dC])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(500 + rng.normal(0, 50), 10), max(50 + rng.normal(0, 5), 1),
                self.P_base, 0.0], dtype=np.float64)
        else:
            self._state = np.array([500.0, 50.0, self.P_base, 0.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[0] = max(self._state[0], 0)
        self._state[1] = max(self._state[1], 0)
        self._state[2] = max(self._state[2], 0.1)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
