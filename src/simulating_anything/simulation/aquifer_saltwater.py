"""Novel Aquifer-Saltwater Intrusion simulation (4D).

Coupled freshwater-saltwater interface dynamics in coastal
aquifers. Freshwater head drives the interface position.
Pumping lowers head, allowing saltwater to intrude.
Sea level rise pushes the interface inland.

State: [h_f, x_int, Q_pump, h_sea] where:
  h_f = freshwater hydraulic head
  x_int = saltwater interface position (distance from coast)
  Q_pump = pumping rate
  h_sea = sea level (slowly rising)
"""
from __future__ import annotations

import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AquiferSaltwaterSimulation(SimulationEnvironment):

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.K_aq = p.get("K_aq", 0.01)
        self.S_y = p.get("S_y", 0.1)
        self.recharge = p.get("recharge", 0.002)
        self.rho_ratio = p.get("rho_ratio", 40.0)
        self.alpha_int = p.get("alpha_int", 0.1)
        self.Q_base = p.get("Q_base", 0.005)
        self.Q_var = p.get("Q_var", 0.002)
        self.slr_rate = p.get("slr_rate", 0.00001)
        self.dt = config.dt
        self._state = np.array([p.get("hf_0", 5.0), p.get("xint_0", 500.0),
                                self.Q_base, p.get("hsea_0", 0.0)], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        hf, xint, Qp, hsea = state
        ghyben = self.rho_ratio * (hf - hsea)
        dh = (self.recharge - Qp - self.K_aq * (hf - hsea)) / self.S_y
        dx = self.alpha_int * (ghyben - xint)
        dQ = self.Q_var * np.sin(2 * np.pi * 0.01 * t)
        dhs = self.slr_rate
        return np.array([dh, dx, dQ, dhs])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(5 + rng.normal(0, 0.5), 0.1), max(500 + rng.normal(0, 50), 10),
                self.Q_base, 0.0], dtype=np.float64)
        else:
            self._state = np.array([5.0, 500.0, self.Q_base, 0.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[0] = max(self._state[0], 0)
        self._state[1] = max(self._state[1], 0)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
