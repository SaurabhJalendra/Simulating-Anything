"""Novel Wastewater Treatment simulation (4D).

Activated sludge process with substrate removal, biomass growth,
dissolved oxygen control, and sludge settling.

State: [S_sub, X_bio, DO, X_sludge] where:
  S_sub = substrate (COD) concentration
  X_bio = biomass concentration (MLSS)
  DO = dissolved oxygen
  X_sludge = settled sludge concentration
"""
from __future__ import annotations
import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

class WastewaterTreatmentSimulation(SimulationEnvironment):
    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.mu_max = p.get("mu_max", 0.5)
        self.K_s = p.get("K_s", 20.0)
        self.K_o = p.get("K_o", 0.5)
        self.Y = p.get("Y_yield", 0.6)
        self.b = p.get("b_decay", 0.05)
        self.kla = p.get("kla", 10.0)
        self.DO_sat = p.get("DO_sat", 8.0)
        self.Q_in = p.get("Q_in", 100.0)
        self.V = p.get("V_reactor", 500.0)
        self.S_in = p.get("S_in", 200.0)
        self.v_settle = p.get("v_settle", 0.1)
        self.dt = config.dt
        self._state = np.array([p.get("S_0", 100.0), p.get("X_0", 3000.0),
                                p.get("DO_0", 4.0), p.get("Xs_0", 5000.0)], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        S, X, DO, Xs = state
        mu = self.mu_max * S / (self.K_s + S) * DO / (self.K_o + DO)
        D = self.Q_in / self.V
        dS = D * (self.S_in - S) - mu * X / self.Y
        dX = mu * X - self.b * X - D * X + self.v_settle * (Xs - X) * 0.01
        dDO = self.kla * (self.DO_sat - DO) - mu * X * 1.5 / self.Y
        dXs = self.v_settle * X - 0.05 * Xs
        return np.array([dS, dX, dDO, dXs])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(100 + rng.normal(0, 10), 1), max(3000 + rng.normal(0, 200), 100),
                max(4 + rng.normal(0, 0.5), 0.1), max(5000 + rng.normal(0, 300), 100)
            ], dtype=np.float64)
        else:
            self._state = np.array([100.0, 3000.0, 4.0, 5000.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.clip(self._state, 0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
