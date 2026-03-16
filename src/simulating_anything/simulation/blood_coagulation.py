"""Novel Blood Coagulation Cascade simulation (4D).

Simplified coagulation cascade coupling thrombin generation with
fibrin polymerization and platelet activation. Positive feedback
loops create threshold-dependent clot formation.

State: [T_thr, F_fib, P_plt, I_inh] where:
  T_thr = thrombin concentration
  F_fib = fibrin concentration
  P_plt = activated platelet count
  I_inh = inhibitor (antithrombin) concentration
"""
from __future__ import annotations
import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

class BloodCoagulationSimulation(SimulationEnvironment):
    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.k_gen = p.get("k_gen", 0.5)
        self.K_thr = p.get("K_thr", 0.5)
        self.k_fib = p.get("k_fib", 0.3)
        self.k_plt = p.get("k_plt", 0.2)
        self.k_inh = p.get("k_inh", 0.1)
        self.d_t = p.get("d_t", 0.2)
        self.d_f = p.get("d_f", 0.05)
        self.d_p = p.get("d_p", 0.1)
        self.I_prod = p.get("I_prod", 0.1)
        self.trigger = p.get("trigger", 0.1)
        self.dt = config.dt
        self._state = np.array([p.get("T_0", 0.1), p.get("F_0", 0.01),
                                p.get("P_0", 0.5), p.get("I_0", 1.0)], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        T, F, P, I = state
        gen = self.k_gen * T ** 2 / (self.K_thr ** 2 + T ** 2) * P + self.trigger
        dT = gen - self.d_t * T * I
        dF = self.k_fib * T - self.d_f * F
        dP = self.k_plt * T * (1 - P) - self.d_p * P
        dI = self.I_prod - self.k_inh * T * I
        return np.array([dT, dF, dP, dI])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([max(0.1+rng.normal(0,0.02),0.01), max(0.01+rng.normal(0,0.005),0.001),
                                    np.clip(0.5+rng.normal(0,0.05),0.01,1), max(1+rng.normal(0,0.1),0.1)], dtype=np.float64)
        else:
            self._state = np.array([0.1, 0.01, 0.5, 1.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5*self.dt*k1)
        k3 = self._rhs(self._state + 0.5*self.dt*k2)
        k4 = self._rhs(self._state + self.dt*k3)
        self._state += (self.dt/6)*(k1+2*k2+2*k3+k4)
        self._state = np.clip(self._state, 0, None)
        self._state[2] = min(self._state[2], 1)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
