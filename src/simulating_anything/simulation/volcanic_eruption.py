"""Novel Magma chamber + eruption simulation (4D)."""
from __future__ import annotations
import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

class VolcanicEruptionSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.dt = config.dt
        self._state = np.array([1.0, 0.5, 0.1, 0.0], dtype=np.float64)
        self._t = 0.0
    def _rhs(self, state):
        x = state
        return np.array([
            0.1*x[1] - 0.05*x[0],
            -0.1*x[0]*x[1] + 0.02*x[2],
            x[0]*0.05 - 0.03*x[2] + 0.01*x[3],
            0.1*x[1]*x[2] - 0.05*x[3],
        ])
    def reset(self, seed=None):
        rng = np.random.default_rng(seed) if seed else None
        if rng:
            self._state = np.array([1+rng.normal(0,.1), .5+rng.normal(0,.05),
                                    .1+rng.normal(0,.01), 0], dtype=np.float64)
        else:
            self._state = np.array([1.0, 0.5, 0.1, 0.0], dtype=np.float64)
        self._state = np.clip(self._state, 0, None)
        self._t = 0.0
        return self.observe()
    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5*self.dt*k1)
        k3 = self._rhs(self._state + 0.5*self.dt*k2)
        k4 = self._rhs(self._state + self.dt*k3)
        self._state += (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self._state = np.clip(self._state, 0, None)
        self._t += self.dt
        return self.observe()
    def observe(self):
        return self._state.copy()
