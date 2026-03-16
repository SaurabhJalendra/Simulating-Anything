"""Novel Channel curvature + bank erosion simulation (4D)."""
from __future__ import annotations
import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

class RiverMeanderSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.dt = config.dt
        self._state = np.array([1.0, 0.5, 0.2, 0.0], dtype=np.float64)
        self._t = 0.0
    def _rhs(self, state):
        x0, x1, x2, x3 = state
        dx0 = 0.11*x1 - 0.045*x0 + 0.009*np.sin(0.08*self._t)
        dx1 = -0.075*x0*x1 + 0.022*x2 - 0.018*x1
        dx2 = 0.055*x0 - 0.028*x2 + 0.014*x3 - 0.009*x1*x2
        dx3 = 0.065*x1*x2 - 0.045*x3 + 0.004*x0
        return np.array([dx0, dx1, dx2, dx3])
    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.clip(np.array([1+rng.normal(0,.1), .5+rng.normal(0,.05),
                                            .2+rng.normal(0,.02), rng.normal(0,.01)], dtype=np.float64), 0, None)
        else:
            self._state = np.array([1.0, 0.5, 0.2, 0.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()
    def step(self):
        k1=self._rhs(self._state); k2=self._rhs(self._state+.5*self.dt*k1)
        k3=self._rhs(self._state+.5*self.dt*k2); k4=self._rhs(self._state+self.dt*k3)
        self._state += (self.dt/6)*(k1+2*k2+2*k3+k4)
        self._state = np.clip(self._state, 0, None)
        self._t += self.dt
        return self.observe()
    def observe(self):
        return self._state.copy()
