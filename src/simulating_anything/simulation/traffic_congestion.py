"""Novel Traffic Congestion simulation (4D).

Coupled traffic flow with signal control. Vehicles queue at
intersection, signal alternates green phases. Demand fluctuates.
Queue spillback creates upstream congestion.

State: [q1, q2, g, D] where:
  q1 = queue length direction 1
  q2 = queue length direction 2
  g = green phase indicator (0-1, smooth)
  D = demand rate
"""
from __future__ import annotations
import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

class TrafficCongestionSimulation(SimulationEnvironment):
    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.s_max = p.get("s_max", 1800.0)
        self.D_base = p.get("D_base", 800.0)
        self.D_amp = p.get("D_amp", 300.0)
        self.cycle = p.get("cycle", 90.0)
        self.green_frac = p.get("green_frac", 0.5)
        self.dt = config.dt
        self._state = np.array([p.get("q1_0",10.0), p.get("q2_0",5.0), 0.5, self.D_base], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        q1, q2, g, D = state
        phase = 0.5 + 0.5 * np.sin(2*np.pi*t/self.cycle)
        s1 = self.s_max * phase
        s2 = self.s_max * (1-phase)
        D_t = self.D_base + self.D_amp * np.sin(2*np.pi*0.001*t)
        dq1 = D_t * self.green_frac - s1 * min(q1/(q1+1), 1)
        dq2 = D_t * (1-self.green_frac) - s2 * min(q2/(q2+1), 1)
        dg = phase - g
        dD = D_t - D
        return np.array([dq1, dq2, dg, dD])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([max(10+rng.normal(0,2),0), max(5+rng.normal(0,1),0),
                                    0.5, self.D_base], dtype=np.float64)
        else:
            self._state = np.array([10.0, 5.0, 0.5, self.D_base], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state+0.5*self.dt*k1, t+0.5*self.dt)
        k3 = self._rhs(self._state+0.5*self.dt*k2, t+0.5*self.dt)
        k4 = self._rhs(self._state+self.dt*k3, t+self.dt)
        self._state += (self.dt/6)*(k1+2*k2+2*k3+k4)
        self._state[:2] = np.clip(self._state[:2], 0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
