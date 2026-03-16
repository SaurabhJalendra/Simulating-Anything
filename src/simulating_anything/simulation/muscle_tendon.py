"""Novel coupled Muscle-Tendon simulation (4D).

Couples Hill-type muscle fiber dynamics with series elastic tendon.
Muscle activation generates force via force-length and force-velocity
relationships. Tendon stores and releases elastic energy. Models
the musculoskeletal system for biomechanics and rehabilitation.

State: [a, l_m, v_m, F_t] where:
  a = muscle activation level (0-1)
  l_m = muscle fiber length
  v_m = muscle fiber velocity
  F_t = tendon force
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class MuscleTendonSimulation(SimulationEnvironment):
    """Coupled Hill-type muscle + series elastic tendon."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.tau_act = p.get("tau_act", 0.01)
        self.tau_deact = p.get("tau_deact", 0.04)
        self.F_max = p.get("F_max", 100.0)
        self.l_opt = p.get("l_opt", 1.0)
        self.v_max = p.get("v_max", 10.0)
        self.k_tendon = p.get("k_tendon", 50.0)
        self.l_slack = p.get("l_slack", 1.0)
        self.b_damp = p.get("b_damp", 0.1)
        self.u_drive = p.get("u_drive", 0.5)

        self.a_0 = p.get("a_0", 0.1)
        self.lm_0 = p.get("lm_0", 1.0)
        self.vm_0 = p.get("vm_0", 0.0)

        self.dt = config.dt
        Ft_0 = self.k_tendon * max(self.lm_0 - self.l_slack, 0)
        self._state = np.array([self.a_0, self.lm_0, self.vm_0, Ft_0], dtype=np.float64)
        self._t = 0.0

    def _fl(self, l_m):
        x = (l_m / self.l_opt - 1.0) / 0.5
        return max(np.exp(-x * x), 0.01)

    def _fv(self, v_m):
        if v_m <= 0:
            return (1.0 + v_m / self.v_max) / (1.0 - v_m / (0.25 * self.v_max))
        return 1.3 - 0.3 * (1.0 + v_m / self.v_max) / (1.0 + 7.56 * v_m / self.v_max)

    def _rhs(self, state, t):
        a, l_m, v_m, F_t = state

        u = self.u_drive * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        tau = self.tau_act if u > a else self.tau_deact
        da = (u - a) / tau

        F_muscle = a * self.F_max * self._fl(l_m) * self._fv(v_m)
        F_tendon = self.k_tendon * max(l_m - self.l_slack, 0)
        dvm = (F_muscle - F_tendon - self.b_damp * v_m) / 1.0

        dlm = v_m
        dFt = self.k_tendon * v_m if l_m > self.l_slack else 0.0

        return np.array([da, dlm, dvm, dFt])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            lm = max(self.lm_0 + rng.normal(0, 0.05), 0.5)
            self._state = np.array([
                np.clip(self.a_0 + rng.normal(0, 0.02), 0.0, 1.0),
                lm, 0.0,
                self.k_tendon * max(lm - self.l_slack, 0),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.a_0, self.lm_0, 0.0,
                                    self.k_tendon * max(self.lm_0 - self.l_slack, 0)], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[0] = np.clip(self._state[0], 0.0, 1.0)
        self._state[3] = self.k_tendon * max(self._state[1] - self.l_slack, 0)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
