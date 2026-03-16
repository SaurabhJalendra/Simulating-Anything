"""Novel coupled Dopamine Reward Learning simulation (4D).

Couples dopamine release dynamics with reward prediction error
(RPE) learning. Dopamine neurons fire in response to unexpected
rewards, driving synaptic plasticity. As learning proceeds,
dopamine shifts from reward to cue, implementing temporal
difference (TD) learning.

State: [D, V, W, R] where:
  D = dopamine concentration
  V = value estimate (learned expectation)
  W = synaptic weight (cue-reward association)
  R = reward signal (periodic + random)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DopamineRewardSimulation(SimulationEnvironment):
    """Coupled dopamine release + TD reward learning."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.tau_d = p.get("tau_d", 0.5)
        self.D_base = p.get("D_base", 0.5)
        self.alpha_rpe = p.get("alpha_rpe", 2.0)
        self.alpha_learn = p.get("alpha_learn", 0.1)
        self.gamma_td = p.get("gamma_td", 0.95)
        self.tau_v = p.get("tau_v", 1.0)
        self.R_amp = p.get("R_amp", 1.0)
        self.R_freq = p.get("R_freq", 0.05)

        self.D_0 = p.get("D_0", 0.5)
        self.V_0 = p.get("V_0", 0.0)
        self.W_0 = p.get("W_0", 0.1)

        self.dt = config.dt
        self._state = np.array([self.D_0, self.V_0, self.W_0, 0.0], dtype=np.float64)
        self._t = 0.0

    def _reward_signal(self, t):
        return self.R_amp * max(np.sin(2 * np.pi * self.R_freq * t), 0)

    def _rhs(self, state, t):
        D, V, W, R = state

        R_new = self._reward_signal(t)
        rpe = R_new - V
        dD = (self.D_base + self.alpha_rpe * max(rpe, 0) - D) / self.tau_d
        dV = self.alpha_learn * rpe
        dW = self.alpha_learn * rpe * D
        dR = R_new - R

        return np.array([dD, dV, dW, dR])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.D_0 + rng.normal(0, 0.1), 0.01),
                self.V_0 + rng.normal(0, 0.05),
                max(self.W_0 + rng.normal(0, 0.02), 0.01),
                0.0,
            ], dtype=np.float64)
        else:
            self._state = np.array([self.D_0, self.V_0, self.W_0, 0.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[0] = max(self._state[0], 0.0)
        self._state[3] = self._reward_signal(self._t + self.dt)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
