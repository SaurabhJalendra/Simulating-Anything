"""Novel coupled Earthquake-Aftershock simulation (4D).

Couples tectonic stress accumulation with fault slip dynamics
and aftershock triggering. Stress builds linearly until reaching
a threshold, triggering a slip event that generates aftershocks
following Omori's law. Models the fundamental seismic cycle.

State: [sigma, u, n, R] where:
  sigma = tectonic shear stress
  u = fault slip displacement
  n = aftershock count rate (Omori decay)
  R = cumulative seismic moment release
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class EarthquakeAftershockSimulation(SimulationEnvironment):
    """Coupled tectonic stress + fault slip + aftershock dynamics."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.v_plate = p.get("v_plate", 0.01)
        self.k_fault = p.get("k_fault", 1.0)
        self.sigma_c = p.get("sigma_c", 1.0)
        self.eta = p.get("eta", 0.5)
        self.mu_f = p.get("mu_f", 0.6)

        self.K_omori = p.get("K_omori", 1.0)
        self.p_omori = p.get("p_omori", 1.1)
        self.c_omori = p.get("c_omori", 0.1)
        self.alpha_n = p.get("alpha_n", 5.0)

        self.sigma_0 = p.get("sigma_0", 0.5)
        self.u_0 = p.get("u_0", 0.0)

        self.dt = config.dt
        self._state = np.array([self.sigma_0, self.u_0, 0.0, 0.0], dtype=np.float64)
        self._t = 0.0
        self._last_event_time = -10.0

    def _rhs(self, state):
        sigma, u, n, R = state

        dsigma = self.k_fault * (self.v_plate - self._slip_rate(sigma))

        slip_rate = self._slip_rate(sigma)
        du = slip_rate

        t_since = self._t - self._last_event_time + self.c_omori
        dn = (-self.p_omori * n / t_since
               + self.alpha_n * max(slip_rate - 0.01, 0))

        dR = slip_rate * sigma

        return np.array([dsigma, du, dn, dR])

    def _slip_rate(self, sigma):
        if sigma > self.sigma_c:
            return self.eta * (sigma - self.mu_f * self.sigma_c)
        return 0.0

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(self.sigma_0 + rng.normal(0, 0.1), 0.0),
                0.0, 0.0, 0.0,
            ], dtype=np.float64)
        else:
            self._state = np.array([self.sigma_0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._t = 0.0
        self._last_event_time = -10.0
        return self.observe()

    def step(self):
        old_sigma = self._state[0]
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if old_sigma < self.sigma_c and self._state[0] >= self.sigma_c:
            self._last_event_time = self._t

        self._state[0] = max(self._state[0], 0.0)
        self._state[2] = max(self._state[2], 0.0)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
