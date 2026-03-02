"""Glucose-Insulin Regulatory System (Bergman Minimal Model).

Bergman's minimal model describes the dynamics of glucose and insulin
after an intravenous glucose tolerance test (IVGTT). Three state variables:

  G  -- plasma glucose concentration (mg/dL)
  X  -- remote (interstitial) insulin action (1/min)
  I  -- plasma insulin concentration (uU/mL)

ODEs:
  dG/dt = -(p1 + X)*G + p1*Gb
  dX/dt = -p2*X + p3*(I - Ib)
  dI/dt = -n*(I - Ib) + gamma*max(G - h, 0)

Key physiological indices:
  SG = p1          -- glucose effectiveness (min^-1)
  SI = p3/p2       -- insulin sensitivity (min^-1 per uU/mL)

Reference: Bergman, Ider, Bowden, Cobelli (1979)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GlucoseInsulinSimulation(SimulationEnvironment):
    """Bergman minimal model for glucose-insulin regulation.

    State vector: [G, X, I] where
        G = plasma glucose (mg/dL)
        X = remote insulin action (1/min)
        I = plasma insulin (uU/mL)

    Parameters:
        p1: glucose effectiveness (min^-1), default 0.028
        p2: insulin action decay rate (min^-1), default 0.025
        p3: insulin action sensitivity (min^-2 per uU/mL), default 5e-6
        n: insulin clearance rate (min^-1), default 0.23
        gamma: pancreatic insulin secretion rate (uU/mL per mg/dL per min), default 0.01
        Gb: basal glucose (mg/dL), default 92
        Ib: basal insulin (uU/mL), default 11
        h: glucose threshold for insulin secretion (mg/dL), default 80
        G0: initial glucose after bolus (mg/dL), default 300
        X0: initial remote insulin action (1/min), default 0
        I0: initial insulin (uU/mL), default Ib
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.p1 = p.get("p1", 0.028)
        self.p2 = p.get("p2", 0.025)
        self.p3 = p.get("p3", 5e-6)
        self.n = p.get("n", 0.23)
        self.gamma = p.get("gamma", 0.01)
        self.Gb = p.get("Gb", 92.0)
        self.Ib = p.get("Ib", 11.0)
        self.h = p.get("h", 80.0)
        self.G0 = p.get("G0", 300.0)
        self.X0 = p.get("X0", 0.0)
        self.I0 = p.get("I0", p.get("Ib", 11.0))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state to post-bolus conditions."""
        self._state = np.array([self.G0, self.X0, self.I0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [G, X, I]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        G, X, ins = y

        dG = -(self.p1 + X) * G + self.p1 * self.Gb
        dX = -self.p2 * X + self.p3 * (ins - self.Ib)
        dI = -self.n * (ins - self.Ib) + self.gamma * max(G - self.h, 0.0)

        return np.array([dG, dX, dI])

    @property
    def glucose_effectiveness(self) -> float:
        """Glucose effectiveness SG = p1 (min^-1).

        Measures the ability of glucose to promote its own disposal
        and suppress hepatic glucose production, independent of insulin.
        """
        return self.p1

    @property
    def insulin_sensitivity(self) -> float:
        """Insulin sensitivity SI = p3/p2 (min^-1 per uU/mL).

        Measures the increase in glucose disappearance ability per unit
        change in plasma insulin concentration.
        """
        return self.p3 / self.p2

    @property
    def basal_state(self) -> np.ndarray:
        """Basal (equilibrium) state [Gb, 0, Ib]."""
        return np.array([self.Gb, 0.0, self.Ib])

    def time_to_half_peak(self, max_steps: int = 10000) -> float:
        """Compute time for glucose to decay to halfway between peak and basal.

        Runs a fresh simulation and measures the time for G to reach
        (G0 + Gb) / 2. Returns time in minutes.
        """
        half_target = (self.G0 + self.Gb) / 2.0
        dt = self.config.dt

        state = self.reset()
        G_prev = state[0]

        for i in range(1, max_steps + 1):
            state = self.step()
            G_curr = state[0]

            if G_curr <= half_target:
                # Linear interpolation for sub-step accuracy
                if G_prev != G_curr:
                    frac = (G_prev - half_target) / (G_prev - G_curr)
                else:
                    frac = 0.5
                return (i - 1 + frac) * dt
            G_prev = G_curr

        return float("inf")

    def steady_state_error(self, n_steps: int = 5000) -> np.ndarray:
        """Run simulation and return final deviation from basal state."""
        self.reset()
        for _ in range(n_steps):
            self.step()
        return self._state - self.basal_state
