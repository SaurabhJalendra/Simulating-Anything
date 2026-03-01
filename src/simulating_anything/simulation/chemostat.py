"""Chemostat (continuous stirred bioreactor) simulation.

Models microbial growth on a single limiting substrate in a well-mixed reactor
with constant dilution rate. The Monod kinetics describe substrate-limited growth:

    dS/dt = D*(S_in - S) - (mu_max*S / (K_s + S)) * X / Y_xs
    dX/dt = (mu_max*S / (K_s + S)) * X - D*X

where S = substrate concentration, X = biomass concentration.

Target rediscoveries:
- Monod growth rate: mu(S) = mu_max * S / (K_s + S)
- Washout dilution: D_c = mu_max * S_in / (K_s + S_in)
- Steady-state biomass: X* = Y_xs * (S_in - S*)
- Steady-state substrate: S* = K_s * D / (mu_max - D)
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class Chemostat(SimulationEnvironment):
    """Chemostat bioreactor with Monod kinetics.

    State vector: [S, X] where S = substrate, X = biomass.

    The system has a nontrivial steady state when D < D_c (washout dilution).
    At D >= D_c, biomass washes out and S -> S_in.

    Parameters:
        D: dilution rate (1/time, default 0.1)
        S_in: inlet substrate concentration (default 10.0)
        mu_max: maximum specific growth rate (default 0.5)
        K_s: half-saturation constant (default 2.0)
        Y_xs: yield coefficient, biomass per substrate consumed (default 0.5)
        S_0: initial substrate concentration (default 5.0)
        X_0: initial biomass concentration (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.D = p.get("D", 0.1)
        self.S_in = p.get("S_in", 10.0)
        self.mu_max = p.get("mu_max", 0.5)
        self.K_s = p.get("K_s", 2.0)
        self.Y_xs = p.get("Y_xs", 0.5)
        self.S_0 = p.get("S_0", 5.0)
        self.X_0 = p.get("X_0", 1.0)

    @property
    def growth_rate(self) -> float:
        """Current Monod growth rate mu(S) = mu_max * S / (K_s + S)."""
        if self._state is None:
            return 0.0
        S = self._state[0]
        return self.mu_max * S / (self.K_s + S)

    @property
    def washout_D(self) -> float:
        """Critical dilution rate above which biomass washes out.

        D_c = mu_max * S_in / (K_s + S_in)
        """
        return self.mu_max * self.S_in / (self.K_s + self.S_in)

    @property
    def steady_state(self) -> tuple[float, float]:
        """Analytical nontrivial steady state (S*, X*).

        S* = K_s * D / (mu_max - D)
        X* = Y_xs * (S_in - S*)

        Returns (S_in, 0.0) if D >= D_c (washout).
        """
        if self.D >= self.washout_D:
            return (self.S_in, 0.0)
        S_star = self.K_s * self.D / (self.mu_max - self.D)
        X_star = self.Y_xs * (self.S_in - S_star)
        return (S_star, X_star)

    def monod_rate(self, S: float) -> float:
        """Monod growth rate for a given substrate concentration."""
        return self.mu_max * S / (self.K_s + S)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize substrate and biomass concentrations."""
        self._state = np.array([self.S_0, self.X_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [S, X]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce non-negativity (physical constraint)
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        S, X = y
        # Monod growth rate
        mu = self.mu_max * S / (self.K_s + S) if S > 0 else 0.0
        dS = self.D * (self.S_in - S) - mu * X / self.Y_xs
        dX = mu * X - self.D * X
        return np.array([dS, dX])
