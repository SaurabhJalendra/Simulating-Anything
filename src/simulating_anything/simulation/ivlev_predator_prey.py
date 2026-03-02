"""Predator-prey model with Ivlev functional response.

The Ivlev functional response f(N) = a*(1 - exp(-b*N)) is a saturating
function that differs from the Holling Type II in that it lacks a handling
time interpretation -- it is purely phenomenological, based on empirical
feeding observations. The response saturates at f(inf) = a with initial
slope f'(0) = a*b.

Equations:
    dN/dt = r*N*(1 - N/K) - a*(1 - exp(-b*N))*P
    dP/dt = e*a*(1 - exp(-b*N))*P - d*P

Default parameters: r=1.0, K=10.0, a=1.5, b=0.3, e=0.5, d=0.3,
                    N_0=5.0, P_0=2.0, dt=0.01

Target rediscoveries:
- Ivlev functional response shape: f(N) = a*(1-exp(-b*N))
- SINDy ODE recovery of coupled equations
- Coexistence equilibrium from predator nullcline
- Stability analysis (limit cycles vs stable spirals)
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class IvlevPredatorPrey(SimulationEnvironment):
    """Predator-prey model with Ivlev functional response.

    State vector: [N, P] where N = prey density, P = predator density.

    Equations:
        dN/dt = r*N*(1 - N/K) - a*(1 - exp(-b*N))*P
        dP/dt = e*a*(1 - exp(-b*N))*P - d*P

    The Ivlev functional response f(N) = a*(1 - exp(-b*N)) satisfies:
        f(0) = 0
        f'(0) = a*b  (initial slope / capture efficiency)
        f(inf) = a   (maximum predation rate)

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 10.0)
        a: maximum predation rate (default 1.5)
        b: capture efficiency / shape parameter (default 0.3)
        e: conversion efficiency, prey to predator biomass (default 0.5)
        d: predator natural mortality rate (default 0.3)
        N_0: initial prey density (default 5.0)
        P_0: initial predator density (default 2.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 10.0)
        self.a = p.get("a", 1.5)
        self.b = p.get("b", 0.3)
        self.e = p.get("e", 0.5)
        self.d = p.get("d", 0.3)
        self.N_0 = p.get("N_0", 5.0)
        self.P_0 = p.get("P_0", 2.0)

    def functional_response(self, N: float | np.ndarray) -> float | np.ndarray:
        """Ivlev functional response: f(N) = a*(1 - exp(-b*N)).

        Saturates at a for large N, with initial slope a*b at N=0.

        Args:
            N: Prey density (scalar or array).

        Returns:
            Per-predator consumption rate.
        """
        return self.a * (1.0 - np.exp(-self.b * N))

    @property
    def prey_population(self) -> float:
        """Current prey density."""
        if self._state is None:
            return 0.0
        return float(self._state[0])

    @property
    def predator_population(self) -> float:
        """Current predator density."""
        if self._state is None:
            return 0.0
        return float(self._state[1])

    @property
    def total_population(self) -> float:
        """Sum of prey and predator densities."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    def coexistence_equilibrium(self) -> tuple[float, float]:
        """Compute the interior coexistence equilibrium (N*, P*).

        From dP/dt = 0:
            e*a*(1 - exp(-b*N*)) = d
            => 1 - exp(-b*N*) = d/(e*a)
            => N* = -ln(1 - d/(e*a)) / b

        Requires d < e*a (predator can sustain itself).

        From dN/dt = 0 at N*:
            P* = r*N*(1 - N*/K) / f(N*)

        Returns:
            Tuple (N_star, P_star).

        Raises:
            ValueError: If coexistence equilibrium does not exist.
        """
        ratio = self.d / (self.e * self.a)
        if ratio >= 1.0:
            raise ValueError(
                "No coexistence equilibrium: predator cannot sustain itself "
                f"(d/(e*a) = {ratio:.4f} >= 1.0)"
            )

        N_star = -np.log(1.0 - ratio) / self.b
        if N_star >= self.K:
            raise ValueError(
                f"No coexistence equilibrium: N* = {N_star:.4f} >= K = {self.K:.4f}"
            )

        f_N = self.functional_response(N_star)
        if f_N <= 0:
            raise ValueError("Functional response at equilibrium is non-positive.")

        P_star = self.r * N_star * (1.0 - N_star / self.K) / f_N
        if P_star <= 0:
            raise ValueError(
                f"No coexistence equilibrium: P* = {P_star:.4f} <= 0"
            )

        return (float(N_star), float(P_star))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [N, P]."""
        self._state = np.array([self.N_0, self.P_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [N, P]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Ensure non-negative populations
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Ivlev predator-prey right-hand side.

        dN/dt = r*N*(1 - N/K) - a*(1 - exp(-b*N))*P
        dP/dt = e*a*(1 - exp(-b*N))*P - d*P
        """
        N, P = y
        f_N = self.functional_response(N)

        dN = self.r * N * (1.0 - N / self.K) - f_N * P
        dP = self.e * f_N * P - self.d * P
        return np.array([dN, dP])
