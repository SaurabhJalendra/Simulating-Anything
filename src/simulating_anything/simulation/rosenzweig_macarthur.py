"""Rosenzweig-MacArthur predator-prey model with Holling Type II functional response.

Extends Lotka-Volterra with saturating predation (Holling Type II) and
logistic prey growth. Exhibits the paradox of enrichment: increasing
carrying capacity K destabilizes the coexistence equilibrium via Hopf
bifurcation, producing limit cycles.

Equations:
    dx/dt = r*x*(1 - x/K) - a*x*y/(1 + a*h*x)
    dy/dt = e*a*x*y/(1 + a*h*x) - d*y

Default parameters: r=1.0, K=10.0, a=0.5, h=0.5, e=0.5, d=0.1
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RosenzweigMacArthur(SimulationEnvironment):
    """Rosenzweig-MacArthur predator-prey model.

    State vector: [x, y] where x = prey, y = predator.

    Equations:
        dx/dt = r*x*(1 - x/K) - a*x*y/(1 + a*h*x)
        dy/dt = e*a*x*y/(1 + a*h*x) - d*y

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 10.0)
        a: attack rate (default 0.5)
        h: handling time (default 0.5)
        e: conversion efficiency (default 0.5)
        d: predator death rate (default 0.1)
        x_0: initial prey population (default 1.0)
        y_0: initial predator population (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 10.0)
        self.a = p.get("a", 0.5)
        self.h = p.get("h", 0.5)
        self.e = p.get("e", 0.5)
        self.d = p.get("d", 0.1)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 1.0)

    @property
    def total_population(self) -> float:
        """Sum of prey and predator populations."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    @property
    def prey_population(self) -> float:
        """Current prey population."""
        if self._state is None:
            return 0.0
        return float(self._state[0])

    @property
    def predator_population(self) -> float:
        """Current predator population."""
        if self._state is None:
            return 0.0
        return float(self._state[1])

    def coexistence_equilibrium(self) -> tuple[float, float]:
        """Compute the interior coexistence equilibrium (x*, y*).

        x* = d / (e*a - a*h*d)
        y* = (r/a) * (1 - x*/K) * (1 + a*h*x*)

        Requires e*a > a*h*d (predator can sustain itself) and x* < K.

        Returns:
            Tuple (x_star, y_star).

        Raises:
            ValueError: If coexistence equilibrium does not exist.
        """
        denominator = self.e * self.a - self.a * self.h * self.d
        if denominator <= 0:
            raise ValueError(
                "No coexistence equilibrium: predator cannot sustain itself "
                f"(e*a={self.e * self.a:.4f} <= a*h*d={self.a * self.h * self.d:.4f})"
            )

        x_star = self.d / denominator
        if x_star >= self.K:
            raise ValueError(
                f"No coexistence equilibrium: x*={x_star:.4f} >= K={self.K:.4f}"
            )

        functional_response_at_eq = 1.0 + self.a * self.h * x_star
        y_star = (self.r / self.a) * (1.0 - x_star / self.K) * functional_response_at_eq
        return (x_star, y_star)

    def is_stable(self) -> bool:
        """Check if the coexistence equilibrium is locally stable.

        The Hopf bifurcation occurs when x* = K/2 (enrichment paradox).
        The equilibrium is stable when x* > K/2, unstable (limit cycle) when x* < K/2.

        Returns:
            True if coexistence equilibrium exists and is stable.
        """
        try:
            x_star, _ = self.coexistence_equilibrium()
        except ValueError:
            return False
        return x_star > self.K / 2.0

    def critical_K(self) -> float:
        """Compute the critical carrying capacity K_c for Hopf bifurcation.

        At K_c, x* = K_c/2, which gives:
            K_c = 2*d / (e*a - a*h*d)

        Returns:
            Critical K value. For K > K_c, limit cycles appear.

        Raises:
            ValueError: If predator cannot sustain itself.
        """
        denominator = self.e * self.a - self.a * self.h * self.d
        if denominator <= 0:
            raise ValueError("Predator cannot sustain itself; no critical K exists.")
        return 2.0 * self.d / denominator

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [x, y]."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [x, y]."""
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
        """Rosenzweig-MacArthur right-hand side.

        dx/dt = r*x*(1 - x/K) - a*x*y/(1 + a*h*x)
        dy/dt = e*a*x*y/(1 + a*h*x) - d*y
        """
        x, pred = y
        functional_response = self.a * x / (1.0 + self.a * self.h * x)

        dx = self.r * x * (1.0 - x / self.K) - functional_response * pred
        dy = self.e * functional_response * pred - self.d * pred
        return np.array([dx, dy])
