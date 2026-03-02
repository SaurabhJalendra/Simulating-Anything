"""Ratio-dependent predator-prey (Arditi-Ginzburg) model simulation.

The Arditi-Ginzburg model replaces the prey-dependent functional response
with a ratio-dependent one, where predation depends on the prey-to-predator
ratio N/P rather than prey density N alone. This eliminates the paradox of
enrichment: increasing carrying capacity K does NOT destabilize the
coexistence equilibrium (unlike Rosenzweig-MacArthur).

Equations:
    dN/dt = r*N*(1 - N/K) - a*N*P/(N + b*P)
    dP/dt = e*a*N*P/(N + b*P) - d*P

Default parameters: r=1.0, K=100.0, a=0.5, b=1.0, e=0.5, d=0.2
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RatioDependentSimulation(SimulationEnvironment):
    """Ratio-dependent (Arditi-Ginzburg) predator-prey model.

    State vector: [N, P] where N = prey, P = predator.

    Equations:
        dN/dt = r*N*(1 - N/K) - a*N*P/(N + b*P)
        dP/dt = e*a*N*P/(N + b*P) - d*P

    The functional response a*N/(N + b*P) depends on the ratio N/P,
    not on N alone. This is the key distinction from prey-dependent
    models (Lotka-Volterra, Rosenzweig-MacArthur).

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 100.0)
        a: maximum predation rate (default 0.5)
        b: ratio-dependence parameter (default 1.0)
        e: conversion efficiency (default 0.5)
        d: predator death rate (default 0.2)
        N_0: initial prey population (default 50.0)
        P_0: initial predator population (default 10.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 100.0)
        self.a = p.get("a", 0.5)
        self.b = p.get("b", 1.0)
        self.e = p.get("e", 0.5)
        self.d = p.get("d", 0.2)
        self.N_0 = p.get("N_0", 50.0)
        self.P_0 = p.get("P_0", 10.0)

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

    @property
    def total_population(self) -> float:
        """Sum of prey and predator populations."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    def coexistence_equilibrium(self) -> tuple[float, float]:
        """Compute the interior coexistence equilibrium (N*, P*).

        At equilibrium, dP/dt = 0 with P > 0 gives:
            e*a*N/(N + b*P) = d  =>  N/(N + b*P) = d/(e*a)

        Let q = d/(e*a). Requires q < 1 (i.e., e*a > d) for predator
        to sustain itself. From the ratio:
            P = N*(1 - q)/(q*b)

        From dN/dt = 0 with N > 0:
            r*(1 - N/K) = a*P/(N + b*P)

        Since N + b*P = N/q, we have P/(N + b*P) = (1-q)/b. Therefore:
            r*(1 - N*/K) = a*(1 - q)/b

        Solving:
            N* = K*(1 - a*(1-q)/(b*r))
            P* = N*(1-q)/(q*b)

        where q = d/(e*a).

        Returns:
            Tuple (N_star, P_star).

        Raises:
            ValueError: If coexistence equilibrium does not exist.
        """
        if self.e * self.a <= self.d:
            raise ValueError(
                "No coexistence equilibrium: predator cannot sustain itself "
                f"(e*a={self.e * self.a:.4f} <= d={self.d:.4f})"
            )

        q = self.d / (self.e * self.a)
        # q < 1 guaranteed by the check above

        N_star = self.K * (1.0 - self.a * (1.0 - q) / (self.b * self.r))
        if N_star <= 0:
            raise ValueError(
                f"No coexistence equilibrium: N*={N_star:.4f} <= 0"
            )

        P_star = N_star * (1.0 - q) / (q * self.b)
        if P_star <= 0:
            raise ValueError(
                f"No coexistence equilibrium: P*={P_star:.4f} <= 0"
            )

        return (N_star, P_star)

    def functional_response(self, N: float, P: float) -> float:
        """Compute the ratio-dependent functional response.

        f(N, P) = a * N / (N + b * P)

        This depends on the ratio N/P, not just N.
        Returns 0 if both N and P are zero.
        """
        denom = N + self.b * P
        if denom <= 0:
            return 0.0
        return self.a * N / denom

    def is_stable(self) -> bool:
        """Check if the coexistence equilibrium is locally stable.

        For the ratio-dependent model, the coexistence equilibrium is
        always stable when it exists (no paradox of enrichment).
        This is in contrast to the Rosenzweig-MacArthur model.

        Returns:
            True if coexistence equilibrium exists (it is always stable).
        """
        try:
            self.coexistence_equilibrium()
            return True
        except ValueError:
            return False

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
        """Ratio-dependent predator-prey right-hand side.

        dN/dt = r*N*(1 - N/K) - a*N*P/(N + b*P)
        dP/dt = e*a*N*P/(N + b*P) - d*P
        """
        N, P = y

        # Handle edge case where both populations are zero
        denom = N + self.b * P
        if denom <= 0:
            return np.array([self.r * N * (1.0 - N / self.K), -self.d * P])

        functional = self.a * N / denom

        dN = self.r * N * (1.0 - N / self.K) - functional * P
        dP = self.e * functional * P - self.d * P

        return np.array([dN, dP])
