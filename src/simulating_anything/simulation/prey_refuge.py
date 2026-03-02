"""Predator-prey model with prey refuge.

Models predator-prey dynamics where a fraction m of prey has access to a
refuge, reducing the effective predation rate. Increasing refuge fraction
stabilizes the system by weakening predator-prey coupling.

Equations:
    dN/dt = r*N*(1 - N/K) - alpha*(1-m)*N*P
    dP/dt = beta*alpha*(1-m)*N*P - delta*P

Parameters:
    r: prey intrinsic growth rate (default 1.0)
    K: prey carrying capacity (default 10.0)
    alpha: predation rate (default 0.5)
    m: refuge fraction, 0 <= m < 1 (default 0.3)
    beta: conversion efficiency (default 0.4)
    delta: predator death rate (default 0.2)

The refuge reduces effective predation from alpha to alpha*(1-m). This shifts
the coexistence equilibrium, increases prey density, and can stabilize
oscillatory dynamics.
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PreyRefugeSimulation(SimulationEnvironment):
    """Predator-prey model with prey refuge.

    State vector: [N, P] where N = prey density, P = predator density.

    Equations:
        dN/dt = r*N*(1 - N/K) - alpha*(1-m)*N*P
        dP/dt = beta*alpha*(1-m)*N*P - delta*P

    The refuge fraction m represents the proportion of prey that is
    inaccessible to predators, reducing the effective predation rate.

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 10.0)
        alpha: predation rate (default 0.5)
        m: refuge fraction, 0 <= m < 1 (default 0.3)
        beta: conversion efficiency (default 0.4)
        delta: predator death rate (default 0.2)
        N_0: initial prey population (default 2.0)
        P_0: initial predator population (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 10.0)
        self.alpha = p.get("alpha", 0.5)
        self.m = p.get("m", 0.3)
        self.beta = p.get("beta", 0.4)
        self.delta = p.get("delta", 0.2)
        self.N_0 = p.get("N_0", 2.0)
        self.P_0 = p.get("P_0", 1.0)

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

    def effective_predation_rate(self) -> float:
        """Compute the effective predation rate alpha*(1-m).

        The refuge reduces predation by shielding fraction m of prey.

        Returns:
            Effective predation rate.
        """
        return self.alpha * (1.0 - self.m)

    def equilibrium(self) -> tuple[float, float]:
        """Compute the coexistence equilibrium (N*, P*).

        N* = delta / (beta * alpha * (1 - m))
        P* = r * (1 - N*/K) / (alpha * (1 - m))

        Requires N* < K for the equilibrium to be biologically meaningful.

        Returns:
            Tuple (N_star, P_star).

        Raises:
            ValueError: If coexistence equilibrium does not exist.
        """
        eff = self.effective_predation_rate()
        if eff <= 0:
            raise ValueError(
                "No coexistence equilibrium: effective predation rate is zero "
                f"(m={self.m})"
            )

        N_star = self.delta / (self.beta * eff)
        if N_star >= self.K:
            raise ValueError(
                f"No coexistence equilibrium: N*={N_star:.4f} >= K={self.K:.4f}. "
                "Predator cannot sustain itself."
            )

        P_star = self.r * (1.0 - N_star / self.K) / eff
        if P_star <= 0:
            raise ValueError(
                f"No coexistence equilibrium: P*={P_star:.4f} <= 0."
            )

        return (N_star, P_star)

    def is_stable(self) -> bool:
        """Check if the coexistence equilibrium is locally stable.

        For this model, the Jacobian at (N*, P*) has trace:
            tr(J) = r - 2*r*N*/K - alpha*(1-m)*P* + 0
        The equilibrium is stable when tr(J) < 0.

        Returns:
            True if coexistence equilibrium exists and is stable.
        """
        try:
            N_star, P_star = self.equilibrium()
        except ValueError:
            return False

        eff = self.effective_predation_rate()
        # Jacobian at equilibrium:
        # J11 = r - 2*r*N*/K - eff*P*
        # J12 = -eff*N*
        # J21 = beta*eff*P*
        # J22 = beta*eff*N* - delta = 0 (at equilibrium)
        j11 = self.r - 2.0 * self.r * N_star / self.K - eff * P_star
        # J22 = 0 at equilibrium since beta*eff*N* = delta
        # Trace = J11 + J22 = J11
        # Determinant = J11*J22 - J12*J21 = 0 - (-eff*N*)*(beta*eff*P*)
        # = beta * eff**2 * N* * P* > 0 always
        # Stable when trace < 0
        trace = j11
        return trace < 0

    def jacobian_eigenvalues(self) -> tuple[complex, complex]:
        """Compute eigenvalues of the Jacobian at the coexistence equilibrium.

        Returns:
            Tuple of two eigenvalues (may be complex).

        Raises:
            ValueError: If coexistence equilibrium does not exist.
        """
        N_star, P_star = self.equilibrium()
        eff = self.effective_predation_rate()

        j11 = self.r - 2.0 * self.r * N_star / self.K - eff * P_star
        j12 = -eff * N_star
        j21 = self.beta * eff * P_star
        j22 = self.beta * eff * N_star - self.delta  # = 0 at equilibrium

        trace = j11 + j22
        det = j11 * j22 - j12 * j21
        discriminant = trace**2 - 4.0 * det

        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            return (
                complex(0.5 * (trace + sqrt_disc)),
                complex(0.5 * (trace - sqrt_disc)),
            )
        else:
            sqrt_disc = np.sqrt(-discriminant)
            return (
                complex(0.5 * trace, 0.5 * sqrt_disc),
                complex(0.5 * trace, -0.5 * sqrt_disc),
            )

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
        """Prey-refuge right-hand side.

        dN/dt = r*N*(1 - N/K) - alpha*(1-m)*N*P
        dP/dt = beta*alpha*(1-m)*N*P - delta*P
        """
        N, P = y
        eff = self.alpha * (1.0 - self.m)

        dN = self.r * N * (1.0 - N / self.K) - eff * N * P
        dP = self.beta * eff * N * P - self.delta * P
        return np.array([dN, dP])
