"""Predator-prey model with Beddington-DeAngelis functional response.

Extends Lotka-Volterra with logistic prey growth and mutual predator
interference.  The Beddington-DeAngelis (BD) functional response generalizes
Holling Type II by adding a predator-interference term, which can stabilize
coexistence that would otherwise exhibit the paradox of enrichment.

Equations:
    dN/dt = r*N*(1 - N/K) - a*N*P / (1 + b*N + c*P)
    dP/dt = e*a*N*P / (1 + b*N + c*P) - d*P

Default parameters: r=1.0, K=10.0, a=1.0, b=0.5, c=0.5, e=0.5, d=0.3
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BeddingtonDeAngelisSimulation(SimulationEnvironment):
    """Predator-prey model with Beddington-DeAngelis functional response.

    State vector: [N, P] where N = prey, P = predator.

    Equations:
        dN/dt = r*N*(1 - N/K) - a*N*P / (1 + b*N + c*P)
        dP/dt = e*a*N*P / (1 + b*N + c*P) - d*P

    The parameter c controls mutual predator interference.  When c=0, this
    reduces to the Rosenzweig-MacArthur (Holling Type II) model.

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 10.0)
        a: attack rate (default 1.0)
        b: prey handling / half-saturation coefficient (default 0.5)
        c: predator mutual interference coefficient (default 0.5)
        e: conversion efficiency (default 0.5)
        d: predator death rate (default 0.3)
        N_0: initial prey population (default 5.0)
        P_0: initial predator population (default 2.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 10.0)
        self.a = p.get("a", 1.0)
        self.b_param = p.get("b", 0.5)
        self.c = p.get("c", 0.5)
        self.e = p.get("e", 0.5)
        self.d = p.get("d", 0.3)
        self.N_0 = p.get("N_0", 5.0)
        self.P_0 = p.get("P_0", 2.0)

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

    def functional_response(self, N: float, P: float) -> float:
        """Beddington-DeAngelis functional response.

        f(N, P) = a*N / (1 + b*N + c*P)

        Args:
            N: prey density
            P: predator density

        Returns:
            Per-predator consumption rate.
        """
        return self.a * N / (1.0 + self.b_param * N + self.c * P)

    def coexistence_equilibrium(self) -> tuple[float, float]:
        """Compute the interior coexistence equilibrium (N*, P*).

        At equilibrium:
            dP/dt = 0 implies f(N*, P*) = d/e
            dN/dt = 0 implies r*N*(1-N/K) = f(N*, P*) * P*

        From dP/dt = 0:
            e*a*N* / (1 + b*N* + c*P*) = d
            => P* = (e*a*N* - d - d*b*N*) / (d*c)   [when d*c > 0]

        From dN/dt = 0, substituting f = d/e:
            r*(1 - N*/K) = (d/e) * P*/N*

        We solve for N* from the combined system.

        Returns:
            Tuple (N_star, P_star).

        Raises:
            ValueError: If coexistence equilibrium does not exist.
        """
        # From dP/dt = 0:  e*a*N / (1 + b*N + c*P) = d
        # Rearranging: P = (e*a*N - d*(1 + b*N)) / (d*c)
        #            = (e*a*N - d - d*b*N) / (d*c)
        #            = ((e*a - d*b)*N - d) / (d*c)
        if self.d * self.c <= 0:
            raise ValueError(
                "Cannot compute coexistence equilibrium: d*c must be positive."
            )

        coeff_N = self.e * self.a - self.d * self.b_param
        # For P* > 0 we need (coeff_N * N* - d) > 0 => N* > d / coeff_N

        # From dN/dt = 0:  r*N*(1 - N/K) = a*N*P / (1 + b*N + c*P)
        # Since at equilibrium a*N*P / (1 + b*N + c*P) = (d/e)*P,
        # we get r*(1 - N/K) = (d*P) / (e*N)   [cancel N > 0]
        # Substitute P = ((coeff_N * N - d) / (d*c)):
        # r*(1 - N/K) = (d * ((coeff_N * N - d) / (d*c))) / (e*N)
        # r*(1 - N/K) = (coeff_N * N - d) / (c * e * N)
        # => r * N * (1 - N/K) * c * e = coeff_N * N - d
        # => r*c*e*N - r*c*e*N^2/K = coeff_N * N - d
        # => (r*c*e/K)*N^2 + (coeff_N - r*c*e)*N - d = 0
        A = self.r * self.c * self.e / self.K
        B = coeff_N - self.r * self.c * self.e
        C = -self.d

        discriminant = B * B - 4.0 * A * C
        if discriminant < 0:
            raise ValueError(
                "No coexistence equilibrium: discriminant is negative."
            )

        sqrt_disc = np.sqrt(discriminant)
        # We want the positive root
        N_star_1 = (-B + sqrt_disc) / (2.0 * A)
        N_star_2 = (-B - sqrt_disc) / (2.0 * A)

        # Pick positive root in (0, K)
        N_star = None
        for candidate in [N_star_1, N_star_2]:
            if 0 < candidate < self.K:
                N_star = candidate
                break

        if N_star is None:
            raise ValueError(
                f"No valid coexistence equilibrium: roots N*={N_star_1:.4f}, "
                f"{N_star_2:.4f} not in (0, K={self.K})."
            )

        P_star = (coeff_N * N_star - self.d) / (self.d * self.c)
        if P_star <= 0:
            raise ValueError(
                f"No coexistence equilibrium: P*={P_star:.4f} <= 0."
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
        """Beddington-DeAngelis right-hand side.

        dN/dt = r*N*(1 - N/K) - a*N*P / (1 + b*N + c*P)
        dP/dt = e*a*N*P / (1 + b*N + c*P) - d*P
        """
        N, P = y
        denom = 1.0 + self.b_param * N + self.c * P
        functional = self.a * N * P / denom

        dN = self.r * N * (1.0 - N / self.K) - functional
        dP = self.e * functional - self.d * P
        return np.array([dN, dP])
