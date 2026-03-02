"""Two-Patch Predator-Prey spatial migration model.

Predator-prey dynamics on two connected patches with migration between them:

    dN1/dt = r*N1*(1 - N1/K) - a*N1*P1 + m_N*(N2 - N1)   (prey patch 1)
    dP1/dt = e*a*N1*P1 - d*P1 + m_P*(P2 - P1)             (pred patch 1)
    dN2/dt = r*N2*(1 - N2/K) - a*N2*P2 + m_N*(N1 - N2)   (prey patch 2)
    dP2/dt = e*a*N2*P2 - d*P2 + m_P*(P1 - P2)             (pred patch 2)

Migration couples the two patches diffusively. Key phenomena:
- High migration -> synchronized patch dynamics
- Low migration -> asynchronous oscillations with potential rescue effect
- Zero migration -> independent predator-prey cycles per patch
- Symmetric ICs + symmetric params -> symmetric dynamics preserved

Default parameters: r=1.0, K=10.0, a=0.5, e=0.4, d=0.2,
                    m_N=0.1 (prey migration), m_P=0.05 (pred migration)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TwoPatchSimulation(SimulationEnvironment):
    """Two-patch predator-prey model with diffusive migration.

    State vector: [N1, P1, N2, P2] where N = prey, P = predator.

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 10.0)
        a: attack rate / predation coefficient (default 0.5)
        e: conversion efficiency (default 0.4)
        d: predator death rate (default 0.2)
        m_N: prey migration rate between patches (default 0.1)
        m_P: predator migration rate between patches (default 0.05)
        N1_0: initial prey on patch 1 (default 5.0)
        P1_0: initial predator on patch 1 (default 2.0)
        N2_0: initial prey on patch 2 (default 3.0)
        P2_0: initial predator on patch 2 (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 10.0)
        self.a = p.get("a", 0.5)
        self.e = p.get("e", 0.4)
        self.d = p.get("d", 0.2)
        self.m_N = p.get("m_N", 0.1)
        self.m_P = p.get("m_P", 0.05)
        self.N1_0 = p.get("N1_0", 5.0)
        self.P1_0 = p.get("P1_0", 2.0)
        self.N2_0 = p.get("N2_0", 3.0)
        self.P2_0 = p.get("P2_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize both patches with specified populations."""
        self._state = np.array(
            [self.N1_0, self.P1_0, self.N2_0, self.P2_0],
            dtype=np.float64,
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        # Clamp populations to be non-negative
        self._state = np.maximum(self._state, 0.0)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [N1, P1, N2, P2]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for the two-patch predator-prey system.

        dN1/dt = r*N1*(1 - N1/K) - a*N1*P1 + m_N*(N2 - N1)
        dP1/dt = e*a*N1*P1 - d*P1 + m_P*(P2 - P1)
        dN2/dt = r*N2*(1 - N2/K) - a*N2*P2 + m_N*(N1 - N2)
        dP2/dt = e*a*N2*P2 - d*P2 + m_P*(P1 - P2)
        """
        N1, P1, N2, P2 = state

        dN1 = (
            self.r * N1 * (1 - N1 / self.K)
            - self.a * N1 * P1
            + self.m_N * (N2 - N1)
        )
        dP1 = (
            self.e * self.a * N1 * P1
            - self.d * P1
            + self.m_P * (P2 - P1)
        )
        dN2 = (
            self.r * N2 * (1 - N2 / self.K)
            - self.a * N2 * P2
            + self.m_N * (N1 - N2)
        )
        dP2 = (
            self.e * self.a * N2 * P2
            - self.d * P2
            + self.m_P * (P1 - P2)
        )

        return np.array([dN1, dP1, dN2, dP2])

    def total_prey(self) -> float:
        """Return the total prey across both patches: N1 + N2."""
        if self._state is None:
            return 0.0
        return float(self._state[0] + self._state[2])

    def total_predator(self) -> float:
        """Return the total predator across both patches: P1 + P2."""
        if self._state is None:
            return 0.0
        return float(self._state[1] + self._state[3])

    def patch_difference(self) -> float:
        """Return the Euclidean distance between patch states.

        Measures how different the two patches are:
        sqrt((N1-N2)^2 + (P1-P2)^2).

        Returns:
            Non-negative distance between patch 1 and patch 2 states.
        """
        if self._state is None:
            return 0.0
        N1, P1, N2, P2 = self._state
        return float(np.sqrt((N1 - N2) ** 2 + (P1 - P2) ** 2))

    def is_synchronized(self, threshold: float = 0.1) -> bool:
        """Check if both patches have approximately equal populations.

        Args:
            threshold: Maximum patch_difference to count as synchronized.

        Returns:
            True if patch states are within threshold of each other.
        """
        return self.patch_difference() < threshold

    def migration_sweep(
        self,
        m_values: np.ndarray,
        sweep_prey: bool = True,
        n_steps: int = 10000,
        n_transient: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep migration rate and measure steady-state patch difference.

        Args:
            m_values: Array of migration rates to sweep.
            sweep_prey: If True, sweep m_N; if False, sweep m_P.
            n_steps: Measurement steps after transient.
            n_transient: Transient steps to discard.

        Returns:
            Dict with 'm', 'mean_diff', and 'mean_total_prey' arrays.
        """
        mean_diffs = np.empty(len(m_values))
        mean_totals = np.empty(len(m_values))

        for i, m_val in enumerate(m_values):
            params = {
                "r": self.r, "K": self.K, "a": self.a,
                "e": self.e, "d": self.d,
                "m_N": float(m_val) if sweep_prey else self.m_N,
                "m_P": self.m_P if sweep_prey else float(m_val),
                "N1_0": self.N1_0, "P1_0": self.P1_0,
                "N2_0": self.N2_0, "P2_0": self.P2_0,
            }
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters=params,
            )
            sim = TwoPatchSimulation(config)
            sim.reset()

            for _ in range(n_transient):
                sim.step()

            diff_sum = 0.0
            total_sum = 0.0
            for _ in range(n_steps):
                sim.step()
                diff_sum += sim.patch_difference()
                total_sum += sim.total_prey()

            mean_diffs[i] = diff_sum / n_steps
            mean_totals[i] = total_sum / n_steps

        return {
            "m": m_values,
            "mean_diff": mean_diffs,
            "mean_total_prey": mean_totals,
        }

    def coexistence_equilibrium(self) -> tuple[float, float]:
        """Compute the single-patch coexistence equilibrium (N*, P*).

        At equilibrium with zero migration, each patch satisfies:
            N* = d / (e*a)
            P* = (r/a) * (1 - N*/K)

        Returns:
            Tuple (N_star, P_star).
        """
        N_star = self.d / (self.e * self.a)
        P_star = (self.r / self.a) * (1 - N_star / self.K)
        return float(N_star), float(P_star)
