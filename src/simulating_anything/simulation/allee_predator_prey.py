"""Predator-prey model with strong Allee effect and Holling Type II response.

The prey exhibits a strong Allee effect (positive density dependence at low
population), creating bistability between extinction and coexistence.

Equations:
    dN/dt = r*N*(N/A - 1)*(1 - N/K) - a*N*P/(1 + h*a*N)
    dP/dt = e*a*N*P/(1 + h*a*N) - m*P

Target rediscoveries:
- Allee effect: prey declines when N < A (strong Allee threshold)
- Bistability: two stable states (extinction vs coexistence)
- Separatrix: boundary between basins of attraction
- Critical predator density for prey extinction
- Saddle-node bifurcation as Allee threshold changes

Default parameters: r=1.0, A=10, K=100, a=0.01, h=0.1, e=0.5, m=0.3
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AlleePredatorPreySimulation(SimulationEnvironment):
    """Predator-prey model with strong Allee effect.

    State vector: [N, P] where N = prey, P = predator.

    The prey growth rate includes a strong Allee effect: the per-capita
    growth rate is negative for N < A, zero at N = A, and positive for
    A < N < K. This creates bistability between extinction (N=0) and
    a coexistence equilibrium when predation is moderate.

    The predator uses a Holling Type II functional response,
    a*N/(1 + h*a*N), which saturates at high prey density.

    Parameters:
        r: intrinsic growth rate of prey (default 1.0)
        A: Allee threshold -- prey declines below this (default 10.0)
        K: prey carrying capacity (default 100.0)
        a: predator attack rate (default 0.01)
        h: handling time per prey item (default 0.1)
        e: conversion efficiency, prey to predator (default 0.5)
        m: predator mortality rate (default 0.3)
        N_0: initial prey population (default 50.0)
        P_0: initial predator population (default 5.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.A = p.get("A", 10.0)
        self.K = p.get("K", 100.0)
        self.a_rate = p.get("a", 0.01)
        self.h = p.get("h", 0.1)
        self.e = p.get("e", 0.5)
        self.m = p.get("m", 0.3)
        self.N_0 = p.get("N_0", 50.0)
        self.P_0 = p.get("P_0", 5.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [N, P]."""
        self._state = np.array([self.N_0, self.P_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with non-negativity enforcement."""
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
        """Right-hand side of the Allee predator-prey system.

        dN/dt = r*N*(N/A - 1)*(1 - N/K) - a*N*P/(1 + h*a*N)
        dP/dt = e*a*N*P/(1 + h*a*N) - m*P
        """
        N, P = y
        # Prey growth with strong Allee effect
        growth = self.r * N * (N / self.A - 1.0) * (1.0 - N / self.K)
        # Holling Type II functional response
        functional_response = self.a_rate * N / (1.0 + self.h * self.a_rate * N)
        # Predation loss
        predation = functional_response * P

        dN = growth - predation
        dP = self.e * functional_response * P - self.m * P
        return np.array([dN, dP])

    def allee_growth(self, N: float | np.ndarray) -> float | np.ndarray:
        """Prey growth rate without predation: r*N*(N/A - 1)*(1 - N/K).

        This is zero at N=0, N=A, and N=K.
        Negative for 0 < N < A (strong Allee effect).
        Positive for A < N < K (normal growth).
        Negative for N > K (overcrowded).

        Args:
            N: Prey population (scalar or array).

        Returns:
            Growth rate (same shape as N).
        """
        return self.r * N * (N / self.A - 1.0) * (1.0 - N / self.K)

    def holling_type2(self, N: float | np.ndarray) -> float | np.ndarray:
        """Holling Type II functional response: a*N/(1 + h*a*N).

        Saturates at 1/h for large N.

        Args:
            N: Prey population (scalar or array).

        Returns:
            Per-predator consumption rate.
        """
        return self.a_rate * N / (1.0 + self.h * self.a_rate * N)

    def find_equilibria(self) -> list[dict[str, float]]:
        """Compute fixed points of the system.

        The system always has (0, 0) as a trivial equilibrium.
        Prey-only equilibria are at N=A and N=K (with P=0).
        Interior equilibria require solving for N* from the predator
        nullcline and substituting into the prey nullcline.

        Returns:
            List of dicts with keys 'N', 'P', 'type'.
        """
        equilibria = []

        # Trivial: extinction
        equilibria.append({"N": 0.0, "P": 0.0, "type": "extinction"})

        # Prey-only: N = A (Allee threshold, unstable saddle)
        equilibria.append({"N": self.A, "P": 0.0, "type": "allee_threshold"})

        # Prey-only: N = K (carrying capacity, stable if no predator invasion)
        equilibria.append({"N": self.K, "P": 0.0, "type": "carrying_capacity"})

        # Interior equilibrium: predator nullcline gives N*
        # dP/dt = 0 => e*a*N/(1+h*a*N) = m => N* = m / (e*a - h*a*m)
        denom = self.e * self.a_rate - self.h * self.a_rate * self.m
        if denom > 0:
            N_star = self.m / denom
            if 0 < N_star < self.K:
                # Prey nullcline at N*: P* from dN/dt = 0
                growth = self.r * N_star * (N_star / self.A - 1.0) * (
                    1.0 - N_star / self.K
                )
                fr = self.holling_type2(N_star)
                if fr > 0:
                    P_star = growth / fr
                    if P_star > 0:
                        eq_type = (
                            "coexistence_stable"
                            if N_star > self.A
                            else "coexistence_unstable"
                        )
                        equilibria.append({
                            "N": float(N_star),
                            "P": float(P_star),
                            "type": eq_type,
                        })

        return equilibria

    def separatrix_distance(self, N: float, P: float) -> float:
        """Approximate distance from the separatrix.

        Uses a simple heuristic: the separatrix roughly passes through
        the saddle point at (A, 0) in the (N, P) plane. The distance
        is approximated as N - A, which is positive in the coexistence
        basin and negative in the extinction basin.

        A more accurate estimate would require computing the stable
        manifold of the saddle, but this provides a useful first-order
        approximation.

        Args:
            N: Prey population.
            P: Predator population.

        Returns:
            Approximate signed distance. Positive means coexistence basin,
            negative means extinction basin.
        """
        return float(N - self.A)

    def extinction_sweep(
        self,
        P0_values: np.ndarray,
        N_0: float = 50.0,
        n_steps: int = 50000,
    ) -> dict[str, np.ndarray]:
        """Sweep initial predator density and track prey extinction.

        For each P_0 value, runs the simulation and determines whether
        the prey population goes extinct (N drops below 0.1).

        Args:
            P0_values: Array of initial predator densities to test.
            N_0: Initial prey population for all runs.
            n_steps: Number of simulation steps per run.

        Returns:
            Dict with P0_values, final_N, final_P, and extinct flag arrays.
        """
        dt = self.config.dt
        final_N = np.zeros(len(P0_values))
        final_P = np.zeros(len(P0_values))
        extinct = np.zeros(len(P0_values), dtype=bool)

        for i, P0 in enumerate(P0_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "N_0": N_0,
                    "P_0": P0,
                },
            )
            sim = AlleePredatorPreySimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()
                N, P = sim.observe()
                if N < 0.1:
                    extinct[i] = True
                    break

            final_N[i] = sim.observe()[0]
            final_P[i] = sim.observe()[1]

        return {
            "P0_values": P0_values,
            "final_N": final_N,
            "final_P": final_P,
            "extinct": extinct,
        }

    def bifurcation_analysis(
        self,
        A_values: np.ndarray,
        n_steps: int = 50000,
    ) -> dict[str, np.ndarray]:
        """Sweep Allee threshold A and track system outcome.

        For each A value, runs the simulation with default IC and
        determines whether the system reaches coexistence or extinction.

        Args:
            A_values: Array of Allee threshold values.
            n_steps: Number of simulation steps per run.

        Returns:
            Dict with A_values, final_N, final_P, and outcome arrays.
        """
        dt = self.config.dt
        final_N = np.zeros(len(A_values))
        final_P = np.zeros(len(A_values))
        extinct = np.zeros(len(A_values), dtype=bool)

        for i, A_val in enumerate(A_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "A": A_val,
                },
            )
            sim = AlleePredatorPreySimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()
                N, P = sim.observe()
                if N < 0.1:
                    extinct[i] = True
                    break

            final_N[i] = sim.observe()[0]
            final_P[i] = sim.observe()[1]

        return {
            "A_values": A_values,
            "final_N": final_N,
            "final_P": final_P,
            "extinct": extinct,
        }
