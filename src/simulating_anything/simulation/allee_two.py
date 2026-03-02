"""Double Allee predator-prey model: both prey AND predator have Allee effects.

Both species have strong Allee effects with density-dependent growth bounded
by carrying capacities. Predator growth also depends on prey via Holling Type
II functional response. The system exhibits rich multistability: coexistence,
prey-only, predator-only, and mutual extinction are all possible.

Equations:
    dN/dt = r_N*N*(N/A_N - 1)*(1 - N/K_N) - a*N*P/(1 + h*N)
    dP/dt = r_P*P*(P/A_P - 1)*(1 - P/K_P) + e*a*N*P/(1 + h*N)

The prey equation has the standard strong Allee effect with Holling Type II
predation. The predator has its own strong Allee effect: intrinsic growth is
negative below A_P, positive between A_P and K_P, bounded by carrying capacity
K_P. Additionally, predators gain from consuming prey.

Target rediscoveries:
- Double Allee bistability: multiple stable states from both thresholds
- Prey Allee threshold A_N: prey declines when N < A_N
- Predator Allee threshold A_P: predator declines when P < A_P (without prey)
- Basin of attraction mapping via IC sweeps
- Equilibria structure (extinction, prey-only, predator-only, coexistence)

Default parameters: r_N=1.0, K_N=100.0, A_N=5.0, r_P=0.5, K_P=50.0, A_P=3.0,
                    a=0.01, h=0.1, e=0.5, N_0=50.0, P_0=10.0
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AlleeTwoSimulation(SimulationEnvironment):
    """Double Allee predator-prey: both species have strong Allee effects.

    State vector: [N, P] where N = prey, P = predator.

    Both species have strong Allee effects: per-capita growth is negative
    below their respective Allee thresholds (A_N for prey, A_P for predator).
    Both are bounded by carrying capacities (K_N, K_P). Predation uses a
    Holling Type II functional response.

    Parameters:
        r_N: intrinsic prey growth rate (default 1.0)
        K_N: prey carrying capacity (default 100.0)
        A_N: prey Allee threshold (default 5.0)
        r_P: intrinsic predator growth rate (default 0.5)
        K_P: predator carrying capacity (default 50.0)
        A_P: predator Allee threshold (default 3.0)
        a: predator attack rate (default 0.01)
        h: handling time per prey item (default 0.1)
        e: conversion efficiency, prey to predator (default 0.5)
        N_0: initial prey population (default 50.0)
        P_0: initial predator population (default 10.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r_N = p.get("r_N", 1.0)
        self.K_N = p.get("K_N", 100.0)
        self.A_N = p.get("A_N", 5.0)
        self.r_P = p.get("r_P", 0.5)
        self.K_P = p.get("K_P", 50.0)
        self.A_P = p.get("A_P", 3.0)
        self.a_rate = p.get("a", 0.01)
        self.h = p.get("h", 0.1)
        self.e = p.get("e", 0.5)
        self.N_0 = p.get("N_0", 50.0)
        self.P_0 = p.get("P_0", 10.0)

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
        """Right-hand side of the double Allee predator-prey system.

        dN/dt = r_N*N*(N/A_N - 1)*(1 - N/K_N) - a*N*P/(1 + h*N)
        dP/dt = r_P*P*(P/A_P - 1)*(1 - P/K_P) + e*a*N*P/(1 + h*N)
        """
        N, P = y
        # Holling Type II functional response
        functional_response = self.a_rate * N / (1.0 + self.h * N)

        # Prey: strong Allee growth minus predation
        prey_growth = self.r_N * N * (N / self.A_N - 1.0) * (1.0 - N / self.K_N)
        dN = prey_growth - functional_response * P

        # Predator: strong Allee growth plus prey-dependent gain
        pred_growth = self.r_P * P * (P / self.A_P - 1.0) * (1.0 - P / self.K_P)
        dP = pred_growth + self.e * functional_response * P
        return np.array([dN, dP])

    def prey_allee_growth(self, N: float | np.ndarray) -> float | np.ndarray:
        """Prey intrinsic growth: r_N*N*(N/A_N - 1)*(1 - N/K_N).

        Zeros at N=0, N=A_N, N=K_N.
        Negative for 0 < N < A_N (strong Allee effect).
        Positive for A_N < N < K_N.
        Negative for N > K_N.

        Args:
            N: Prey population (scalar or array).

        Returns:
            Growth rate (same shape as N).
        """
        return self.r_N * N * (N / self.A_N - 1.0) * (1.0 - N / self.K_N)

    def predator_allee_growth(self, P: float | np.ndarray) -> float | np.ndarray:
        """Predator intrinsic growth: r_P*P*(P/A_P - 1)*(1 - P/K_P).

        Zeros at P=0, P=A_P, P=K_P.
        Negative for 0 < P < A_P (strong Allee effect).
        Positive for A_P < P < K_P.
        Negative for P > K_P.

        Args:
            P: Predator population (scalar or array).

        Returns:
            Growth rate (same shape as P).
        """
        return self.r_P * P * (P / self.A_P - 1.0) * (1.0 - P / self.K_P)

    def holling_type2(self, N: float | np.ndarray) -> float | np.ndarray:
        """Holling Type II functional response: a*N/(1 + h*N).

        Saturates at a/h for large N.

        Args:
            N: Prey population (scalar or array).

        Returns:
            Per-predator consumption rate.
        """
        return self.a_rate * N / (1.0 + self.h * N)

    def find_equilibria(self) -> list[dict[str, float]]:
        """Compute fixed points of the double Allee system.

        Boundary equilibria: (0,0), (A_N,0), (K_N,0), (0,A_P), (0,K_P).
        Interior equilibria found by nullcline scanning.

        Returns:
            List of dicts with keys 'N', 'P', 'type'.
        """
        equilibria = []

        # Boundary equilibria
        equilibria.append({"N": 0.0, "P": 0.0, "type": "extinction"})
        equilibria.append({"N": self.A_N, "P": 0.0, "type": "prey_allee_threshold"})
        equilibria.append({"N": self.K_N, "P": 0.0, "type": "prey_carrying_capacity"})
        equilibria.append({"N": 0.0, "P": self.A_P, "type": "pred_allee_threshold"})
        equilibria.append({"N": 0.0, "P": self.K_P, "type": "pred_carrying_capacity"})

        # Interior equilibria by nullcline scanning
        # Prey nullcline (dN/dt = 0, N > 0):
        #   r_N*(N/A_N - 1)*(1 - N/K_N) = a*P/(1+h*N)
        #   P = r_N*(N/A_N - 1)*(1 - N/K_N)*(1+h*N)/a
        # Predator nullcline (dP/dt = 0, P > 0):
        #   r_P*(P/A_P - 1)*(1 - P/K_P) + e*fr = 0
        #   This is a cubic in P for each N value.
        N_scan = np.linspace(0.01, self.K_N * 1.1, 2000)
        prev_diff = None
        for N_val in N_scan:
            fr = self.holling_type2(N_val)
            # P from prey nullcline
            prey_factor = (
                self.r_N * (N_val / self.A_N - 1.0) * (1.0 - N_val / self.K_N)
            )
            if self.a_rate == 0:
                prev_diff = None
                continue
            P_prey = prey_factor * (1.0 + self.h * N_val) / self.a_rate
            if P_prey <= 0:
                prev_diff = None
                continue

            # P from predator nullcline: solve
            #   r_P*(P/A_P - 1)*(1 - P/K_P) = -e*fr
            # Let f(P) = r_P*(P/A_P - 1)*(1 - P/K_P) + e*fr
            # We need f(P_prey) = 0
            f_val = (
                self.r_P * (P_prey / self.A_P - 1.0) * (1.0 - P_prey / self.K_P)
                + self.e * fr
            )
            if prev_diff is not None and f_val * prev_diff < 0:
                eq_type = (
                    "coexistence_stable"
                    if N_val > self.A_N and P_prey > self.A_P
                    else "coexistence_unstable"
                )
                is_dup = any(
                    abs(eq["N"] - float(N_val)) < 1.0
                    and abs(eq["P"] - float(P_prey)) < 1.0
                    for eq in equilibria
                )
                if not is_dup:
                    equilibria.append({
                        "N": float(N_val),
                        "P": float(P_prey),
                        "type": eq_type,
                    })
            prev_diff = f_val

        return equilibria

    def basin_of_attraction_sweep(
        self,
        N_range: np.ndarray | None = None,
        P_range: np.ndarray | None = None,
        n_steps: int = 50000,
    ) -> dict[str, np.ndarray]:
        """Map basins of attraction over a grid of initial conditions.

        For each (N_0, P_0) pair, runs the simulation and classifies the
        final state as coexistence, prey-only, predator-only, or extinction.

        Args:
            N_range: Array of initial prey densities.
            P_range: Array of initial predator densities.
            n_steps: Steps per simulation run.

        Returns:
            Dict with N_range, P_range, final_N, final_P, and outcome grids.
        """
        if N_range is None:
            N_range = np.linspace(1.0, 80.0, 15)
        if P_range is None:
            P_range = np.linspace(0.5, 30.0, 15)

        nN, nP = len(N_range), len(P_range)
        final_N = np.zeros((nN, nP))
        final_P = np.zeros((nN, nP))
        outcome = np.zeros((nN, nP), dtype=int)
        # 0=extinction, 1=prey-only, 2=predator-only, 3=coexistence

        dt = self.config.dt
        for i, N0 in enumerate(N_range):
            for j, P0 in enumerate(P_range):
                config = SimulationConfig(
                    domain=self.config.domain,
                    dt=dt,
                    n_steps=n_steps,
                    parameters={
                        **{k: v for k, v in self.config.parameters.items()},
                        "N_0": N0,
                        "P_0": P0,
                    },
                )
                sim = AlleeTwoSimulation(config)
                sim.reset()
                for _ in range(n_steps):
                    sim.step()
                N_f, P_f = sim.observe()
                final_N[i, j] = N_f
                final_P[i, j] = P_f

                if N_f < 0.1 and P_f < 0.1:
                    outcome[i, j] = 0
                elif N_f >= 0.1 and P_f < 0.1:
                    outcome[i, j] = 1
                elif N_f < 0.1 and P_f >= 0.1:
                    outcome[i, j] = 2
                else:
                    outcome[i, j] = 3

        return {
            "N_range": N_range,
            "P_range": P_range,
            "final_N": final_N,
            "final_P": final_P,
            "outcome": outcome,
        }

    def predator_allee_sweep(
        self,
        A_P_values: np.ndarray,
        n_steps: int = 50000,
    ) -> dict[str, np.ndarray]:
        """Sweep predator Allee threshold A_P and track outcomes."""
        dt = self.config.dt
        final_N = np.zeros(len(A_P_values))
        final_P = np.zeros(len(A_P_values))
        pred_survives = np.zeros(len(A_P_values), dtype=bool)

        for i, A_P_val in enumerate(A_P_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "A_P": A_P_val,
                },
            )
            sim = AlleeTwoSimulation(config)
            sim.reset()
            for _ in range(n_steps):
                sim.step()
            final_N[i] = sim.observe()[0]
            final_P[i] = sim.observe()[1]
            pred_survives[i] = final_P[i] > 0.1

        return {
            "A_P_values": A_P_values,
            "final_N": final_N,
            "final_P": final_P,
            "pred_survives": pred_survives,
        }

    def prey_allee_sweep(
        self,
        A_N_values: np.ndarray,
        n_steps: int = 50000,
    ) -> dict[str, np.ndarray]:
        """Sweep prey Allee threshold A_N and track outcomes."""
        dt = self.config.dt
        final_N = np.zeros(len(A_N_values))
        final_P = np.zeros(len(A_N_values))
        prey_survives = np.zeros(len(A_N_values), dtype=bool)

        for i, A_N_val in enumerate(A_N_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "A_N": A_N_val,
                },
            )
            sim = AlleeTwoSimulation(config)
            sim.reset()
            for _ in range(n_steps):
                sim.step()
            final_N[i] = sim.observe()[0]
            final_P[i] = sim.observe()[1]
            prey_survives[i] = final_N[i] > 0.1

        return {
            "A_N_values": A_N_values,
            "final_N": final_N,
            "final_P": final_P,
            "prey_survives": prey_survives,
        }
