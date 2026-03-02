"""Predator-prey model with toxic defense (group defense via toxin production).

Prey produce a toxin that reduces predator consumption efficiency at high prey
density. The c*N^2 term in the functional response denominator represents
group toxic defense, creating an Allee-like effect on predation.

Equations:
    dN/dt = r*N*(1 - N/K) - a*N*P/(1 + a*h*N + c*N^2)
    dP/dt = e*a*N*P/(1 + a*h*N + c*N^2) - d*P

When c=0, this reduces to the standard Rosenzweig-MacArthur model with
Holling Type II functional response. For c > 0, predation efficiency drops
at high prey density due to collective toxin production.

Key physics:
- c=0 reduces to standard Holling Type II (Rosenzweig-MacArthur)
- For c > c_crit, prey can escape predation at high density
- Possible bistability: coexistence vs prey-only equilibrium

Default parameters: r=1.0, K=100.0, a=0.5, h=0.02, e=0.4, d=0.2,
c=0.001, N_0=50.0, P_0=10.0
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorPreyToxinSimulation(SimulationEnvironment):
    """Predator-prey model with toxic defense.

    State vector: [N, P] where N = prey, P = predator.

    The prey have logistic growth and produce a toxin at high density
    that inhibits predator consumption. The modified functional response
    is a*N/(1 + a*h*N + c*N^2), where the c*N^2 term represents
    group toxic defense.

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 100.0)
        a: predator attack rate (default 0.5)
        h: handling time per prey item (default 0.02)
        e: conversion efficiency, prey to predator (default 0.4)
        d: predator mortality rate (default 0.2)
        c: toxin strength parameter (default 0.001)
        N_0: initial prey population (default 50.0)
        P_0: initial predator population (default 10.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 100.0)
        self.a = p.get("a", 0.5)
        self.h = p.get("h", 0.02)
        self.e = p.get("e", 0.4)
        self.d = p.get("d", 0.2)
        self.c = p.get("c", 0.001)
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
        """Right-hand side of the toxin predator-prey system.

        dN/dt = r*N*(1 - N/K) - a*N*P/(1 + a*h*N + c*N^2)
        dP/dt = e*a*N*P/(1 + a*h*N + c*N^2) - d*P
        """
        N, P = y
        # Logistic prey growth
        growth = self.r * N * (1.0 - N / self.K)
        # Modified functional response with toxin defense
        fr = self.functional_response(N)
        # Predation loss
        predation = fr * P

        dN = growth - predation
        dP = self.e * fr * P - self.d * P
        return np.array([dN, dP])

    def functional_response(self, N: float | np.ndarray) -> float | np.ndarray:
        """Modified functional response with toxin defense.

        f(N) = a*N / (1 + a*h*N + c*N^2)

        At low N, behaves like Holling Type II: a*N/(1 + a*h*N).
        At high N, the c*N^2 term dominates the denominator,
        causing the response to decrease ~ a/(c*N), reflecting
        group toxic defense.

        When c=0, this is exactly Holling Type II.

        Args:
            N: Prey population (scalar or array).

        Returns:
            Per-predator consumption rate.
        """
        return self.a * N / (1.0 + self.a * self.h * N + self.c * N**2)

    def functional_response_holling2(
        self, N: float | np.ndarray
    ) -> float | np.ndarray:
        """Standard Holling Type II for comparison (c=0 case).

        f(N) = a*N / (1 + a*h*N)

        Args:
            N: Prey population (scalar or array).

        Returns:
            Per-predator consumption rate without toxin.
        """
        return self.a * N / (1.0 + self.a * self.h * N)

    def equilibrium_prey(self) -> float | None:
        """Compute prey equilibrium N* from the predator nullcline.

        At equilibrium, dP/dt = 0 requires:
            e*a*N*/(1 + a*h*N* + c*N*^2) = d

        Rearranging: e*a*N* = d*(1 + a*h*N* + c*N*^2)
        => d*c*N*^2 + (d*a*h - e*a)*N* + d = 0

        Returns:
            Positive prey equilibrium N*, or None if no valid root exists.
        """
        # Quadratic: d*c*N^2 + (d*a*h - e*a)*N + d = 0
        A_coef = self.d * self.c
        B_coef = self.d * self.a * self.h - self.e * self.a
        C_coef = self.d

        if A_coef == 0:
            # Linear case (c=0): standard Holling Type II equilibrium
            if B_coef == 0:
                return None
            N_star = -C_coef / B_coef
            return float(N_star) if N_star > 0 else None

        discriminant = B_coef**2 - 4.0 * A_coef * C_coef
        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        N1 = (-B_coef + sqrt_disc) / (2.0 * A_coef)
        N2 = (-B_coef - sqrt_disc) / (2.0 * A_coef)

        # Return the smallest positive root that is below K
        candidates = [n for n in [N1, N2] if 0 < n < self.K]
        if not candidates:
            return None
        return float(min(candidates))

    def equilibrium_predator(self, N_star: float) -> float:
        """Compute predator equilibrium P* given prey equilibrium N*.

        At equilibrium, dN/dt = 0:
            r*N*(1-N/K) = f(N)*P
            P* = r*N*(1-N/K) / f(N)

        Args:
            N_star: Prey equilibrium density.

        Returns:
            Predator equilibrium density.
        """
        growth = self.r * N_star * (1.0 - N_star / self.K)
        fr = self.functional_response(N_star)
        if fr <= 0:
            return 0.0
        return float(growth / fr)

    def find_equilibria(self) -> list[dict[str, float]]:
        """Compute all fixed points of the system.

        Returns:
            List of dicts with keys 'N', 'P', 'type'.
        """
        equilibria = []

        # Trivial: extinction
        equilibria.append({"N": 0.0, "P": 0.0, "type": "extinction"})

        # Prey-only: N = K, P = 0
        equilibria.append({"N": self.K, "P": 0.0, "type": "prey_only"})

        # Interior equilibrium from predator nullcline
        N_star = self.equilibrium_prey()
        if N_star is not None and 0 < N_star < self.K:
            P_star = self.equilibrium_predator(N_star)
            if P_star > 0:
                equilibria.append({
                    "N": float(N_star),
                    "P": float(P_star),
                    "type": "coexistence",
                })

        return equilibria

    def is_oscillatory(self, states: np.ndarray, threshold: float = 0.05) -> bool:
        """Detect whether a trajectory exhibits oscillatory behavior.

        Uses the amplitude of the second half of prey trajectory
        relative to carrying capacity.

        Args:
            states: Array of shape (n_steps, 2) with [N, P] columns.
            threshold: Minimum relative amplitude to classify as oscillatory.

        Returns:
            True if oscillations are detected.
        """
        half = len(states) // 2
        N_half = states[half:, 0]
        if len(N_half) < 10:
            return False

        amplitude = np.max(N_half) - np.min(N_half)
        relative_amplitude = amplitude / self.K
        return bool(relative_amplitude > threshold)

    def toxin_sweep(
        self,
        c_values: np.ndarray,
        n_steps: int = 50000,
    ) -> dict[str, np.ndarray]:
        """Sweep toxin strength c and measure system outcomes.

        For each c value, runs the simulation and records final
        populations and oscillatory behavior.

        Args:
            c_values: Array of toxin strength values.
            n_steps: Number of simulation steps per run.

        Returns:
            Dict with c_values, final_N, final_P, and oscillatory arrays.
        """
        dt = self.config.dt
        final_N = np.zeros(len(c_values))
        final_P = np.zeros(len(c_values))
        oscillatory = np.zeros(len(c_values), dtype=bool)

        for i, c_val in enumerate(c_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "c": float(c_val),
                },
            )
            sim = PredatorPreyToxinSimulation(config)
            sim.reset()

            states_list = [sim.observe().copy()]
            for _ in range(n_steps):
                sim.step()
                states_list.append(sim.observe().copy())

            traj = np.array(states_list)
            final_N[i] = traj[-1, 0]
            final_P[i] = traj[-1, 1]
            oscillatory[i] = sim.is_oscillatory(traj)

        return {
            "c_values": c_values,
            "final_N": final_N,
            "final_P": final_P,
            "oscillatory": oscillatory,
        }
