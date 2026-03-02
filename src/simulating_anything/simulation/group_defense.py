"""Predator-prey model with group defense (Andrews/Monod-Haldane functional response).

Prey exhibit group defense -- the predation rate decreases at high prey density
due to collective anti-predator behavior. The functional response is non-monotone:
it increases then decreases, with a maximum at N_max = 1/sqrt(c).

Equations:
    dN/dt = r*N*(1 - N/K) - a*N*P/(1 + b*N + c*N^2)
    dP/dt = e*a*N*P/(1 + b*N + c*N^2) - d*P

The functional response f(N) = a*N/(1 + b*N + c*N^2) is the Andrews (or
Monod-Haldane) function. Unlike Holling Type II which saturates monotonically,
this response peaks at N_max = 1/sqrt(c) and then declines due to group defense.

Key properties:
- Non-monotone functional response (group defense)
- Can have multiple positive equilibria (bistability)
- Heteroclinic bifurcation possible
- Maximum functional response at N = 1/sqrt(c)

Default parameters: r=1.0, K=15.0, a=1.5, b=0.1, c=0.05, e=0.5, d=0.3
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GroupDefenseSimulation(SimulationEnvironment):
    """Predator-prey model with group defense (Andrews functional response).

    State vector: [N, P] where N = prey density, P = predator density.

    The prey exhibits group defense: at high prey density, collective behavior
    reduces predation effectiveness. This is modeled by the non-monotone
    Andrews (Monod-Haldane) functional response:

        f(N) = a*N / (1 + b*N + c*N^2)

    which peaks at N_max = 1/sqrt(c) and then declines.

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 15.0)
        a: attack rate (default 1.5)
        b: handling time coefficient (default 0.1)
        c: group defense coefficient (default 0.05)
        e: conversion efficiency, prey to predator (default 0.5)
        d: predator mortality rate (default 0.3)
        N_0: initial prey population (default 5.0)
        P_0: initial predator population (default 2.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 15.0)
        self.a = p.get("a", 1.5)
        self.b = p.get("b", 0.1)
        self.c = p.get("c", 0.05)
        self.e = p.get("e", 0.5)
        self.d = p.get("d", 0.3)
        self.N_0 = p.get("N_0", 5.0)
        self.P_0 = p.get("P_0", 2.0)

    def functional_response(self, N: float | np.ndarray) -> float | np.ndarray:
        """Andrews (Monod-Haldane) functional response.

        f(N) = a*N / (1 + b*N + c*N^2)

        This is non-monotone: it increases for N < 1/sqrt(c) and decreases
        for N > 1/sqrt(c). The maximum value is a/(b + 2*sqrt(c)).

        Args:
            N: Prey density (scalar or array).

        Returns:
            Per-predator consumption rate (same shape as N).
        """
        return self.a * N / (1.0 + self.b * N + self.c * N**2)

    def functional_response_peak(self) -> float:
        """Prey density at which the functional response is maximized.

        N_max = 1 / sqrt(c)

        Returns:
            N_max value.
        """
        return 1.0 / np.sqrt(self.c)

    def functional_response_max_value(self) -> float:
        """Maximum value of the functional response.

        f(N_max) = a / (b + 2*sqrt(c))

        Returns:
            Peak functional response value.
        """
        return self.a / (self.b + 2.0 * np.sqrt(self.c))

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

    def find_equilibria(self) -> list[dict[str, float | str]]:
        """Compute fixed points of the system.

        The system always has:
        - (0, 0): extinction
        - (K, 0): prey-only carrying capacity

        Interior equilibria require solving dP/dt = 0 for N*, which gives
        e*a*N* / (1 + b*N* + c*N*^2) = d. This is a quadratic in N*:
            e*a*N* = d*(1 + b*N* + c*N*^2)
            d*c*N*^2 + (d*b - e*a)*N* + d = 0

        Each positive root N* yields P* from the prey nullcline.

        Returns:
            List of dicts with keys 'N', 'P', 'type'.
        """
        equilibria: list[dict[str, float | str]] = [
            {"N": 0.0, "P": 0.0, "type": "extinction"},
            {"N": self.K, "P": 0.0, "type": "carrying_capacity"},
        ]

        # Interior equilibria from predator nullcline: e*f(N) = d
        # d*c*N^2 + (d*b - e*a)*N + d = 0
        A_coeff = self.d * self.c
        B_coeff = self.d * self.b - self.e * self.a
        C_coeff = self.d

        discriminant = B_coeff**2 - 4.0 * A_coeff * C_coeff
        if discriminant < 0:
            return equilibria

        sqrt_disc = np.sqrt(discriminant)
        for sign in [-1.0, 1.0]:
            N_star = (-B_coeff + sign * sqrt_disc) / (2.0 * A_coeff)
            if N_star <= 0 or N_star >= self.K:
                continue

            # Compute P* from prey nullcline: dN/dt = 0
            # r*N*(1 - N/K) = f(N)*P => P = r*N*(1 - N/K) / f(N)
            fr = self.functional_response(N_star)
            if fr <= 0:
                continue

            growth = self.r * N_star * (1.0 - N_star / self.K)
            P_star = growth / fr
            if P_star > 0:
                equilibria.append({
                    "N": float(N_star),
                    "P": float(P_star),
                    "type": "coexistence",
                })

        return equilibria

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
        """Right-hand side of the group defense predator-prey system.

        dN/dt = r*N*(1 - N/K) - a*N*P/(1 + b*N + c*N^2)
        dP/dt = e*a*N*P/(1 + b*N + c*N^2) - d*P
        """
        N, P = y
        fr = self.functional_response(N)

        dN = self.r * N * (1.0 - N / self.K) - fr * P
        dP = self.e * fr * P - self.d * P
        return np.array([dN, dP])

    def group_defense_sweep(
        self,
        c_values: np.ndarray,
        n_steps: int = 50000,
    ) -> dict[str, np.ndarray]:
        """Sweep group defense coefficient c and track dynamics.

        For each c value, runs a long simulation and measures final
        populations and oscillation amplitude in the second half.

        Args:
            c_values: Array of group defense coefficients to test.
            n_steps: Number of simulation steps per run.

        Returns:
            Dict with c_values, final populations, and amplitude data.
        """
        dt = self.config.dt
        final_N = np.zeros(len(c_values))
        final_P = np.zeros(len(c_values))
        N_amplitude = np.zeros(len(c_values))
        P_amplitude = np.zeros(len(c_values))

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
            sim = GroupDefenseSimulation(config)
            sim.reset()

            states = []
            for _ in range(n_steps):
                sim.step()
                states.append(sim.observe().copy())

            trajectory = np.array(states)
            half = n_steps // 2
            N_half = trajectory[half:, 0]
            P_half = trajectory[half:, 1]

            final_N[i] = trajectory[-1, 0]
            final_P[i] = trajectory[-1, 1]
            N_amplitude[i] = float(np.max(N_half) - np.min(N_half))
            P_amplitude[i] = float(np.max(P_half) - np.min(P_half))

        return {
            "c_values": c_values,
            "final_N": final_N,
            "final_P": final_P,
            "N_amplitude": N_amplitude,
            "P_amplitude": P_amplitude,
        }
