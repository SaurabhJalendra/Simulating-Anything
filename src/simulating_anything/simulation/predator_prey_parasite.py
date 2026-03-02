"""Predator-prey-parasite 3-species eco-epidemiological model.

A tri-trophic model with prey (N), predator (P), and parasite (Z) that infects
prey. The parasite acts as a biological control agent on prey, while predators
can consume parasitized prey preferentially.

Equations:
    dN/dt = r*N*(1 - N/K) - a*N*P - lambda*N*Z
    dP/dt = e*a*N*P - d*P
    dZ/dt = lambda*N*Z - mu*Z - alpha*Z*P

Key phenomena:
- Tri-trophic interactions: prey-predator-parasite
- Parasite can regulate prey population (biological control)
- Multiple equilibria: prey-only, prey-predator, prey-parasite, coexistence
- Hopf bifurcation possible with all 3 species
- Predator consumption of parasitized prey (alpha term)

Target rediscoveries:
- ODE recovery via SINDy
- Equilibrium classification (prey-only, 2-species, 3-species)
- Lambda sweep: parasite effect on prey regulation
- Biological control analysis
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorPreyParasiteSimulation(SimulationEnvironment):
    """Predator-prey-parasite eco-epidemiological model.

    State vector: [N, P, Z] where N = prey, P = predator, Z = parasite.

    Equations:
        dN/dt = r*N*(1 - N/K) - a*N*P - lambda_*N*Z
        dP/dt = e*a*N*P - d*P
        dZ/dt = lambda_*N*Z - mu*Z - alpha*Z*P

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: carrying capacity (default 10.0)
        a: predation rate (default 0.1)
        e: conversion efficiency (default 0.5)
        d: predator death rate (default 0.2)
        lambda_: infection rate (default 0.15)
        mu: parasite mortality rate (default 0.3)
        alpha: predator consumption of parasitized prey (default 0.05)
        N_0: initial prey population (default 5.0)
        P_0: initial predator population (default 2.0)
        Z_0: initial parasite population (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 10.0)
        self.a = p.get("a", 0.1)
        self.e = p.get("e", 0.5)
        self.d = p.get("d", 0.2)
        self.lambda_ = p.get("lambda_", 0.15)
        self.mu = p.get("mu", 0.3)
        self.alpha = p.get("alpha", 0.05)
        self.N_0 = p.get("N_0", 5.0)
        self.P_0 = p.get("P_0", 2.0)
        self.Z_0 = p.get("Z_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [N, P, Z] with optional perturbation."""
        rng = np.random.default_rng(seed)
        perturbation = rng.uniform(-0.01, 0.01, size=3)
        self._state = np.array(
            [self.N_0, self.P_0, self.Z_0], dtype=np.float64
        ) + perturbation
        # Ensure non-negative after perturbation
        self._state = np.maximum(self._state, 0.0)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with non-negativity clamping."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [N, P, Z]."""
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
        """Predator-prey-parasite right-hand side.

        dN/dt = r*N*(1 - N/K) - a*N*P - lambda_*N*Z
        dP/dt = e*a*N*P - d*P
        dZ/dt = lambda_*N*Z - mu*Z - alpha*Z*P
        """
        N, P, Z = y

        dN = self.r * N * (1.0 - N / self.K) - self.a * N * P - self.lambda_ * N * Z
        dP = self.e * self.a * N * P - self.d * P
        dZ = self.lambda_ * N * Z - self.mu * Z - self.alpha * Z * P

        return np.array([dN, dP, dZ])

    @property
    def total_population(self) -> float:
        """Sum of all three species populations."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    @property
    def is_coexisting(self) -> bool:
        """True if all three species are above the extinction threshold."""
        if self._state is None:
            return False
        threshold = 1e-6
        return bool(np.all(self._state > threshold))

    def compute_equilibria(self) -> dict[str, np.ndarray | None]:
        """Compute possible equilibrium states.

        Returns dict with keys:
            - 'trivial': [0, 0, 0]
            - 'prey_only': [K, 0, 0]
            - 'prey_predator': [N*, P*, 0] (Z=0 subsystem, LV-like)
            - 'prey_parasite': [N*, 0, Z*] (P=0 subsystem)
            - 'coexistence': [N*, P*, Z*] or None if not feasible
        """
        equilibria: dict[str, np.ndarray | None] = {}

        # Trivial: all zero
        equilibria["trivial"] = np.array([0.0, 0.0, 0.0])

        # Prey-only: N=K, P=0, Z=0
        equilibria["prey_only"] = np.array([self.K, 0.0, 0.0])

        # Prey-predator (Z=0): from dP/dt=0 => N* = d/(e*a)
        N_pp = self.d / (self.e * self.a) if self.e * self.a > 0 else None
        if N_pp is not None and 0 < N_pp < self.K:
            # From dN/dt=0 with Z=0: r*N*(1-N/K) - a*N*P = 0 => P = r*(1-N/K)/a
            P_pp = self.r * (1.0 - N_pp / self.K) / self.a
            if P_pp > 0:
                equilibria["prey_predator"] = np.array([N_pp, P_pp, 0.0])
            else:
                equilibria["prey_predator"] = None
        else:
            equilibria["prey_predator"] = None

        # Prey-parasite (P=0): from dZ/dt=0 => N* = mu/lambda_
        if self.lambda_ > 0:
            N_pz = self.mu / self.lambda_
            if 0 < N_pz < self.K:
                # From dN/dt=0 with P=0: r*N*(1-N/K) - lambda_*N*Z = 0
                # => Z = r*(1-N/K)/lambda_
                Z_pz = self.r * (1.0 - N_pz / self.K) / self.lambda_
                if Z_pz > 0:
                    equilibria["prey_parasite"] = np.array([N_pz, 0.0, Z_pz])
                else:
                    equilibria["prey_parasite"] = None
            else:
                equilibria["prey_parasite"] = None
        else:
            equilibria["prey_parasite"] = None

        # Coexistence: all three species > 0
        # From dP/dt=0: N* = d/(e*a)
        # From dZ/dt=0: lambda_*N*Z - mu*Z - alpha*Z*P = 0
        #   => lambda_*N - mu - alpha*P = 0 (for Z>0)
        #   => P* = (lambda_*N* - mu)/alpha
        # From dN/dt=0: r*(1-N*/K) - a*P* - lambda_*Z* = 0
        #   => Z* = (r*(1-N*/K) - a*P*)/lambda_
        if self.e * self.a > 0 and self.alpha > 0 and self.lambda_ > 0:
            N_co = self.d / (self.e * self.a)
            P_co = (self.lambda_ * N_co - self.mu) / self.alpha
            if N_co > 0 and N_co < self.K and P_co > 0:
                Z_co = (self.r * (1.0 - N_co / self.K) - self.a * P_co) / self.lambda_
                if Z_co > 0:
                    equilibria["coexistence"] = np.array([N_co, P_co, Z_co])
                else:
                    equilibria["coexistence"] = None
            else:
                equilibria["coexistence"] = None
        else:
            equilibria["coexistence"] = None

        return equilibria

    def parameter_sweep(
        self,
        param_name: str,
        param_range: np.ndarray,
        n_steps: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep a single parameter and record long-time averages.

        Args:
            param_name: Name of the parameter to sweep.
            param_range: Array of parameter values.
            n_steps: Steps per simulation.

        Returns:
            Dict with param_values, N_avg, P_avg, Z_avg arrays.
        """
        N_avgs = []
        P_avgs = []
        Z_avgs = []

        for val in param_range:
            params = dict(self.config.parameters)
            params[param_name] = float(val)
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters=params,
            )
            sim = PredatorPreyParasiteSimulation(config)
            sim.reset(seed=42)

            # Run to approach steady state
            for _ in range(n_steps):
                sim.step()

            # Average over last 20% for steady-state estimate
            tail_steps = max(1, int(n_steps * 0.2))
            tail_states = []
            for _ in range(tail_steps):
                sim.step()
                tail_states.append(sim.observe().copy())

            tail_states_arr = np.array(tail_states)
            N_avgs.append(np.mean(tail_states_arr[:, 0]))
            P_avgs.append(np.mean(tail_states_arr[:, 1]))
            Z_avgs.append(np.mean(tail_states_arr[:, 2]))

        return {
            "param_values": param_range,
            "N_avg": np.array(N_avgs),
            "P_avg": np.array(P_avgs),
            "Z_avg": np.array(Z_avgs),
        }
