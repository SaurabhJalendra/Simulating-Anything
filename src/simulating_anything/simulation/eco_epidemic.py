"""Eco-epidemiological predator-prey model with disease in prey.

Combines predator-prey dynamics with disease transmission. Susceptible prey (S)
can become infected (I), and predators (P) preferentially consume infected prey
via Holling Type II functional responses.

Equations:
    dS/dt = r*S*(1 - (S+I)/K) - beta*S*I - a1*S*P/(1 + h1*a1*S)
    dI/dt = beta*S*I - a2*I*P/(1 + h2*a2*I) - d*I
    dP/dt = e1*a1*S*P/(1 + h1*a1*S) + e2*a2*I*P/(1 + h2*a2*I) - m*P

Key phenomena:
- Disease-free equilibrium (S*, 0, P*)
- Endemic equilibrium (S*, I*, P*)
- Predators as biological disease control
- R0 = beta*K/d (without predators)
- Holling Type II functional responses with preferential predation on infected prey
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class EcoEpidemicSimulation(SimulationEnvironment):
    """Eco-epidemiological model: predator-prey with disease in prey.

    State vector: [S, I, P] where S = susceptible prey, I = infected prey,
    P = predator.

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: carrying capacity (default 100.0)
        beta: disease transmission rate (default 0.01)
        a1: predation rate on susceptible prey (default 0.1)
        a2: predation rate on infected prey (default 0.3, a2 > a1)
        h1: handling time for susceptible prey (default 0.1)
        h2: handling time for infected prey (default 0.1)
        e1: conversion efficiency from susceptible prey (default 0.5)
        e2: conversion efficiency from infected prey (default 0.3)
        d_disease: disease-induced mortality rate (default 0.2)
        m: predator natural death rate (default 0.3)
        S_0: initial susceptible prey (default 50.0)
        I_0: initial infected prey (default 10.0)
        P_0: initial predator population (default 5.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 100.0)
        self.beta = p.get("beta", 0.01)
        self.a1 = p.get("a1", 0.1)
        self.a2 = p.get("a2", 0.3)
        self.h1 = p.get("h1", 0.1)
        self.h2 = p.get("h2", 0.1)
        self.e1 = p.get("e1", 0.5)
        self.e2 = p.get("e2", 0.3)
        self.d_disease = p.get("d_disease", 0.2)
        self.m = p.get("m", 0.3)
        self.S_0 = p.get("S_0", 50.0)
        self.I_0 = p.get("I_0", 10.0)
        self.P_0 = p.get("P_0", 5.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [S, I, P]."""
        self._state = np.array(
            [self.S_0, self.I_0, self.P_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [S, I, P]."""
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
        """Eco-epidemiological right-hand side.

        dS/dt = r*S*(1 - (S+I)/K) - beta*S*I - a1*S*P/(1 + h1*a1*S)
        dI/dt = beta*S*I - a2*I*P/(1 + h2*a2*I) - d*I
        dP/dt = e1*a1*S*P/(1 + h1*a1*S) + e2*a2*I*P/(1 + h2*a2*I) - m*P
        """
        S, Ip, P = y  # noqa: N806 (S, I, P for susceptible, infected, predator)

        # Holling Type II functional responses
        fr_s = self.a1 * S / (1.0 + self.h1 * self.a1 * S)
        fr_i = self.a2 * Ip / (1.0 + self.h2 * self.a2 * Ip)

        total_prey = S + Ip
        dS = (
            self.r * S * (1.0 - total_prey / self.K)
            - self.beta * S * Ip
            - fr_s * P
        )
        dI = (
            self.beta * S * Ip
            - fr_i * P
            - self.d_disease * Ip
        )
        dP = (
            self.e1 * fr_s * P
            + self.e2 * fr_i * P
            - self.m * P
        )

        return np.array([dS, dI, dP])

    @property
    def R0_no_predators(self) -> float:
        """Basic reproduction number without predators: R0 = beta*K/d."""
        return self.beta * self.K / self.d_disease

    def disease_free_equilibrium(self) -> tuple[float, float]:
        """Compute disease-free equilibrium (S*, P*) with I=0.

        At disease-free equilibrium:
            dS/dt = r*S*(1 - S/K) - a1*S*P/(1 + h1*a1*S) = 0
            dP/dt = e1*a1*S*P/(1 + h1*a1*S) - m*P = 0

        From dP/dt = 0 (with P > 0):
            e1*a1*S / (1 + h1*a1*S) = m
            S* = m / (e1*a1 - h1*a1*m)

        From dS/dt = 0:
            P* = r*(1 - S*/K) * (1 + h1*a1*S*) / a1

        Returns:
            Tuple (S_star, P_star).

        Raises:
            ValueError: If disease-free equilibrium with predators does not exist.
        """
        denom = self.e1 * self.a1 - self.h1 * self.a1 * self.m
        if denom <= 0:
            raise ValueError(
                "No disease-free coexistence: predator cannot sustain on "
                "healthy prey alone."
            )
        S_star = self.m / denom
        if S_star >= self.K:
            raise ValueError(
                f"No disease-free coexistence: S*={S_star:.4f} >= K={self.K}"
            )
        P_star = (
            self.r * (1.0 - S_star / self.K)
            * (1.0 + self.h1 * self.a1 * S_star) / self.a1
        )
        return (S_star, max(0.0, P_star))

    def compute_R0(self) -> float:
        """Effective R0 for disease invasion in presence of predators.

        At the disease-free equilibrium (S*, 0, P*), disease invades if
        the linearized growth rate of I is positive:
            beta*S* - a2*P*/(1 + 0) - d > 0

        The effective R0 = beta*S* / (d + a2*P*).
        """
        try:
            S_star, P_star = self.disease_free_equilibrium()
        except ValueError:
            # No predator equilibrium; R0 without predators
            return self.R0_no_predators
        return self.beta * S_star / (self.d_disease + self.a2 * P_star)

    def predator_effect_sweep(
        self,
        m_values: np.ndarray,
        n_steps: int = 50000,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Sweep predator mortality to measure disease prevalence.

        Higher predator mortality means weaker predation, which should
        increase disease prevalence (predators as biological control).

        Args:
            m_values: Array of predator mortality rates to sweep.
            n_steps: Steps per simulation.
            dt: Timestep override (uses config.dt if None).

        Returns:
            Dict with m_values, disease_prevalence, predator_pop, S, I, P.
        """
        if dt is None:
            dt = self.config.dt

        prevalence = []
        pred_pop = []
        S_final = []
        I_final = []
        P_final = []

        for m_val in m_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "m": m_val,
                },
            )
            sim = EcoEpidemicSimulation(config)
            sim.reset()

            # Run to near steady state
            for _ in range(n_steps):
                sim.step()

            # Average over last portion for steady-state estimate
            states = []
            for _ in range(int(n_steps * 0.1)):
                sim.step()
                states.append(sim.observe().copy())

            states = np.array(states)
            S_avg = np.mean(states[:, 0])
            I_avg = np.mean(states[:, 1])
            P_avg = np.mean(states[:, 2])

            total_prey = S_avg + I_avg
            prev = I_avg / total_prey if total_prey > 1e-10 else 0.0

            prevalence.append(prev)
            pred_pop.append(P_avg)
            S_final.append(S_avg)
            I_final.append(I_avg)
            P_final.append(P_avg)

        return {
            "m_values": m_values,
            "disease_prevalence": np.array(prevalence),
            "predator_pop": np.array(pred_pop),
            "S": np.array(S_final),
            "I": np.array(I_final),
            "P": np.array(P_final),
        }

    def bifurcation_analysis(
        self,
        beta_values: np.ndarray,
        n_steps: int = 50000,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Sweep beta to track equilibria and identify disease invasion.

        Args:
            beta_values: Array of disease transmission rates.
            n_steps: Steps per simulation.
            dt: Timestep override.

        Returns:
            Dict with beta_values, S, I, P, oscillating flags.
        """
        if dt is None:
            dt = self.config.dt

        S_final = []
        I_final = []
        P_final = []
        oscillating = []

        for beta_val in beta_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "beta": beta_val,
                },
            )
            sim = EcoEpidemicSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            # Collect post-transient data
            states = []
            for _ in range(int(n_steps * 0.1)):
                sim.step()
                states.append(sim.observe().copy())

            states = np.array(states)
            S_avg = np.mean(states[:, 0])
            I_avg = np.mean(states[:, 1])
            P_avg = np.mean(states[:, 2])

            # Check for oscillations: std / mean > threshold
            S_std = np.std(states[:, 0])
            is_osc = (S_std / max(S_avg, 1e-10)) > 0.05

            S_final.append(S_avg)
            I_final.append(I_avg)
            P_final.append(P_avg)
            oscillating.append(is_osc)

        return {
            "beta_values": beta_values,
            "S": np.array(S_final),
            "I": np.array(I_final),
            "P": np.array(P_final),
            "oscillating": np.array(oscillating),
        }
