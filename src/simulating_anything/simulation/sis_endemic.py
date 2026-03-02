"""SIS (Susceptible-Infected-Susceptible) endemic model simulation.

Simplest compartmental epidemic model without permanent immunity:

    dS/dt = -beta * S * I / N + gamma * I
    dI/dt =  beta * S * I / N - gamma * I

where:
    S, I = susceptible and infected populations
    beta  = transmission rate
    gamma = recovery rate
    N     = S + I = constant total population (conservation law)

Key properties:
    R0 = beta / gamma -- basic reproduction number
    Endemic equilibrium: I* = N * (1 - 1/R0) when R0 > 1
    Disease-free equilibrium (DFE): I = 0, stable when R0 < 1
    Transcritical bifurcation at R0 = 1

Target rediscoveries:
- R0 = beta / gamma
- Endemic equilibrium I* = N * (1 - 1/R0)
- SINDy recovery of SIS ODEs
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SISEndemicSimulation(SimulationEnvironment):
    """SIS (Susceptible-Infected-Susceptible) endemic model.

    State vector: [S, I] (population counts)

    Equations:
        dS/dt = -beta * S * I / N + gamma * I
        dI/dt =  beta * S * I / N - gamma * I

    where N = S + I is the constant total population.

    Key quantities:
        R0 = beta / gamma -- basic reproduction number
        I* = N * (1 - 1/R0) -- endemic equilibrium (R0 > 1)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.5)       # Transmission rate
        self.gamma = p.get("gamma", 0.2)     # Recovery rate
        self.N0 = p.get("N0", 1000.0)        # Total population
        self.I_0 = p.get("I_0", 10.0)        # Initial infected

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, I].

        Ensures S + I = N0 exactly.
        """
        I_init = min(self.I_0, self.N0)
        self._state = np.array(
            [self.N0 - I_init, I_init],
            dtype=np.float64,
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current compartment values [S, I]."""
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
        # Clamp non-negative
        self._state = np.maximum(self._state, 0.0)
        # Renormalize to preserve S + I = N0 exactly
        total = self._state.sum()
        if total > 0:
            self._state *= self.N0 / total

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """SIS right-hand side.

        dS/dt = -beta * S * I / N + gamma * I
        dI/dt =  beta * S * I / N - gamma * I

        N = S + I (constant total population).
        """
        S, Ip = y  # noqa: N806 (Ip = infected population)
        N = S + Ip
        if N <= 0:
            return np.zeros(2)

        force_of_infection = self.beta * S * Ip / N

        dS = -force_of_infection + self.gamma * Ip
        dI = force_of_infection - self.gamma * Ip

        return np.array([dS, dI])

    @property
    def R0(self) -> float:
        """Basic reproduction number: R0 = beta / gamma."""
        return self.beta / self.gamma

    def endemic_equilibrium(self) -> np.ndarray:
        """Endemic equilibrium state [S*, I*].

        When R0 > 1:
            I* = N * (1 - 1/R0) = N * (1 - gamma/beta)
            S* = N / R0 = N * gamma / beta

        When R0 <= 1, returns the disease-free equilibrium.
        """
        r0 = self.R0
        if r0 <= 1.0:
            return self.disease_free_equilibrium()
        I_star = self.N0 * (1.0 - 1.0 / r0)
        S_star = self.N0 / r0
        return np.array([S_star, I_star])

    def disease_free_equilibrium(self) -> np.ndarray:
        """Disease-free equilibrium: [N0, 0]."""
        return np.array([self.N0, 0.0])

    @property
    def total_population(self) -> float:
        """Current total population S + I (should equal N0)."""
        if self._state is None:
            return self.N0
        return float(self._state.sum())
