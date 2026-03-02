"""SIRS (Susceptible-Infected-Recovered-Susceptible) epidemic model simulation.

SIR model with waning immunity -- recovered individuals return to susceptible:

    dS/dt = -beta * S * I / N + xi * R
    dI/dt =  beta * S * I / N - gamma * I
    dR/dt =  gamma * I - xi * R

where:
    S, I, R = susceptible, infected, recovered populations
    beta  = transmission rate
    gamma = recovery rate
    xi    = immunity waning rate (1/xi = mean duration of immunity)
    N     = total population S + I + R (constant)

Target rediscoveries:
- R0 = beta / gamma
- Endemic equilibrium: I* = N * (1 - 1/R0) * xi / (xi + gamma) when R0 > 1
- Damped oscillations toward endemic equilibrium (unlike SIR)
- SINDy recovery of SIRS ODEs with waning immunity term
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SIRSSimulation(SimulationEnvironment):
    """SIRS (Susceptible-Infected-Recovered-Susceptible) epidemic model.

    State vector: [S, I, R] (population counts, S + I + R = N = const)

    Equations:
        dS/dt = -beta * S * I / N + xi * R
        dI/dt =  beta * S * I / N - gamma * I
        dR/dt =  gamma * I - xi * R

    Key quantities:
        R0 = beta / gamma (basic reproduction number)
        Endemic equilibrium exists when R0 > 1
        Waning immunity xi -> 0 recovers standard SIR
        Waning immunity xi -> inf recovers SIS dynamics
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.5)       # Transmission rate
        self.gamma = p.get("gamma", 0.15)    # Recovery rate
        self.xi = p.get("xi", 0.05)          # Immunity waning rate
        self.N0 = p.get("N0", 1000.0)        # Total population (constant)
        self.I_0 = p.get("I_0", 10.0)        # Initial infected
        self.R_0_init = p.get("R_0_init", 0.0)  # Initial recovered

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, I, R].

        Normalizes initial population so S + I + R = N0.
        """
        living_total = (self.N0 - self.I_0 - self.R_0_init) + self.I_0 + self.R_0_init
        scale = self.N0 / living_total if living_total > 0 else 1.0
        S_0 = (self.N0 - self.I_0 - self.R_0_init) * scale
        I_0 = self.I_0 * scale
        R_0 = self.R_0_init * scale
        self._state = np.array([S_0, I_0, R_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current compartment values [S, I, R]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step with renormalization."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Clamp non-negative
        self._state = np.maximum(self._state, 0.0)
        # Renormalize to ensure S + I + R = N0
        total = self._state.sum()
        if total > 0:
            self._state *= self.N0 / total

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """SIRS right-hand side.

        dS/dt = -beta * S * I / N + xi * R
        dI/dt =  beta * S * I / N - gamma * I
        dR/dt =  gamma * I - xi * R
        """
        S, Ip, R = y
        N = S + Ip + R
        if N <= 0:
            return np.zeros(3)

        force_of_infection = self.beta * S * Ip / N

        dS = -force_of_infection + self.xi * R
        dI = force_of_infection - self.gamma * Ip
        dR = self.gamma * Ip - self.xi * R

        return np.array([dS, dI, dR])

    @property
    def R0(self) -> float:
        """Basic reproduction number: R0 = beta / gamma."""
        return self.beta / self.gamma

    def endemic_equilibrium(self) -> np.ndarray | None:
        """Compute the endemic equilibrium [S*, I*, R*].

        At steady state (dI/dt = 0 => S* = N*gamma/beta = N/R0,
        dR/dt = 0 => R* = gamma*I*/xi):
            S* = N / R0
            I* = N * (1 - 1/R0) * xi / (xi + gamma)
            R* = gamma * I* / xi

        Returns:
            Endemic equilibrium array, or None if R0 <= 1.
        """
        r0 = self.R0
        if r0 <= 1.0:
            return None

        N = self.N0
        S_star = N / r0
        # From S* + I*(1 + gamma/xi) = N => I* = N*(1-1/R0)*xi/(xi+gamma)
        I_star = N * (1.0 - 1.0 / r0) * self.xi / (self.xi + self.gamma)
        R_star = N - S_star - I_star

        return np.array([S_star, I_star, R_star])

    def disease_free_equilibrium(self) -> np.ndarray:
        """Disease-free equilibrium: [N, 0, 0]."""
        return np.array([self.N0, 0.0, 0.0])

    @property
    def total_population(self) -> float:
        """Current total population S + I + R (should equal N0)."""
        if self._state is None:
            return self.N0
        return float(self._state.sum())

    def effective_reproduction_number(self) -> float:
        """Effective reproduction number R_eff = R0 * S / N.

        At the endemic equilibrium, R_eff = 1.
        """
        if self._state is None:
            return self.R0
        S = self._state[0]
        N = self._state.sum()
        if N <= 0:
            return 0.0
        return self.R0 * S / N

    def sir_limit_final_size(self) -> float:
        """Theoretical SIR final size (xi=0 limit) using Newton's method.

        The final size R_inf satisfies:
            R_inf = 1 - S(0)/N * exp(-R0 * R_inf)

        Returns:
            Fraction of initial population ultimately infected (SIR limit).
        """
        r0 = self.R0
        if r0 <= 1.0:
            return 0.0
        s0_frac = (self.N0 - self.I_0 - self.R_0_init) / self.N0
        R_inf = 0.9
        for _ in range(50):
            f = R_inf - 1.0 + s0_frac * np.exp(-r0 * R_inf)
            df = 1.0 + s0_frac * r0 * np.exp(-r0 * R_inf)
            R_inf -= f / df
        return max(0.0, R_inf)
