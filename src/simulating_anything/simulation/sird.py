"""SIRD (Susceptible-Infected-Recovered-Deceased) epidemic model simulation.

Extension of the SIR model with a deceased compartment to model disease mortality:

    dS/dt = -beta * S * I / N
    dI/dt =  beta * S * I / N - gamma * I - mu * I
    dR/dt =  gamma * I
    dD/dt =  mu * I

where:
    S, I, R, D = susceptible, infected, recovered, deceased populations
    beta   = transmission rate
    gamma  = recovery rate
    mu     = disease-induced mortality rate
    N      = living population (S + I + R, excludes D)

Target rediscoveries:
- R0 = beta / (gamma + mu)
- Case Fatality Rate CFR = mu / (gamma + mu)
- Infection Fatality Rate (from simulation)
- SINDy recovery of SIRD ODEs
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SIRDSimulation(SimulationEnvironment):
    """SIRD (Susceptible-Infected-Recovered-Deceased) epidemic model.

    State vector: [S, I, R, D] (population counts)

    Equations:
        dS/dt = -beta * S * I / N
        dI/dt =  beta * S * I / N - gamma * I - mu * I
        dR/dt =  gamma * I
        dD/dt =  mu * I

    where N = S + I + R is the living population (D excluded).

    Key quantities:
        R0  = beta / (gamma + mu) -- basic reproduction number
        CFR = mu / (gamma + mu)   -- case fatality rate
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.4)       # Transmission rate
        self.gamma = p.get("gamma", 0.1)     # Recovery rate
        self.mu = p.get("mu", 0.02)          # Disease-induced mortality rate
        self.N0 = p.get("N0", 1000.0)        # Initial total population
        self.S_0 = p.get("S_0", 990.0)       # Initial susceptible
        self.I_0 = p.get("I_0", 10.0)        # Initial infected
        self.R_0_init = p.get("R_0_init", 0.0)  # Initial recovered
        self.D_0 = p.get("D_0", 0.0)         # Initial deceased

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, I, R, D].

        Normalizes initial living population (S + I + R) to N0 - D_0.
        """
        living_total = self.S_0 + self.I_0 + self.R_0_init
        target_living = self.N0 - self.D_0
        scale = target_living / living_total if living_total > 0 else 1.0
        self._state = np.array(
            [
                self.S_0 * scale,
                self.I_0 * scale,
                self.R_0_init * scale,
                self.D_0,
            ],
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
        """Return current compartment values [S, I, R, D]."""
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

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """SIRD right-hand side.

        dS/dt = -beta * S * I / N
        dI/dt =  beta * S * I / N - gamma * I - mu * I
        dR/dt =  gamma * I
        dD/dt =  mu * I

        N = S + I + R (living population, excludes D).
        """
        S, Ip, R, D = y  # noqa: N806 (Ip = infected population)
        N = S + Ip + R  # Living population
        if N <= 0:
            return np.zeros(4)

        force_of_infection = self.beta * S * Ip / N

        dS = -force_of_infection
        dI = force_of_infection - self.gamma * Ip - self.mu * Ip
        dR = self.gamma * Ip
        dD = self.mu * Ip

        return np.array([dS, dI, dR, dD])

    @property
    def R0(self) -> float:
        """Basic reproduction number: R0 = beta / (gamma + mu)."""
        return self.beta / (self.gamma + self.mu)

    def case_fatality_rate(self) -> float:
        """Case fatality rate: CFR = mu / (gamma + mu).

        The fraction of infected individuals who die (rather than recover).
        """
        return self.mu / (self.gamma + self.mu)

    def infection_fatality_rate(self) -> float:
        """Infection fatality rate from current simulation state.

        IFR = D / (R + D), the fraction of resolved cases that died.
        Returns 0.0 if no cases have resolved yet.
        """
        if self._state is None:
            return 0.0
        R, D = self._state[2], self._state[3]
        resolved = R + D
        if resolved <= 0:
            return 0.0
        return float(D / resolved)

    def final_size(self) -> float:
        """Theoretical final epidemic size using Newton's method.

        The final size R_inf satisfies:
            R_inf = 1 - S(0)/N0 * exp(-R0 * R_inf)

        Returns fraction of initial population ultimately infected.
        """
        r0 = self.R0
        if r0 <= 1.0:
            return 0.0
        s0_frac = self.S_0 / self.N0
        # Newton's method for final size equation
        R_inf = 0.9
        for _ in range(50):
            f = R_inf - 1.0 + s0_frac * np.exp(-r0 * R_inf)
            df = 1.0 + s0_frac * r0 * np.exp(-r0 * R_inf)
            R_inf -= f / df
        return max(0.0, R_inf)

    def peak_infected(self) -> float:
        """Theoretical peak infected fraction.

        I_peak = I_0/N + S_0/N - (1/R0) * (1 + ln(R0 * S_0/N))
        """
        r0 = self.R0
        s0_frac = self.S_0 / self.N0
        i0_frac = self.I_0 / self.N0
        if r0 * s0_frac <= 1:
            return i0_frac
        return i0_frac + s0_frac - (1.0 / r0) * (1.0 + np.log(r0 * s0_frac))

    def disease_free_equilibrium(self) -> np.ndarray:
        """Disease-free equilibrium: [N0, 0, 0, 0]."""
        return np.array([self.N0, 0.0, 0.0, 0.0])

    @property
    def living_population(self) -> float:
        """Current living population N = S + I + R."""
        if self._state is None:
            return self.N0
        return float(self._state[0] + self._state[1] + self._state[2])

    @property
    def total_population(self) -> float:
        """Current total population S + I + R + D (should equal N0)."""
        if self._state is None:
            return self.N0
        return float(self._state.sum())
