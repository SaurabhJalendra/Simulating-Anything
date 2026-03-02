"""SEI (Susceptible-Exposed-Infected) epidemic model simulation.

An epidemic model with NO recovery -- once infected, individuals remain
infected permanently (or die at rate mu). This models diseases with long
infectious periods where recovery is negligible (e.g., HIV, herpes, TB
without treatment).

Equations:
    dS/dt = -beta * S * I / N
    dE/dt =  beta * S * I / N - sigma * E
    dI/dt =  sigma * E - mu * I

where:
    S, E, I = susceptible, exposed, infected populations
    beta   = transmission rate
    sigma  = rate of progression from exposed to infected (1/latency period)
    mu     = disease-induced mortality rate (removal from I)
    N      = S + E + I (total living population, may decrease if mu > 0)

Target rediscoveries:
- R0 = beta * sigma / (sigma * mu) = beta / mu (when sigma cancels)
- More precisely: R0 = beta / mu (next-generation matrix)
- Latent period = 1 / sigma
- Mean infectious duration = 1 / mu
- SINDy recovery of SEI ODEs
- Eventual infection: if R0 > 1, S -> 0 as t -> inf
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SEISimulation(SimulationEnvironment):
    """SEI (Susceptible-Exposed-Infected) epidemic model with no recovery.

    State vector: [S, E, I] (population counts, N = S + E + I may decrease)

    Equations:
        dS/dt = -beta * S * I / N
        dE/dt =  beta * S * I / N - sigma * E
        dI/dt =  sigma * E - mu * I

    Key quantities:
        R0 = beta / mu (basic reproduction number)
        Latent period = 1 / sigma
        Infectious duration = 1 / mu

    When mu = 0, infected individuals accumulate permanently and
    N = S + E + I is conserved. When mu > 0, the living population
    decreases over time as infected individuals die.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.5)       # Transmission rate
        self.sigma = p.get("sigma", 0.2)     # 1 / latency period
        self.mu = p.get("mu", 0.05)          # Disease-induced mortality rate
        self.N0 = p.get("N0", 1000.0)        # Initial total population
        self.S_0 = p.get("S_0", 985.0)       # Initial susceptible
        self.E_0 = p.get("E_0", 10.0)        # Initial exposed
        self.I_0 = p.get("I_0", 5.0)         # Initial infected

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, E, I].

        Normalizes initial conditions so that S + E + I = N0.
        """
        total = self.S_0 + self.E_0 + self.I_0
        scale = self.N0 / total if total > 0 else 1.0
        self._state = np.array(
            [
                self.S_0 * scale,
                self.E_0 * scale,
                self.I_0 * scale,
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
        """Return current compartment values [S, E, I]."""
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
        """SEI right-hand side.

        dS/dt = -beta * S * I / N
        dE/dt =  beta * S * I / N - sigma * E
        dI/dt =  sigma * E - mu * I

        N = S + E + I (living population).
        """
        S, E, Ip = y  # noqa: N806 (Ip = infected population)
        N = S + E + Ip
        if N <= 0:
            return np.zeros(3)

        force_of_infection = self.beta * S * Ip / N

        dS = -force_of_infection
        dE = force_of_infection - self.sigma * E
        dI = self.sigma * E - self.mu * Ip

        return np.array([dS, dE, dI])

    @property
    def R0(self) -> float:
        """Basic reproduction number: R0 = beta / mu.

        From the next-generation matrix: an infected individual transmits
        at rate beta (when S ~ N) and is removed at rate mu, giving
        R0 = beta / mu. The latency period (1/sigma) affects timing
        but not R0 in the SEI framework.
        """
        if self.mu <= 0:
            return float("inf")
        return self.beta / self.mu

    @property
    def latent_period(self) -> float:
        """Mean latent period = 1 / sigma."""
        return 1.0 / self.sigma

    @property
    def infectious_duration(self) -> float:
        """Mean time an individual remains infected = 1 / mu.

        Returns inf if mu = 0 (no removal from infected class).
        """
        if self.mu <= 0:
            return float("inf")
        return 1.0 / self.mu

    @property
    def living_population(self) -> float:
        """Current living population N = S + E + I."""
        if self._state is None:
            return self.N0
        return float(self._state.sum())

    def disease_free_equilibrium(self) -> np.ndarray:
        """Disease-free equilibrium: [N0, 0, 0]."""
        return np.array([self.N0, 0.0, 0.0])

    def endemic_equilibrium(self) -> np.ndarray | None:
        """Endemic equilibrium when R0 > 1 and mu > 0.

        At endemic equilibrium (dS/dt = dE/dt = dI/dt = 0):
            S* = N* / R0 = N* * mu / beta
            E* = mu * I* / sigma
            I* = (N* - S*) / (1 + mu/sigma)

        where N* depends on the steady-state balance. For the SEI model
        with mu > 0, the population eventually reaches a steady state where
        new infections balance mortality.

        Returns None if R0 <= 1 or mu = 0.
        """
        if self.mu <= 0 or self.R0 <= 1.0:
            return None

        # At steady state with constant N*:
        # Force of infection = beta * S* * I* / N* = sigma * E* (from dE/dt=0)
        # sigma * E* = mu * I* (from dI/dt=0)
        # So E* = mu * I* / sigma
        # And S* = sigma * E* * N* / (beta * I*) = mu * N* / beta
        # N* = S* + E* + I* = mu*N*/beta + mu*I*/sigma + I*
        # N*(1 - mu/beta) = I*(1 + mu/sigma)
        # I* = N* * (1 - mu/beta) / (1 + mu/sigma)
        # But N* is unknown; use N0 as approximation for the initial phase
        N_star = self.N0
        S_star = self.mu * N_star / self.beta
        I_star = N_star * (1.0 - self.mu / self.beta) / (1.0 + self.mu / self.sigma)
        E_star = self.mu * I_star / self.sigma

        if I_star < 0 or E_star < 0 or S_star < 0:
            return None

        return np.array([S_star, E_star, I_star])

    def jacobian(self, y: np.ndarray | None = None) -> np.ndarray:
        """Compute the Jacobian matrix at the given state.

        Args:
            y: State vector [S, E, I]. Uses current state if None.

        Returns:
            3x3 Jacobian matrix J[i,j] = d(f_i)/d(y_j).
        """
        if y is None:
            y = self._state
        S, E, Ip = y  # noqa: N806 (Ip = infected population)
        N = S + E + Ip
        if N <= 0:
            return np.zeros((3, 3))

        b = self.beta
        s = self.sigma
        m = self.mu

        # Partial derivatives of force = beta * S * Ip / N
        # Using quotient rule with N = S + E + Ip:
        N2 = N * N
        df_dS = b * Ip * (E + Ip) / N2
        df_dE = -b * S * Ip / N2
        df_dI = b * S * (S + E) / N2

        J = np.array([
            [-df_dS, -df_dE, -df_dI],
            [df_dS, df_dE - s, df_dI],
            [0.0, s, -m],
        ])
        return J

    def compute_divergence(self, y: np.ndarray | None = None) -> float:
        """Compute the divergence of the vector field (trace of Jacobian).

        Args:
            y: State vector [S, E, I]. Uses current state if None.

        Returns:
            Divergence value.
        """
        J = self.jacobian(y)
        return float(np.trace(J))

    def fraction_infected_total(self) -> float:
        """Fraction of initial population currently in E + I.

        Returns (E + I) / N0.
        """
        if self._state is None:
            return (self.E_0 + self.I_0) / self.N0
        return float((self._state[1] + self._state[2]) / self.N0)

    def cumulative_deaths(self) -> float:
        """Cumulative deaths = N0 - (S + E + I).

        Only meaningful when mu > 0.
        """
        if self._state is None:
            return 0.0
        return max(0.0, self.N0 - float(self._state.sum()))
