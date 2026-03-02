"""SEIRS (Susceptible-Exposed-Infectious-Recovered-Susceptible) epidemic model.

Combines the latent period of SEIR with the waning immunity of SIRS to produce
recurrent epidemic dynamics with damped oscillations toward an endemic equilibrium.

Equations:
    dS/dt = -beta * S * I / N + xi * R
    dE/dt =  beta * S * I / N - sigma * E
    dI/dt =  sigma * E - gamma * I
    dR/dt =  gamma * I - xi * R

where:
    S, E, I, R = susceptible, exposed, infectious, recovered populations
    beta   = transmission rate
    sigma  = rate of progression from exposed to infectious (1/latent period)
    gamma  = recovery rate (1/infectious period)
    xi     = immunity waning rate (1/immunity duration)
    N      = total population (S + E + I + R, conserved)

Target rediscoveries:
- R0 = beta / gamma (basic reproduction number, without vital dynamics)
- Endemic equilibrium I* when R0 > 1
- Damped oscillations toward endemic state
- Immunity duration = 1 / xi
- Population conservation: S + E + I + R = N
- SINDy recovery of SEIRS ODEs
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SEIRSSimulation(SimulationEnvironment):
    """SEIRS (Susceptible-Exposed-Infectious-Recovered-Susceptible) epidemic model.

    State vector: [S, E, I, R] (population counts, S + E + I + R = N)

    Equations:
        dS/dt = -beta * S * I / N + xi * R
        dE/dt =  beta * S * I / N - sigma * E
        dI/dt =  sigma * E - gamma * I
        dR/dt =  gamma * I - xi * R

    Key quantities:
        R0 = beta / gamma (basic reproduction number)
        Endemic equilibrium exists when R0 > 1
        Immunity duration = 1 / xi
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.5)       # Transmission rate
        self.sigma = p.get("sigma", 0.2)     # 1 / latent period
        self.gamma = p.get("gamma", 0.1)     # Recovery rate
        self.xi = p.get("xi", 0.05)          # Immunity waning rate
        self.N0 = p.get("N0", 1000.0)        # Total population
        self.S_0 = p.get("S_0", 990.0)       # Initial susceptible
        self.E_0 = p.get("E_0", 5.0)         # Initial exposed
        self.I_0 = p.get("I_0", 5.0)         # Initial infectious
        self.R_0_init = p.get("R_0_init", 0.0)  # Initial recovered

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, E, I, R].

        Normalizes initial conditions so that S + E + I + R = N0.
        """
        total = self.S_0 + self.E_0 + self.I_0 + self.R_0_init
        scale = self.N0 / total if total > 0 else 1.0
        self._state = np.array(
            [
                self.S_0 * scale,
                self.E_0 * scale,
                self.I_0 * scale,
                self.R_0_init * scale,
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
        """Return current compartment values [S, E, I, R]."""
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
        """SEIRS right-hand side.

        dS/dt = -beta * S * I / N + xi * R
        dE/dt =  beta * S * I / N - sigma * E
        dI/dt =  sigma * E - gamma * I
        dR/dt =  gamma * I - xi * R
        """
        S, E, Ip, R = y  # noqa: N806 (Ip = infectious population)
        N = S + E + Ip + R
        if N <= 0:
            return np.zeros(4)

        force_of_infection = self.beta * S * Ip / N

        dS = -force_of_infection + self.xi * R
        dE = force_of_infection - self.sigma * E
        dI = self.sigma * E - self.gamma * Ip
        dR = self.gamma * Ip - self.xi * R

        return np.array([dS, dE, dI, dR])

    @property
    def basic_reproduction_number(self) -> float:
        """Basic reproduction number R0 = beta / gamma."""
        return self.beta / self.gamma

    @property
    def R0(self) -> float:
        """Alias for basic_reproduction_number."""
        return self.basic_reproduction_number

    @property
    def latent_period(self) -> float:
        """Mean latent period = 1 / sigma."""
        return 1.0 / self.sigma

    @property
    def infectious_period(self) -> float:
        """Mean infectious period = 1 / gamma."""
        return 1.0 / self.gamma

    @property
    def immunity_duration(self) -> float:
        """Mean immunity duration = 1 / xi."""
        if self.xi <= 0:
            return float("inf")
        return 1.0 / self.xi

    @property
    def generation_time(self) -> float:
        """Mean generation time = 1/sigma + 1/gamma."""
        return self.latent_period + self.infectious_period

    def endemic_equilibrium(self) -> np.ndarray | None:
        """Compute the endemic equilibrium (exists only when R0 > 1).

        At endemic equilibrium with SEIRS dynamics:
            I* = 0 is always an equilibrium (DFE).
            For the non-trivial endemic equilibrium:
                S* = N * gamma / beta = N / R0
                I* = (xi * N / (gamma + xi)) * (1 - 1/R0) * (sigma / (sigma + xi_eff))
            where the exact values come from setting all derivatives to zero.

        Simplified endemic equilibrium:
            S* = N * gamma / beta
            E* = (gamma / sigma) * I*
            From dS/dt = 0: beta * S* * I* / N = xi * R*
            From dR/dt = 0: R* = (gamma / xi) * I*
            From dE/dt = 0: E* = (beta * S* * I*) / (N * sigma)
            S* + E* + I* + R* = N

        Returns None if R0 <= 1 (disease-free equilibrium is the only stable one).
        """
        r0 = self.basic_reproduction_number
        if r0 <= 1.0:
            return None

        # S* = N / R0
        S_star = self.N0 / r0

        # From dR/dt = 0: R* = (gamma / xi) * I*
        # From dE/dt = 0: E* = (gamma / sigma) * I*  (using S*I*/N = gamma*I*/R0... )
        # Actually from dE/dt=0: beta*S*I/N = sigma*E => E* = beta*S*I/(N*sigma)
        # Since S* = N/R0 = N*gamma/beta: E* = (gamma/sigma)*I*
        # S* + E* + I* + R* = N
        # N/R0 + (gamma/sigma)*I* + I* + (gamma/xi)*I* = N
        # I* * (gamma/sigma + 1 + gamma/xi) = N * (1 - 1/R0)
        coeff = self.gamma / self.sigma + 1.0 + self.gamma / self.xi
        I_star = self.N0 * (1.0 - 1.0 / r0) / coeff

        if I_star <= 0:
            return None

        E_star = (self.gamma / self.sigma) * I_star
        R_star = (self.gamma / self.xi) * I_star

        return np.array([S_star, E_star, I_star, R_star])

    def disease_free_equilibrium(self) -> np.ndarray:
        """Disease-free equilibrium: [N, 0, 0, 0]."""
        return np.array([self.N0, 0.0, 0.0, 0.0])

    @property
    def total_population(self) -> float:
        """Current total population N = S + E + I + R."""
        if self._state is None:
            return self.N0
        return float(self._state.sum())
