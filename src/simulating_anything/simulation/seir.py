"""SEIR epidemic model simulation.

Susceptible-Exposed-Infectious-Recovered compartmental model with a latent
period between exposure and infectiousness.

Equations:
    dS/dt = -beta * S * I / N
    dE/dt =  beta * S * I / N - sigma * E
    dI/dt =  sigma * E - gamma * I
    dR/dt =  gamma * I

where:
    S, E, I, R = susceptible, exposed, infectious, recovered populations
    beta   = transmission rate
    sigma  = rate of progression from exposed to infectious (1/latent period)
    gamma  = recovery rate (1/infectious period)
    N      = total population (S + E + I + R, conserved)

Target rediscoveries:
- R0 = beta / gamma (basic reproduction number)
- Latent period = 1 / sigma
- Infectious period = 1 / gamma
- Population conservation: S + E + I + R = N
- SINDy recovery of SEIR ODEs
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SEIRSimulation(SimulationEnvironment):
    """SEIR (Susceptible-Exposed-Infectious-Recovered) epidemic model.

    State vector: [S, E, I, R] (population counts, S + E + I + R = N)

    Equations:
        dS/dt = -beta * S * I / N
        dE/dt =  beta * S * I / N - sigma * E
        dI/dt =  sigma * E - gamma * I
        dR/dt =  gamma * I

    Key quantity: R0 = beta / gamma (basic reproduction number).
    Epidemic grows when R0 > 1.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.5)       # Transmission rate
        self.sigma = p.get("sigma", 0.2)     # 1 / latent period
        self.gamma = p.get("gamma", 0.1)     # Recovery rate
        self.N = p.get("N", 1000.0)          # Total population
        self.S_0 = p.get("S_0", 990.0)       # Initial susceptible
        self.E_0 = p.get("E_0", 5.0)         # Initial exposed
        self.I_0 = p.get("I_0", 5.0)         # Initial infectious
        self.R_0_init = p.get("R_0_init", 0.0)  # Initial recovered

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, E, I, R].

        Normalizes initial conditions so that S + E + I + R = N.
        """
        total = self.S_0 + self.E_0 + self.I_0 + self.R_0_init
        scale = self.N / total if total > 0 else 1.0
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
        """SEIR right-hand side.

        dS/dt = -beta * S * I / N
        dE/dt =  beta * S * I / N - sigma * E
        dI/dt =  sigma * E - gamma * I
        dR/dt =  gamma * I
        """
        S, E, Ip, R = y  # noqa: N806 (Ip = infectious population)
        N = S + E + Ip + R
        if N <= 0:
            return np.zeros(4)

        force_of_infection = self.beta * S * Ip / N

        dS = -force_of_infection
        dE = force_of_infection - self.sigma * E
        dI = self.sigma * E - self.gamma * Ip
        dR = self.gamma * Ip

        return np.array([dS, dE, dI, dR])

    @property
    def basic_reproduction_number(self) -> float:
        """Basic reproduction number R0 = beta / gamma."""
        return self.beta / self.gamma

    @property
    def latent_period(self) -> float:
        """Mean latent period = 1 / sigma."""
        return 1.0 / self.sigma

    @property
    def infectious_period(self) -> float:
        """Mean infectious period = 1 / gamma."""
        return 1.0 / self.gamma

    @property
    def generation_time(self) -> float:
        """Mean generation time = 1/sigma + 1/gamma."""
        return self.latent_period + self.infectious_period

    def jacobian(self, y: np.ndarray | None = None) -> np.ndarray:
        """Compute the Jacobian matrix at the given state.

        Args:
            y: State vector [S, E, I, R]. Uses current state if None.

        Returns:
            4x4 Jacobian matrix J[i,j] = d(f_i)/d(y_j).
        """
        if y is None:
            y = self._state
        S, E, Ip, R = y  # noqa: N806 (Ip = infectious population)
        N = S + E + Ip + R
        if N <= 0:
            return np.zeros((4, 4))

        b = self.beta
        s = self.sigma
        g = self.gamma

        # Partial derivatives of force_of_infection = beta * S * Ip / N
        # Using quotient rule with N = S + E + Ip + R:
        # d(S*Ip/N)/dS = Ip*(N-S)/N^2 = Ip*(E+Ip+R)/N^2
        # d(S*Ip/N)/dE = -S*Ip/N^2
        # d(S*Ip/N)/dI = S*(N-Ip)/N^2 = S*(S+E+R)/N^2
        # d(S*Ip/N)/dR = -S*Ip/N^2
        N2 = N * N
        df_dS = b * Ip * (E + Ip + R) / N2
        df_dE = -b * S * Ip / N2
        df_dI = b * S * (S + E + R) / N2
        df_dR = -b * S * Ip / N2

        J = np.array([
            [-df_dS, -df_dE, -df_dI, -df_dR],
            [df_dS, df_dE - s, df_dI, df_dR],
            [0.0, s, -g, 0.0],
            [0.0, 0.0, g, 0.0],
        ])
        return J

    def compute_divergence(self, y: np.ndarray | None = None) -> float:
        """Compute the divergence of the vector field (trace of Jacobian).

        For SEIR the divergence = -(sigma + gamma + beta*(...)) which is
        always negative, indicating volume contraction in phase space.

        Args:
            y: State vector [S, E, I, R]. Uses current state if None.

        Returns:
            Divergence value (should be negative).
        """
        J = self.jacobian(y)
        return float(np.trace(J))

    @property
    def peak_infected_theory(self) -> float:
        """Theoretical peak infected fraction (approximate).

        For SEIR with S(0) ~ N:
            I_peak ~ N * (1 - 1/R0 - ln(R0)/R0)

        This is an approximation that holds when E(0), I(0) << N.
        """
        r0 = self.basic_reproduction_number
        if r0 <= 1.0:
            return self.I_0
        return self.N * (1.0 - 1.0 / r0 - np.log(r0) / r0)

    def final_size(self) -> float:
        """Compute theoretical final epidemic size using Newton's method.

        The final size R_inf satisfies:
            R_inf = N * (1 - S(0)/N * exp(-R0 * R_inf / N))

        Returns fraction of population ultimately infected.
        """
        r0 = self.basic_reproduction_number
        if r0 <= 1.0:
            return 0.0
        s0_frac = self.S_0 / self.N
        # Newton's method for final size equation
        R_inf = 0.9
        for _ in range(50):
            f = R_inf - 1.0 + s0_frac * np.exp(-r0 * R_inf)
            df = 1.0 + s0_frac * r0 * np.exp(-r0 * R_inf)
            R_inf -= f / df
        return max(0.0, R_inf)

    def disease_free_equilibrium(self) -> np.ndarray:
        """Disease-free equilibrium: [N, 0, 0, 0]."""
        return np.array([self.N, 0.0, 0.0, 0.0])

    @property
    def total_population(self) -> float:
        """Current total population N = S + E + I + R."""
        if self._state is None:
            return self.N
        return float(self._state.sum())
