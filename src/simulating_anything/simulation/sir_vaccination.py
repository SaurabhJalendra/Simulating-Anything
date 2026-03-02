"""SIR with vaccination and vital dynamics simulation.

Extension of the SIR model with vaccination and demographic turnover:

    dS/dt = mu*N - beta*S*I/N - nu*S - mu*S
    dI/dt = beta*S*I/N - gamma*I - mu*I
    dR/dt = gamma*I + nu*S - mu*R

where:
    S, I, R = susceptible, infected, recovered populations
    beta  = transmission rate
    gamma = recovery rate
    mu    = birth/death rate (vital dynamics)
    nu    = vaccination rate
    N     = total population (S + I + R, conserved when mu_birth = mu_death)

Target rediscoveries:
- R_0 = beta / (gamma + mu)
- R_eff = R_0 * (1 - p), where p = vaccination coverage at endemic equilibrium
- Herd immunity threshold: p_c = 1 - 1/R_0
- Critical vaccination rate: nu_c such that R_eff = 1
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SIRVaccinationSimulation(SimulationEnvironment):
    """SIR model with vaccination and vital dynamics.

    State vector: [S, I, R] (population counts, S + I + R = N)

    Equations:
        dS/dt = mu*N - beta*S*I/N - nu*S - mu*S
        dI/dt = beta*S*I/N - gamma*I - mu*I
        dR/dt = gamma*I + nu*S - mu*R

    The total population N = S + I + R is conserved because the birth
    rate (mu*N) equals the total death rate (mu*(S + I + R) = mu*N).
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.3)
        self.gamma = p.get("gamma", 0.1)
        self.mu = p.get("mu", 0.01)
        self.nu = p.get("nu", 0.0)
        self.N = p.get("N", 1000.0)
        self.S_0 = p.get("S_0", self.N * 0.99)
        self.I_0 = p.get("I_0", self.N * 0.01)
        self.R_0_init = p.get("R_0_init", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, I, R]."""
        # Normalize to ensure S + I + R = N
        total = self.S_0 + self.I_0 + self.R_0_init
        scale = self.N / total if total > 0 else 1.0
        self._state = np.array(
            [self.S_0 * scale, self.I_0 * scale, self.R_0_init * scale],
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
        """Return current compartment values [S, I, R]."""
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
        """SIR with vaccination right-hand side."""
        S, I, R = y
        N = S + I + R
        if N <= 0:
            return np.zeros(3)

        force_of_infection = self.beta * S * I / N

        dS = self.mu * N - force_of_infection - self.nu * S - self.mu * S
        dI = force_of_infection - self.gamma * I - self.mu * I
        dR = self.gamma * I + self.nu * S - self.mu * R

        return np.array([dS, dI, dR])

    def compute_r0(self) -> float:
        """Basic reproduction number: R_0 = beta / (gamma + mu)."""
        return self.beta / (self.gamma + self.mu)

    def compute_r_eff(self) -> float:
        """Effective reproduction number accounting for vaccination.

        At the disease-free equilibrium with vaccination, the susceptible
        fraction is S*/N = mu / (nu + mu). Thus:

            R_eff = R_0 * (S*/N) = R_0 * mu / (nu + mu)
                  = R_0 * (1 - nu/(nu + mu))
        """
        r0 = self.compute_r0()
        vacc_coverage = (
            self.nu / (self.nu + self.mu)
            if (self.nu + self.mu) > 0
            else 0.0
        )
        return r0 * (1.0 - vacc_coverage)

    def herd_immunity_threshold(self) -> float:
        """Critical vaccination fraction: p_c = 1 - 1/R_0.

        Returns 0 if R_0 <= 1 (no vaccination needed).
        """
        r0 = self.compute_r0()
        if r0 <= 1.0:
            return 0.0
        return 1.0 - 1.0 / r0

    def disease_free_equilibrium(self) -> np.ndarray:
        """Disease-free equilibrium (DFE) with vaccination.

        S* = mu*N / (nu + mu)
        I* = 0
        R* = nu*N / (nu + mu)
        """
        denom = self.nu + self.mu
        if denom <= 0:
            return np.array([self.N, 0.0, 0.0])
        S_star = self.mu * self.N / denom
        R_star = self.nu * self.N / denom
        return np.array([S_star, 0.0, R_star])

    def endemic_equilibrium(self) -> np.ndarray | None:
        """Endemic equilibrium (exists only when R_eff > 1).

        Returns None if R_eff <= 1 (disease cannot persist).

        At endemic equilibrium:
            S* = N * (gamma + mu) / beta
            I* = mu*N/(gamma+mu) * (1 - 1/R_eff)
        """
        r_eff = self.compute_r_eff()
        if r_eff <= 1.0:
            return None

        # From dI/dt = 0: S* = N*(gamma+mu)/beta
        S_star = self.N * (self.gamma + self.mu) / self.beta

        # I* = N*mu/(gamma+mu) * (1 - 1/R_eff)
        I_star = (
            self.mu * self.N / (self.gamma + self.mu)
            * (1.0 - 1.0 / r_eff)
        )

        R_star = self.N - S_star - I_star
        if I_star < 0 or S_star < 0 or R_star < 0:
            return None

        return np.array([S_star, I_star, R_star])

    def critical_vaccination_rate(self) -> float:
        """Critical vaccination rate nu_c for disease elimination.

        R_eff = 1 when nu_c = mu * (R_0 - 1).
        Returns 0 if R_0 <= 1.
        """
        r0 = self.compute_r0()
        if r0 <= 1.0:
            return 0.0
        return self.mu * (r0 - 1.0)

    @property
    def total_population(self) -> float:
        """Current total population N = S + I + R."""
        if self._state is None:
            return self.N
        return float(self._state.sum())
