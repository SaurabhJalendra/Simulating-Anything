"""SIS epidemic model with adaptive behavioral response.

An SIS epidemic where susceptible individuals reduce contact rate in response
to disease prevalence, creating a feedback loop:
    high infection -> behavioral change -> reduced transmission.

ODEs:
    dI/dt = beta(I) * S * I - gamma * I
    S = 1 - I  (normalized population)
    beta(I) = beta0 / (1 + kappa * I)  -- behavioral contact reduction

Parameters:
    beta0  = baseline transmission rate (no behavioral response)
    gamma  = recovery rate
    kappa  = behavioral sensitivity (kappa=0 is standard SIS)
    I_0    = initial infected fraction

Key physics:
- Without behavior (kappa=0): endemic equilibrium I* = 1 - gamma/beta0
- With behavior: I* = (beta0 - gamma) / (beta0 + kappa*gamma)
- R0_eff = beta0/gamma for small I (same threshold as standard SIS)
- Higher kappa -> lower endemic level
- Can create damped oscillations toward equilibrium
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SISAdaptiveSimulation(SimulationEnvironment):
    """SIS epidemic model with adaptive behavioral response.

    State vector: [I] (infected fraction; S = 1 - I computed internally).

    Equations:
        dI/dt = beta0 * S * I / (1 + kappa * I) - gamma * I
        S = 1 - I

    The effective transmission rate beta(I) = beta0 / (1 + kappa*I) decreases
    as prevalence rises, modeling behavioral avoidance.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta0 = p.get("beta0", 0.5)
        self.gamma = p.get("gamma", 0.1)
        self.kappa = p.get("kappa", 5.0)
        self.I_0 = p.get("I_0", 0.01)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize infected fraction."""
        self._state = np.array([np.clip(self.I_0, 0.0, 1.0)], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current observables [I, S, beta_eff]."""
        inf = self._state[0]
        sus = 1.0 - inf
        beta_eff = self.behavioral_beta(inf)
        return np.array([inf, sus, beta_eff], dtype=np.float64)

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Clamp I to [0, 1]
        self._state = np.clip(self._state, 0.0, 1.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """SIS adaptive right-hand side.

        dI/dt = beta0 * S * I / (1 + kappa * I) - gamma * I
        where S = 1 - I.
        """
        inf = y[0]
        sus = 1.0 - inf
        beta_eff = self.beta0 / (1.0 + self.kappa * inf)
        d_inf = beta_eff * sus * inf - self.gamma * inf
        return np.array([d_inf])

    def behavioral_beta(self, prevalence: float) -> float:
        """Compute effective transmission rate given prevalence.

        beta(I) = beta0 / (1 + kappa * I)
        """
        return self.beta0 / (1.0 + self.kappa * prevalence)

    def effective_R0(self, prevalence: float) -> float:
        """Compute effective reproduction number at given prevalence.

        R_eff(I) = beta(I) * S / gamma = beta0 * (1 - I) / (gamma * (1 + kappa * I))
        """
        sus = 1.0 - prevalence
        beta_eff = self.behavioral_beta(prevalence)
        if self.gamma <= 0:
            return float("inf")
        return beta_eff * sus / self.gamma

    def endemic_equilibrium(self) -> float | None:
        """Compute the endemic equilibrium infected fraction I*.

        At equilibrium, dI/dt = 0 with I > 0 gives:
            beta0 * (1 - I*) / (1 + kappa * I*) = gamma
        Solving: I* = (beta0 - gamma) / (beta0 + kappa * gamma)

        Returns None if R0 <= 1 (disease-free equilibrium is stable).
        """
        r0 = self.beta0 / self.gamma if self.gamma > 0 else float("inf")
        if r0 <= 1.0:
            return None
        i_star = (self.beta0 - self.gamma) / (self.beta0 + self.kappa * self.gamma)
        if i_star <= 0 or i_star >= 1:
            return None
        return i_star

    @property
    def basic_R0(self) -> float:
        """Basic reproduction number R0 = beta0 / gamma.

        This is the R0 at zero prevalence (no behavioral response).
        """
        if self.gamma <= 0:
            return float("inf")
        return self.beta0 / self.gamma
