"""SIR epidemic model simulation.

Standard compartmental model: Susceptible -> Infected -> Recovered.
Target rediscovery: basic reproduction number R0 = beta/gamma.
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SIRSimulation(SimulationEnvironment):
    """SIR (Susceptible-Infected-Recovered) epidemic model.

    State vector: [S, I, R] (fractions, sum = 1)

    Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

    Key quantity: R0 = beta / gamma (basic reproduction number).
    Epidemic grows when R0 > 1 (i.e., S(0) * R0 > 1).
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.3)      # Transmission rate
        self.gamma = p.get("gamma", 0.1)     # Recovery rate
        self.S_0 = p.get("S_0", 0.99)        # Initial susceptible fraction
        self.I_0 = p.get("I_0", 0.01)        # Initial infected fraction
        self.R_0_init = p.get("R_0_init", 0.0)  # Initial recovered fraction

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments."""
        total = self.S_0 + self.I_0 + self.R_0_init
        self._state = np.array(
            [self.S_0 / total, self.I_0 / total, self.R_0_init / total],
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
        """Return current compartment fractions [S, I, R]."""
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
        self._state = np.maximum(self._state, 0.0)
        # Renormalize to ensure S + I + R = 1
        self._state /= self._state.sum()

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """SIR right-hand side."""
        S, I, R = y
        dS = -self.beta * S * I
        dI = self.beta * S * I - self.gamma * I
        dR = self.gamma * I
        return np.array([dS, dI, dR])

    @property
    def R0(self) -> float:
        """Basic reproduction number."""
        return self.beta / self.gamma

    @property
    def peak_infected(self) -> float:
        """Theoretical peak infected fraction.

        I_peak = I_0 + S_0 - (1/R0) * (1 + ln(R0 * S_0))
        """
        r0 = self.R0
        if r0 * self.S_0 <= 1:
            return self.I_0
        return self.I_0 + self.S_0 - (1 / r0) * (1 + np.log(r0 * self.S_0))

    @property
    def final_size(self) -> float:
        """Theoretical final epidemic size (fraction infected).

        Solved from: R_inf = 1 - S_0 * exp(-R0 * R_inf)
        """
        r0 = self.R0
        if r0 <= 1:
            return 0.0
        # Newton's method for final size equation
        R_inf = 0.9
        for _ in range(50):
            f = R_inf - 1 + self.S_0 * np.exp(-r0 * R_inf)
            df = 1 + self.S_0 * r0 * np.exp(-r0 * R_inf)
            R_inf -= f / df
        return max(0.0, R_inf)
