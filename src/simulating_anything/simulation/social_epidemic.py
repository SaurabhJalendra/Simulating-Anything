"""Novel coupled Social-Epidemic simulation (4D).

Couples a continuous opinion dynamics model (Deffuant-like bounded
confidence) with SIR epidemic dynamics. Public opinion on vaccination
modulates vaccination uptake, which reduces disease susceptibility.
Disease prevalence feeds back to shift public opinion toward vaccination.

State: [x, sigma, S, I] where:
  x = mean opinion on vaccination (0=anti-vax, 1=pro-vax)
  sigma = opinion variance (polarization)
  S = susceptible fraction
  I = infected fraction

Coupling:
  - Mean opinion x determines vaccination rate: v = v_max * x^n
  - Infection prevalence I shifts opinion toward pro-vax: dx += c1 * I
  - High polarization sigma slows opinion convergence
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SocialEpidemicSimulation(SimulationEnvironment):
    """Coupled opinion dynamics + SIR epidemic simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Opinion dynamics parameters
        self.mu_x = p.get("mu_x", 0.5)       # opinion drift toward center
        self.D_x = p.get("D_x", 0.02)        # opinion diffusion (noise)
        self.sigma_decay = p.get("sigma_decay", 0.1)  # polarization decay

        # SIR parameters
        self.beta = p.get("beta", 0.4)        # base transmission rate
        self.gamma_epi = p.get("gamma_epi", 0.1)  # recovery rate
        self.mu_pop = p.get("mu_pop", 0.01)   # birth/death rate

        # Vaccination parameters
        self.v_max = p.get("v_max", 0.05)     # max vaccination rate
        self.n_hill = p.get("n_hill", 2.0)    # Hill coefficient for uptake

        # Coupling parameters
        self.coupling_IS = p.get("coupling_IS", 1.0)   # I -> opinion shift
        self.coupling_xv = p.get("coupling_xv", 1.0)   # x -> vaccination rate

        # Initial conditions
        self.x_0 = p.get("x_0", 0.5)
        self.sigma_0 = p.get("sigma_0", 0.2)
        self.S_0 = p.get("S_0", 0.95)
        self.I_0 = p.get("I_0", 0.04)

        self.dt = config.dt
        self._state = np.array(
            [self.x_0, self.sigma_0, self.S_0, self.I_0], dtype=np.float64,
        )
        self._t = 0.0

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        x, sigma, S, I = state

        # Vaccination rate (Hill function of opinion)
        x_clipped = np.clip(x, 0.01, 0.99)
        v_rate = self.v_max * self.coupling_xv * x_clipped ** self.n_hill

        # Opinion dynamics with disease feedback
        # Disease prevalence pushes opinion toward pro-vax
        dx = (-self.mu_x * (x - 0.5)  # drift toward center
              + self.coupling_IS * I * (1.0 - x)  # disease fear -> pro-vax
              + self.D_x * (0.5 - x))  # diffusion

        # Polarization dynamics (decays with consensus, increases with disease)
        dsigma = (-self.sigma_decay * sigma
                  + 0.1 * I * (1.0 - sigma))  # disease increases uncertainty

        # SIR with vaccination
        dS = (-self.beta * S * I
              - v_rate * S
              + self.mu_pop * (1.0 - S))
        dI = (self.beta * S * I
              - self.gamma_epi * I
              - self.mu_pop * I)

        return np.array([dx, dsigma, dS, dI])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                np.clip(self.x_0 + rng.normal(0, 0.05), 0.01, 0.99),
                np.clip(self.sigma_0 + rng.normal(0, 0.02), 0.01, 0.5),
                np.clip(self.S_0 + rng.normal(0, 0.02), 0.01, 0.99),
                np.clip(self.I_0 + rng.normal(0, 0.01), 0.001, 0.2),
            ], dtype=np.float64)
        else:
            self._state = np.array(
                [self.x_0, self.sigma_0, self.S_0, self.I_0], dtype=np.float64,
            )
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state = self._state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[0] = np.clip(self._state[0], 0.0, 1.0)
        self._state[1] = np.clip(self._state[1], 0.0, 1.0)
        self._state[2:] = np.clip(self._state[2:], 0.0, 1.0)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [x, sigma, S, I]."""
        return self._state.copy()
