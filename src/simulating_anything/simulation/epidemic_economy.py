"""Novel coupled Epidemic-Economy simulation (4D).

Couples SIR disease dynamics with a simple Goodwin-type economic
oscillator. Economic activity modulates disease transmission (more
activity = more contacts), while disease reduces labor supply and
economic output. Creates boom-bust-epidemic cycles with no known
analytical solution.

State: [S, I, w, u] where:
  S = susceptible fraction
  I = infected fraction
  w = wage share (workers' share of output)
  u = employment rate

Coupling:
  - Employment u modulates transmission: beta_eff = beta * (1 + c1 * u)
  - Infection I reduces employment: u_loss = c2 * I
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class EpidemicEconomySimulation(SimulationEnvironment):
    """Coupled SIR epidemic + Goodwin economic oscillator."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # SIR parameters
        self.beta = p.get("beta", 0.4)        # base transmission rate
        self.gamma_epi = p.get("gamma_epi", 0.1)  # recovery rate
        self.mu = p.get("mu_pop", 0.01)       # birth/death rate

        # Goodwin economic parameters
        self.alpha_e = p.get("alpha_e", 0.02)   # productivity growth
        self.beta_e = p.get("beta_e", 0.04)     # labor force growth
        self.rho_e = p.get("rho_e", 0.95)       # Phillips curve threshold
        self.phi_e = p.get("phi_e", 1.5)        # Phillips curve slope
        self.sigma_e = p.get("sigma_e", 0.5)    # output-capital ratio

        # Coupling parameters
        self.coupling_uS = p.get("coupling_uS", 0.3)   # u -> transmission
        self.coupling_Iu = p.get("coupling_Iu", 0.5)    # I -> employment loss

        # Initial conditions
        self.S_0 = p.get("S_0", 0.95)
        self.I_0 = p.get("I_0", 0.04)
        self.w_0 = p.get("w_0", 0.6)
        self.u_0 = p.get("u_0", 0.9)

        self.dt = config.dt
        self._state = np.array(
            [self.S_0, self.I_0, self.w_0, self.u_0], dtype=np.float64,
        )
        self._t = 0.0

    def _phillips_curve(self, u: float) -> float:
        """Phillips curve: wage growth as function of employment."""
        return self.phi_e * (u - self.rho_e)

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        S, I, w, u = state

        # Effective employment (disease reduces labor)
        u_eff = u * (1.0 - self.coupling_Iu * I)
        u_eff = max(u_eff, 0.0)

        # Effective transmission (economic activity increases contacts)
        beta_eff = self.beta * (1.0 + self.coupling_uS * u_eff)

        # SIR dynamics with vital dynamics
        R = max(1.0 - S - I, 0.0)
        dS = -beta_eff * S * I + self.mu * (1.0 - S)
        dI = beta_eff * S * I - self.gamma_epi * I - self.mu * I

        # Goodwin dynamics with disease feedback
        # Wage share: grows when employment high (Phillips curve)
        dw = w * (self._phillips_curve(u_eff) - self.alpha_e)

        # Employment rate: grows with investment, falls with productivity
        du = u * (self.sigma_e * (1.0 - w) - (self.alpha_e + self.beta_e))

        return np.array([dS, dI, dw, du])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                np.clip(self.S_0 + rng.normal(0, 0.02), 0.01, 0.99),
                np.clip(self.I_0 + rng.normal(0, 0.01), 0.001, 0.2),
                np.clip(self.w_0 + rng.normal(0, 0.05), 0.1, 0.9),
                np.clip(self.u_0 + rng.normal(0, 0.05), 0.3, 0.99),
            ], dtype=np.float64)
        else:
            self._state = np.array(
                [self.S_0, self.I_0, self.w_0, self.u_0], dtype=np.float64,
            )
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state = self._state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce bounds
        self._state[:2] = np.clip(self._state[:2], 0.0, 1.0)
        self._state[2:] = np.clip(self._state[2:], 0.01, 1.0)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [S, I, w, u]."""
        return self._state.copy()
