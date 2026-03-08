"""Coupled climate-epidemic system: ENSO-driven disease dynamics.

Novel coupled system combining Vallis-type ENSO oscillation with SIR
epidemic dynamics where climate state modulates disease transmission.

Physical motivation: Many vector-borne diseases (malaria, dengue, Zika)
show strong seasonal and interannual variability driven by climate modes
like ENSO. The coupling is bidirectional in the sense that temperature
affects mosquito breeding rates and thus transmission, while large
epidemics can indirectly affect local conditions (e.g., land use changes).

State variables (6D):
  T: sea surface temperature anomaly (ENSO)
  h: thermocline depth anomaly (ENSO)
  tau: wind stress anomaly (ENSO)
  S: susceptible fraction (epidemic)
  I: infected fraction (epidemic)
  R: recovered fraction (epidemic)

Key coupling: beta_eff = beta_0 * (1 + coupling * T)
Temperature anomaly modulates disease transmission rate.

No known analytical solution for the coupled system. Targets:
- How does ENSO amplitude affect epidemic peak timing?
- Is there a critical coupling strength for epidemic synchronization?
- What scaling law relates ENSO period to epidemic periodicity?
- Can SINDy recover the coupled ODE from data?
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ClimateEpidemicSimulation(SimulationEnvironment):
    """Coupled ENSO + SIR epidemic simulation.

    The ENSO subsystem follows a simplified Vallis-type model:
        dT/dt = mu_c * T - omega * h + coupling_TI * I
        dh/dt = omega * T - epsilon * h
        dtau/dt = -alpha * tau + gamma_v * T

    The SIR subsystem with climate-dependent transmission:
        beta_eff = beta_0 * (1 + coupling_TC * T)
        dS/dt = -beta_eff * S * I + mu_pop * (1 - S)
        dI/dt = beta_eff * S * I - gamma_epi * I - mu_pop * I
        dR/dt = gamma_epi * I - mu_pop * R

    Args:
        config: SimulationConfig with parameters.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # ENSO parameters
        self.mu_c = p.get("mu_c", 0.5)         # growth rate
        self.omega = p.get("omega", 2.0)        # oscillation frequency
        self.epsilon = p.get("epsilon", 0.5)    # damping
        self.alpha = p.get("alpha", 1.0)        # wind stress decay
        self.gamma_v = p.get("gamma_v", 1.0)    # wind-temperature coupling

        # SIR parameters
        self.beta_0 = p.get("beta_0", 0.5)     # base transmission rate
        self.gamma_epi = p.get("gamma_epi", 0.1)  # recovery rate
        self.mu_pop = p.get("mu_pop", 0.01)     # birth/death rate

        # Coupling parameters
        self.coupling_TC = p.get("coupling_TC", 0.3)  # T -> beta modulation
        self.coupling_TI = p.get("coupling_TI", 0.0)  # I -> T feedback

        # Initial conditions
        self.T_0 = p.get("T_0", 0.5)
        self.h_0 = p.get("h_0", 0.0)
        self.tau_0 = p.get("tau_0", 0.0)
        self.S_0 = p.get("S_0", 0.95)
        self.I_0 = p.get("I_0", 0.04)
        self.R_0_init = p.get("R_0_init", 0.01)

        self.dt = config.dt
        self._state = np.array(
            [self.T_0, self.h_0, self.tau_0,
             self.S_0, self.I_0, self.R_0_init],
            dtype=np.float64,
        )

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        """Compute right-hand side of the coupled ODE system."""
        T, h, tau, S, I, R = state

        # ENSO dynamics (simplified Vallis)
        dT = self.mu_c * T - self.omega * h + self.coupling_TI * I
        dh = self.omega * T - self.epsilon * h
        dtau = -self.alpha * tau + self.gamma_v * T

        # Climate-dependent transmission rate
        beta_eff = self.beta_0 * (1.0 + self.coupling_TC * T)
        beta_eff = max(beta_eff, 0.0)  # transmission can't be negative

        # SIR dynamics with vital dynamics
        dS = -beta_eff * S * I + self.mu_pop * (1.0 - S)
        dI = beta_eff * S * I - self.gamma_epi * I - self.mu_pop * I
        dR = self.gamma_epi * I - self.mu_pop * R

        return np.array([dT, dh, dtau, dS, dI, dR])

    def _rk4_step(self, state: np.ndarray) -> np.ndarray:
        """Fourth-order Runge-Kutta integration step."""
        k1 = self._rhs(state)
        k2 = self._rhs(state + 0.5 * self.dt * k1)
        k3 = self._rhs(state + 0.5 * self.dt * k2)
        k4 = self._rhs(state + self.dt * k3)
        new_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Enforce SIR constraints: populations stay in [0, 1]
        new_state[3:6] = np.clip(new_state[3:6], 0.0, 1.0)
        # Normalize SIR to sum to 1
        total = new_state[3] + new_state[4] + new_state[5]
        if total > 0:
            new_state[3:6] /= total

        return new_state

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to initial conditions."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            # Small perturbation for varied trajectories
            self._state = np.array([
                self.T_0 + rng.normal(0, 0.1),
                self.h_0 + rng.normal(0, 0.05),
                self.tau_0,
                self.S_0 + rng.normal(0, 0.01),
                self.I_0 + rng.normal(0, 0.005),
                self.R_0_init,
            ])
            self._state[3:6] = np.clip(self._state[3:6], 0.001, 0.999)
            total = self._state[3] + self._state[4] + self._state[5]
            self._state[3:6] /= total
        else:
            self._state = np.array(
                [self.T_0, self.h_0, self.tau_0,
                 self.S_0, self.I_0, self.R_0_init],
                dtype=np.float64,
            )
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._state = self._rk4_step(self._state)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [T, h, tau, S, I, R]."""
        return self._state.copy()

    @property
    def climate_state(self) -> np.ndarray:
        """Return ENSO state [T, h, tau]."""
        return self._state[:3].copy()

    @property
    def epidemic_state(self) -> np.ndarray:
        """Return SIR state [S, I, R]."""
        return self._state[3:6].copy()

    def effective_beta(self) -> float:
        """Compute the current effective transmission rate."""
        T = self._state[0]
        return max(self.beta_0 * (1.0 + self.coupling_TC * T), 0.0)

    def effective_r0(self) -> float:
        """Compute the current effective reproduction number."""
        return self.effective_beta() / self.gamma_epi
