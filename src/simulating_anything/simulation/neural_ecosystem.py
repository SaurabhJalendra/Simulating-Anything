"""Novel coupled Neural-Ecosystem simulation (4D).

Couples Wilson-Cowan excitatory-inhibitory neural populations with
Lotka-Volterra predator-prey dynamics. Neural activity (representing
foraging drive) modulates prey capture rate, while prey abundance
feeds back as a reward signal to excitatory neurons.

State: [E, I_n, N, P] where:
  E = excitatory neural population activity
  I_n = inhibitory neural population activity
  N = prey density
  P = predator density

Coupling:
  - Excitatory activity E modulates predation rate: a_eff = a * (1 + c1 * E)
  - Prey abundance N stimulates excitatory neurons: I_ext = c2 * N
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class NeuralEcosystemSimulation(SimulationEnvironment):
    """Coupled Wilson-Cowan neural + Lotka-Volterra predator-prey."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Wilson-Cowan neural parameters
        self.tau_e = p.get("tau_e", 1.0)     # excitatory time constant
        self.tau_i = p.get("tau_i", 1.0)     # inhibitory time constant
        self.w_ee = p.get("w_ee", 12.0)      # E -> E connection weight
        self.w_ei = p.get("w_ei", 4.0)       # I -> E connection weight
        self.w_ie = p.get("w_ie", 13.0)      # E -> I connection weight
        self.w_ii = p.get("w_ii", 11.0)      # I -> I connection weight
        self.theta_e = p.get("theta_e", 2.8) # excitatory threshold
        self.theta_i = p.get("theta_i", 4.0) # inhibitory threshold
        self.I_ext_base = p.get("I_ext_base", 1.0)  # base external input

        # Lotka-Volterra parameters
        self.alpha = p.get("alpha_lv", 1.1)  # prey growth rate
        self.beta_lv = p.get("beta_lv", 0.4)  # base predation rate
        self.gamma_lv = p.get("gamma_lv", 0.4)  # predator death rate
        self.delta_lv = p.get("delta_lv", 0.1)  # conversion efficiency

        # Coupling parameters
        self.coupling_EN = p.get("coupling_EN", 0.2)   # E -> predation rate
        self.coupling_NE = p.get("coupling_NE", 0.1)   # N -> excitatory input

        # Initial conditions
        self.E_0 = p.get("E_0", 0.3)
        self.In_0 = p.get("In_0", 0.2)
        self.N_0 = p.get("N_0", 10.0)
        self.P_0 = p.get("P_0", 5.0)

        self.dt = config.dt
        self._state = np.array(
            [self.E_0, self.In_0, self.N_0, self.P_0], dtype=np.float64,
        )
        self._t = 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        E, In, N, P = state

        # Effective external input (prey abundance as reward signal)
        I_ext = self.I_ext_base + self.coupling_NE * N

        # Effective predation rate (neural drive enhances foraging)
        beta_eff = self.beta_lv * (1.0 + self.coupling_EN * E)

        # Wilson-Cowan E-I dynamics
        input_e = self.w_ee * E - self.w_ei * In + I_ext
        input_i = self.w_ie * E - self.w_ii * In

        dE = (-E + self._sigmoid(input_e - self.theta_e)) / self.tau_e
        dIn = (-In + self._sigmoid(input_i - self.theta_i)) / self.tau_i

        # Lotka-Volterra with neural-modulated predation
        dN = self.alpha * N - beta_eff * N * P
        dP = self.delta_lv * beta_eff * N * P - self.gamma_lv * P

        return np.array([dE, dIn, dN, dP])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                np.clip(self.E_0 + rng.normal(0, 0.05), 0.01, 0.99),
                np.clip(self.In_0 + rng.normal(0, 0.05), 0.01, 0.99),
                max(self.N_0 + rng.normal(0, 1.0), 0.1),
                max(self.P_0 + rng.normal(0, 0.5), 0.1),
            ], dtype=np.float64)
        else:
            self._state = np.array(
                [self.E_0, self.In_0, self.N_0, self.P_0], dtype=np.float64,
            )
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state = self._state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce positivity for populations
        self._state[0:2] = np.clip(self._state[0:2], 0.0, 1.0)
        self._state[2:] = np.clip(self._state[2:], 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [E, I_n, N, P]."""
        return self._state.copy()
