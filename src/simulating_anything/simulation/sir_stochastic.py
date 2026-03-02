"""Stochastic SIR epidemic model with demographic noise.

SDE system (Euler-Maruyama):
    dS = (-beta*S*I/N)*dt + sigma_S * sqrt(beta*S*I/N) * dW_1
    dI = (beta*S*I/N - gamma*I)*dt + sigma_I * sqrt(beta*S*I/N + gamma*I) * dW_2
    dR = (gamma*I)*dt + sigma_R * sqrt(gamma*I) * dW_3

Noise scales with sqrt of transition rates (demographic stochasticity).
Population S+I+R is approximately conserved (with fluctuations from noise).
All compartments are clamped to non-negative values after each step.

Target rediscoveries:
- Ensemble mean converges to deterministic SIR
- Variance scales with sigma^2 and inversely with population size N
- Extinction probability increases for R0 near 1 due to stochastic effects
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SIRStochasticSimulation(SimulationEnvironment):
    """Stochastic SIR epidemic model with demographic noise.

    State vector: [S, I, R] (population counts, not fractions)

    SDE system:
        dS = (-beta*S*I/N)*dt + sigma_S * sqrt(beta*S*I/N) * dW_1
        dI = (beta*S*I/N - gamma*I)*dt + sigma_I * sqrt(beta*S*I/N + gamma*I) * dW_2
        dR = (gamma*I)*dt + sigma_R * sqrt(gamma*I) * dW_3

    where:
        N = S + I + R (total population)
        sigma_S = sigma_I = sigma_R = sigma (global noise scale)
        dW_i are independent Wiener increments

    Parameters:
        beta: transmission rate (default 0.5)
        gamma: recovery rate (default 0.1)
        N0: initial total population (default 1000.0)
        sigma: noise scale (default 0.1)
        I_0: initial infected count (default 10.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.5)
        self.gamma = p.get("gamma", 0.1)
        self.N0 = p.get("N0", 1000.0)
        self.sigma = p.get("sigma", 0.1)
        self.I_0 = p.get("I_0", 10.0)
        self._rng: np.random.Generator = np.random.default_rng(config.seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, I, R].

        S = N0 - I_0, I = I_0, R = 0.
        """
        self._rng = np.random.default_rng(seed)
        S0 = self.N0 - self.I_0
        self._state = np.array([S0, self.I_0, 0.0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using Euler-Maruyama."""
        self._euler_maruyama_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current compartment values [S, I, R]."""
        return self._state

    def _euler_maruyama_step(self) -> None:
        """Single Euler-Maruyama step for the stochastic SIR system."""
        dt = self.config.dt
        S, Ip, R = self._state  # noqa: N806 (Ip = infected population)

        N = S + Ip + R
        if N <= 0:
            return

        # Transition rates (clamped to non-negative for sqrt)
        infection_rate = max(0.0, self.beta * S * Ip / N)
        recovery_rate = max(0.0, self.gamma * Ip)

        # Deterministic drift
        dS_det = -infection_rate
        dI_det = infection_rate - recovery_rate
        dR_det = recovery_rate

        # Stochastic diffusion (3 independent Wiener increments)
        dW = self._rng.normal(0.0, np.sqrt(dt), size=3)

        # Noise scales with sqrt of transition rates (demographic stochasticity)
        noise_S = self.sigma * np.sqrt(infection_rate)
        noise_I = self.sigma * np.sqrt(infection_rate + recovery_rate)
        noise_R = self.sigma * np.sqrt(recovery_rate)

        # Euler-Maruyama update
        S_new = S + dS_det * dt + noise_S * dW[0]
        I_new = Ip + dI_det * dt + noise_I * dW[1]
        R_new = R + dR_det * dt + noise_R * dW[2]

        # Non-negativity clamping
        self._state = np.maximum(np.array([S_new, I_new, R_new]), 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Deterministic SIR right-hand side (no noise)."""
        S, Ip, R = y  # noqa: N806 (Ip = infected population)
        N = S + Ip + R
        if N <= 0:
            return np.zeros(3)
        infection_rate = self.beta * S * Ip / N
        recovery_rate = self.gamma * Ip
        dS = -infection_rate
        dI = infection_rate - recovery_rate
        dR = recovery_rate
        return np.array([dS, dI, dR])

    @property
    def R0(self) -> float:
        """Basic reproduction number: R0 = beta / gamma."""
        return self.beta / self.gamma

    def run_deterministic(self, n_steps: int | None = None) -> np.ndarray:
        """Run deterministic SIR (RK4, no noise) for comparison.

        Returns:
            Array of shape (n_steps+1, 3) with [S, I, R] at each timestep.
        """
        if n_steps is None:
            n_steps = self.config.n_steps
        dt = self.config.dt
        S0 = self.N0 - self.I_0
        state = np.array([S0, self.I_0, 0.0], dtype=np.float64)
        trajectory = [state.copy()]

        for _ in range(n_steps):
            k1 = self._derivatives(state)
            k2 = self._derivatives(state + 0.5 * dt * k1)
            k3 = self._derivatives(state + 0.5 * dt * k2)
            k4 = self._derivatives(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            state = np.maximum(state, 0.0)
            trajectory.append(state.copy())

        return np.array(trajectory)

    def run_ensemble(
        self,
        n_runs: int,
        n_steps: int | None = None,
        base_seed: int = 0,
    ) -> np.ndarray:
        """Run an ensemble of stochastic simulations.

        Args:
            n_runs: number of independent stochastic runs.
            n_steps: steps per run (default from config).
            base_seed: base seed; each run uses base_seed + i.

        Returns:
            Array of shape (n_runs, n_steps+1, 3).
        """
        if n_steps is None:
            n_steps = self.config.n_steps
        ensemble = np.zeros((n_runs, n_steps + 1, 3))

        for i in range(n_runs):
            state = self.reset(seed=base_seed + i)
            ensemble[i, 0] = state.copy()
            for t in range(n_steps):
                state = self.step()
                ensemble[i, t + 1] = state.copy()

        return ensemble
