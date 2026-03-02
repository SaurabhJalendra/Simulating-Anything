"""SIR model with vital dynamics (births and deaths) simulation.

Standard SIR compartmental model with constant birth/death rate:

    dS/dt = mu*N - beta*S*I/N - mu*S
    dI/dt = beta*S*I/N - gamma*I - mu*I
    dR/dt = gamma*I - mu*R

where N = S + I + R is conserved (births balance deaths at rate mu).

Unlike the basic SIR model, the disease can reach an endemic equilibrium
because newborns replenish the susceptible pool. The disease persists when
R0 = beta / (gamma + mu) > 1.

Target rediscoveries:
- R0 = beta / (gamma + mu)
- Endemic equilibrium: S* = 1/R0, I* = mu*(R0 - 1)/beta
- Transcritical bifurcation at R0 = 1
- Damped oscillations to endemic equilibrium
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SIRVitalSimulation(SimulationEnvironment):
    """SIR epidemic model with vital dynamics (births and deaths).

    State vector: [S, I, R] (fractions of total population, S + I + R = 1).

    Equations (fraction form, N = 1):
        dS/dt = mu - beta*S*I - mu*S
        dI/dt = beta*S*I - gamma*I - mu*I
        dR/dt = gamma*I - mu*R

    The key difference from the basic SIR is that births (at rate mu) continuously
    replenish the susceptible pool. This allows an endemic equilibrium where the
    disease persists indefinitely when R0 > 1.

    Parameters:
        beta: transmission rate (default 0.4)
        gamma: recovery rate (default 0.1)
        mu: birth/death rate (default 0.01)
        S_0: initial susceptible fraction (default 0.99)
        I_0: initial infected fraction (default 0.01)
        R_0_init: initial recovered fraction (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.4)
        self.gamma = p.get("gamma", 0.1)
        self.mu = p.get("mu", 0.01)
        self.S_0 = p.get("S_0", 0.99)
        self.I_0 = p.get("I_0", 0.01)
        self.R_0_init = p.get("R_0_init", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, I, R] as fractions summing to 1."""
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
        # Enforce non-negativity (physical constraint)
        self._state = np.maximum(self._state, 0.0)
        # Renormalize to ensure S + I + R = 1
        self._state /= self._state.sum()

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """SIR with vital dynamics right-hand side (fraction form, N=1).

        dS/dt = mu - beta*S*I - mu*S
        dI/dt = beta*S*I - gamma*I - mu*I
        dR/dt = gamma*I - mu*R
        """
        S, I, R = y
        dS = self.mu - self.beta * S * I - self.mu * S
        dI = self.beta * S * I - self.gamma * I - self.mu * I
        dR = self.gamma * I - self.mu * R
        return np.array([dS, dI, dR])

    def compute_R0(self) -> float:
        """Basic reproduction number: R0 = beta / (gamma + mu)."""
        return self.beta / (self.gamma + self.mu)

    def endemic_equilibrium(self) -> tuple[float, float, float] | None:
        """Analytical endemic equilibrium (exists only when R0 > 1).

        S* = 1 / R0 = (gamma + mu) / beta
        I* = mu * (R0 - 1) / beta
        R* = gamma * I* / mu

        Returns:
            Tuple (S*, I*, R*) if R0 > 1, else None.
        """
        r0 = self.compute_R0()
        if r0 <= 1.0:
            return None
        S_star = 1.0 / r0
        I_star = self.mu * (r0 - 1.0) / self.beta
        R_star = self.gamma * I_star / self.mu
        return (S_star, I_star, R_star)

    def disease_free_equilibrium(self) -> tuple[float, float, float]:
        """Disease-free equilibrium: (S, I, R) = (1, 0, 0).

        Always exists; stable when R0 < 1.
        """
        return (1.0, 0.0, 0.0)

    def parameter_sweep(
        self,
        param_name: str,
        param_range: np.ndarray,
        n_steps: int = 10000,
    ) -> dict[str, np.ndarray]:
        """Sweep a single parameter and record final infected fraction.

        Runs the simulation to approximate equilibrium for each parameter value,
        recording the final I level (endemic equilibrium or disease-free).

        Args:
            param_name: Name of the parameter to sweep (e.g., "beta", "gamma", "mu").
            param_range: Array of parameter values to sweep.
            n_steps: Number of integration steps per simulation.

        Returns:
            Dictionary with parameter values, final S/I/R, and R0 for each run.
        """
        all_param = []
        all_S_final = []
        all_I_final = []
        all_R_final = []
        all_R0 = []

        for val in param_range:
            # Build parameters dict with the swept value
            params = {
                "beta": self.beta,
                "gamma": self.gamma,
                "mu": self.mu,
                "S_0": self.S_0,
                "I_0": self.I_0,
                "R_0_init": self.R_0_init,
            }
            params[param_name] = float(val)

            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters=params,
            )
            sim = SIRVitalSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            state = sim.observe()
            all_param.append(float(val))
            all_S_final.append(state[0])
            all_I_final.append(state[1])
            all_R_final.append(state[2])
            all_R0.append(sim.compute_R0())

        return {
            param_name: np.array(all_param),
            "S_final": np.array(all_S_final),
            "I_final": np.array(all_I_final),
            "R_final": np.array(all_R_final),
            "R0": np.array(all_R0),
        }
