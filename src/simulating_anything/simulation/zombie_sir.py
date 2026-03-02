"""Zombie SIR epidemic model (Munz et al. 2009).

Modified SIR compartmental model with resurrection dynamics:
    dS/dt = -beta*S*Z + Pi           (Susceptible: attacked by zombies, births)
    dI/dt = beta*S*Z - rho*I - delta*I  (Infected: bitten, can die or turn)
    dZ/dt = rho*I + zeta*R - alpha*S*Z  (Zombie: from infected, resurrection, killed)
    dR/dt = delta*I + alpha*S*Z - zeta*R  (Removed: dead, killed zombies, resurrect)

Key phenomena:
- Zombie apocalypse dynamics with 4 compartments (S, I, Z, R)
- Resurrection term (zeta*R) means dead can return as zombies
- Competition between human kill rate (alpha) and zombie bite rate (beta)
- Without intervention (low alpha), zombies always win
- Population conservation: S + I + Z + R = N + Pi*t (when Pi > 0)
- Doomsday equilibrium: S=I=R=0, Z=N (zombies win)
- Coexistence only possible with sufficiently large alpha*S > beta*Z
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ZombieSIRSimulation(SimulationEnvironment):
    """Zombie SIR (Susceptible-Infected-Zombie-Removed) epidemic model.

    State vector: [S, I, Z, R] (absolute counts, not fractions).

    Parameters:
        beta: Transmission rate (zombie bites susceptible) (default 0.0095)
        alpha: Kill rate (susceptible kills zombie) (default 0.005)
        zeta: Resurrection rate (removed rise as zombie) (default 0.0001)
        delta: Natural death rate of infected (default 0.0001)
        rho: Zombification rate (infected become zombie) (default 0.5)
        Pi: Birth rate of new susceptibles (default 0.0)
        N: Total initial population (default 500)
        S_0: Initial susceptibles (default N-1)
        I_0: Initial infected (default 0)
        Z_0: Initial zombies (default 1)
        R_0_init: Initial removed (default 0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.beta = p.get("beta", 0.0095)
        self.alpha = p.get("alpha", 0.005)
        self.zeta = p.get("zeta", 0.0001)
        self.delta = p.get("delta", 0.0001)
        self.rho = p.get("rho", 0.5)
        self.Pi = p.get("Pi", 0.0)
        self.N = p.get("N", 500.0)
        self.S_0 = p.get("S_0", self.N - 1.0)
        self.I_0 = p.get("I_0", 0.0)
        self.Z_0 = p.get("Z_0", 1.0)
        self.R_0_init = p.get("R_0_init", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments [S, I, Z, R]."""
        self._state = np.array(
            [self.S_0, self.I_0, self.Z_0, self.R_0_init],
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
        """Return current compartment counts [S, I, Z, R]."""
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
        # Ensure non-negative populations
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Zombie SIR right-hand side.

        dS/dt = -beta*S*Z + Pi
        dI/dt = beta*S*Z - rho*I - delta*I
        dZ/dt = rho*I + zeta*R - alpha*S*Z
        dR/dt = delta*I + alpha*S*Z - zeta*R
        """
        S, Ip, Z, R = y  # noqa: N806 (Ip = infected population)
        dS = -self.beta * S * Z + self.Pi
        dI = self.beta * S * Z - self.rho * Ip - self.delta * Ip
        dZ = self.rho * Ip + self.zeta * R - self.alpha * S * Z
        dR = self.delta * Ip + self.alpha * S * Z - self.zeta * R
        return np.array([dS, dI, dZ, dR])

    def compute_basic_reproduction(self) -> float:
        """Compute zombie basic reproduction number analog.

        R0_zombie = beta * S_0 / (alpha * S_0) = beta / alpha
        This is the ratio of zombie creation rate to zombie destruction rate
        at the disease-free equilibrium S = S_0.

        When R0_zombie > 1, the zombie outbreak grows.
        """
        if self.alpha <= 0:
            return float("inf")
        return self.beta / self.alpha

    def compute_equilibria(self) -> dict[str, np.ndarray]:
        """Compute equilibrium points of the zombie SIR system.

        The system has two main equilibria:
        1. Disease-free equilibrium (DFE): Z=0, I=0 (no zombies)
           S = N + Pi*t (grows if Pi > 0), R = 0
        2. Doomsday equilibrium: S=0, I=0, R=0, Z=N
           (all humans become zombies)

        Returns:
            Dict with named equilibria as numpy arrays [S, I, Z, R].
        """
        equilibria = {}

        # Disease-free equilibrium (only stable if alpha > beta initially)
        equilibria["disease_free"] = np.array(
            [self.N, 0.0, 0.0, 0.0], dtype=np.float64
        )

        # Doomsday: all become zombies
        equilibria["doomsday"] = np.array(
            [0.0, 0.0, self.N, 0.0], dtype=np.float64
        )

        # With resurrection (zeta > 0), even R compartment empties into Z
        # so true doomsday with zeta > 0 is all in Z
        if self.zeta > 0:
            equilibria["doomsday_resurrection"] = np.array(
                [0.0, 0.0, self.N, 0.0], dtype=np.float64
            )

        return equilibria

    def alpha_sweep(
        self,
        alpha_values: np.ndarray,
        n_steps: int = 10000,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Sweep kill rate alpha to find human survival threshold.

        Args:
            alpha_values: Array of alpha (kill rate) values.
            n_steps: Steps per simulation.
            dt: Timestep override.

        Returns:
            Dict with alpha_values, final_S, final_Z, survived flags.
        """
        if dt is None:
            dt = self.config.dt

        final_S = []
        final_I = []
        final_Z = []
        final_R = []
        survived = []

        for alpha_val in alpha_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "alpha": alpha_val,
                },
            )
            sim = ZombieSIRSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            state = sim.observe()
            final_S.append(state[0])
            final_I.append(state[1])
            final_Z.append(state[2])
            final_R.append(state[3])
            # Humans survive if S > 1% of initial population
            survived.append(state[0] > 0.01 * self.N)

        return {
            "alpha_values": alpha_values,
            "final_S": np.array(final_S),
            "final_I": np.array(final_I),
            "final_Z": np.array(final_Z),
            "final_R": np.array(final_R),
            "survived": np.array(survived),
        }

    def outbreak_dynamics(
        self,
        n_steps: int = 5000,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Run a single outbreak and record detailed dynamics.

        Args:
            n_steps: Number of simulation steps.
            dt: Timestep override.

        Returns:
            Dict with time, S, I, Z, R arrays and summary statistics.
        """
        if dt is None:
            dt = self.config.dt

        config = SimulationConfig(
            domain=self.config.domain,
            dt=dt,
            n_steps=n_steps,
            parameters=self.config.parameters,
        )
        sim = ZombieSIRSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        states = np.array(states)
        time = np.arange(n_steps + 1) * dt

        # Peak zombie count and time
        peak_Z_idx = np.argmax(states[:, 2])

        return {
            "time": time,
            "S": states[:, 0],
            "I": states[:, 1],
            "Z": states[:, 2],
            "R": states[:, 3],
            "peak_Z": float(states[peak_Z_idx, 2]),
            "peak_Z_time": float(time[peak_Z_idx]),
            "final_S": float(states[-1, 0]),
            "final_Z": float(states[-1, 2]),
        }
