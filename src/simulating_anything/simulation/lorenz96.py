"""Lorenz-96 model simulation -- atmospheric chaos on a circle.

The Lorenz-96 equations model advection on a latitude circle:
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F

for i = 0, ..., N-1 with periodic boundary conditions.

Target rediscoveries:
- Chaos transition: positive Lyapunov exponent for F >= ~8
- Energy statistics: mean(x^2/2) as a function of F
- Attractor dimension scaling with N and F
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class Lorenz96(SimulationEnvironment):
    """Lorenz-96 model: a toy model of atmospheric dynamics.

    State vector: [x_0, x_1, ..., x_{N-1}] on a periodic circle.

    The equations exhibit:
    - For F < ~3: decay to uniform fixed point x_i = F
    - For F ~ 3-8: periodic or quasi-periodic orbits
    - For F >= ~8: fully developed chaos

    Parameters:
        N: number of sites (default 36)
        F: constant forcing (default 8.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 36))
        self.F = p.get("F", 8.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state: x_i = F with a small perturbation at site 0."""
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        self._state = np.full(self.N, self.F, dtype=np.float64)
        # Small perturbation at one site to break symmetry
        self._state[0] += 0.01 * rng.standard_normal()
        self._step_count = 0
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state.copy()

    def observe(self) -> np.ndarray:
        """Return current state [x_0, ..., x_{N-1}]."""
        return self._state.copy()

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Lorenz-96 equations with periodic boundary conditions.

        dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F

        Uses numpy roll for efficient periodic indexing.
        """
        x = state
        # x_{i+1}, x_{i-1}, x_{i-2} via circular shifts
        x_plus1 = np.roll(x, -1)   # x_{i+1}
        x_minus1 = np.roll(x, 1)   # x_{i-1}
        x_minus2 = np.roll(x, 2)   # x_{i-2}
        return (x_plus1 - x_minus2) * x_minus1 - x + self.F

    @property
    def energy(self) -> float:
        """Compute the energy: mean(x^2 / 2)."""
        return float(np.mean(self._state**2) / 2.0)

    @property
    def mean_state(self) -> float:
        """Compute the spatial mean of the state."""
        return float(np.mean(self._state))

    @property
    def max_amplitude(self) -> float:
        """Compute the maximum absolute deviation from the mean."""
        return float(np.max(np.abs(self._state - np.mean(self._state))))

    @property
    def fixed_point(self) -> np.ndarray:
        """The uniform fixed point x_i = F for all i."""
        return np.full(self.N, self.F, dtype=np.float64)

    def estimate_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the method of Wolf et al. (1985): track two nearby trajectories,
        renormalize when they diverge too far.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        # Perturb one site
        perturbation = np.zeros(self.N)
        perturbation[0] = eps
        state2 = state1 + perturbation

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance both states with RK4
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Compute distance and renormalize
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)
