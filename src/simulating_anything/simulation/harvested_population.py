"""Logistic growth with harvesting simulation.

Models a single-species population subject to constant-rate harvesting:

    dx/dt = r*x*(1 - x/K) - H

where x = population density, r = intrinsic growth rate, K = carrying capacity,
and H = constant harvest rate.

Target rediscoveries:
- Saddle-node bifurcation at H_c = r*K/4 (maximum sustainable yield)
- Equilibria: x* = (K/2) +/- sqrt(K^2/4 - H*K/r)
- Extinction below the lower equilibrium (unstable threshold)
- For H < H_c: two positive equilibria (upper stable, lower unstable)
- For H = H_c: single saddle-node at x* = K/2
- For H > H_c: no positive equilibria, population collapses to zero
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class HarvestedPopulationSimulation(SimulationEnvironment):
    """Logistic growth with constant harvesting: dx/dt = r*x*(1 - x/K) - H.

    State: [x] (single scalar population, stored as 1D array for consistency).

    The population is clamped to x >= 0 to represent extinction (once the
    population reaches zero, it cannot recover under constant harvesting).

    Parameters:
        r: intrinsic growth rate (default 1.0)
        K: carrying capacity (default 1.0)
        H: constant harvest rate (default 0.0)
        x_0: initial population density (default K/2, or 0.5 if K not given)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 1.0)
        self.H = p.get("H", 0.0)
        self.x_0 = p.get("x_0", self.K / 2.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state near K/2."""
        self._state = np.array([self.x_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4, clamping x >= 0."""
        self._rk4_step()
        # Clamp to prevent negative population (extinction absorbing state)
        if self._state[0] < 0.0:
            self._state[0] = 0.0
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integrator."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dx/dt = r*x*(1 - x/K) - H."""
        x = y[0]
        dxdt = self.r * x * (1.0 - x / self.K) - self.H
        return np.array([dxdt])

    def compute_msy(self) -> float:
        """Maximum sustainable yield: H_c = r*K/4.

        This is the largest constant harvest rate at which a positive
        equilibrium still exists (the saddle-node bifurcation point).
        """
        return self.r * self.K / 4.0

    def find_equilibria(self) -> list[dict[str, float]]:
        """Find equilibria by solving r*x*(1 - x/K) = H.

        The equation r*x*(1 - x/K) - H = 0 is a quadratic:
            -(r/K)*x^2 + r*x - H = 0
            (r/K)*x^2 - r*x + H = 0

        Solutions: x* = (K/2) +/- sqrt(K^2/4 - H*K/r)

        Returns:
            List of dicts with 'x' and 'stability' keys.
            Empty list if discriminant < 0 (H > H_c, no positive equilibria).
        """
        r, K, H = self.r, self.K, self.H
        discriminant = K**2 / 4.0 - H * K / r

        equilibria: list[dict[str, float]] = []

        if discriminant < -1e-12:
            # No real equilibria: H > H_c
            return equilibria

        if abs(discriminant) < 1e-12:
            # Saddle-node: exactly one equilibrium at K/2
            equilibria.append({
                "x": K / 2.0,
                "stability": "saddle-node",
                "eigenvalue": 0.0,
            })
            return equilibria

        sqrt_disc = np.sqrt(max(discriminant, 0.0))

        # Upper equilibrium (stable)
        x_upper = K / 2.0 + sqrt_disc
        eig_upper = self._jacobian_eigenvalue(x_upper)
        equilibria.append({
            "x": x_upper,
            "stability": "stable" if eig_upper < 0 else "unstable",
            "eigenvalue": eig_upper,
        })

        # Lower equilibrium (unstable)
        x_lower = K / 2.0 - sqrt_disc
        if x_lower > 0:
            eig_lower = self._jacobian_eigenvalue(x_lower)
            equilibria.append({
                "x": x_lower,
                "stability": "stable" if eig_lower < 0 else "unstable",
                "eigenvalue": eig_lower,
            })

        return equilibria

    def _jacobian_eigenvalue(self, x: float) -> float:
        """Eigenvalue of the 1D Jacobian at equilibrium x.

        df/dx = r*(1 - x/K) + r*x*(-1/K) = r*(1 - 2*x/K)
        """
        return self.r * (1.0 - 2.0 * x / self.K)

    def is_extinct(self) -> bool:
        """Check if population has gone extinct (x <= 0)."""
        return self._state[0] <= 0.0

    def time_to_extinction(self, max_steps: int = 100000) -> float:
        """Simulate until population hits zero and return the time.

        Returns float('inf') if the population survives for max_steps.
        """
        dt = self.config.dt
        self.reset()
        for i in range(1, max_steps + 1):
            self.step()
            if self._state[0] <= 0.0:
                return i * dt
        return float("inf")
