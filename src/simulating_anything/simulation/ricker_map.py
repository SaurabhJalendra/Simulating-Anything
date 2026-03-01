"""Ricker map simulation (discrete population dynamics).

The Ricker map is a classic fisheries science model:
  x_{n+1} = x_n * exp(r * (1 - x_n / K))

Target rediscoveries:
- Period-doubling route to chaos as r increases
- Overcompensation: population overshoots carrying capacity K
- Chaos onset at r ~ 2.0
- Fixed point x* = K (nontrivial), x* = 0 (trivial/unstable for r > 0)
- Lyapunov exponent: lambda = ln|f'(x*)| where f'(x) = exp(r(1-x/K)) * (1 - r*x/K)
- At x* = K: f'(K) = 1 - r, so |lambda| = ln|1 - r|, chaos when |1-r| > 1 => r > 2
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RickerMapSimulation(SimulationEnvironment):
    """Ricker map: x_{n+1} = x_n * exp(r * (1 - x_n / K)).

    State: [x] (single scalar population, stored as 1D array for consistency).

    Parameters:
        r: intrinsic growth rate (default 2.0)
        K: carrying capacity (default 1.0)
        x_0: initial population (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 2.0)
        self.K = p.get("K", 1.0)
        self.x_0 = p.get("x_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state near K/2."""
        self._state = np.array([self.x_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the Ricker map."""
        x = self._state[0]
        self._state[0] = x * np.exp(self.r * (1.0 - x / self.K))
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state."""
        return self._state

    def find_fixed_points(self) -> list[dict[str, float]]:
        """Find the fixed points of the Ricker map.

        The Ricker map x_{n+1} = x * exp(r*(1-x/K)) has two fixed points:
        - x* = 0 (trivial, unstable for r > 0)
        - x* = K (nontrivial, stable for 0 < r < 2)

        Returns:
            List of dicts with 'x', 'stability', and 'eigenvalue' keys.
        """
        fps = []

        # Trivial fixed point x* = 0
        # f'(0) = exp(r) > 1 for r > 0, so always unstable
        eigenvalue_0 = np.exp(self.r)
        fps.append({
            "x": 0.0,
            "stability": "unstable" if self.r > 0 else "stable",
            "eigenvalue": eigenvalue_0,
        })

        # Nontrivial fixed point x* = K
        # f'(K) = exp(r*(1-K/K)) * (1 - r*K/K) = 1*(1-r) = 1-r
        eigenvalue_K = 1.0 - self.r
        if abs(eigenvalue_K) < 1.0:
            stability = "stable"
        else:
            stability = "unstable"
        fps.append({
            "x": self.K,
            "stability": stability,
            "eigenvalue": eigenvalue_K,
        })

        return fps

    def compute_lyapunov(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """Compute the Lyapunov exponent for the current parameters.

        lambda = lim (1/N) sum log|f'(x_n)|
        where f'(x) = exp(r*(1-x/K)) * (1 - r*x/K)
        """
        x = self.x_0
        r = self.r
        K = self.K

        # Transient to reach the attractor
        for _ in range(n_transient):
            x = x * np.exp(r * (1.0 - x / K))
            if x <= 0 or not np.isfinite(x):
                return float("nan")

        # Accumulate Lyapunov sum
        log_sum = 0.0
        for _ in range(n_iterations):
            # f'(x) = exp(r*(1 - x/K)) * (1 - r*x/K)
            exp_term = np.exp(r * (1.0 - x / K))
            derivative = abs(exp_term * (1.0 - r * x / K))
            if derivative > 0:
                log_sum += np.log(derivative)
            else:
                log_sum += -100.0  # Effectively -infinity
            x = x * exp_term
            if x <= 0 or not np.isfinite(x):
                return float("nan")

        return log_sum / n_iterations

    def detect_period(
        self,
        max_period: int = 64,
        n_transient: int = 1000,
    ) -> int:
        """Detect the period of the orbit at current r.

        Returns -1 if chaotic or period exceeds max_period.
        """
        x = self.x_0
        r = self.r
        K = self.K

        for _ in range(n_transient):
            x = x * np.exp(r * (1.0 - x / K))

        # Record orbit
        orbit = [x]
        for _ in range(max_period * 2):
            x = x * np.exp(r * (1.0 - x / K))
            orbit.append(x)

        # Check for period-p by comparing x_n with x_{n+p}
        for p in range(1, max_period + 1):
            is_periodic = True
            for i in range(max_period):
                if abs(orbit[i] - orbit[i + p]) > 1e-6:
                    is_periodic = False
                    break
            if is_periodic:
                return p

        return -1  # Chaotic or period > max_period

    def bifurcation_diagram(
        self,
        r_values: np.ndarray,
        n_transient: int = 500,
        n_plot: int = 100,
    ) -> dict[str, np.ndarray]:
        """Generate bifurcation diagram data (x vs r)."""
        all_r = []
        all_x = []
        K = self.K

        for r in r_values:
            x = self.K / 2.0  # Start near K/2
            # Transient
            for _ in range(n_transient):
                x = x * np.exp(r * (1.0 - x / K))
            # Collect steady state
            for _ in range(n_plot):
                x = x * np.exp(r * (1.0 - x / K))
                all_r.append(r)
                all_x.append(x)

        return {
            "r": np.array(all_r),
            "x": np.array(all_x),
        }
