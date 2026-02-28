"""Logistic map simulation (discrete-time chaos).

Target rediscoveries:
- Period-doubling route to chaos
- Feigenbaum constant: delta ~ 4.669 (ratio of successive bifurcation intervals)
- Critical r values: period-1 stable for r < 3, period-2 for r < 3.449,
  period-4 for r < 3.544, chaos onset ~ r = 3.5699...
- Lyapunov exponent as function of r
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LogisticMapSimulation(SimulationEnvironment):
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n).

    State: [x] (single scalar, but stored as 1D array for consistency).

    Parameters:
        r: growth rate parameter (default 3.5)
        x_0: initial value (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 3.5)
        self.x_0 = p.get("x_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state."""
        self._state = np.array([self.x_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the logistic map."""
        x = self._state[0]
        self._state[0] = self.r * x * (1 - x)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state."""
        return self._state

    def lyapunov_exponent(self, n_iterations: int = 10000, n_transient: int = 1000) -> float:
        """Compute the Lyapunov exponent for the current r.

        lambda = lim (1/N) sum log|f'(x_n)| = lim (1/N) sum log|r*(1-2*x_n)|
        """
        x = self.x_0
        # Transient
        for _ in range(n_transient):
            x = self.r * x * (1 - x)

        # Accumulate
        log_sum = 0.0
        for _ in range(n_iterations):
            derivative = abs(self.r * (1 - 2 * x))
            if derivative > 0:
                log_sum += np.log(derivative)
            else:
                log_sum += -100  # Effectively -infinity
            x = self.r * x * (1 - x)

        return log_sum / n_iterations

    def bifurcation_diagram(
        self,
        r_values: np.ndarray,
        n_transient: int = 500,
        n_plot: int = 100,
    ) -> dict[str, np.ndarray]:
        """Generate bifurcation diagram data."""
        all_r = []
        all_x = []

        for r in r_values:
            x = 0.5
            # Transient
            for _ in range(n_transient):
                x = r * x * (1 - x)
            # Collect steady state
            for _ in range(n_plot):
                x = r * x * (1 - x)
                all_r.append(r)
                all_x.append(x)

        return {
            "r": np.array(all_r),
            "x": np.array(all_x),
        }

    def detect_period(self, max_period: int = 64, n_transient: int = 1000) -> int:
        """Detect the period of the orbit at current r."""
        x = self.x_0
        for _ in range(n_transient):
            x = self.r * x * (1 - x)

        # Record orbit
        orbit = [x]
        for _ in range(max_period * 2):
            x = self.r * x * (1 - x)
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
