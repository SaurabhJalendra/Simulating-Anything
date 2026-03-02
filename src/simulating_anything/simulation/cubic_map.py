"""Cubic map simulation (1D discrete chaos with odd symmetry).

The cubic map is a one-dimensional discrete dynamical system:
  x_{n+1} = r * x_n - x_n^3

Target rediscoveries:
- Period-doubling route to chaos as r increases
- Fixed points: x* = 0 (always), x* = +/-sqrt(r-1) for r > 1
- Odd symmetry: f(-x) = -f(x), so orbits come in symmetric pairs
- Stability of x*=0: |f'(0)| = |r|, stable for |r| < 1
- Stability of x*=+/-sqrt(r-1): f'(x*) = r - 3(r-1) = 3 - 2r,
  stable for |3 - 2r| < 1, i.e., 1 < r < 2
- Period-doubling onset at r = 2 (nontrivial fixed point loses stability)
- Lyapunov exponent: lambda = lim (1/N) sum log|r - 3*x_n^2|
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CubicMapSimulation(SimulationEnvironment):
    """Cubic map: x_{n+1} = r * x_n - x_n^3.

    State: [x] (single scalar, stored as 1D array for consistency).

    Parameters:
        r: control parameter (default 2.5)
        x_0: initial value (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 2.5)
        self.x_0 = p.get("x_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state."""
        self._state = np.array([self.x_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the cubic map."""
        x = self._state[0]
        self._state[0] = self.r * x - x**3
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state."""
        return self._state

    def find_fixed_points(self) -> list[dict[str, float]]:
        """Find the fixed points of the cubic map.

        Fixed points satisfy x = r*x - x^3, i.e., x^3 + x - r*x = 0,
        so x * (x^2 - (r - 1)) = 0.

        Solutions:
        - x* = 0 (always exists)
        - x* = +/-sqrt(r-1) (exist only when r > 1)

        Stability via f'(x) = r - 3*x^2:
        - x*=0: f'(0) = r, stable if |r| < 1
        - x*=+/-sqrt(r-1): f'(x*) = r - 3*(r-1) = 3 - 2r,
          stable if |3 - 2r| < 1, i.e., 1 < r < 2

        Returns:
            List of dicts with 'x', 'stability', and 'eigenvalue' keys.
        """
        fps = []

        # Trivial fixed point x* = 0
        eigenvalue_0 = self.r
        stability_0 = "stable" if abs(eigenvalue_0) < 1 else "unstable"
        fps.append({
            "x": 0.0,
            "stability": stability_0,
            "eigenvalue": eigenvalue_0,
        })

        # Nontrivial fixed points x* = +/-sqrt(r-1) for r > 1
        if self.r > 1.0:
            x_star = np.sqrt(self.r - 1.0)
            eigenvalue_nontrivial = 3.0 - 2.0 * self.r
            stability_nt = (
                "stable" if abs(eigenvalue_nontrivial) < 1 else "unstable"
            )
            fps.append({
                "x": x_star,
                "stability": stability_nt,
                "eigenvalue": eigenvalue_nontrivial,
            })
            fps.append({
                "x": -x_star,
                "stability": stability_nt,
                "eigenvalue": eigenvalue_nontrivial,
            })

        return fps

    def compute_lyapunov(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """Compute the Lyapunov exponent for the current r.

        lambda = lim (1/N) sum log|f'(x_n)| = lim (1/N) sum log|r - 3*x_n^2|
        """
        x = self.x_0

        # Transient to reach the attractor
        for _ in range(n_transient):
            try:
                x = self.r * x - x**3
            except (OverflowError, FloatingPointError):
                return float("nan")
            if not np.isfinite(x):
                return float("nan")

        # Accumulate
        log_sum = 0.0
        for _ in range(n_iterations):
            try:
                derivative = abs(self.r - 3.0 * x**2)
            except (OverflowError, FloatingPointError):
                return float("nan")
            if derivative > 0:
                log_sum += np.log(derivative)
            else:
                log_sum += -100.0  # Effectively -infinity
            try:
                x = self.r * x - x**3
            except (OverflowError, FloatingPointError):
                return float("nan")
            if not np.isfinite(x):
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

        for _ in range(n_transient):
            x = self.r * x - x**3

        # Record orbit
        orbit = [x]
        for _ in range(max_period * 2):
            x = self.r * x - x**3
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

        for r in r_values:
            x = 0.5  # Starting point
            # Transient
            for _ in range(n_transient):
                x = r * x - x**3
                # Escape guard: if orbit diverges, skip
                if abs(x) > 1e10:
                    break
            if abs(x) > 1e10:
                continue
            # Collect steady state
            for _ in range(n_plot):
                x = r * x - x**3
                if abs(x) > 1e10:
                    break
                all_r.append(r)
                all_x.append(x)

        return {
            "r": np.array(all_r),
            "x": np.array(all_x),
        }

    def compute_invariant_density(
        self,
        n_iterations: int = 100000,
        n_transient: int = 1000,
        n_bins: int = 200,
    ) -> dict[str, np.ndarray]:
        """Estimate the invariant density of the attractor via histogram.

        Iterates the map past a transient and then bins the visited x values.

        Returns:
            Dict with 'bin_centers' and 'density' arrays.
        """
        x = self.x_0

        # Transient
        for _ in range(n_transient):
            x = self.r * x - x**3
            if not np.isfinite(x):
                return {"bin_centers": np.array([]), "density": np.array([])}

        # Collect orbit
        orbit = np.empty(n_iterations)
        for i in range(n_iterations):
            x = self.r * x - x**3
            if not np.isfinite(x):
                orbit = orbit[:i]
                break
            orbit[i] = x

        if len(orbit) < 10:
            return {"bin_centers": np.array([]), "density": np.array([])}

        counts, bin_edges = np.histogram(orbit, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return {
            "bin_centers": bin_centers,
            "density": counts,
        }
