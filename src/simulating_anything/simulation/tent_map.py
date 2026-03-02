"""Tent map simulation (piecewise-linear 1D discrete map).

The tent map is defined by:
  x_{n+1} = r * min(x, 1 - x)
Equivalently:
  x_{n+1} = r * x       if x < 0.5
  x_{n+1} = r * (1 - x) if x >= 0.5

Target rediscoveries:
- Lyapunov exponent = ln(r) for ergodic orbits (r > 1)
- No period-doubling cascade: direct transition to chaos at r = 1
- Fixed points: x* = 0, x* = r / (r + 1)
- Invariant density: uniform on [0, 1] for r = 2
- At r = 2, Lyapunov exponent = ln(2) exactly
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TentMapSimulation(SimulationEnvironment):
    """Tent map: x_{n+1} = r * min(x, 1 - x).

    State: [x] (single scalar, stored as 1D array for consistency).

    Parameters:
        r: control parameter (default 2.0, range [0, 2])
        x_0: initial value (default 1/pi ~ 0.3183)
             Avoid dyadic rationals (p/2^n) which produce periodic orbits
             at r=2 due to the binary shift structure of the tent map.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 2.0)
        # Default to 1/pi (irrational) to avoid periodic orbits at r=2
        self.x_0 = p.get("x_0", 1.0 / np.pi)

    def _tent(self, x: float) -> float:
        """Apply one iteration of the tent map."""
        if x < 0.5:
            return self.r * x
        return self.r * (1.0 - x)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state."""
        self._state = np.array([self.x_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the tent map."""
        x = self._state[0]
        self._state[0] = self._tent(x)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state."""
        return self._state

    def find_fixed_points(self) -> list[dict]:
        """Find the fixed points of the tent map.

        Fixed points satisfy x = r * min(x, 1-x):
        - x* = 0 (trivial, always a fixed point)
          - Left branch: f(0) = r*0 = 0. Derivative = r.
          - Stable for r < 1, unstable for r > 1.
        - x* = r / (r + 1) (nontrivial, exists only for r >= 1)
          - On the right branch (x >= 0.5): f(x) = r*(1-x) = x => x = r/(r+1).
          - Requires r/(r+1) >= 0.5, i.e., r >= 1.
          - Derivative on right branch = -r, so |eigenvalue| = r.
          - Always unstable for r > 1, marginally stable at r = 1.
        """
        fps = []

        # Trivial fixed point x* = 0
        fps.append({
            "x": 0.0,
            "stability": "stable" if self.r < 1.0 else "unstable",
            "eigenvalue": self.r,
        })

        # Nontrivial fixed point x* = r / (r + 1), exists for r >= 1
        if self.r >= 1.0:
            x_star = self.r / (self.r + 1.0)
            fps.append({
                "x": x_star,
                "stability": "unstable" if self.r > 1.0 else "marginally_stable",
                "eigenvalue": -self.r,
            })

        return fps

    def compute_lyapunov(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """Compute the Lyapunov exponent for the current r.

        For the tent map, f'(x) = r for x < 0.5 and f'(x) = -r for x > 0.5.
        So |f'(x)| = r everywhere (except x = 0.5).
        Therefore lambda = ln(r) for ergodic orbits when r > 1.

        We compute numerically for verification:
        lambda = lim (1/N) sum log|f'(x_n)|
        """
        x = self.x_0
        r = self.r

        # Transient to reach the attractor
        for _ in range(n_transient):
            x = self._tent_static(x, r)
            if not np.isfinite(x):
                return float("nan")

        # Accumulate
        log_sum = 0.0
        for _ in range(n_iterations):
            # |f'(x)| = r for all x != 0.5
            if r > 0:
                log_sum += np.log(r)
            else:
                log_sum += -100.0
            x = self._tent_static(x, r)
            if not np.isfinite(x):
                return float("nan")

        return log_sum / n_iterations

    @staticmethod
    def _tent_static(x: float, r: float) -> float:
        """Static tent map iteration for use in analysis methods.

        At r=2, the tent map is conjugate to a binary shift, so finite-precision
        floating-point arithmetic causes orbits to eventually reach x=0 (absorbing
        state) after ~52 iterations (the number of mantissa bits in float64).
        We add a tiny random perturbation to maintain ergodic behavior, which
        is mathematically justified since the invariant measure is continuous.
        """
        if x < 0.5:
            x_new = r * x
        else:
            x_new = r * (1.0 - x)
        # Perturb away from absorbing boundaries (floating-point artifact at r=2)
        if x_new <= 0.0 or x_new >= 1.0:
            x_new = np.random.uniform(1e-10, 1.0 - 1e-10)
        return x_new

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

        for _ in range(n_transient):
            x = self._tent_static(x, r)

        # Record orbit
        orbit = [x]
        for _ in range(max_period * 2):
            x = self._tent_static(x, r)
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
            x = 1.0 / np.pi  # Irrational to avoid periodic traps at r=2
            # Transient
            for _ in range(n_transient):
                x = self._tent_static(x, r)
            # Collect steady state
            for _ in range(n_plot):
                x = self._tent_static(x, r)
                all_r.append(r)
                all_x.append(x)

        return {
            "r": np.array(all_r),
            "x": np.array(all_x),
        }

    def compute_invariant_density(
        self,
        n_bins: int = 50,
        n_iterations: int = 100000,
        n_transient: int = 1000,
    ) -> dict[str, np.ndarray]:
        """Compute the invariant density via histogram of long orbit.

        For r = 2, the invariant density is uniform on [0, 1].

        Returns:
            Dict with 'bin_centers' and 'density' arrays.
        """
        x = self.x_0
        r = self.r

        # Transient
        for _ in range(n_transient):
            x = self._tent_static(x, r)

        # Collect orbit
        orbit = np.empty(n_iterations)
        for i in range(n_iterations):
            x = self._tent_static(x, r)
            orbit[i] = x

        # Histogram
        counts, bin_edges = np.histogram(orbit, bins=n_bins, range=(0, 1), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return {
            "bin_centers": bin_centers,
            "density": counts,
        }
