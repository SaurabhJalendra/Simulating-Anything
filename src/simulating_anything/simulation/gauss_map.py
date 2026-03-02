"""Gauss map simulation (Gaussian iterated map, 1D discrete chaos).

The Gauss map is a one-dimensional discrete dynamical system:
  x_{n+1} = exp(-alpha * x_n^2) + beta

where alpha > 0 controls the curvature and beta shifts the map vertically.

Target rediscoveries:
- Period-doubling route to chaos as beta varies
- Fixed points: solve x = exp(-alpha * x^2) + beta numerically
- Lyapunov exponent: lambda = lim (1/N) sum log|f'(x_n)|
  where f'(x) = -2 * alpha * x * exp(-alpha * x^2)
- Chaotic and periodic windows in the beta parameter space
- Derivative at fixed point determines stability: |f'(x*)| < 1 => stable
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GaussMapSimulation(SimulationEnvironment):
    """Gauss map: x_{n+1} = exp(-alpha * x_n^2) + beta.

    State: [x] (single scalar, stored as 1D array for consistency).

    Parameters:
        alpha: curvature parameter (default 6.2, must be > 0)
        beta: vertical shift parameter (default -0.5)
        x_0: initial value (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 6.2)
        self.beta = p.get("beta", -0.5)
        self.x_0 = p.get("x_0", 0.5)

    def _map(self, x: float) -> float:
        """Apply the Gauss map: f(x) = exp(-alpha * x^2) + beta."""
        return np.exp(-self.alpha * x**2) + self.beta

    def _derivative(self, x: float) -> float:
        """Compute f'(x) = -2 * alpha * x * exp(-alpha * x^2)."""
        return -2.0 * self.alpha * x * np.exp(-self.alpha * x**2)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state."""
        self._state = np.array([self.x_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the Gauss map."""
        x = self._state[0]
        self._state[0] = self._map(x)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state."""
        return self._state

    def find_fixed_points(
        self,
        n_search: int = 200,
        x_range: tuple[float, float] = (-2.0, 2.0),
    ) -> list[dict]:
        """Find fixed points of the Gauss map numerically.

        Fixed points satisfy g(x) = f(x) - x = exp(-alpha * x^2) + beta - x = 0.
        Uses sign-change detection on a fine grid followed by Brent's method.

        Args:
            n_search: number of grid points for sign-change scanning.
            x_range: range to search for fixed points.

        Returns:
            List of dicts with 'x', 'stability', and 'eigenvalue' keys.
        """
        def residual(x: float) -> float:
            return self._map(x) - x

        x_grid = np.linspace(x_range[0], x_range[1], n_search)
        g_vals = np.array([residual(x) for x in x_grid])

        fps = []
        found_x = []

        for i in range(len(x_grid) - 1):
            if g_vals[i] * g_vals[i + 1] < 0:
                try:
                    x_root = brentq(residual, x_grid[i], x_grid[i + 1])
                except ValueError:
                    continue

                # Check it is not a duplicate
                is_new = all(abs(x_root - xf) > 1e-8 for xf in found_x)
                if not is_new:
                    continue

                found_x.append(x_root)
                eigenvalue = self._derivative(x_root)
                if abs(eigenvalue) < 1.0:
                    stability = "stable"
                else:
                    stability = "unstable"

                fps.append({
                    "x": float(x_root),
                    "stability": stability,
                    "eigenvalue": float(eigenvalue),
                })

        return fps

    def compute_lyapunov(
        self,
        n_transient: int = 1000,
        n_compute: int = 5000,
    ) -> float:
        """Compute the Lyapunov exponent for the current parameters.

        lambda = lim (1/N) sum log|f'(x_n)|
        where f'(x) = -2 * alpha * x * exp(-alpha * x^2).

        Args:
            n_transient: iterations to discard before accumulating.
            n_compute: iterations to accumulate the Lyapunov sum.

        Returns:
            The estimated Lyapunov exponent.
        """
        x = self.x_0

        # Transient to reach the attractor
        for _ in range(n_transient):
            x = self._map(x)
            if not np.isfinite(x):
                return float("nan")

        # Accumulate
        log_sum = 0.0
        for _ in range(n_compute):
            deriv = abs(self._derivative(x))
            if deriv > 0:
                log_sum += np.log(deriv)
            else:
                # f'(x) = 0 at x = 0; this is a super-stable orbit point
                log_sum += -100.0  # Effectively -infinity
            x = self._map(x)
            if not np.isfinite(x):
                return float("nan")

        return log_sum / n_compute

    def detect_period(
        self,
        max_period: int = 64,
        n_transient: int = 1000,
    ) -> int:
        """Detect the period of the orbit at current parameters.

        Returns -1 if chaotic or period exceeds max_period.
        """
        x = self.x_0

        for _ in range(n_transient):
            x = self._map(x)

        # Record orbit
        orbit = [x]
        for _ in range(max_period * 2):
            x = self._map(x)
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

    def bifurcation_data(
        self,
        param_name: str = "beta",
        param_range: tuple[float, float] = (-1.0, 1.0),
        n_params: int = 200,
        n_transient: int = 500,
        n_plot: int = 100,
    ) -> dict[str, np.ndarray]:
        """Generate bifurcation diagram data.

        Args:
            param_name: which parameter to sweep ('alpha' or 'beta').
            param_range: (min, max) for the sweep.
            n_params: number of parameter values to sample.
            n_transient: iterations to discard per parameter value.
            n_plot: orbit points to record per parameter value.

        Returns:
            Dict with 'param' and 'x' arrays for plotting.
        """
        param_values = np.linspace(param_range[0], param_range[1], n_params)
        all_param = []
        all_x = []

        for pval in param_values:
            if param_name == "beta":
                alpha = self.alpha
                beta = pval
            elif param_name == "alpha":
                alpha = pval
                beta = self.beta
            else:
                raise ValueError(f"Unknown param_name: {param_name}")

            x = 0.5  # Starting point for each parameter
            # Transient
            for _ in range(n_transient):
                x = np.exp(-alpha * x**2) + beta
                if not np.isfinite(x) or abs(x) > 1e10:
                    break
            if not np.isfinite(x) or abs(x) > 1e10:
                continue

            # Collect attractor points
            for _ in range(n_plot):
                x = np.exp(-alpha * x**2) + beta
                if not np.isfinite(x) or abs(x) > 1e10:
                    break
                all_param.append(pval)
                all_x.append(x)

        return {
            "param": np.array(all_param),
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
            x = self._map(x)
            if not np.isfinite(x):
                return {"bin_centers": np.array([]), "density": np.array([])}

        # Collect orbit
        orbit = np.empty(n_iterations)
        for i in range(n_iterations):
            x = self._map(x)
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
