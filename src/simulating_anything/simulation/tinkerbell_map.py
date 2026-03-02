"""Tinkerbell map simulation -- 2D discrete chaotic map.

The Tinkerbell map is defined by:
    x_{n+1} = x_n^2 - y_n^2 + a*x_n + b*y_n
    y_{n+1} = 2*x_n*y_n + c*x_n + d*y_n

Target rediscoveries:
- Strange attractor at classic parameters (a=0.9, b=-0.6013, c=2.0, d=0.5)
- Lyapunov exponent as function of a (positive for chaotic regime)
- Fixed points from quadratic system
- Jacobian determinant det(J) = (2x+a)(2x+d) - (-2y+b)(2y+c)
- Bifurcation diagram with period-doubling route to chaos
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TinkerbellMapSimulation(SimulationEnvironment):
    """Tinkerbell map: a 2D discrete dynamical system.

    x_{n+1} = x_n^2 - y_n^2 + a*x_n + b*y_n
    y_{n+1} = 2*x_n*y_n + c*x_n + d*y_n

    State vector: [x, y] (shape (2,)).

    Parameters:
        a: first parameter (default 0.9, classic attractor)
        b: second parameter (default -0.6013)
        c: third parameter (default 2.0)
        d: fourth parameter (default 0.5)
        x_0: initial x coordinate (default -0.72)
        y_0: initial y coordinate (default -0.64)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.9)
        self.b = p.get("b", -0.6013)
        self.c = p.get("c", 2.0)
        self.d = p.get("d", 0.5)
        self.x_0 = p.get("x_0", -0.72)
        self.y_0 = p.get("y_0", -0.64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the Tinkerbell map."""
        x, y = self._state
        x_new = x**2 - y**2 + self.a * x + self.b * y
        y_new = 2.0 * x * y + self.c * x + self.d * y
        self._state = np.array([x_new, y_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y]."""
        return self._state

    def compute_jacobian(self, x: float, y: float) -> np.ndarray:
        """Compute the 2x2 Jacobian matrix at point (x, y).

        J = [[2x + a, -2y + b],
             [2y + c, 2x + d]]
        """
        return np.array([
            [2.0 * x + self.a, -2.0 * y + self.b],
            [2.0 * y + self.c, 2.0 * x + self.d],
        ], dtype=np.float64)

    def jacobian_determinant_at(self, x: float, y: float) -> float:
        """Compute det(J) at point (x, y).

        det(J) = (2x + a)(2x + d) - (-2y + b)(2y + c)
        """
        jac = self.compute_jacobian(x, y)
        return float(jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0])

    def find_fixed_points(self) -> list[np.ndarray]:
        """Find fixed points numerically by solving f(z) - z = 0.

        Fixed points satisfy:
            x^2 - y^2 + a*x + b*y = x  =>  x^2 + (a-1)*x - y^2 + b*y = 0
            2*x*y + c*x + d*y = y      =>  2*x*y + c*x + (d-1)*y = 0

        Returns a list of distinct fixed points found from multiple
        initial guesses.
        """
        def residual(z: np.ndarray) -> np.ndarray:
            x, y = z
            fx = x**2 - y**2 + self.a * x + self.b * y - x
            fy = 2.0 * x * y + self.c * x + self.d * y - y
            return np.array([fx, fy])

        # Seed guesses from a grid plus analytically-motivated points.
        # The nontrivial fixed point can lie far from the origin (e.g.
        # near (1.0, -1.3) for classic parameters), so we use a broad grid.
        guesses: list[list[float]] = [
            [0.0, 0.0],
        ]
        # Systematic grid covering likely fixed-point locations
        for gx in np.linspace(-2.0, 2.0, 9):
            for gy in np.linspace(-2.0, 2.0, 9):
                guesses.append([gx, gy])

        found: list[np.ndarray] = []
        for g in guesses:
            sol, info, ier, _ = fsolve(residual, g, full_output=True)
            if ier == 1 and np.max(np.abs(info["fvec"])) < 1e-10:
                is_new = True
                for existing in found:
                    if np.linalg.norm(sol - existing) < 1e-6:
                        is_new = False
                        break
                if is_new:
                    found.append(np.array(sol, dtype=np.float64))

        return found

    def compute_lyapunov(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """Estimate the largest Lyapunov exponent via QR method.

        Tracks the Jacobian product along the orbit using repeated
        QR decomposition to prevent numerical overflow.
        """
        x, y = self.x_0, self.y_0

        # Transient to reach the attractor
        for _ in range(n_transient):
            x_new = x**2 - y**2 + self.a * x + self.b * y
            y_new = 2.0 * x * y + self.c * x + self.d * y
            x, y = x_new, y_new
            # Check for divergence during transient
            if x**2 + y**2 > 1e10:
                return float("nan")

        # Accumulate Lyapunov exponent via QR method
        q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_iterations):
            jac = self.compute_jacobian(x, y)
            m = jac @ q
            q, r = np.linalg.qr(m)
            diag = np.abs(np.diag(r))
            if np.any(diag == 0):
                break
            lyap_sum += np.log(diag)

            # Advance the orbit
            x_new = x**2 - y**2 + self.a * x + self.b * y
            y_new = 2.0 * x * y + self.c * x + self.d * y
            x, y = x_new, y_new
            if x**2 + y**2 > 1e10:
                return float("nan")

        return float(lyap_sum[0] / n_iterations)

    def compute_lyapunov_spectrum(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> np.ndarray:
        """Compute the full Lyapunov spectrum (both exponents).

        Returns array of shape (2,) with [lambda_1, lambda_2] sorted descending.
        """
        x, y = self.x_0, self.y_0

        for _ in range(n_transient):
            x_new = x**2 - y**2 + self.a * x + self.b * y
            y_new = 2.0 * x * y + self.c * x + self.d * y
            x, y = x_new, y_new
            if x**2 + y**2 > 1e10:
                return np.array([float("nan"), float("nan")])

        q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_iterations):
            jac = self.compute_jacobian(x, y)
            m = jac @ q
            q, r = np.linalg.qr(m)
            diag = np.abs(np.diag(r))
            if np.any(diag == 0):
                break
            lyap_sum += np.log(diag)

            x_new = x**2 - y**2 + self.a * x + self.b * y
            y_new = 2.0 * x * y + self.c * x + self.d * y
            x, y = x_new, y_new
            if x**2 + y**2 > 1e10:
                return np.array([float("nan"), float("nan")])

        spectrum = lyap_sum / n_iterations
        return np.sort(spectrum)[::-1]

    def bifurcation_diagram(
        self,
        a_values: np.ndarray,
        n_transient: int = 500,
        n_plot: int = 100,
    ) -> dict[str, np.ndarray]:
        """Generate bifurcation diagram data (x vs a)."""
        all_a: list[float] = []
        all_x: list[float] = []

        for a_val in a_values:
            x, y = self.x_0, self.y_0
            diverged = False
            # Transient
            for _ in range(n_transient):
                x_new = x**2 - y**2 + a_val * x + self.b * y
                y_new = 2.0 * x * y + self.c * x + self.d * y
                x, y = x_new, y_new
                if x**2 + y**2 > 1e10:
                    diverged = True
                    break
            if diverged:
                continue
            # Collect attractor points
            for _ in range(n_plot):
                x_new = x**2 - y**2 + a_val * x + self.b * y
                y_new = 2.0 * x * y + self.c * x + self.d * y
                x, y = x_new, y_new
                if x**2 + y**2 > 1e10:
                    break
                all_a.append(a_val)
                all_x.append(x)

        return {
            "a": np.array(all_a),
            "x": np.array(all_x),
        }

    def detect_period(
        self,
        max_period: int = 64,
        n_transient: int = 1000,
    ) -> int:
        """Detect the period of the orbit at current parameters.

        Returns -1 if chaotic or period exceeds max_period.
        """
        x, y = self.x_0, self.y_0
        for _ in range(n_transient):
            x_new = x**2 - y**2 + self.a * x + self.b * y
            y_new = 2.0 * x * y + self.c * x + self.d * y
            x, y = x_new, y_new
            if x**2 + y**2 > 1e10:
                return -1

        # Record orbit
        orbit_x = [x]
        orbit_y = [y]
        for _ in range(max_period * 2):
            x_new = x**2 - y**2 + self.a * x + self.b * y
            y_new = 2.0 * x * y + self.c * x + self.d * y
            x, y = x_new, y_new
            if x**2 + y**2 > 1e10:
                return -1
            orbit_x.append(x)
            orbit_y.append(y)

        # Check for period-p
        for p in range(1, max_period + 1):
            is_periodic = True
            for i in range(max_period):
                dx = abs(orbit_x[i] - orbit_x[i + p])
                dy = abs(orbit_y[i] - orbit_y[i + p])
                if dx > 1e-6 or dy > 1e-6:
                    is_periodic = False
                    break
            if is_periodic:
                return p

        return -1

    def collect_attractor(
        self,
        n_steps: int = 10000,
        n_transient: int = 1000,
    ) -> np.ndarray:
        """Collect attractor points after transient.

        Returns array of shape (n_steps, 2) with the attractor trajectory.
        """
        x, y = self.x_0, self.y_0

        for _ in range(n_transient):
            x_new = x**2 - y**2 + self.a * x + self.b * y
            y_new = 2.0 * x * y + self.c * x + self.d * y
            x, y = x_new, y_new

        points = np.zeros((n_steps, 2))
        for i in range(n_steps):
            x_new = x**2 - y**2 + self.a * x + self.b * y
            y_new = 2.0 * x * y + self.c * x + self.d * y
            x, y = x_new, y_new
            points[i] = [x, y]

        return points
