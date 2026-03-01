"""Ikeda map simulation -- 2D discrete chaotic map modeling a nonlinear optical cavity.

Target rediscoveries:
- Chaotic attractor for u > 0.9, period-doubling route for 0.6 < u < 0.8
- Dissipative: attractor area contraction |det(J)| = u^2 < 1
- Lyapunov exponent as function of u (positive for chaotic regime)
- Fixed points: solve x = 1 + u*(x*cos(t) - y*sin(t)), y = u*(x*sin(t) + y*cos(t))
  where t = 0.4 - 6/(1 + x^2 + y^2)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class IkedaMapSimulation(SimulationEnvironment):
    """Ikeda map for a nonlinear optical resonator.

    x_{n+1} = 1 + u * (x_n * cos(t_n) - y_n * sin(t_n))
    y_{n+1} = u * (x_n * sin(t_n) + y_n * cos(t_n))

    where t_n = 0.4 - 6 / (1 + x_n^2 + y_n^2).

    State vector: [x, y] (shape (2,)).

    Parameters:
        u: coupling parameter (default 0.9). Controls chaos:
            u < 0.6: single fixed point
            0.6 < u < 0.8: period-doubling
            u > 0.9: chaotic attractor
        x_0: initial x coordinate (default 0.0)
        y_0: initial y coordinate (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.u = p.get("u", 0.9)
        self.x_0 = p.get("x_0", 0.0)
        self.y_0 = p.get("y_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the Ikeda map."""
        x, y = self._state
        t = self.compute_t(x, y)
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        x_new = 1.0 + self.u * (x * cos_t - y * sin_t)
        y_new = self.u * (x * sin_t + y * cos_t)
        self._state = np.array([x_new, y_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y]."""
        return self._state

    @staticmethod
    def compute_t(x: float, y: float) -> float:
        """Compute the phase modulation t_n = 0.4 - 6/(1 + x^2 + y^2)."""
        return 0.4 - 6.0 / (1.0 + x**2 + y**2)

    def compute_jacobian(self, x: float, y: float) -> np.ndarray:
        """Compute the 2x2 Jacobian matrix at point (x, y).

        The Jacobian of the Ikeda map is derived from the chain rule applied
        to the map equations including the dependence of t on (x, y).
        """
        r2 = x**2 + y**2
        denom = (1.0 + r2) ** 2
        t = self.compute_t(x, y)
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        # dt/dx and dt/dy
        dt_dx = 12.0 * x / denom
        dt_dy = 12.0 * y / denom

        # Partial derivatives of x_{n+1} = 1 + u*(x*cos(t) - y*sin(t))
        # with respect to x and y, noting t depends on x and y
        a_term = x * cos_t - y * sin_t  # rotation output x-component
        b_term = x * sin_t + y * cos_t  # rotation output y-component

        # da_term/dx = cos_t + (-x*sin_t - y*cos_t) * dt_dx = cos_t - b_term * dt_dx
        # da_term/dy = -sin_t + (-x*sin_t - y*cos_t) * dt_dy = -sin_t - b_term * dt_dy

        # db_term/dx = sin_t + (x*cos_t - y*sin_t) * dt_dx = sin_t + a_term * dt_dx
        # db_term/dy = cos_t + (x*cos_t - y*sin_t) * dt_dy = cos_t + a_term * dt_dy

        j00 = self.u * (cos_t - b_term * dt_dx)
        j01 = self.u * (-sin_t - b_term * dt_dy)
        j10 = self.u * (sin_t + a_term * dt_dx)
        j11 = self.u * (cos_t + a_term * dt_dy)

        return np.array([[j00, j01], [j10, j11]], dtype=np.float64)

    def find_fixed_points(self) -> list[np.ndarray]:
        """Find fixed points numerically by solving f(z) - z = 0.

        Returns a list of distinct fixed points found from multiple initial guesses.
        """
        def residual(z: np.ndarray) -> np.ndarray:
            x, y = z
            t = self.compute_t(x, y)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            fx = 1.0 + self.u * (x * cos_t - y * sin_t) - x
            fy = self.u * (x * sin_t + y * cos_t) - y
            return np.array([fx, fy])

        # Try multiple initial guesses to find distinct fixed points
        guesses = [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [-2.0, -2.0],
            [1.5, 1.5],
            [0.5, -0.5],
        ]

        found = []
        for g in guesses:
            sol, info, ier, _ = fsolve(residual, g, full_output=True)
            if ier == 1 and np.max(np.abs(info["fvec"])) < 1e-10:
                # Check if this is a new fixed point
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
        n_steps: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """Estimate the largest Lyapunov exponent via QR method.

        Tracks the Jacobian product along the orbit using repeated
        QR decomposition to prevent numerical overflow.
        """
        x, y = self.x_0, self.y_0

        # Transient to reach the attractor
        for _ in range(n_transient):
            t = self.compute_t(x, y)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            x_new = 1.0 + self.u * (x * cos_t - y * sin_t)
            y_new = self.u * (x * sin_t + y * cos_t)
            x, y = x_new, y_new

        # Accumulate Lyapunov exponent via QR method
        q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_steps):
            jac = self.compute_jacobian(x, y)
            m = jac @ q
            q, r = np.linalg.qr(m)
            lyap_sum += np.log(np.abs(np.diag(r)))

            # Advance the orbit
            t = self.compute_t(x, y)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            x_new = 1.0 + self.u * (x * cos_t - y * sin_t)
            y_new = self.u * (x * sin_t + y * cos_t)
            x, y = x_new, y_new

        return float(lyap_sum[0] / n_steps)

    def bifurcation_sweep(
        self,
        u_values: np.ndarray,
        n_transient: int = 500,
        n_record: int = 100,
    ) -> dict[str, np.ndarray]:
        """Generate bifurcation diagram data (x vs u).

        For each u value, iterates the map past the transient and then
        records n_record points from the attractor.
        """
        all_u = []
        all_x = []
        all_y = []

        for u_val in u_values:
            x, y = 0.0, 0.0
            # Transient
            for _ in range(n_transient):
                t = self.compute_t(x, y)
                cos_t = np.cos(t)
                sin_t = np.sin(t)
                x_new = 1.0 + u_val * (x * cos_t - y * sin_t)
                y_new = u_val * (x * sin_t + y * cos_t)
                x, y = x_new, y_new
            # Collect attractor points
            for _ in range(n_record):
                t = self.compute_t(x, y)
                cos_t = np.cos(t)
                sin_t = np.sin(t)
                x_new = 1.0 + u_val * (x * cos_t - y * sin_t)
                y_new = u_val * (x * sin_t + y * cos_t)
                x, y = x_new, y_new
                all_u.append(u_val)
                all_x.append(x)
                all_y.append(y)

        return {
            "u": np.array(all_u),
            "x": np.array(all_x),
            "y": np.array(all_y),
        }

    def compute_attractor_dimension(self, n_steps: int = 10000) -> float:
        """Estimate the correlation dimension of the attractor.

        Uses the Grassberger-Procaccia algorithm: for a set of orbit points,
        compute the correlation integral C(r) = (2/N^2) * #{|z_i - z_j| < r}
        and fit the scaling C(r) ~ r^D to estimate dimension D.
        """
        x, y = self.x_0, self.y_0

        # Transient
        for _ in range(1000):
            t = self.compute_t(x, y)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            x_new = 1.0 + self.u * (x * cos_t - y * sin_t)
            y_new = self.u * (x * sin_t + y * cos_t)
            x, y = x_new, y_new

        # Collect orbit points
        points = np.zeros((n_steps, 2))
        for i in range(n_steps):
            t = self.compute_t(x, y)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            x_new = 1.0 + self.u * (x * cos_t - y * sin_t)
            y_new = self.u * (x * sin_t + y * cos_t)
            x, y = x_new, y_new
            points[i] = [x, y]

        # Compute pairwise distances (subsample for speed)
        max_pts = min(n_steps, 2000)
        idx = np.random.default_rng(42).choice(n_steps, size=max_pts, replace=False)
        pts = points[idx]

        dists = []
        for i in range(max_pts):
            for j in range(i + 1, max_pts):
                d = np.sqrt((pts[i, 0] - pts[j, 0]) ** 2 + (pts[i, 1] - pts[j, 1]) ** 2)
                if d > 0:
                    dists.append(d)
        dists = np.sort(dists)

        if len(dists) < 10:
            return 0.0

        # Correlation integral at multiple radii
        n_pairs = len(dists)
        r_min = dists[int(0.01 * n_pairs)]
        r_max = dists[int(0.50 * n_pairs)]
        if r_min <= 0 or r_max <= r_min:
            return 0.0

        radii = np.logspace(np.log10(r_min), np.log10(r_max), 20)
        log_r = []
        log_c = []
        for r in radii:
            count = np.searchsorted(dists, r)
            if count > 0:
                c = count / n_pairs
                log_r.append(np.log(r))
                log_c.append(np.log(c))

        if len(log_r) < 5:
            return 0.0

        # Linear fit to log-log
        coeffs = np.polyfit(log_r, log_c, 1)
        return float(coeffs[0])

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
            t = self.compute_t(x, y)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            x_new = 1.0 + self.u * (x * cos_t - y * sin_t)
            y_new = self.u * (x * sin_t + y * cos_t)
            x, y = x_new, y_new

        # Record orbit
        orbit_x = [x]
        orbit_y = [y]
        for _ in range(max_period * 2):
            t = self.compute_t(x, y)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            x_new = 1.0 + self.u * (x * cos_t - y * sin_t)
            y_new = self.u * (x * sin_t + y * cos_t)
            x, y = x_new, y_new
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
