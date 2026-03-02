"""Lozi map simulation -- 2D piecewise-linear discrete chaotic map.

Target rediscoveries:
- Strange attractor at a=1.7, b=0.5 (piecewise-linear analog of Henon map)
- Lyapunov exponent as function of a (positive in chaotic regime)
- Fixed points: x* = 1/(1+a-b), y* = b*x* (for x* > 0 branch)
- Area contraction: Jacobian determinant |det(J)| = |b| per iteration
- For b=0: reduces to 1D tent-like map x_{n+1} = 1 - a*|x_n|
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LoziMapSimulation(SimulationEnvironment):
    """Lozi map: x_{n+1} = 1 - a*|x_n| + y_n, y_{n+1} = b*x_n.

    A piecewise-linear analog of the Henon map. The absolute value
    nonlinearity |x| replaces the quadratic x^2, producing a strange
    attractor with piecewise-linear structure.

    State vector: [x, y] (shape (2,)).

    Parameters:
        a: nonlinearity parameter (default 1.7, classic chaotic attractor)
        b: contraction parameter (default 0.5)
        x_0: initial x coordinate (default 0.0)
        y_0: initial y coordinate (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.7)
        self.b = p.get("b", 0.5)
        self.x_0 = p.get("x_0", 0.0)
        self.y_0 = p.get("y_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the Lozi map."""
        x, y = self._state
        x_new = 1.0 - self.a * abs(x) + y
        y_new = self.b * x
        self._state = np.array([x_new, y_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y]."""
        return self._state

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Lozi map.

        Fixed points satisfy x = 1 - a*|x| + b*x (since y = b*x).

        For x > 0: x = 1 - a*x + b*x => x*(1 + a - b) = 1
            x* = 1/(1 + a - b), y* = b*x*

        For x < 0: x = 1 + a*x + b*x => x*(1 - a - b) = 1
            x* = 1/(1 - a - b), y* = b*x*
            This solution only valid if 1 - a - b != 0 and x* < 0.
        """
        fps = []

        # Positive branch: x > 0
        denom_pos = 1.0 + self.a - self.b
        if abs(denom_pos) > 1e-15:
            x_pos = 1.0 / denom_pos
            if x_pos > 0:
                fps.append(np.array([x_pos, self.b * x_pos], dtype=np.float64))

        # Negative branch: x < 0
        denom_neg = 1.0 - self.a - self.b
        if abs(denom_neg) > 1e-15:
            x_neg = 1.0 / denom_neg
            if x_neg < 0:
                fps.append(np.array([x_neg, self.b * x_neg], dtype=np.float64))

        return fps

    def jacobian_at(self, x: float) -> np.ndarray:
        """Compute the 2x2 Jacobian matrix at a point.

        The Lozi map Jacobian depends on the sign of x:
            J = [[-a*sign(x), 1], [b, 0]]  for x != 0
            J is undefined at x = 0 (piecewise-linear kink).
        """
        if x > 0:
            sign_x = 1.0
        elif x < 0:
            sign_x = -1.0
        else:
            sign_x = 0.0  # Subgradient convention
        return np.array([
            [-self.a * sign_x, 1.0],
            [self.b, 0.0],
        ], dtype=np.float64)

    @property
    def jacobian_determinant(self) -> float:
        """Jacobian determinant of the Lozi map.

        J = [[-a*sign(x), 1], [b, 0]], so det(J) = 0 - b = -b.
        The area contraction factor per iteration is |det(J)| = |b|.
        This is constant (independent of x), same as the Henon map.
        """
        return -self.b

    def compute_lyapunov(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """Estimate the largest Lyapunov exponent via QR method.

        Uses the standard QR-based method for 2D maps: track the Jacobian
        product along the orbit via repeated QR decomposition.
        """
        x, y = self.x_0, self.y_0

        # Transient to reach the attractor
        for _ in range(n_transient):
            x_new = 1.0 - self.a * abs(x) + y
            y_new = self.b * x
            x, y = x_new, y_new

        # Accumulate Lyapunov exponent via QR method
        q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_iterations):
            # Jacobian at current point
            jac = self.jacobian_at(x)

            # Map the tangent vectors
            m = jac @ q
            # QR decomposition to re-orthonormalize
            q, r = np.linalg.qr(m)
            # Accumulate log of diagonal elements (stretching factors)
            lyap_sum += np.log(np.abs(np.diag(r)))

            # Advance the orbit
            x_new = 1.0 - self.a * abs(x) + y
            y_new = self.b * x
            x, y = x_new, y_new

        # Largest Lyapunov exponent is the first component
        return float(lyap_sum[0] / n_iterations)

    def compute_lyapunov_spectrum(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> np.ndarray:
        """Compute both Lyapunov exponents via QR method.

        Returns array [lambda_1, lambda_2] where lambda_1 >= lambda_2.
        For a dissipative map: lambda_1 + lambda_2 = ln|b|.
        """
        x, y = self.x_0, self.y_0

        for _ in range(n_transient):
            x_new = 1.0 - self.a * abs(x) + y
            y_new = self.b * x
            x, y = x_new, y_new

        q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_iterations):
            jac = self.jacobian_at(x)
            m = jac @ q
            q, r = np.linalg.qr(m)
            lyap_sum += np.log(np.abs(np.diag(r)))

            x_new = 1.0 - self.a * abs(x) + y
            y_new = self.b * x
            x, y = x_new, y_new

        return lyap_sum / n_iterations

    def bifurcation_diagram(
        self,
        a_values: np.ndarray,
        n_transient: int = 500,
        n_plot: int = 100,
    ) -> dict[str, np.ndarray]:
        """Generate bifurcation diagram data (x vs a)."""
        all_a = []
        all_x = []

        for a in a_values:
            x, y = 0.0, 0.0
            # Transient
            for _ in range(n_transient):
                x_new = 1.0 - a * abs(x) + y
                y_new = self.b * x
                x, y = x_new, y_new
            # Collect attractor points
            for _ in range(n_plot):
                x_new = 1.0 - a * abs(x) + y
                y_new = self.b * x
                x, y = x_new, y_new
                all_a.append(a)
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
            x_new = 1.0 - self.a * abs(x) + y
            y_new = self.b * x
            x, y = x_new, y_new

        # Record orbit
        orbit_x = [x]
        orbit_y = [y]
        for _ in range(max_period * 2):
            x_new = 1.0 - self.a * abs(x) + y
            y_new = self.b * x
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

    def compute_attractor_dimension(self, n_steps: int = 10000) -> float:
        """Estimate the correlation dimension of the attractor.

        Uses the Grassberger-Procaccia algorithm: for a set of orbit points,
        compute the correlation integral C(r) = (2/N^2) * count(|z_i - z_j| < r)
        and fit the scaling C(r) ~ r^D to estimate dimension D.
        """
        x, y = self.x_0, self.y_0

        # Transient
        for _ in range(1000):
            x_new = 1.0 - self.a * abs(x) + y
            y_new = self.b * x
            x, y = x_new, y_new

        # Collect orbit points
        points = np.zeros((n_steps, 2))
        for i in range(n_steps):
            x_new = 1.0 - self.a * abs(x) + y
            y_new = self.b * x
            x, y = x_new, y_new
            points[i] = [x, y]

        # Compute pairwise distances (subsample for speed)
        max_pts = min(n_steps, 2000)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_steps, size=max_pts, replace=False)
        pts = points[idx]

        dists = []
        for i in range(max_pts):
            for j in range(i + 1, max_pts):
                d = np.sqrt(
                    (pts[i, 0] - pts[j, 0]) ** 2
                    + (pts[i, 1] - pts[j, 1]) ** 2
                )
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
