"""Henon map simulation -- 2D discrete chaotic map.

Target rediscoveries:
- Period-doubling route to chaos as a increases
- Lyapunov exponent as function of a (~0.42 at a=1.4, b=0.3)
- Fixed points: x* = (-(1-b) +/- sqrt((1-b)^2 + 4a)) / (2a), y* = b*x*
- Area contraction: Jacobian determinant = |b| per iteration
- For b=0: reduces to 1D quadratic map x_{n+1} = 1 - a*x_n^2
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class HenonMapSimulation(SimulationEnvironment):
    """Henon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n.

    State vector: [x, y] (shape (2,)).

    Parameters:
        a: nonlinearity parameter (default 1.4, classic chaotic attractor)
        b: contraction parameter (default 0.3)
        x_0: initial x coordinate (default 0.0)
        y_0: initial y coordinate (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.4)
        self.b = p.get("b", 0.3)
        self.x_0 = p.get("x_0", 0.0)
        self.y_0 = p.get("y_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the Henon map."""
        x, y = self._state
        x_new = 1.0 - self.a * x**2 + y
        y_new = self.b * x
        self._state = np.array([x_new, y_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y]."""
        return self._state

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the two fixed points of the Henon map.

        Fixed points satisfy x = 1 - a*x^2 + b*x, i.e. a*x^2 + (1-b)*x - 1 = 0.
        Solutions: x* = (-(1-b) +/- sqrt((1-b)^2 + 4a)) / (2a), y* = b*x*.
        """
        discriminant = (1 - self.b) ** 2 + 4 * self.a
        if discriminant < 0:
            return []
        sqrt_disc = np.sqrt(discriminant)
        x1 = (-(1 - self.b) + sqrt_disc) / (2 * self.a)
        x2 = (-(1 - self.b) - sqrt_disc) / (2 * self.a)
        return [
            np.array([x1, self.b * x1], dtype=np.float64),
            np.array([x2, self.b * x2], dtype=np.float64),
        ]

    @property
    def jacobian_determinant(self) -> float:
        """Jacobian determinant of the Henon map.

        J = [[-2*a*x, 1], [b, 0]], so det(J) = -b.
        The area contraction factor per iteration is |det(J)| = |b|.
        """
        return -self.b

    def compute_lyapunov(
        self,
        n_iterations: int = 10000,
        n_transient: int = 1000,
    ) -> float:
        """Estimate the largest Lyapunov exponent.

        Uses the standard QR-based method for 2D maps: track the Jacobian
        product along the orbit via repeated QR decomposition.
        """
        x, y = self.x_0, self.y_0

        # Transient to reach the attractor
        for _ in range(n_transient):
            x_new = 1.0 - self.a * x**2 + y
            y_new = self.b * x
            x, y = x_new, y_new

        # Accumulate Lyapunov exponent via QR method
        # Start with identity tangent vector
        q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_iterations):
            # Jacobian at current point
            jac = np.array([
                [-2.0 * self.a * x, 1.0],
                [self.b, 0.0],
            ])

            # Map the tangent vectors
            m = jac @ q
            # QR decomposition to re-orthonormalize
            q, r = np.linalg.qr(m)
            # Accumulate log of diagonal elements (stretching factors)
            lyap_sum += np.log(np.abs(np.diag(r)))

            # Advance the orbit
            x_new = 1.0 - self.a * x**2 + y
            y_new = self.b * x
            x, y = x_new, y_new

        # Largest Lyapunov exponent is the first component
        return float(lyap_sum[0] / n_iterations)

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
                x_new = 1.0 - a * x**2 + y
                y_new = self.b * x
                x, y = x_new, y_new
            # Collect attractor points
            for _ in range(n_plot):
                x_new = 1.0 - a * x**2 + y
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
            x_new = 1.0 - self.a * x**2 + y
            y_new = self.b * x
            x, y = x_new, y_new

        # Record orbit
        orbit_x = [x]
        orbit_y = [y]
        for _ in range(max_period * 2):
            x_new = 1.0 - self.a * x**2 + y
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
