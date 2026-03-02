"""Circle map (Arnold tongue) simulation -- 1D map on the unit circle.

The circle map is: theta_{n+1} = (theta_n + Omega - (K/(2*pi)) * sin(2*pi*theta_n)) mod 1

Target rediscoveries:
- Devil's staircase: w(Omega) at K=1 is a complete devil's staircase
- Mode-locking: rational winding numbers w = p/q form Arnold tongues
- K=0: pure rotation (winding number = Omega exactly)
- K=1: critical line, Arnold tongues touch, measure-zero set of irrational w
- K>1: overlapping tongues, chaos possible
- Lyapunov exponent: zero for K=0, can be positive for K>1
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CircleMapSimulation(SimulationEnvironment):
    """Circle map on [0, 1): theta_{n+1} = (theta_n + Omega - K/(2pi) sin(2pi theta_n)) mod 1.

    State: [theta] (single scalar stored as 1D array for consistency).

    Parameters:
        omega: driving frequency (default 0.5)
        K: nonlinearity strength (default 1.0)
        theta_0: initial angle on [0, 1) (default 0.1)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.omega = p.get("omega", 0.5)
        self.K = p.get("K", 1.0)
        self.theta_0 = p.get("theta_0", 0.1)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state to theta_0."""
        self._state = np.array([self.theta_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Apply one iteration of the circle map."""
        theta = self._state[0]
        theta_new = (
            theta + self.omega - (self.K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
        )
        self._state[0] = theta_new % 1.0
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [theta]."""
        return self._state

    def compute_winding_number(
        self,
        n_transient: int = 500,
        n_compute: int = 2000,
        omega: float | None = None,
        K: float | None = None,
        theta_0: float | None = None,
    ) -> float:
        """Compute the winding number w = lim (Theta_N - Theta_0) / N.

        Uses the unwrapped (non-modular) angle to track total rotation.

        Args:
            n_transient: iterations to discard before measurement.
            n_compute: iterations used for the winding number estimate.
            omega: driving frequency (defaults to self.omega).
            K: nonlinearity (defaults to self.K).
            theta_0: initial condition (defaults to self.theta_0).

        Returns:
            Winding number (average rotation per iteration).
        """
        if omega is None:
            omega = self.omega
        if K is None:
            K = self.K
        if theta_0 is None:
            theta_0 = self.theta_0

        theta = theta_0
        # Unwrapped angle -- tracks cumulative rotation without mod
        theta_unwrapped = theta

        # Transient: iterate without modding to accumulate unwrapped angle
        for _ in range(n_transient):
            theta_new = theta + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
            theta_unwrapped += (theta_new - theta)
            theta = theta_new % 1.0

        # Record starting unwrapped angle
        start_unwrapped = theta_unwrapped

        # Compute phase: accumulate unwrapped rotation
        for _ in range(n_compute):
            theta_new = theta + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
            theta_unwrapped += (theta_new - theta)
            theta = theta_new % 1.0

        return (theta_unwrapped - start_unwrapped) / n_compute

    def compute_lyapunov(
        self,
        n_transient: int = 500,
        n_compute: int = 5000,
        omega: float | None = None,
        K: float | None = None,
        theta_0: float | None = None,
    ) -> float:
        """Compute the Lyapunov exponent of the circle map.

        lambda = (1/N) * sum(log|f'(theta_n)|)
        where f'(theta) = 1 - K * cos(2*pi*theta).

        Args:
            n_transient: transient iterations before accumulation.
            n_compute: iterations for Lyapunov computation.
            omega: driving frequency (defaults to self.omega).
            K: nonlinearity (defaults to self.K).
            theta_0: initial condition (defaults to self.theta_0).

        Returns:
            Lyapunov exponent (nats per iteration).
        """
        if omega is None:
            omega = self.omega
        if K is None:
            K = self.K
        if theta_0 is None:
            theta_0 = self.theta_0

        theta = theta_0

        # Transient
        for _ in range(n_transient):
            theta = (
                theta + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
            ) % 1.0

        # Accumulate log|f'(theta_n)|
        log_sum = 0.0
        for _ in range(n_compute):
            derivative = abs(1.0 - K * np.cos(2 * np.pi * theta))
            if derivative > 0:
                log_sum += np.log(derivative)
            else:
                log_sum += -100.0  # Effectively -infinity
            theta = (
                theta + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)
            ) % 1.0

        return log_sum / n_compute

    def arnold_tongue_data(
        self,
        omega_range: tuple[float, float] = (0.0, 1.0),
        K_range: tuple[float, float] = (0.0, 2.0),
        n_omega: int = 100,
        n_K: int = 50,
    ) -> dict[str, np.ndarray]:
        """Compute winding number grid over (Omega, K) parameter space.

        This maps out the Arnold tongue structure: regions where the winding
        number locks to rational values p/q.

        Args:
            omega_range: (min, max) for driving frequency.
            K_range: (min, max) for nonlinearity.
            n_omega: number of Omega grid points.
            n_K: number of K grid points.

        Returns:
            Dict with 'omega_values', 'K_values', and 'winding_numbers' (2D array).
        """
        omega_values = np.linspace(omega_range[0], omega_range[1], n_omega)
        K_values = np.linspace(K_range[0], K_range[1], n_K)
        winding_numbers = np.zeros((n_K, n_omega), dtype=np.float64)

        for i, K in enumerate(K_values):
            for j, omega in enumerate(omega_values):
                w = self.compute_winding_number(
                    n_transient=200,
                    n_compute=1000,
                    omega=omega,
                    K=K,
                )
                winding_numbers[i, j] = w

        return {
            "omega_values": omega_values,
            "K_values": K_values,
            "winding_numbers": winding_numbers,
        }

    def devils_staircase(
        self,
        n_omega: int = 500,
        omega_range: tuple[float, float] = (0.0, 1.0),
        K: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Compute the devil's staircase w(Omega) at a fixed K.

        At K=1 (critical), the winding number as a function of Omega forms
        a complete devil's staircase: constant on a dense set of intervals
        (mode-locked tongues), with irrational values on a measure-zero set.

        Args:
            n_omega: number of Omega values to sample.
            omega_range: range of Omega to sweep.
            K: nonlinearity parameter.

        Returns:
            Dict with 'omega' and 'winding_number' arrays.
        """
        omega_values = np.linspace(omega_range[0], omega_range[1], n_omega)
        winding_numbers = np.zeros(n_omega, dtype=np.float64)

        for i, omega in enumerate(omega_values):
            winding_numbers[i] = self.compute_winding_number(
                n_transient=500,
                n_compute=2000,
                omega=omega,
                K=K,
            )

        return {
            "omega": omega_values,
            "winding_number": winding_numbers,
        }

    def detect_mode_locking(
        self,
        omega_values: np.ndarray,
        winding_numbers: np.ndarray,
        tolerance: float = 1e-4,
        max_q: int = 12,
    ) -> list[dict[str, float]]:
        """Detect mode-locked plateaus (rational winding numbers p/q).

        Scans the winding number data for intervals where w is approximately
        constant at a rational value p/q with q <= max_q.

        Args:
            omega_values: array of Omega values.
            winding_numbers: corresponding winding numbers.
            tolerance: max deviation from p/q to count as locked.
            max_q: maximum denominator to consider.

        Returns:
            List of dicts with keys 'p', 'q', 'winding_number', 'omega_center',
            'omega_width' for each detected tongue.
        """
        # Build list of Farey fractions p/q with q <= max_q
        rationals = set()
        for q in range(1, max_q + 1):
            for p in range(0, q + 1):
                rationals.add((p, q))

        tongues = []
        for p, q in sorted(rationals, key=lambda x: x[0] / x[1]):
            target = p / q
            locked = np.abs(winding_numbers - target) < tolerance
            if not np.any(locked):
                continue

            indices = np.where(locked)[0]
            omega_center = float(np.mean(omega_values[indices]))
            omega_width = float(omega_values[indices[-1]] - omega_values[indices[0]])

            tongues.append({
                "p": int(p),
                "q": int(q),
                "winding_number": target,
                "omega_center": omega_center,
                "omega_width": omega_width,
                "n_points": int(len(indices)),
            })

        return tongues
