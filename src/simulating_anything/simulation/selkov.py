"""Selkov glycolysis model simulation.

Target rediscoveries:
- ODE: dx/dt = -x + a*y + x^2*y, dy/dt = b - a*y - x^2*y
- Hopf bifurcation: b_c = a*(1 + a^2)^2 / (1 + a^2)  (simplified form)
- Fixed point: (x*, y*) satisfying x* = b - a*y* - x*^2*y* = 0
- Limit cycle oscillations above Hopf threshold
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SelkovSimulation(SimulationEnvironment):
    """Selkov glycolysis model: dx/dt = -x + a*y + x^2*y, dy/dt = b - a*y - x^2*y.

    State vector: [x, y] where x = ADP (product), y = F6P (substrate).

    The system has a unique fixed point where:
        x* = b,  y* = b / (a + b^2)

    Hopf bifurcation occurs when the trace of the Jacobian at the fixed point
    crosses zero. The Jacobian eigenvalue condition gives:
        b_c^2 = a * (1 - a - 2*a*b_c^2 / (a + b_c^2)^2)  (implicit)

    For small a, the Hopf boundary can be approximated, but we compute it
    numerically for accuracy.

    Parameters:
        a: linear substrate consumption rate (default 0.08)
        b: substrate input rate (default 0.6)
        x_0: initial ADP concentration
        y_0: initial F6P concentration
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.08)
        self.b = p.get("b", 0.6)
        self.x_0 = p.get("x_0", 0.5)
        self.y_0 = p.get("y_0", 0.5)

    @property
    def fixed_point(self) -> tuple[float, float]:
        """The unique fixed point (x*, y*).

        Setting dx/dt = 0 and dy/dt = 0:
            -x + a*y + x^2*y = 0  =>  y*(a + x^2) = x  =>  y = x / (a + x^2)
            b - a*y - x^2*y = 0   =>  b = y*(a + x^2) = x  =>  x* = b
        Therefore x* = b, y* = b / (a + b^2).
        """
        x_star = self.b
        y_star = self.b / (self.a + self.b**2)
        return (x_star, y_star)

    @property
    def hopf_threshold(self) -> float:
        """Critical b for Hopf bifurcation (numerical root-finding).

        The Jacobian at the fixed point (x*=b, y*=b/(a+b^2)) is:
            J = [[-1 + 2*x*y,   a + x^2],
                 [-2*x*y,      -(a + x^2)]]

        Hopf bifurcation occurs when trace(J) = 0:
            -1 + 2*x*y - (a + x^2) = 0

        Substituting x*=b, y*=b/(a+b^2):
            -1 + 2*b^2/(a+b^2) - a - b^2 = 0
        """
        return compute_hopf_b(self.a)

    @property
    def is_oscillatory(self) -> bool:
        """True if b is above Hopf bifurcation threshold."""
        return self.b > self.hopf_threshold

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize concentrations."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        x, yc = y
        dx = -x + self.a * yc + x**2 * yc
        dy = self.b - self.a * yc - x**2 * yc
        return np.array([dx, dy])

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period via zero crossings of x - x*."""
        if not self.is_oscillatory:
            return float("inf")

        dt = self.config.dt
        x_star = self.fixed_point[0]

        # Transient
        transient_steps = int(500 / dt)
        for _ in range(transient_steps):
            self.step()

        # Detect upward crossings of x = x_star
        crossings: list[float] = []
        prev_x = self._state[0]
        for _ in range(int(n_periods * 100 / dt)):
            self.step()
            x = self._state[0]
            if prev_x < x_star and x >= x_star:
                frac = (x_star - prev_x) / (x - prev_x) if x != prev_x else 0.5
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_x = x

        if len(crossings) < 2:
            return float("inf")

        return float(np.mean(np.diff(crossings)))

    def measure_amplitude(self, transient_time: float = 500.0) -> float:
        """Measure peak-to-peak amplitude of x after transient."""
        dt = self.config.dt

        # Skip transient
        for _ in range(int(transient_time / dt)):
            self.step()

        # Collect x values
        x_vals: list[float] = []
        for _ in range(int(200 / dt)):
            self.step()
            x_vals.append(self._state[0])

        return float(max(x_vals) - min(x_vals))


def compute_hopf_b(a: float) -> float:
    """Compute critical b for Hopf bifurcation at given a.

    Trace of Jacobian at fixed point (x*=b, y*=b/(a+b^2)):
        tr(J) = -1 + 2*b^2/(a+b^2) - (a + b^2)

    Hopf occurs when tr(J) = 0.
    """
    def trace_eq(b: float) -> float:
        return -1.0 + 2.0 * b**2 / (a + b**2) - a - b**2

    # Initial guess: for small a, b_c ~ sqrt(a)*(1+a)
    b_guess = max(0.1, (a * (1 + a**2))**0.5)
    sol = fsolve(trace_eq, b_guess, full_output=True)
    b_c = float(sol[0][0])

    # Ensure we found a positive root
    if b_c < 0:
        # Try a different initial guess
        sol2 = fsolve(trace_eq, 0.5, full_output=True)
        b_c = float(sol2[0][0])

    return abs(b_c)


def compute_hopf_boundary(
    a_values: np.ndarray,
) -> np.ndarray:
    """Compute the Hopf bifurcation boundary b_c(a) for an array of a values."""
    b_critical = np.array([compute_hopf_b(a) for a in a_values])
    return b_critical
