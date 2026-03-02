"""Bazykin predator-prey model with intraspecific competition and Holling Type II.

Extends classical predator-prey with saturating functional response (Holling
Type II) and intraspecific competition in the predator. Exhibits multiple
bifurcations: Hopf, saddle-node, and homoclinic.

Equations:
    dx/dt = x*(1 - x) - x*y/(1 + alpha*x)
    dy/dt = -gamma*y + x*y/(1 + alpha*x) - delta*y^2

Default parameters: alpha=0.1, gamma=0.1, delta=0.01
"""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BazykinSimulation(SimulationEnvironment):
    """Bazykin predator-prey model with intraspecific predator competition.

    State vector: [x, y] where x = prey, y = predator.

    Equations:
        dx/dt = x*(1 - x) - x*y/(1 + alpha*x)
        dy/dt = -gamma*y + x*y/(1 + alpha*x) - delta*y^2

    Parameters:
        alpha: handling time / half-saturation (default 0.1)
        gamma: predator death rate (default 0.1)
        delta: predator intraspecific competition (default 0.01)
        x_0: initial prey population (default 0.5)
        y_0: initial predator population (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 0.1)
        self.gamma_param = p.get("gamma", 0.1)
        self.delta = p.get("delta", 0.01)
        self.x_0 = p.get("x_0", 0.5)
        self.y_0 = p.get("y_0", 0.5)

    @property
    def total_population(self) -> float:
        """Sum of prey and predator populations."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    @property
    def prey_population(self) -> float:
        """Current prey population."""
        if self._state is None:
            return 0.0
        return float(self._state[0])

    @property
    def predator_population(self) -> float:
        """Current predator population."""
        if self._state is None:
            return 0.0
        return float(self._state[1])

    def coexistence_equilibrium(self) -> tuple[float, float]:
        """Compute the interior coexistence equilibrium (x*, y*).

        At equilibrium:
            x*(1-x*) = x*y*/(1+alpha*x*)   => y* = (1-x*)(1+alpha*x*)
            -gamma*y* + x*y*/(1+alpha*x*) - delta*y*^2 = 0

        From the second equation (y* != 0):
            x*/(1+alpha*x*) = gamma + delta*y*

        Substituting y* from the first:
            x*/(1+alpha*x*) = gamma + delta*(1-x*)(1+alpha*x*)

        This is solved numerically via bisection on (0, 1).

        Returns:
            Tuple (x_star, y_star).

        Raises:
            ValueError: If no valid coexistence equilibrium exists.
        """
        gamma = self.gamma_param

        def f(x: float) -> float:
            denom = 1.0 + self.alpha * x
            y_prey = (1.0 - x) * denom
            return x / denom - gamma - self.delta * y_prey

        # Check bracket on (0, 1)
        eps = 1e-10
        f_lo = f(eps)
        f_hi = f(1.0 - eps)

        if f_lo * f_hi > 0:
            raise ValueError(
                "No coexistence equilibrium found in (0, 1): "
                f"f(0+)={f_lo:.6f}, f(1-)={f_hi:.6f}"
            )

        # Bisection
        lo, hi = eps, 1.0 - eps
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(mid) * f(lo) < 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-12:
                break

        x_star = 0.5 * (lo + hi)
        y_star = (1.0 - x_star) * (1.0 + self.alpha * x_star)

        if x_star <= 0 or y_star <= 0:
            raise ValueError(
                f"Coexistence equilibrium not positive: "
                f"x*={x_star:.6f}, y*={y_star:.6f}"
            )

        return (x_star, y_star)

    def jacobian(self, x: float, y: float) -> np.ndarray:
        """Compute the Jacobian matrix at a given state (x, y).

        Returns:
            2x2 numpy array of partial derivatives.
        """
        gamma = self.gamma_param
        denom = 1.0 + self.alpha * x
        denom2 = denom ** 2

        # Partial derivatives of dx/dt = x*(1-x) - x*y/(1+alpha*x)
        dfdx = 1.0 - 2.0 * x - y / denom2
        dfdy = -x / denom

        # Partial derivatives of dy/dt = -gamma*y + x*y/(1+alpha*x) - delta*y^2
        dgdx = y / denom2
        dgdy = -gamma + x / denom - 2.0 * self.delta * y

        return np.array([[dfdx, dfdy], [dgdx, dgdy]])

    def is_stable(self) -> bool:
        """Check if the coexistence equilibrium is locally stable.

        Stability requires trace(J) < 0 and det(J) > 0.

        Returns:
            True if coexistence equilibrium exists and is stable.
        """
        try:
            x_star, y_star = self.coexistence_equilibrium()
        except ValueError:
            return False

        J = self.jacobian(x_star, y_star)
        tr = J[0, 0] + J[1, 1]
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        return bool(tr < 0 and det > 0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [x, y]."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [x, y]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Ensure non-negative populations
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Bazykin right-hand side.

        dx/dt = x*(1 - x) - x*y/(1 + alpha*x)
        dy/dt = -gamma*y + x*y/(1 + alpha*x) - delta*y^2
        """
        gamma = self.gamma_param
        x, pred = y
        functional_response = x / (1.0 + self.alpha * x)

        dx = x * (1.0 - x) - functional_response * pred
        dy = (
            -gamma * pred
            + functional_response * pred
            - self.delta * pred ** 2
        )
        return np.array([dx, dy])
