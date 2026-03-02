"""Predator-Two-Prey ecological simulation.

Implements a 3-species system with 1 predator and 2 prey species sharing
a common predator (apparent competition):

    dx1/dt = r1*x1*(1 - x1/K1) - a1*x1*y
    dx2/dt = r2*x2*(1 - x2/K2) - a2*x2*y
    dy/dt  = -d*y + b1*x1*y + b2*x2*y

where x1, x2 are prey populations and y is the predator population.

Key phenomena:
- Apparent competition between prey species via shared predator
- Coexistence vs competitive exclusion depending on parameters
- Logistic prey growth with Holling Type I functional response
- Limit cycles and oscillatory dynamics possible
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorTwoPreySimulation(SimulationEnvironment):
    """Predator-Two-Prey model with apparent competition.

    State vector: [x1, x2, y] where x1, x2 are prey populations and
    y is the predator population.

    Equations:
        dx1/dt = r1*x1*(1 - x1/K1) - a1*x1*y
        dx2/dt = r2*x2*(1 - x2/K2) - a2*x2*y
        dy/dt  = -d*y + b1*x1*y + b2*x2*y

    Parameters (via config.parameters):
        r1: prey 1 intrinsic growth rate (default 1.0)
        r2: prey 2 intrinsic growth rate (default 0.8)
        K1: prey 1 carrying capacity (default 10.0)
        K2: prey 2 carrying capacity (default 8.0)
        a1: predation rate on prey 1 (default 0.5)
        a2: predation rate on prey 2 (default 0.4)
        b1: predator conversion from prey 1 (default 0.2)
        b2: predator conversion from prey 2 (default 0.15)
        d:  predator death rate (default 0.6)
        x1_0: initial prey 1 population (default 5.0)
        x2_0: initial prey 2 population (default 4.0)
        y_0:  initial predator population (default 2.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.r1 = p.get("r1", 1.0)
        self.r2 = p.get("r2", 0.8)
        self.K1 = p.get("K1", 10.0)
        self.K2 = p.get("K2", 8.0)
        self.a1 = p.get("a1", 0.5)
        self.a2 = p.get("a2", 0.4)
        self.b1 = p.get("b1", 0.2)
        self.b2 = p.get("b2", 0.15)
        self.d = p.get("d", 0.6)

        self.x1_0 = p.get("x1_0", 5.0)
        self.x2_0 = p.get("x2_0", 4.0)
        self.y_0 = p.get("y_0", 2.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [x1, x2, y]."""
        self._state = np.array(
            [self.x1_0, self.x2_0, self.y_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with non-negativity enforcement."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [x1, x2, y]."""
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
        # Enforce non-negative populations
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Predator-Two-Prey right-hand side.

        dx1/dt = r1*x1*(1 - x1/K1) - a1*x1*y
        dx2/dt = r2*x2*(1 - x2/K2) - a2*x2*y
        dy/dt  = -d*y + b1*x1*y + b2*x2*y
        """
        x1, x2, yp = y
        dx1 = self.r1 * x1 * (1.0 - x1 / self.K1) - self.a1 * x1 * yp
        dx2 = self.r2 * x2 * (1.0 - x2 / self.K2) - self.a2 * x2 * yp
        dy = -self.d * yp + self.b1 * x1 * yp + self.b2 * x2 * yp
        return np.array([dx1, dx2, dy])

    def jacobian(self, state: np.ndarray | None = None) -> np.ndarray:
        """Compute the Jacobian matrix at the given state (or current state).

        J = [[r1*(1 - 2*x1/K1) - a1*y,  0,                     -a1*x1],
             [0,                          r2*(1 - 2*x2/K2) - a2*y, -a2*x2],
             [b1*y,                       b2*y,                  -d + b1*x1 + b2*x2]]

        Args:
            state: State [x1, x2, y] to evaluate at. Uses current if None.

        Returns:
            3x3 Jacobian matrix.
        """
        if state is None:
            state = self._state
        x1, x2, yp = state

        J = np.array([
            [
                self.r1 * (1.0 - 2.0 * x1 / self.K1) - self.a1 * yp,
                0.0,
                -self.a1 * x1,
            ],
            [
                0.0,
                self.r2 * (1.0 - 2.0 * x2 / self.K2) - self.a2 * yp,
                -self.a2 * x2,
            ],
            [
                self.b1 * yp,
                self.b2 * yp,
                -self.d + self.b1 * x1 + self.b2 * x2,
            ],
        ], dtype=np.float64)
        return J

    def fixed_points(self) -> list[np.ndarray]:
        """Compute analytically known fixed points.

        Fixed points:
        1. Trivial: (0, 0, 0)
        2. Prey 1 only: (K1, 0, 0)
        3. Prey 2 only: (0, K2, 0)
        4. Both prey, no predator: (K1, K2, 0)
        5. Prey 1 + predator (x2=0): solve x1*(r1/K1) = r1 - a1*y and
           b1*x1 = d  =>  x1 = d/b1, y = (r1/a1)*(1 - d/(b1*K1))
        6. Prey 2 + predator (x1=0): x2 = d/b2, y = (r2/a2)*(1 - d/(b2*K2))
        7. Interior coexistence: all three positive (solve coupled system)

        Returns:
            List of fixed point arrays. Only returns points with all
            non-negative components.
        """
        fps = []

        # (0, 0, 0)
        fps.append(np.array([0.0, 0.0, 0.0]))

        # (K1, 0, 0)
        fps.append(np.array([self.K1, 0.0, 0.0]))

        # (0, K2, 0)
        fps.append(np.array([0.0, self.K2, 0.0]))

        # (K1, K2, 0)
        fps.append(np.array([self.K1, self.K2, 0.0]))

        # Prey 1 + predator boundary: x1 = d/b1, x2 = 0
        if self.b1 > 0:
            x1_star = self.d / self.b1
            y_star = (self.r1 / self.a1) * (1.0 - x1_star / self.K1)
            if x1_star >= 0 and y_star >= 0:
                fps.append(np.array([x1_star, 0.0, y_star]))

        # Prey 2 + predator boundary: x2 = d/b2, x1 = 0
        if self.b2 > 0:
            x2_star = self.d / self.b2
            y_star = (self.r2 / self.a2) * (1.0 - x2_star / self.K2)
            if x2_star >= 0 and y_star >= 0:
                fps.append(np.array([0.0, x2_star, y_star]))

        # Interior coexistence fixed point (all three positive):
        # From dy/dt = 0: b1*x1 + b2*x2 = d
        # From dx1/dt = 0: y = (r1/a1)*(1 - x1/K1)
        # From dx2/dt = 0: y = (r2/a2)*(1 - x2/K2)
        # Equating the two y expressions and combining with predator nullcline:
        fp_interior = self._interior_fixed_point()
        if fp_interior is not None:
            fps.append(fp_interior)

        return fps

    def _interior_fixed_point(self) -> np.ndarray | None:
        """Solve for the interior fixed point where all species coexist.

        From the nullclines:
            y = (r1/a1)*(1 - x1/K1)   ... (i)
            y = (r2/a2)*(1 - x2/K2)   ... (ii)
            b1*x1 + b2*x2 = d         ... (iii)

        From (i) and (ii): x2 = K2*(1 - (a2/r2)*(r1/a1)*(1 - x1/K1))
        Substitute into (iii) to solve for x1.

        Returns:
            Interior fixed point array [x1*, x2*, y*] if feasible, else None.
        """
        # From (i): y = (r1/a1)*(1 - x1/K1)
        # From (ii): x2 = K2*(1 - (a2*y)/r2) = K2 - K2*a2*y/r2
        # Substituting y from (i):
        # x2 = K2 - K2*(a2/r2)*(r1/a1)*(1 - x1/K1)
        # x2 = K2 - K2*a2*r1/(r2*a1) + K2*a2*r1/(r2*a1*K1)*x1
        # Let c = a2*r1/(r2*a1)
        c = (self.a2 * self.r1) / (self.r2 * self.a1)
        # x2 = K2*(1 - c) + K2*c/K1 * x1
        # Substitute into (iii): b1*x1 + b2*(K2*(1-c) + K2*c/K1*x1) = d
        # b1*x1 + b2*K2*(1-c) + b2*K2*c/K1*x1 = d
        # x1*(b1 + b2*K2*c/K1) = d - b2*K2*(1-c)
        denom = self.b1 + self.b2 * self.K2 * c / self.K1
        if abs(denom) < 1e-15:
            return None

        x1_star = (self.d - self.b2 * self.K2 * (1.0 - c)) / denom
        x2_star = self.K2 * (1.0 - c) + self.K2 * c / self.K1 * x1_star
        y_star = (self.r1 / self.a1) * (1.0 - x1_star / self.K1)

        if x1_star > 0 and x2_star > 0 and y_star > 0:
            return np.array([x1_star, x2_star, y_star], dtype=np.float64)
        return None

    def compute_divergence(self, state: np.ndarray | None = None) -> float:
        """Compute the divergence of the vector field (trace of Jacobian).

        For dissipative systems, divergence < 0 implies volume contraction.

        Args:
            state: State [x1, x2, y] to evaluate at. Uses current if None.

        Returns:
            Divergence value (scalar).
        """
        J = self.jacobian(state)
        return float(np.trace(J))

    def n_surviving(self, threshold: float = 1e-3) -> int:
        """Count species with population above threshold."""
        return int(np.sum(self._state > threshold))

    @property
    def total_population(self) -> float:
        """Sum of all species populations."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    @property
    def is_coexisting(self) -> bool:
        """True if all three species are above the extinction threshold."""
        if self._state is None:
            return False
        return bool(np.all(self._state > 1e-6))

    def apparent_competition_index(self) -> float:
        """Measure of apparent competition strength.

        The shared predator mediates indirect competition between prey.
        This index quantifies the interaction strength:
            ACI = (b1*a2 + b2*a1) / d

        Higher values mean stronger indirect competition between prey species.

        Returns:
            Apparent competition index.
        """
        return (self.b1 * self.a2 + self.b2 * self.a1) / self.d
