"""Four-species Lotka-Volterra food web simulation.

Two prey and two predator species with intra-prey competition and
predator-prey coupling:

    dx1/dt = x1*(r1 - a11*x1 - a12*x2 - b1*y1)
    dx2/dt = x2*(r2 - a21*x1 - a22*x2 - b2*y2)
    dy1/dt = y1*(-d1 + c1*x1)
    dy2/dt = y2*(-d2 + c2*x2)

Target rediscoveries:
- SINDy ODE recovery (4 coupled equations)
- Coexistence equilibrium: x1*=d1/c1, x2*=d2/c2, y1*=..., y2*=...
- Jacobian eigenvalue stability analysis
- Competition strength sweep (a12, a21 vary)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FourSpeciesLVSimulation(SimulationEnvironment):
    """Four-species Lotka-Volterra food web: two prey, two predators.

    State vector: [x1, x2, y1, y2] where
        x1, x2 = prey species densities
        y1, y2 = predator species densities

    Equations:
        dx1/dt = x1*(r1 - a11*x1 - a12*x2 - b1*y1)
        dx2/dt = x2*(r2 - a21*x1 - a22*x2 - b2*y2)
        dy1/dt = y1*(-d1 + c1*x1)
        dy2/dt = y2*(-d2 + c2*x2)

    Parameters:
        r1, r2: prey intrinsic growth rates (default 1.0, 0.8)
        a11, a12, a21, a22: competition coefficients among prey
            (defaults: 0.1, 0.05, 0.05, 0.1)
        b1, b2: predation rates (defaults: 0.5, 0.5)
        c1, c2: conversion efficiencies (defaults: 0.3, 0.3)
        d1, d2: predator death rates (defaults: 0.4, 0.4)
        x1_0, x2_0: initial prey densities (defaults: 0.5, 0.5)
        y1_0, y2_0: initial predator densities (defaults: 0.3, 0.3)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Prey growth rates
        self.r1 = p.get("r1", 1.0)
        self.r2 = p.get("r2", 0.8)

        # Prey competition coefficients
        self.a11 = p.get("a11", 0.1)
        self.a12 = p.get("a12", 0.05)
        self.a21 = p.get("a21", 0.05)
        self.a22 = p.get("a22", 0.1)

        # Predation rates
        self.b1 = p.get("b1", 0.5)
        self.b2 = p.get("b2", 0.5)

        # Conversion efficiencies
        self.c1 = p.get("c1", 0.3)
        self.c2 = p.get("c2", 0.3)

        # Predator death rates
        self.d1 = p.get("d1", 0.4)
        self.d2 = p.get("d2", 0.4)

        # Initial conditions
        self.x1_0 = p.get("x1_0", 0.5)
        self.x2_0 = p.get("x2_0", 0.5)
        self.y1_0 = p.get("y1_0", 0.3)
        self.y2_0 = p.get("y2_0", 0.3)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [x1, x2, y1, y2]."""
        self._state = np.array(
            [self.x1_0, self.x2_0, self.y1_0, self.y2_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [x1, x2, y1, y2]."""
        return self._state

    @property
    def total_population(self) -> float:
        """Sum of all four species populations."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    @property
    def is_coexisting(self) -> bool:
        """True if all four species are above the extinction threshold."""
        if self._state is None:
            return False
        threshold = 1e-6
        return bool(np.all(self._state > threshold))

    def n_surviving(self, threshold: float = 1e-3) -> int:
        """Count species with population above threshold."""
        return int(np.sum(self._state > threshold))

    def coexistence_equilibrium(self) -> np.ndarray:
        """Compute the interior coexistence equilibrium analytically.

        At equilibrium each derivative is zero. Assuming all species are
        positive, dividing out x_i or y_i gives:

            From dy1/dt = 0: x1* = d1/c1
            From dy2/dt = 0: x2* = d2/c2
            From dx1/dt = 0: y1* = (r1 - a11*x1* - a12*x2*) / b1
            From dx2/dt = 0: y2* = (r2 - a21*x1* - a22*x2*) / b2

        The equilibrium is feasible only if y1* > 0 and y2* > 0.

        Returns:
            numpy array [x1*, x2*, y1*, y2*] of the coexistence equilibrium.
        """
        x1_star = self.d1 / self.c1
        x2_star = self.d2 / self.c2
        y1_star = (self.r1 - self.a11 * x1_star - self.a12 * x2_star) / self.b1
        y2_star = (self.r2 - self.a21 * x1_star - self.a22 * x2_star) / self.b2
        return np.array([x1_star, x2_star, y1_star, y2_star], dtype=np.float64)

    def jacobian_at_equilibrium(self) -> np.ndarray:
        """Compute the Jacobian matrix at the coexistence equilibrium.

        For the system:
            f1 = x1*(r1 - a11*x1 - a12*x2 - b1*y1)
            f2 = x2*(r2 - a21*x1 - a22*x2 - b2*y2)
            f3 = y1*(-d1 + c1*x1)
            f4 = y2*(-d2 + c2*x2)

        At the interior equilibrium where the per-capita growth rates vanish:
            J[0,0] = -a11*x1*, J[0,1] = -a12*x1*, J[0,2] = -b1*x1*, J[0,3] = 0
            J[1,0] = -a21*x2*, J[1,1] = -a22*x2*, J[1,2] = 0,       J[1,3] = -b2*x2*
            J[2,0] = c1*y1*,   J[2,1] = 0,        J[2,2] = 0,        J[2,3] = 0
            J[3,0] = 0,        J[3,1] = c2*y2*,   J[3,2] = 0,        J[3,3] = 0

        Returns:
            4x4 Jacobian matrix.
        """
        eq = self.coexistence_equilibrium()
        x1s, x2s, y1s, y2s = eq

        J = np.zeros((4, 4), dtype=np.float64)

        # df1/d(x1,x2,y1,y2)
        J[0, 0] = -self.a11 * x1s
        J[0, 1] = -self.a12 * x1s
        J[0, 2] = -self.b1 * x1s
        J[0, 3] = 0.0

        # df2/d(x1,x2,y1,y2)
        J[1, 0] = -self.a21 * x2s
        J[1, 1] = -self.a22 * x2s
        J[1, 2] = 0.0
        J[1, 3] = -self.b2 * x2s

        # df3/d(x1,x2,y1,y2)
        J[2, 0] = self.c1 * y1s
        J[2, 1] = 0.0
        J[2, 2] = 0.0
        J[2, 3] = 0.0

        # df4/d(x1,x2,y1,y2)
        J[3, 0] = 0.0
        J[3, 1] = self.c2 * y2s
        J[3, 2] = 0.0
        J[3, 3] = 0.0

        return J

    def stability_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the Jacobian at the coexistence equilibrium.

        All real parts negative => locally stable coexistence.

        Returns:
            Complex eigenvalue array of length 4.
        """
        J = self.jacobian_at_equilibrium()
        return np.linalg.eigvals(J)

    def is_stable_coexistence(self) -> bool:
        """Check if the coexistence equilibrium is feasible and stable.

        Feasible: all components > 0.
        Stable: all eigenvalue real parts < 0.
        """
        eq = self.coexistence_equilibrium()
        if np.any(np.isnan(eq)) or np.any(eq <= 0):
            return False
        eigs = self.stability_eigenvalues()
        return bool(np.all(np.real(eigs) < 0))

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
        """Four-species Lotka-Volterra food web right-hand side."""
        x1, x2, y1, y2 = y

        dx1 = x1 * (self.r1 - self.a11 * x1 - self.a12 * x2 - self.b1 * y1)
        dx2 = x2 * (self.r2 - self.a21 * x1 - self.a22 * x2 - self.b2 * y2)
        dy1 = y1 * (-self.d1 + self.c1 * x1)
        dy2 = y2 * (-self.d2 + self.c2 * x2)

        return np.array([dx1, dx2, dy1, dy2])
