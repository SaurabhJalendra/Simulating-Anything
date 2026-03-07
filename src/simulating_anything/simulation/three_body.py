"""Restricted three-body gravitational simulation.

The general three-body problem has no closed-form solution (Poincare, 1890).
This simulation explores the restricted problem where a test mass moves in
the gravitational field of two heavy bodies (e.g., Sun-Jupiter-asteroid).

Discovery targets:
- Lagrange point locations (known analytically)
- Stability boundaries around L4/L5 (partially known)
- Periodic orbit families (Halo orbits, figure-8s)
- Lyapunov exponents and chaotic transition boundaries
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ThreeBodySimulation(SimulationEnvironment):
    """Circular restricted three-body problem (CR3BP).

    Two primary masses (m1, m2) orbit their center of mass in circles.
    A test mass (massless) moves in their gravitational field.
    Coordinates in the rotating frame (co-rotating with the primaries).

    State: [x, y, vx, vy] in rotating frame.
    Parameters:
    - mu: mass ratio m2/(m1+m2), e.g. 0.01 for Sun-Jupiter
    - x0, y0, vx0, vy0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.mu = config.parameters.get("mu", 0.01)
        self.x0 = config.parameters.get("x0", 0.5)
        self.y0 = config.parameters.get("y0", 0.5)
        self.vx0 = config.parameters.get("vx0", 0.0)
        self.vy0 = config.parameters.get("vy0", 0.0)
        self.dt = config.dt

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._step_count = 0
        self._state = np.array([self.x0, self.y0, self.vx0, self.vy0], dtype=np.float64)
        return self._state.copy()

    def step(self) -> np.ndarray:
        self._state = self._rk4_step(self._state, self.dt)
        self._step_count += 1
        return self._state.copy()

    def observe(self) -> np.ndarray:
        return self._state.copy()

    def _rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._derivatives(state)
        k2 = self._derivatives(state + 0.5 * dt * k1)
        k3 = self._derivatives(state + 0.5 * dt * k2)
        k4 = self._derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """CR3BP equations of motion in the rotating frame."""
        x, y, vx, vy = state
        mu = self.mu

        # Positions of primaries in rotating frame
        # m1 at (-mu, 0), m2 at (1-mu, 0)
        r1 = np.sqrt((x + mu) ** 2 + y ** 2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)

        # Avoid singularities
        r1 = max(r1, 1e-10)
        r2 = max(r2, 1e-10)

        # Accelerations in rotating frame (includes Coriolis and centrifugal)
        ax = (
            2 * vy
            + x
            - (1 - mu) * (x + mu) / r1 ** 3
            - mu * (x - 1 + mu) / r2 ** 3
        )
        ay = (
            -2 * vx
            + y
            - (1 - mu) * y / r1 ** 3
            - mu * y / r2 ** 3
        )

        return np.array([vx, vy, ax, ay])

    def jacobi_constant(self, state: np.ndarray | None = None) -> float:
        """Compute the Jacobi constant (conserved quantity in CR3BP).

        C_J = -(vx^2 + vy^2) + 2*Omega
        where Omega is the pseudo-potential.
        """
        if state is None:
            state = self._state
        x, y, vx, vy = state
        mu = self.mu

        r1 = np.sqrt((x + mu) ** 2 + y ** 2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
        r1 = max(r1, 1e-10)
        r2 = max(r2, 1e-10)

        # Pseudo-potential
        omega = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2

        return -(vx ** 2 + vy ** 2) + 2 * omega

    def lagrange_points(self) -> dict[str, np.ndarray]:
        """Compute the 5 Lagrange points for current mass ratio.

        L1, L2, L3: collinear (on x-axis), found by root-finding.
        L4, L5: equilateral triangular points (exact).
        """
        mu = self.mu

        # L4 and L5 are exact equilateral triangle points
        l4 = np.array([0.5 - mu, np.sqrt(3) / 2])
        l5 = np.array([0.5 - mu, -np.sqrt(3) / 2])

        # L1, L2, L3: solve quintic on x-axis (y=0)
        # Use Newton's method for each
        l1 = self._find_collinear_lp(1 - mu - 0.1)  # Between m1 and m2
        l2 = self._find_collinear_lp(1 - mu + 0.1)  # Beyond m2
        l3 = self._find_collinear_lp(-1.0 - mu)  # Beyond m1

        return {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5}

    def _find_collinear_lp(self, x0: float) -> np.ndarray:
        """Find collinear Lagrange point by Newton's method."""
        mu = self.mu
        x = x0
        for _ in range(100):
            r1 = abs(x + mu)
            r2 = abs(x - 1 + mu)
            r1 = max(r1, 1e-15)
            r2 = max(r2, 1e-15)
            s1 = np.sign(x + mu) if r1 > 1e-15 else 1.0
            s2 = np.sign(x - 1 + mu) if r2 > 1e-15 else 1.0

            # f(x) = x - (1-mu)*s1/r1^2 - mu*s2/r2^2
            f = x - (1 - mu) * s1 / r1 ** 2 - mu * s2 / r2 ** 2
            # f'(x) = 1 + 2*(1-mu)/r1^3 + 2*mu/r2^3
            fp = 1 + 2 * (1 - mu) / r1 ** 3 + 2 * mu / r2 ** 3

            if abs(fp) < 1e-15:
                break
            dx = f / fp
            x -= dx
            if abs(dx) < 1e-12:
                break
        return np.array([x, 0.0])

    def compute_lyapunov(self, n_steps: int = 10000, dt: float | None = None) -> float:
        """Estimate maximum Lyapunov exponent via variational method."""
        if dt is None:
            dt = self.dt
        state = np.array([self.x0, self.y0, self.vx0, self.vy0])

        # Perturbation vector
        delta = np.array([1e-8, 0, 0, 0])
        delta = delta / np.linalg.norm(delta)

        lyap_sum = 0.0
        for i in range(n_steps):
            state_pert = state + delta * 1e-8
            state = self._rk4_step(state, dt)
            state_pert = self._rk4_step(state_pert, dt)

            diff = state_pert - state
            d = np.linalg.norm(diff)
            if d > 0:
                lyap_sum += np.log(d / 1e-8)
                delta = diff / d
            else:
                delta = np.array([1e-8, 0, 0, 0])
                delta = delta / np.linalg.norm(delta)

        return lyap_sum / (n_steps * dt)
