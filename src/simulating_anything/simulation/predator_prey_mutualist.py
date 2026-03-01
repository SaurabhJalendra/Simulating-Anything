"""Predator-Prey-Mutualist 3-species ODE simulation.

A 3-species system where prey (x) and mutualist (z) benefit each other,
while predator (y) consumes prey via a Holling Type II functional response.

Equations:
    dx/dt = r*x*(1 - x/K) - a*x*y/(1 + b*x) + m*x*z/(1 + n*z)
    dy/dt = -d*y + e*a*x*y/(1 + b*x)
    dz/dt = s*z*(1 - z/C) + p*x*z/(1 + n*z)

Key physics:
    - Holling Type II functional response for predation: a*x/(1 + b*x)
    - Mutualistic interaction saturates: m*z/(1 + n*z) and p*x/(1 + n*z)
    - Logistic growth for prey (capacity K) and mutualist (capacity C)
    - Mutualism stabilizes predator-prey oscillations
    - System can exhibit multiple equilibria
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class PredatorPreyMutualistSimulation(SimulationEnvironment):
    """Predator-Prey-Mutualist 3-species model.

    State vector: [x, y, z] where x = prey, y = predator, z = mutualist.

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 10.0)
        a: predation rate (default 1.0)
        b: predator handling time (default 0.1)
        m: mutualism benefit to prey (default 0.5)
        n: mutualism saturation constant (default 0.2)
        d: predator death rate (default 0.4)
        e: predation conversion efficiency (default 0.6)
        s: mutualist intrinsic growth rate (default 0.8)
        C: mutualist carrying capacity (default 8.0)
        p: mutualism benefit to mutualist (default 0.3)
        x_0: initial prey population (default 5.0)
        y_0: initial predator population (default 2.0)
        z_0: initial mutualist population (default 3.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        par = config.parameters
        self.r = par.get("r", 1.0)
        self.K = par.get("K", 10.0)
        self.a = par.get("a", 1.0)
        self.b = par.get("b", 0.1)
        self.m = par.get("m", 0.5)
        self.n = par.get("n", 0.2)
        self.d = par.get("d", 0.4)
        self.e = par.get("e", 0.6)
        self.s = par.get("s", 0.8)
        self.C = par.get("C", 8.0)
        self.p = par.get("p", 0.3)
        self.x_0 = par.get("x_0", 5.0)
        self.y_0 = par.get("y_0", 2.0)
        self.z_0 = par.get("z_0", 3.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [x, y, z]."""
        self._state = np.array(
            [self.x_0, self.y_0, self.z_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [x, y, z]."""
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

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Predator-Prey-Mutualist right-hand side.

        dx/dt = r*x*(1 - x/K) - a*x*y/(1 + b*x) + m*x*z/(1 + n*z)
        dy/dt = -d*y + e*a*x*y/(1 + b*x)
        dz/dt = s*z*(1 - z/C) + p*x*z/(1 + n*z)
        """
        x, y, z = state

        # Holling Type II functional response
        functional_response = self.a * x / (1.0 + self.b * x)
        # Mutualism saturation
        mutualism_factor = 1.0 / (1.0 + self.n * z)

        dx = (
            self.r * x * (1.0 - x / self.K)
            - functional_response * y
            + self.m * x * z * mutualism_factor
        )
        dy = -self.d * y + self.e * functional_response * y
        dz = (
            self.s * z * (1.0 - z / self.C)
            + self.p * x * z * mutualism_factor
        )

        return np.array([dx, dy, dz])

    @property
    def total_population(self) -> float:
        """Sum of all three species populations."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    @property
    def is_coexisting(self) -> bool:
        """True if all three species are above the extinction threshold."""
        if self._state is None:
            return False
        threshold = 1e-6
        return bool(np.all(self._state > threshold))

    def find_equilibria(
        self,
        n_initial: int = 20,
        tol: float = 1e-10,
        max_iter: int = 200,
    ) -> list[np.ndarray]:
        """Find equilibria using Newton iteration from multiple initial guesses.

        Searches for fixed points where dx/dt = dy/dt = dz/dt = 0 in the
        positive octant (x >= 0, y >= 0, z >= 0).

        Args:
            n_initial: Number of random initial guesses.
            tol: Convergence tolerance on the residual norm.
            max_iter: Maximum Newton iterations per guess.

        Returns:
            List of unique equilibrium points found (as numpy arrays).
        """
        rng = np.random.default_rng(42)
        found: list[np.ndarray] = []

        # Always try the trivial equilibrium and boundary cases
        initial_guesses = [
            np.array([0.0, 0.0, 0.0]),
            np.array([self.K, 0.0, 0.0]),
            np.array([0.0, 0.0, self.C]),
            np.array([self.K, 0.0, self.C]),
        ]

        # Add random guesses in the positive octant
        for _ in range(n_initial):
            guess = rng.uniform(0.01, max(self.K, self.C), size=3)
            initial_guesses.append(guess)

        for guess in initial_guesses:
            eq = self._newton_solve(guess, tol, max_iter)
            if eq is not None:
                # Check if this is a new equilibrium (not already found)
                is_new = True
                for existing in found:
                    if np.allclose(eq, existing, atol=1e-6):
                        is_new = False
                        break
                if is_new:
                    found.append(eq)

        return found

    def _newton_solve(
        self,
        guess: np.ndarray,
        tol: float,
        max_iter: int,
    ) -> np.ndarray | None:
        """Solve for equilibrium using Newton-Raphson method.

        Returns the equilibrium point if converged, None otherwise.
        """
        y = guess.copy()

        for _ in range(max_iter):
            f = self._derivatives(y)
            if np.linalg.norm(f) < tol:
                # Converged; only accept if all components non-negative
                if np.all(y >= -1e-8):
                    return np.maximum(y, 0.0)
                return None

            J = self._jacobian(y)
            try:
                delta = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None

            y = y + delta

            # Bail out if values blow up
            if np.any(np.abs(y) > 1e6):
                return None

        return None

    def _jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix of the system at the given state.

        Returns 3x3 matrix of partial derivatives.
        """
        x, y, z = state
        r, K, a, b = self.r, self.K, self.a, self.b
        m, n, d_val, e_val = self.m, self.n, self.d, self.e
        s, C, p = self.s, self.C, self.p

        denom_x = (1.0 + b * x) ** 2
        denom_z = (1.0 + n * z)
        denom_z2 = denom_z ** 2

        # Partial derivatives of dx/dt
        dx_dx = r * (1.0 - 2.0 * x / K) - a * y / denom_x + m * z / denom_z
        dx_dy = -a * x / (1.0 + b * x)
        dx_dz = m * x / denom_z2

        # Partial derivatives of dy/dt
        dy_dx = e_val * a * y / denom_x
        dy_dy = -d_val + e_val * a * x / (1.0 + b * x)
        dy_dz = 0.0

        # Partial derivatives of dz/dt
        dz_dx = p * z / denom_z
        dz_dy = 0.0
        dz_dz = s * (1.0 - 2.0 * z / C) + p * x / denom_z2

        return np.array([
            [dx_dx, dx_dy, dx_dz],
            [dy_dx, dy_dy, dy_dz],
            [dz_dx, dz_dy, dz_dz],
        ])

    def stability_eigenvalues(self, equilibrium: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of the Jacobian at an equilibrium point.

        Args:
            equilibrium: A 3D state vector at equilibrium.

        Returns:
            Array of 3 eigenvalues (may be complex).
        """
        J = self._jacobian(equilibrium)
        return np.linalg.eigvals(J)
