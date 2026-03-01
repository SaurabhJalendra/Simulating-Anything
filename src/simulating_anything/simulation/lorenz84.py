"""Lorenz-84 atmospheric circulation model.

The Lorenz-84 model (Lorenz, 1984) is a simplified atmospheric circulation model:
    dx/dt = -y^2 - z^2 - a*x + a*F
    dy/dt = x*y - b*x*z - y + G
    dz/dt = b*x*y + x*z - z

Variables:
    x: strength of the westerly wind current
    y: cosine phase of large-scale eddies
    z: sine phase of large-scale eddies

Parameters:
    a: damping (typically 0.25)
    b: rotation/advection (typically 4.0)
    F: symmetric thermal forcing (Hadley circulation driver)
    G: asymmetric thermal forcing (wave excitation)

Key physics:
    - Hadley circulation fixed point at (x=F, y=0, z=0) for G=0
    - Quasi-periodic and chaotic routes to chaos as F increases
    - Coexisting attractors for certain parameter combinations
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class Lorenz84Simulation(SimulationEnvironment):
    """Lorenz-84 atmospheric circulation model.

    State vector: [x, y, z]

    Parameters:
        a: damping coefficient (default 0.25)
        b: rotation coefficient (default 4.0)
        F: symmetric thermal forcing (default 8.0)
        G: asymmetric thermal forcing (default 1.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.25)
        self.b = p.get("b", 4.0)
        self.F = p.get("F", 8.0)
        self.G = p.get("G", 1.0)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Lorenz-84 state.

        If seed is provided, adds small random perturbation to initial conditions.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
            perturbation = rng.normal(0, 0.01, size=3)
            self._state = np.array(
                [self.x_0, self.y_0, self.z_0], dtype=np.float64
            ) + perturbation
        else:
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
        """Return current state [x, y, z]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Lorenz-84 equations.

        dx/dt = -y^2 - z^2 - a*x + a*F
        dy/dt = x*y - b*x*z - y + G
        dz/dt = b*x*y + x*z - z
        """
        x, y, z = state
        dx = -y**2 - z**2 - self.a * x + self.a * self.F
        dy = x * y - self.b * x * z - y + self.G
        dz = self.b * x * y + x * z - z
        return np.array([dx, dy, dz])

    def find_fixed_points(self) -> list[np.ndarray]:
        """Find fixed points numerically via Newton iteration.

        The Hadley fixed point (x=F, y=0, z=0) is always a fixed point when G=0.
        For G != 0, fixed points are found numerically.

        Returns:
            List of fixed point arrays.
        """
        fixed_points = []

        # Analytical Hadley fixed point for G=0
        if abs(self.G) < 1e-10:
            fixed_points.append(np.array([self.F, 0.0, 0.0]))

        # Newton iteration from multiple initial guesses
        guesses = [
            np.array([self.F, 0.0, 0.0]),
            np.array([self.F, 1.0, 0.0]),
            np.array([self.F, 0.0, 1.0]),
            np.array([self.F, -1.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([self.F / 2, 1.0, 1.0]),
        ]

        for guess in guesses:
            fp = self._newton_find_fp(guess)
            if fp is not None:
                # Check if this is a new fixed point
                is_new = True
                for existing in fixed_points:
                    if np.linalg.norm(fp - existing) < 1e-6:
                        is_new = False
                        break
                if is_new:
                    fixed_points.append(fp)

        return fixed_points

    def _newton_find_fp(
        self, guess: np.ndarray, max_iter: int = 100, tol: float = 1e-10
    ) -> np.ndarray | None:
        """Find a fixed point using Newton's method."""
        state = guess.copy()

        for _ in range(max_iter):
            f = self._derivatives(state)
            if np.linalg.norm(f) < tol:
                return state

            # Jacobian
            J = self._jacobian(state)
            try:
                delta = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None

            state = state + delta

            # Divergence check
            if np.linalg.norm(state) > 1e6:
                return None

        # Check convergence
        if np.linalg.norm(self._derivatives(state)) < 1e-8:
            return state
        return None

    def _jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[df1/dx, df1/dy, df1/dz],
             [df2/dx, df2/dy, df2/dz],
             [df3/dx, df3/dy, df3/dz]]
        """
        x, y, z = state
        return np.array([
            [-self.a, -2 * y, -2 * z],
            [y - self.b * z, x - 1, -self.b * x],
            [self.b * y + z, self.b * x, x - 1],
        ])

    def compute_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the Wolf et al. (1985) renormalization method.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0, 0])

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance state1
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance state2
            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Compute distance and renormalize
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    @property
    def hadley_fixed_point(self) -> np.ndarray:
        """The Hadley circulation fixed point: (F, 0, 0).

        This is an exact fixed point when G=0.
        For small G, the actual fixed point is perturbed from this.
        """
        return np.array([self.F, 0.0, 0.0])
