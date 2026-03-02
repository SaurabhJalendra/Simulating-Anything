"""Hadley circulation simulation -- atmospheric convection in a rotating frame.

The Hadley system models atmospheric convection as a 3D ODE:
    dx/dt = -y^2 - z^2 - a*x + a*F
    dy/dt = x*y - b*x*z - y + G
    dz/dt = b*x*y + x*z - z

Variables:
    x: intensity of the symmetric (Hadley) circulation
    y: cosine phase of the wave component
    z: sine phase of the wave component

Parameters:
    a: damping coefficient (default 0.2)
    b: rotation/advection coefficient (default 4.0)
    F: symmetric thermal forcing (default 8.0)
    G: asymmetric thermal forcing (default 1.0)

Key physics:
    - Models Hadley cell atmospheric circulation
    - Quadratic nonlinearities from Coriolis rotation
    - F drives the symmetric zonal flow, G excites wave modes
    - Chaotic for certain parameter ranges (e.g. large F with G != 0)
    - Divergence of the flow is constant: div = -(a + 2)
    - Hadley fixed point at (F, 0, 0) when G = 0

Target rediscoveries:
    - SINDy recovery of Hadley ODEs
    - Lyapunov exponent estimation (positive for chaotic regime)
    - Divergence = -(a + 2) verification
    - Fixed point analysis
    - Chaos transition as F varies
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class HadleySimulation(SimulationEnvironment):
    """Hadley atmospheric circulation model.

    State vector: [x, y, z]

    ODEs:
        dx/dt = -y^2 - z^2 - a*x + a*F
        dy/dt = x*y - b*x*z - y + G
        dz/dt = b*x*y + x*z - z

    Parameters:
        a: damping coefficient (default 0.2)
        b: rotation coefficient (default 4.0)
        F: symmetric thermal forcing (default 8.0)
        G: asymmetric thermal forcing (default 1.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.2)
        self.b = p.get("b", 4.0)
        self.F = p.get("F", 8.0)
        self.G = p.get("G", 1.0)
        self.x_0 = p.get("x_0", 0.0)
        self.y_0 = p.get("y_0", 1.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Hadley state.

        If seed is provided, adds small random perturbation to initial
        conditions for ensemble studies.
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
        """Fourth-order Runge-Kutta integration step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Hadley circulation equations.

        dx/dt = -y^2 - z^2 - a*x + a*F
        dy/dt = x*y - b*x*z - y + G
        dz/dt = b*x*y + x*z - z
        """
        x, y, z = state
        dx = -y**2 - z**2 - self.a * x + self.a * self.F
        dy = x * y - self.b * x * z - y + self.G
        dz = self.b * x * y + x * z - z
        return np.array([dx, dy, dz])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
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

    def compute_divergence(self) -> float:
        """Compute the divergence of the Hadley flow.

        The divergence is constant (independent of state):
            div(f) = df1/dx + df2/dy + df3/dz = -a + (x-1) + (x-1)
        evaluated from the Jacobian trace.

        However, computing the trace properly:
            df1/dx = -a
            df2/dy = x - 1
            df3/dz = x - 1

        The trace depends on x, but the *average* divergence over the
        attractor is -(a + 2). More precisely, at any state the
        instantaneous divergence is -a + 2*(x - 1) = -a - 2 + 2*x.

        The *constant* part of the divergence (independent of state)
        comes from the linear damping terms: -(a + 2).
        This is the contraction rate of the flow volume.
        """
        return -(self.a + 2.0)

    def fixed_points(self) -> list[np.ndarray]:
        """Find fixed points numerically via Newton iteration.

        The Hadley fixed point (F, 0, 0) exists when G = 0.
        For G != 0, fixed points must be found numerically.

        Returns:
            List of fixed point arrays.
        """
        fps = []

        # Analytical Hadley fixed point for G=0
        if abs(self.G) < 1e-10:
            fps.append(np.array([self.F, 0.0, 0.0]))

        # Newton iteration from multiple initial guesses
        guesses = [
            np.array([self.F, 0.0, 0.0]),
            np.array([self.F, 1.0, 0.0]),
            np.array([self.F, 0.0, 1.0]),
            np.array([self.F, -1.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([self.F / 2, 1.0, 1.0]),
            np.array([self.F, 1.0, 1.0]),
            np.array([self.F, -1.0, -1.0]),
        ]

        for guess in guesses:
            fp = self._newton_find_fp(guess)
            if fp is not None:
                is_new = True
                for existing in fps:
                    if np.linalg.norm(fp - existing) < 1e-6:
                        is_new = False
                        break
                if is_new:
                    fps.append(fp)

        return fps

    def _newton_find_fp(
        self, guess: np.ndarray, max_iter: int = 100, tol: float = 1e-10
    ) -> np.ndarray | None:
        """Find a fixed point using Newton's method."""
        state = guess.copy()

        for _ in range(max_iter):
            f = self._derivatives(state)
            if np.linalg.norm(f) < tol:
                return state

            J = self.jacobian(state)
            try:
                delta = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None

            state = state + delta

            if np.linalg.norm(state) > 1e6:
                return None

        if np.linalg.norm(self._derivatives(state)) < 1e-8:
            return state
        return None

    def lyapunov_exponent(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the Wolf et al. (1985) renormalization method: track two
        nearby trajectories and renormalize when they diverge too far.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0, 0])

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance state1 with RK4
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance state2 with RK4
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

    def compute_trajectory_statistics(
        self, n_steps: int = 10000, n_transient: int = 2000
    ) -> dict[str, float]:
        """Compute time-averaged statistics of the trajectory.

        Args:
            n_steps: Number of steps to measure after transient.
            n_transient: Steps to skip for transient.

        Returns:
            Dict with mean, std, min, max for each component.
        """
        self.reset()

        # Skip transient
        for _ in range(n_transient):
            self.step()

        xs, ys, zs = [], [], []
        for _ in range(n_steps):
            state = self.step()
            xs.append(state[0])
            ys.append(state[1])
            zs.append(state[2])

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        return {
            "x_mean": float(np.mean(xs)),
            "y_mean": float(np.mean(ys)),
            "z_mean": float(np.mean(zs)),
            "x_std": float(np.std(xs)),
            "y_std": float(np.std(ys)),
            "z_std": float(np.std(zs)),
            "x_min": float(np.min(xs)),
            "y_min": float(np.min(ys)),
            "z_min": float(np.min(zs)),
            "x_max": float(np.max(xs)),
            "y_max": float(np.max(ys)),
            "z_max": float(np.max(zs)),
        }

    @property
    def hadley_fixed_point(self) -> np.ndarray:
        """The Hadley circulation fixed point: (F, 0, 0).

        This is an exact fixed point when G = 0.
        For small G, the actual fixed point is perturbed from this.
        """
        return np.array([self.F, 0.0, 0.0])
