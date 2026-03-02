"""Lorenz-Stenflo system simulation -- electromagnetic wave-plasma interaction.

4D extension of the Lorenz system incorporating electromagnetic rotation:
    dx/dt = sigma*(y - x) + s*w
    dy/dt = r*x - y - x*z
    dz/dt = x*y - b*z
    dw/dt = -x - sigma*w

When the Stenflo parameter s=0, the system reduces to the classic Lorenz attractor
(w decouples and decays exponentially). For s > 0, the fourth variable w introduces
electromagnetic coupling that can produce hyperchaos (two positive Lyapunov exponents).

Target rediscoveries:
- SINDy recovery of all 4 ODEs
- Reduction to Lorenz when s=0
- Lyapunov exponent estimation (positive for chaotic regime)
- s-parameter sweep showing transition from Lorenz-like chaos to hyperchaos
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LorenzStenfloSimulation(SimulationEnvironment):
    """Lorenz-Stenflo system: 4D chaotic attractor from plasma wave interactions.

    State vector: [x, y, z, w]

    ODEs:
        dx/dt = sigma*(y - x) + s*w
        dy/dt = r*x - y - x*z
        dz/dt = x*y - b*z
        dw/dt = -x - sigma*w

    Parameters:
        sigma: Prandtl-like parameter (default 10.0)
        r: Rayleigh-like parameter (default 28.0)
        b: geometry parameter (default 8/3)
        s: Stenflo rotation parameter (default 1.0); s=0 recovers Lorenz
        x_0, y_0, z_0, w_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.sigma = p.get("sigma", 10.0)
        self.r = p.get("r", 28.0)
        self.b = p.get("b", 8.0 / 3.0)
        self.s = p.get("s", 1.0)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)
        self.w_0 = p.get("w_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Lorenz-Stenflo state."""
        self._state = np.array(
            [self.x_0, self.y_0, self.z_0, self.w_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y, z, w]."""
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
        """Lorenz-Stenflo equations.

        dx/dt = sigma*(y - x) + s*w
        dy/dt = r*x - y - x*z
        dz/dt = x*y - b*z
        dw/dt = -x - sigma*w
        """
        x, y, z, w = state
        dx = self.sigma * (y - x) + self.s * w
        dy = self.r * x - y - x * z
        dz = x * y - self.b * z
        dw = -x - self.sigma * w
        return np.array([dx, dy, dz, dw])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Lorenz-Stenflo system.

        Setting derivatives to zero:
            sigma*(y - x) + s*w = 0        (1)
            r*x - y - x*z = 0              (2)
            x*y - b*z = 0                  (3)
            -x - sigma*w = 0               (4)

        From (4): w = -x/sigma
        Substituting into (1): sigma*(y-x) + s*(-x/sigma) = 0
            => sigma*y - sigma*x - s*x/sigma = 0
            => y = x + s*x/sigma^2 = x*(1 + s/sigma^2)

        Let alpha = 1 + s/sigma^2. Then y = alpha*x.
        From (3): z = x*y/b = alpha*x^2/b
        Substituting into (2): r*x - alpha*x - x*(alpha*x^2/b) = 0
            => x*(r - alpha - alpha*x^2/b) = 0

        So x=0 (origin) or x^2 = b*(r - alpha)/alpha.
        Non-origin fixed points exist when r > alpha = 1 + s/sigma^2.
        """
        points = [np.array([0.0, 0.0, 0.0, 0.0])]

        alpha = 1.0 + self.s / self.sigma**2
        if self.r > alpha:
            x_sq = self.b * (self.r - alpha) / alpha
            x_val = np.sqrt(x_sq)
            y_val = alpha * x_val
            z_val = alpha * x_val**2 / self.b
            w_val = -x_val / self.sigma

            points.append(np.array([x_val, y_val, z_val, w_val]))
            points.append(np.array([-x_val, -y_val, z_val, -w_val]))
        return points

    def estimate_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the method of Wolf et al. (1985): track two nearby trajectories,
        renormalize when they diverge too far.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0, 0, 0])

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance both states with RK4
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Compute distance
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                # Renormalize
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def reduces_to_lorenz(self, n_steps: int = 5000) -> dict[str, float]:
        """Verify that with s=0, the xyz dynamics match pure Lorenz.

        Runs the Lorenz-Stenflo system with s=0 and w_0=0, then checks that
        the w component stays near zero and xyz matches Lorenz derivatives.

        Returns:
            Dict with max |w| and max xyz-derivative deviation from Lorenz.
        """
        config = SimulationConfig(
            domain=self.config.domain,
            dt=self.config.dt,
            n_steps=n_steps,
            parameters={
                "sigma": self.sigma,
                "r": self.r,
                "b": self.b,
                "s": 0.0,
                "x_0": self.x_0,
                "y_0": self.y_0,
                "z_0": self.z_0,
                "w_0": 0.0,
            },
        )
        sim_s0 = LorenzStenfloSimulation(config)
        sim_s0.reset()

        max_w = 0.0
        max_lorenz_dev = 0.0

        for _ in range(n_steps):
            state = sim_s0.step()
            x, y, z, w = state
            max_w = max(max_w, abs(w))

            # Compare xyz derivatives with pure Lorenz
            ls_derivs = sim_s0._derivatives(state)
            lorenz_dx = self.sigma * (y - x)
            lorenz_dy = self.r * x - y - x * z
            lorenz_dz = x * y - self.b * z

            dev = np.sqrt(
                (ls_derivs[0] - lorenz_dx) ** 2
                + (ls_derivs[1] - lorenz_dy) ** 2
                + (ls_derivs[2] - lorenz_dz) ** 2
            )
            max_lorenz_dev = max(max_lorenz_dev, dev)

        return {
            "max_w": max_w,
            "max_lorenz_deviation": max_lorenz_dev,
        }

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

        # Collect data
        xs, ys, zs, ws = [], [], [], []
        for _ in range(n_steps):
            state = self.step()
            xs.append(state[0])
            ys.append(state[1])
            zs.append(state[2])
            ws.append(state[3])

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        ws = np.array(ws)

        return {
            "x_mean": float(np.mean(xs)),
            "y_mean": float(np.mean(ys)),
            "z_mean": float(np.mean(zs)),
            "w_mean": float(np.mean(ws)),
            "x_std": float(np.std(xs)),
            "y_std": float(np.std(ys)),
            "z_std": float(np.std(zs)),
            "w_std": float(np.std(ws)),
            "x_range": float(np.max(xs) - np.min(xs)),
            "y_range": float(np.max(ys) - np.min(ys)),
            "z_range": float(np.max(zs) - np.min(zs)),
            "w_range": float(np.max(ws) - np.min(ws)),
        }
