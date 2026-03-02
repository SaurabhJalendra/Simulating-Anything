"""Qi chaotic system simulation -- 4D ODE with rich hyperchaotic dynamics.

The Qi system is a 4-dimensional autonomous chaotic system that extends
Lorenz-type dynamics with a fourth variable w coupled through cross-product
nonlinearities:
    dx/dt = a*(y - x) + y*z
    dy/dt = c*x - y - x*z
    dz/dt = x*y - b*z
    dw/dt = -d*w + x*z

Classic parameters: a=10.0, b=8/3, c=28.0, d=1.0

The system exhibits chaotic attractors with sensitivity to initial conditions
and parameter values. The divergence of the flow is -(a + 1 + b + d), which
is negative for the classic parameters, ensuring dissipation and bounded
attractors.

Target rediscoveries:
- SINDy recovery of 4D ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- d-parameter sweep mapping chaos transitions
- Fixed point analysis (origin + symmetric pair)
- Attractor boundedness and statistics
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class QiSimulation(SimulationEnvironment):
    """Qi 4D chaotic system: a Lorenz-type extension with fourth variable.

    State vector: [x, y, z, w]

    ODEs:
        dx/dt = a*(y - x) + y*z
        dy/dt = c*x - y - x*z
        dz/dt = x*y - b*z
        dw/dt = -d*w + x*z

    Parameters:
        a: Prandtl-like diffusion parameter (default 10.0)
        b: geometry/damping parameter (default 8/3)
        c: Rayleigh-like driving parameter (default 28.0)
        d: w-variable damping rate (default 1.0)
        x_0, y_0, z_0, w_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 10.0)
        self.b = p.get("b", 8.0 / 3.0)
        self.c = p.get("c", 28.0)
        self.d = p.get("d", 1.0)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)
        self.w_0 = p.get("w_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state to [x_0, y_0, z_0, w_0]."""
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
        """Qi system ODEs.

        dx/dt = a*(y - x) + y*z
        dy/dt = c*x - y - x*z
        dz/dt = x*y - b*z
        dw/dt = -d*w + x*z
        """
        x, y, z, w = state
        dx = self.a * (y - x) + y * z
        dy = self.c * x - y - x * z
        dz = x * y - self.b * z
        dw = -self.d * w + x * z
        return np.array([dx, dy, dz, dw])

    def _jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian matrix of the Qi system.

        J = [[-a,     a+z,   y,   0 ],
             [c-z,   -1,    -x,   0 ],
             [y,      x,    -b,   0 ],
             [z,      0,     x,  -d ]]
        """
        x, y, z, _w = state
        return np.array([
            [-self.a, self.a + z, y, 0.0],
            [self.c - z, -1.0, -x, 0.0],
            [y, x, -self.b, 0.0],
            [z, 0.0, x, -self.d],
        ])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Qi system.

        Setting derivatives to zero:
            a*(y - x) + y*z = 0           (1)
            c*x - y - x*z = 0             (2)
            x*y - b*z = 0                 (3)
            -d*w + x*z = 0                (4)

        From (3): z = x*y / b
        From (4): w = x*z / d = x^2*y / (b*d)

        Substituting z into (2):
            c*x - y - x*(x*y/b) = 0
            => c*x - y*(1 + x^2/b) = 0
            => y = c*x*b / (b + x^2)

        Substituting y and z into (1):
            a*(y - x) + y*(x*y/b) = 0
            => a*(y - x) + x*y^2/b = 0

        x=0 gives origin. For non-zero x, solve numerically.
        """
        points = [np.array([0.0, 0.0, 0.0, 0.0])]

        # For the standard parameters, try to find symmetric non-trivial fixed points
        # Using the relations derived above, solve for x numerically
        try:
            from scipy.optimize import fsolve

            def _fp_equations(vars: np.ndarray) -> np.ndarray:
                x, y, z, w = vars
                return np.array([
                    self.a * (y - x) + y * z,
                    self.c * x - y - x * z,
                    x * y - self.b * z,
                    -self.d * w + x * z,
                ])

            # Try multiple initial guesses to find non-trivial fixed points
            found = []
            guesses = [
                [5.0, 5.0, 20.0, 50.0],
                [-5.0, -5.0, 20.0, 50.0],
                [10.0, 10.0, 25.0, 100.0],
                [-10.0, -10.0, 25.0, 100.0],
            ]
            for guess in guesses:
                sol = fsolve(_fp_equations, guess, full_output=True)
                root, info, ier, _msg = sol
                if ier == 1 and np.linalg.norm(info["fvec"]) < 1e-10:
                    # Check not a duplicate
                    is_dup = False
                    for existing in points + found:
                        if np.linalg.norm(root - existing) < 1e-6:
                            is_dup = True
                            break
                    if not is_dup:
                        found.append(np.array(root))
            points.extend(found)
        except ImportError:
            # Fall back to analytical solution for simple cases
            pass

        return points

    def estimate_lyapunov_spectrum(
        self, n_steps: int = 50000, dt: float | None = None, n_transient: int = 5000
    ) -> np.ndarray:
        """Estimate the full Lyapunov spectrum using QR decomposition.

        Uses the standard algorithm of Benettin et al. (1980): evolve
        the state and a set of tangent vectors, periodically reorthonormalize
        via QR decomposition, accumulate the log of the diagonal elements.

        Returns:
            Array of 4 Lyapunov exponents, sorted largest to smallest.
        """
        if dt is None:
            dt = self.config.dt

        dim = 4
        state = self._state.copy()

        # Skip transient
        for _ in range(n_transient):
            k1 = self._derivatives(state)
            k2 = self._derivatives(state + 0.5 * dt * k1)
            k3 = self._derivatives(state + 0.5 * dt * k2)
            k4 = self._derivatives(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Initialize orthonormal tangent vectors
        Q = np.eye(dim)
        lyap_sum = np.zeros(dim)
        n_renorm = 0
        renorm_interval = 10

        for step_i in range(n_steps):
            # Advance state with RK4
            k1 = self._derivatives(state)
            k2 = self._derivatives(state + 0.5 * dt * k1)
            k3 = self._derivatives(state + 0.5 * dt * k2)
            k4 = self._derivatives(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance tangent vectors using the Jacobian
            J = self._jacobian(state)
            Q = Q + dt * (J @ Q)

            # Periodically reorthonormalize
            if (step_i + 1) % renorm_interval == 0:
                Q, R = np.linalg.qr(Q)
                diag = np.abs(np.diag(R))
                diag = np.maximum(diag, 1e-300)
                lyap_sum += np.log(diag)
                n_renorm += 1

        if n_renorm == 0:
            return np.zeros(dim)

        total_time = n_renorm * renorm_interval * dt
        exponents = lyap_sum / total_time
        return np.sort(exponents)[::-1]

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
        state2 = state1 + np.array([eps, 0.0, 0.0, 0.0])

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

            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def kaplan_yorke_dimension(
        self, spectrum: np.ndarray | None = None, n_steps: int = 60000
    ) -> float:
        """Compute the Kaplan-Yorke dimension from the Lyapunov spectrum.

        D_KY = j + (sum of first j exponents) / |lambda_{j+1}|

        where j is the largest integer such that the sum of the first j
        exponents is non-negative.
        """
        if spectrum is None:
            spectrum = self.estimate_lyapunov_spectrum(n_steps=n_steps)

        sorted_exp = np.sort(spectrum)[::-1]
        cumsum = np.cumsum(sorted_exp)

        j = 0
        for i in range(len(sorted_exp)):
            if cumsum[i] >= 0:
                j = i + 1
            else:
                break

        if j == 0:
            return 0.0
        if j >= len(sorted_exp):
            return float(len(sorted_exp))

        d_ky = j + cumsum[j - 1] / abs(sorted_exp[j])
        return float(d_ky)

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
