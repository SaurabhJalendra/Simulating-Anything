"""Rossler hyperchaotic system simulation -- 4D extension with two positive Lyapunov exponents.

Target rediscoveries:
- Two positive Lyapunov exponents (hyperchaos vs ordinary chaos)
- SINDy recovery of 4D ODEs
- Attractor dimension estimation (Kaplan-Yorke)
- Lyapunov spectrum computation via QR decomposition

Equations:
    dx/dt = -(y + z)
    dy/dt = x + a*y + w
    dz/dt = b + x*z
    dw/dt = -c*z + d*w

Classic hyperchaotic parameters: a=0.25, b=3.0, c=0.5, d=0.05
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RosslerHyperchaosSimulation(SimulationEnvironment):
    """Rossler hyperchaotic system: a 4D extension with two positive Lyapunov exponents.

    State vector: [x, y, z, w]

    ODEs:
        dx/dt = -(y + z)
        dy/dt = x + a*y + w
        dz/dt = b + x*z
        dw/dt = -c*z + d*w

    Parameters:
        a: coupling/feedback strength (classic: 0.25)
        b: z-dynamics offset (classic: 3.0)
        c: w-to-z coupling (classic: 0.5)
        d: w self-feedback (classic: 0.05)
        x_0, y_0, z_0, w_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.25)
        self.b = p.get("b", 3.0)
        self.c = p.get("c", 0.5)
        self.d = p.get("d", 0.05)
        self.x_0 = p.get("x_0", -10.0)
        self.y_0 = p.get("y_0", -6.0)
        self.z_0 = p.get("z_0", 0.0)
        self.w_0 = p.get("w_0", 10.0)

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
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Rossler hyperchaos ODEs."""
        x, y, z, w = state
        dx = -(y + z)
        dy = x + self.a * y + w
        dz = self.b + x * z
        dw = -self.c * z + self.d * w
        return np.array([dx, dy, dz, dw])

    def _jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian matrix of the Rossler hyperchaotic system.

        J = [[ 0,  -1, -1,  0],
             [ 1,   a,  0,  1],
             [ z,   0,  x,  0],
             [ 0,   0, -c,  d]]
        """
        x, _y, z, _w = state
        return np.array([
            [0.0, -1.0, -1.0, 0.0],
            [1.0, self.a, 0.0, 1.0],
            [z, 0.0, x, 0.0],
            [0.0, 0.0, -self.c, self.d],
        ])

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
        renorm_interval = 10  # QR every 10 steps for stability

        for step_i in range(n_steps):
            # Advance state
            k1 = self._derivatives(state)
            k2 = self._derivatives(state + 0.5 * dt * k1)
            k3 = self._derivatives(state + 0.5 * dt * k2)
            k4 = self._derivatives(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance tangent vectors using the Jacobian (linearized flow)
            J = self._jacobian(state)
            # First-order approximation: Q_new = (I + dt * J) @ Q
            Q = Q + dt * (J @ Q)

            # Periodically reorthonormalize
            if (step_i + 1) % renorm_interval == 0:
                Q, R = np.linalg.qr(Q)
                # Accumulate log of diagonal (stretching rates)
                diag = np.abs(np.diag(R))
                diag = np.maximum(diag, 1e-300)  # avoid log(0)
                lyap_sum += np.log(diag)
                n_renorm += 1

        if n_renorm == 0:
            return np.zeros(dim)

        total_time = n_renorm * renorm_interval * dt
        exponents = lyap_sum / total_time
        return np.sort(exponents)[::-1]  # largest first

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

    def is_hyperchaotic(
        self, n_steps: int = 60000, dt: float | None = None
    ) -> bool:
        """Check whether the system has two positive Lyapunov exponents.

        A system is hyperchaotic if at least two Lyapunov exponents
        are positive, indicating expansion in two independent directions.
        """
        spectrum = self.estimate_lyapunov_spectrum(n_steps=n_steps, dt=dt)
        n_positive = int(np.sum(spectrum > 0.001))
        return n_positive >= 2

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

        # Find j: largest index where cumulative sum >= 0
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

        # D_KY = j + cumsum[j-1] / |lambda_{j+1}|
        d_ky = j + cumsum[j - 1] / abs(sorted_exp[j])
        return float(d_ky)
