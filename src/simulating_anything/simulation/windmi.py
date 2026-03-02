"""WINDMI (solar wind-magnetosphere-ionosphere) model simulation.

The WINDMI system is a 3D ODE model for solar wind-magnetosphere coupling,
describing substorm dynamics in the magnetosphere. It is a jerk-type system
(like Genesio-Tesi) with exponential nonlinearity:

    dx/dt = y
    dy/dt = z
    dz/dt = -a*z - y + b - exp(x)

Parameters:
    a: damping coefficient controlling dissipation rate (classic: 0.7)
    b: solar wind driving input parameter (classic: 2.5 for chaos)

The exponential nonlinearity exp(x) models the nonlinear magnetospheric
current response. The system has a unique fixed point at
x* = ln(b), y* = 0, z* = 0.

Target rediscoveries:
- SINDy recovery of WINDMI ODEs
- Lyapunov exponent estimation for chaos detection
- b-parameter sweep (solar wind input) for substorm threshold
- Fixed point verification (x* = ln(b))
- Jerk form identification: x''' + a*x'' + x' = b - exp(x)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class WindmiSimulation(SimulationEnvironment):
    """WINDMI solar wind-magnetosphere-ionosphere coupling model.

    State vector: [x, y, z]

    ODEs:
        dx/dt = y
        dy/dt = z
        dz/dt = -a*z - y + b - exp(x)

    Equivalently, the third-order scalar jerk equation:
        x''' + a*x'' + x' = b - exp(x)

    The fixed point is at (ln(b), 0, 0) for b > 0.
    For the classic parameters a=0.7, b=2.5, the system exhibits
    chaotic substorm-like dynamics.

    Parameters:
        a: dissipation rate (classic: 0.7)
        b: solar wind driving input (classic: 2.5 for chaos)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.7)
        self.b = p.get("b", 2.5)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize WINDMI state."""
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
        """WINDMI equations: dx=y, dy=z, dz=-a*z - y + b - exp(x)."""
        x, y, z = state
        dx = y
        dy = z
        dz = -self.a * z - y + self.b - np.exp(x)
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed point of the WINDMI system.

        Setting derivatives to zero:
            y = 0
            z = 0
            -a*0 - 0 + b - exp(x) = 0  =>  exp(x) = b  =>  x = ln(b)

        The unique fixed point is (ln(b), 0, 0) for b > 0.
        For b <= 0, no real fixed point exists.
        """
        if self.b <= 0:
            return []
        x_eq = np.log(self.b)
        return [np.array([x_eq, 0.0, 0.0], dtype=np.float64)]

    @property
    def jacobian_at_fixed_point(self) -> np.ndarray:
        """Compute the Jacobian matrix at the fixed point (ln(b), 0, 0).

        The general Jacobian is:
            J = [[0,       1,  0],
                 [0,       0,  1],
                 [-exp(x), -1, -a]]

        At x = ln(b): exp(x) = b, so:
            J = [[0,  1,  0],
                 [0,  0,  1],
                 [-b, -1, -a]]

        The characteristic polynomial is:
            lambda^3 + a*lambda^2 + lambda + b = 0
        """
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-self.b, -1.0, -self.a],
        ], dtype=np.float64)

    def eigenvalues_at_fixed_point(self) -> np.ndarray:
        """Compute eigenvalues of the Jacobian at the fixed point.

        These are the roots of lambda^3 + a*lambda^2 + lambda + b = 0.
        """
        return np.linalg.eigvals(self.jacobian_at_fixed_point)

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
        state2 = state1 + np.array([eps, 0, 0])

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

            # Bail out if trajectory diverges
            if not (np.all(np.isfinite(state1)) and np.all(np.isfinite(state2))):
                break

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

    def measure_period(
        self, n_transient: int = 5000, n_measure: int = 20000
    ) -> float:
        """Measure the oscillation period by detecting zero crossings of x.

        Crossings are detected relative to the fixed point x* = ln(b),
        since the attractor oscillates around it.

        Returns the average period, or np.inf if no complete cycle is detected.
        """
        dt = self.config.dt
        x_center = np.log(self.b) if self.b > 0 else 0.0

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Detect positive-going crossings of x through x_center
        crossings: list[float] = []
        prev_x = self._state[0] - x_center
        for _ in range(n_measure):
            self.step()
            curr_x = self._state[0] - x_center
            if prev_x < 0 and curr_x >= 0:
                frac = (
                    -prev_x / (curr_x - prev_x)
                    if curr_x != prev_x
                    else 0.5
                )
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_x = curr_x

        if len(crossings) < 2:
            return np.inf

        periods = np.diff(crossings)
        return float(np.mean(periods))

    def compute_jerk(self, state: np.ndarray) -> float:
        """Compute the jerk (third derivative of x) at a given state.

        From the jerk form: x''' = -a*x'' - x' + b - exp(x)
        With x' = y, x'' = z:
            jerk = -a*z - y + b - exp(x)
        """
        x, y, z = state
        return -self.a * z - y + self.b - np.exp(x)

    def bifurcation_sweep(
        self,
        b_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep parameter b (solar wind input) and record attractor statistics.

        For each b value, computes the Lyapunov exponent after
        skipping transients.

        Args:
            b_values: Array of 'b' parameter values to sweep.
            n_transient: Steps to skip for transient.
            n_measure: Steps to use for Lyapunov estimation.

        Returns:
            Dict with b values, Lyapunov exponents, and attractor types.
        """
        lyapunov_exps = []
        attractor_types = []

        for b_val in b_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_transient + n_measure,
                parameters={
                    "a": self.a,
                    "b": b_val,
                    "x_0": self.x_0,
                    "y_0": self.y_0,
                    "z_0": self.z_0,
                },
            )
            sim = WindmiSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Estimate Lyapunov
            lam = sim.estimate_lyapunov(n_steps=n_measure, dt=self.config.dt)
            lyapunov_exps.append(lam)

            # Classify attractor
            if lam > 0.01:
                atype = "chaotic"
            elif lam < -0.01:
                atype = "stable"
            else:
                atype = "marginal"
            attractor_types.append(atype)

        return {
            "b": b_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "attractor_type": np.array(attractor_types),
        }
