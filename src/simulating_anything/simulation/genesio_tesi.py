"""Genesio-Tesi chaotic system simulation.

The Genesio-Tesi system is a 3D chaotic ODE derived from a third-order
polynomial jerk equation:
    dx/dt = y
    dy/dt = z
    dz/dt = -c*x - b*y - a*z + x^2

Parameters:
    a: linear damping of the z equation (classic: 0.44)
    b: coupling coefficient for y (classic: 1.1)
    c: coupling coefficient for x (classic: 1.0)

The system has a unique equilibrium at the origin and exhibits chaos
through a Hopf bifurcation mechanism. It is one of the simplest
polynomial systems displaying chaotic behavior.

Target rediscoveries:
- SINDy recovery of Genesio-Tesi ODEs
- Lyapunov exponent estimation for chaos detection
- Hopf bifurcation sweep (vary a)
- Fixed point verification (origin)
- Jerk form identification: x''' + a*x'' + b*x' + c*x = x^2
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GenesioTesiSimulation(SimulationEnvironment):
    """Genesio-Tesi system: a simple 3D polynomial chaotic ODE.

    State vector: [x, y, z]

    ODEs:
        dx/dt = y
        dy/dt = z
        dz/dt = -c*x - b*y - a*z + x^2

    Equivalently, the third-order scalar jerk equation:
        x''' + a*x'' + b*x' + c*x = x^2

    The origin (0, 0, 0) is always an equilibrium. A second fixed point
    exists at (c, 0, 0) when c > 0. For the classic parameters
    a=0.44, b=1.1, c=1.0 the system exhibits chaos.

    Parameters:
        a: damping in z-equation (classic: 0.44)
        b: coupling for y (classic: 1.1)
        c: coupling for x (classic: 1.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.44)
        self.b = p.get("b", 1.1)
        self.c = p.get("c", 1.0)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.1)
        self.z_0 = p.get("z_0", 0.1)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Genesio-Tesi state."""
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
        """Genesio-Tesi equations: dx=y, dy=z, dz=-c*x-b*y-a*z+x^2."""
        x, y, z = state
        dx = y
        dy = z
        dz = -self.c * x - self.b * y - self.a * z + x**2
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Genesio-Tesi system.

        Setting derivatives to zero:
            y = 0
            z = 0
            -c*x - b*0 - a*0 + x^2 = 0  =>  x*(-c + x) = 0

        Two fixed points:
            FP1: (0, 0, 0)  -- origin, always exists
            FP2: (c, 0, 0)  -- exists when c != 0
        """
        points = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        if abs(self.c) > 1e-15:
            points.append(np.array([self.c, 0.0, 0.0], dtype=np.float64))
        return points

    @property
    def jacobian_at_origin(self) -> np.ndarray:
        """Compute the Jacobian matrix at the origin.

        J = [[0, 1, 0],
             [0, 0, 1],
             [-c + 2*x, -b, -a]]

        At origin (x=0): J = [[0, 1, 0], [0, 0, 1], [-c, -b, -a]]

        The characteristic polynomial is:
            lambda^3 + a*lambda^2 + b*lambda + c = 0
        """
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-self.c, -self.b, -self.a],
        ], dtype=np.float64)

    @property
    def jacobian_at_c(self) -> np.ndarray:
        """Compute the Jacobian matrix at the non-trivial fixed point (c, 0, 0).

        At x=c: J = [[0, 1, 0], [0, 0, 1], [-c + 2c, -b, -a]]
                   = [[0, 1, 0], [0, 0, 1], [c, -b, -a]]
        """
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [self.c, -self.b, -self.a],
        ], dtype=np.float64)

    def eigenvalues_at_origin(self) -> np.ndarray:
        """Compute eigenvalues of the Jacobian at the origin.

        These are the roots of lambda^3 + a*lambda^2 + b*lambda + c = 0.
        """
        return np.linalg.eigvals(self.jacobian_at_origin)

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

        Returns the average period, or np.inf if no complete cycle is detected.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Detect positive-going zero crossings of x
        crossings: list[float] = []
        prev_x = self._state[0]
        for _ in range(n_measure):
            self.step()
            curr_x = self._state[0]
            if prev_x < 0 and curr_x >= 0:
                frac = -prev_x / (curr_x - prev_x) if curr_x != prev_x else 0.5
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_x = curr_x

        if len(crossings) < 2:
            return np.inf

        periods = np.diff(crossings)
        return float(np.mean(periods))

    def compute_jerk(self, state: np.ndarray) -> float:
        """Compute the jerk (third derivative of x) at a given state.

        From the jerk form: x''' = -a*x'' - b*x' - c*x + x^2
        With x' = y, x'' = z:
            jerk = -a*z - b*y - c*x + x^2
        """
        x, y, z = state
        return -self.a * z - self.b * y - self.c * x + x**2

    def bifurcation_sweep(
        self,
        a_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep parameter a and record attractor statistics.

        For each a value, computes the Lyapunov exponent after
        skipping transients.

        Args:
            a_values: Array of 'a' parameter values to sweep.
            n_transient: Steps to skip for transient.
            n_measure: Steps to use for Lyapunov estimation.

        Returns:
            Dict with a values, Lyapunov exponents, and attractor types.
        """
        lyapunov_exps = []
        attractor_types = []

        for a_val in a_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_transient + n_measure,
                parameters={
                    "a": a_val,
                    "b": self.b,
                    "c": self.c,
                    "x_0": self.x_0,
                    "y_0": self.y_0,
                    "z_0": self.z_0,
                },
            )
            sim = GenesioTesiSimulation(config)
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
            "a": a_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "attractor_type": np.array(attractor_types),
        }
