"""Vallis ENSO model simulation -- El Nino-Southern Oscillation dynamics.

The Vallis system is a 3D ODE model of coupled ocean-atmosphere dynamics:
    dx/dt = B*y - C*(x - p)
    dy/dt = -y + x*z
    dz/dt = -z - x*y + 1

where:
    x: ocean temperature anomaly (sea surface temperature deviation)
    y: thermocline depth anomaly
    z: wind stress anomaly

Parameters:
    B: coupling strength between thermocline depth and temperature (default 102.0)
    C: damping rate for temperature anomaly (default 3.0)
    p: base temperature offset (default 0.0)

The system has constant divergence div(F) = -(C + 2), since:
    d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz = -C + (-1) + (-1) = -(C + 2)

For the default C=3, this gives divergence = -5, meaning the system is
uniformly dissipative. Chaos represents the irregular ENSO cycle observed
in nature: quasi-periodic warm (El Nino) and cold (La Nina) events.

Target rediscoveries:
- SINDy recovery of Vallis ODEs
- Lyapunov exponent estimation (positive for chaotic regime)
- B-parameter sweep mapping chaos transition
- Fixed point analysis
- Constant divergence = -(C + 2) verification
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class VallisSimulation(SimulationEnvironment):
    """Vallis ENSO model: coupled ocean-atmosphere dynamics.

    State vector: [x, y, z]

    ODEs:
        dx/dt = B*y - C*(x - p)
        dy/dt = -y + x*z
        dz/dt = -z - x*y + 1

    The trace of the Jacobian is -C + (-1) + (-1) = -(C + 2), so the system
    is uniformly dissipative with constant divergence.

    Parameters:
        B: coupling strength (default 102.0)
        C: damping rate (default 3.0)
        p: base temperature offset (default 0.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.B = p.get("B", 102.0)
        self.C = p.get("C", 3.0)
        self.p = p.get("p", 0.0)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.2)
        self.z_0 = p.get("z_0", 0.3)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Vallis state."""
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
        """Vallis equations: dx=B*y-C*(x-p), dy=-y+x*z, dz=-z-x*y+1."""
        x, y, z = state
        dx = self.B * y - self.C * (x - self.p)
        dy = -y + x * z
        dz = -z - x * y + 1.0
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Vallis system.

        Setting derivatives to zero:
            B*y - C*(x - p) = 0      =>  y = C*(x - p) / B
            -y + x*z = 0             =>  z = y / x  (for x != 0)
            -z - x*y + 1 = 0         =>  z = 1 - x*y

        Combining the z equations: y/x = 1 - x*y  =>  y = x - x^2*y
        =>  y*(1 + x^2) = x  =>  y = x / (1 + x^2)

        Substituting into the first equation:
            B * x / (1 + x^2) = C * (x - p)

        For p=0: B*x / (1+x^2) = C*x  =>  x=0 or B/(1+x^2) = C
        =>  x^2 = B/C - 1  (exists when B > C)

        General case requires solving a cubic. We use numerical approach.
        """
        points = []

        # For p=0, analytic solutions exist
        if abs(self.p) < 1e-12:
            # Origin-like: x=0, y=0, z=1
            points.append(np.array([0.0, 0.0, 1.0], dtype=np.float64))

            # Non-origin: x^2 = B/C - 1
            if self.B > self.C and self.C > 0:
                x_eq = np.sqrt(self.B / self.C - 1.0)
                for sign in [1.0, -1.0]:
                    x = sign * x_eq
                    y = x / (1.0 + x**2)
                    z = y / x if abs(x) > 1e-14 else 1.0
                    points.append(
                        np.array([x, y, z], dtype=np.float64)
                    )
        else:
            # Numerical fixed point search for general p
            # Start from several initial guesses
            from scipy.optimize import fsolve

            def residual(state_vec):
                return self._derivatives(np.array(state_vec))

            guesses = [
                [self.p, 0.0, 1.0],
                [self.p + 1.0, 0.1, 0.9],
                [self.p - 1.0, -0.1, 0.9],
            ]
            found = []
            for g in guesses:
                sol, info, ier, _ = fsolve(residual, g, full_output=True)
                if ier == 1 and np.linalg.norm(info["fvec"]) < 1e-10:
                    # Check if this is a new fixed point
                    is_new = True
                    for existing in found:
                        if np.linalg.norm(sol - existing) < 1e-6:
                            is_new = False
                            break
                    if is_new:
                        found.append(sol.copy())
            points = [
                np.array(fp, dtype=np.float64) for fp in found
            ]

        return points

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state.

        J = [[-C,   B,    0 ],
             [ z,  -1,    x ],
             [-y,  -x,   -1 ]]

        The trace is always -C + (-1) + (-1) = -(C + 2) (constant dissipation).
        """
        x, y, z = state
        return np.array([
            [-self.C, self.B, 0.0],
            [z, -1.0, x],
            [-y, -x, -1.0],
        ], dtype=np.float64)

    def compute_divergence(self, state: np.ndarray) -> float:
        """Compute the divergence (trace of Jacobian) at any state.

        For the Vallis system:
            div = d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz
                = -C + (-1) + (-1) = -(C + 2)

        The divergence is constant everywhere (independent of state),
        making the Vallis system uniformly dissipative.
        """
        return -(self.C + 2.0)

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
            if not (
                np.all(np.isfinite(state1))
                and np.all(np.isfinite(state2))
            ):
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

    def bifurcation_sweep(
        self,
        B_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep parameter B and record attractor statistics.

        For each B value, computes the Lyapunov exponent after
        skipping transients. The Vallis system transitions to chaos
        as B increases.

        Args:
            B_values: Array of 'B' parameter values to sweep.
            n_transient: Steps to skip for transient.
            n_measure: Steps to use for Lyapunov estimation.

        Returns:
            Dict with B values, Lyapunov exponents, and attractor types.
        """
        lyapunov_exps = []
        attractor_types = []

        for B_val in B_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_transient + n_measure,
                parameters={
                    "B": B_val,
                    "C": self.C,
                    "p": self.p,
                    "x_0": self.x_0,
                    "y_0": self.y_0,
                    "z_0": self.z_0,
                },
            )
            sim = VallisSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Estimate Lyapunov
            lam = sim.estimate_lyapunov(
                n_steps=n_measure, dt=self.config.dt
            )
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
            "B": B_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "attractor_type": np.array(attractor_types),
        }
