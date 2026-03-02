"""Nose-Hoover thermostat simulation.

The Nose-Hoover system is a 3D ODE modelling a harmonic oscillator coupled
to a heat bath via a thermostat variable:
    dx/dt = y
    dy/dt = -x + y*z
    dz/dt = a - y^2

This is Sprott's system A, and one of the simplest systems that exhibits
time-reversible, measure-preserving chaos. The parameter a (default 1.0)
controls the target kinetic energy; for a=1 the system is chaotic with
a strange attractor.

The flow has divergence div(F) = d(y)/dy_partial_z_component = z from the
y*z term. However, the system preserves the Gibbs measure exp(-H/a) where
H = (x^2 + y^2)/2 + z^2/2, making it a thermostatted Hamiltonian system.

Target rediscoveries:
- SINDy recovery of Nose-Hoover ODEs
- Positive Lyapunov exponent at a=1.0 (chaos)
- Conserved quantity / measure verification
- Parameter sweep: chaos vs periodic vs quasiperiodic behavior
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class NoseHooverSimulation(SimulationEnvironment):
    """Nose-Hoover thermostat: Hamiltonian + heat bath coupling in 3D.

    State vector: [x, y, z]

    ODEs:
        dx/dt = y
        dy/dt = -x + y*z
        dz/dt = a - y^2

    The system is volume-preserving only in a generalized sense (preserves a
    measure with density proportional to exp(-z^2/(2a)) when a>0). The
    Euclidean divergence is z, so the flow expands where z>0 and contracts
    where z<0, but averaged over the attractor this balances out.

    Parameters:
        a: thermostat parameter controlling target kinetic energy (default 1.0)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.0)
        self.x_0 = p.get("x_0", 0.0)
        self.y_0 = p.get("y_0", 5.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Nose-Hoover state."""
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
        """Nose-Hoover equations: dx/dt=y, dy/dt=-x+y*z, dz/dt=a-y^2."""
        x, y, z = state
        dx = y
        dy = -x + y * z
        dz = self.a - y**2
        return np.array([dx, dy, dz])

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute fixed points of the Nose-Hoover system.

        Setting derivatives to zero:
            y = 0  (from dx/dt = 0)
            -x + y*z = -x = 0  =>  x = 0  (from dy/dt = 0)
            a - y^2 = a = 0  (from dz/dt = 0)

        For a != 0, there are no fixed points. The z equation requires
        y^2 = a, but y = 0 from the first equation, so a must be 0.
        When a = 0, the origin (0, 0, z) is a line of fixed points.
        """
        if abs(self.a) < 1e-15:
            # a=0: line of fixed points at (0, 0, z) for any z
            return [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        return []

    def compute_divergence(self, state: np.ndarray) -> float:
        """Compute the divergence of the vector field at a point.

        div(F) = d(dx/dt)/dx + d(dy/dt)/dy + d(dz/dt)/dz
               = 0 + z + 0 = z

        The divergence equals z, so the flow is volume-preserving only
        at the z=0 plane. On the attractor, the time-averaged divergence
        is zero, confirming the system preserves a measure.
        """
        return float(state[2])

    def check_volume_preservation(
        self, n_steps: int = 10000, n_transient: int = 2000
    ) -> dict[str, float]:
        """Verify that the time-averaged divergence is near zero on the attractor.

        The Nose-Hoover system preserves the measure exp(-z^2/(2a)), so
        while the pointwise divergence = z is nonzero, its time average
        over the attractor should be approximately zero.

        Args:
            n_steps: Number of steps to average over.
            n_transient: Steps to skip for transient.

        Returns:
            Dict with mean and std of the divergence, plus max absolute value.
        """
        self.reset()

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect divergence values
        div_values = []
        for _ in range(n_steps):
            state = self.step()
            div_values.append(self.compute_divergence(state))

        div_arr = np.array(div_values)
        return {
            "mean_divergence": float(np.mean(div_arr)),
            "std_divergence": float(np.std(div_arr)),
            "max_abs_divergence": float(np.max(np.abs(div_arr))),
            "near_zero": bool(abs(np.mean(div_arr)) < 0.5),
        }

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
        crossings = []
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

    def bifurcation_sweep(
        self,
        a_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep parameter a and record attractor statistics.

        For each a value, computes the Lyapunov exponent and trajectory
        statistics after skipping transients.

        Args:
            a_values: Array of parameter values to sweep.
            n_transient: Steps to skip for transient.
            n_measure: Steps to use for Lyapunov estimation.

        Returns:
            Dict with a values, Lyapunov exponents, and attractor types.
        """
        lyapunov_exps = []
        attractor_types = []
        amplitudes = []

        for a_val in a_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_transient + n_measure,
                parameters={
                    "a": a_val,
                    "x_0": self.x_0,
                    "y_0": self.y_0,
                    "z_0": self.z_0,
                },
            )
            sim = NoseHooverSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Estimate Lyapunov
            lam = sim.estimate_lyapunov(n_steps=n_measure, dt=self.config.dt)
            lyapunov_exps.append(lam)

            # Measure amplitude
            sim.reset()
            for _ in range(n_transient):
                sim.step()
            vals = []
            for _ in range(min(5000, n_measure)):
                state = sim.step()
                vals.append(np.linalg.norm(state))
            amplitudes.append(np.max(vals))

            # Classify
            if lam > 0.01:
                atype = "chaotic"
            elif lam < -0.01:
                atype = "fixed_point"
            else:
                atype = "marginal"
            attractor_types.append(atype)

        return {
            "a": a_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
            "max_amplitude": np.array(amplitudes),
            "attractor_type": np.array(attractor_types),
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

    def compute_hamiltonian(self, state: np.ndarray) -> float:
        """Compute the Hamiltonian-like quantity H = (x^2 + y^2)/2.

        In the Nose-Hoover thermostatted system, y^2 represents kinetic
        energy and x^2 represents potential energy. While H is not
        conserved (that is the point of the thermostat), the time-averaged
        <y^2> should equal a (the target temperature).
        """
        x, y, _z = state
        return 0.5 * (x**2 + y**2)

    def check_temperature_equilibration(
        self, n_steps: int = 10000, n_transient: int = 2000
    ) -> dict[str, float]:
        """Verify that the thermostat equilibrates: <y^2> ~ a.

        The Nose-Hoover thermostat enforces that the time-averaged
        kinetic energy <y^2> equals the target parameter a. This is
        the defining property of the thermostat.

        Args:
            n_steps: Number of steps to average over.
            n_transient: Steps to skip for transient.

        Returns:
            Dict with mean kinetic energy, target a, and relative error.
        """
        self.reset()

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect y^2 values
        y2_values = []
        for _ in range(n_steps):
            state = self.step()
            y2_values.append(state[1] ** 2)

        mean_y2 = float(np.mean(y2_values))
        rel_error = abs(mean_y2 - self.a) / max(self.a, 1e-15)

        return {
            "mean_y_squared": mean_y2,
            "target_a": self.a,
            "relative_error": rel_error,
            "equilibrated": bool(rel_error < 0.3),
        }
