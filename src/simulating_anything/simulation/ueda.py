"""Ueda oscillator simulation -- forced cubic oscillator with strange attractor.

The Ueda oscillator is a Duffing-type system without linear restoring force:
    dx/dt = y
    dy/dt = -delta*y - x^3 + B*cos(t)

This system was studied by Yoshisuke Ueda in the 1960s as one of the first
examples of a strange attractor in a periodically forced nonlinear oscillator.

Target rediscoveries:
- ODE recovery via SINDy: dy/dt = -delta*y - x^3 + B*cos(t)
- Chaos onset as B increases (period-doubling cascade)
- Lyapunov exponent as function of B
- Poincare section structure (strobe at T = 2*pi)
- Strange attractor at delta=0.05, B=7.5
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class UedaSimulation(SimulationEnvironment):
    """Ueda oscillator: forced cubic oscillator with strange attractor.

    State vector: [x, y, t_phase] where t_phase tracks the forcing phase.

    ODEs:
        dx/dt = y
        dy/dt = -delta*y - x^3 + B*cos(t)

    The observation includes t_phase so that Poincare sections can be
    constructed by sampling at multiples of the forcing period T = 2*pi.

    Parameters:
        delta: damping coefficient (default 0.05)
        B: forcing amplitude (default 7.5, chaotic regime)
        x_0: initial displacement (default 2.5)
        y_0: initial velocity (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.delta = p.get("delta", 0.05)
        self.B = p.get("B", 7.5)
        self.x_0 = p.get("x_0", 2.5)
        self.y_0 = p.get("y_0", 0.0)
        self._t = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize displacement, velocity, and phase."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with time-dependent forcing."""
        self._rk4_step()
        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [x, y, t_phase].

        t_phase is the forcing phase modulo 2*pi, useful for Poincare sections.
        """
        t_phase = self._t % (2.0 * np.pi)
        return np.array(
            [self._state[0], self._state[1], t_phase], dtype=np.float64
        )

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta with explicit time dependence."""
        dt = self.config.dt
        y = self._state
        t = self._t

        k1 = self._derivatives(y, t)
        k2 = self._derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._derivatives(y + dt * k3, t + dt)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._t += dt

    def _derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """Compute dy/dt for the Ueda oscillator.

        Args:
            state: State vector [x, y].
            t: Current time (needed for forcing term).

        Returns:
            Derivatives [dx/dt, dy/dt].
        """
        x, y = state
        dx_dt = y
        dy_dt = -self.delta * y - x**3 + self.B * np.cos(t)
        return np.array([dx_dt, dy_dt])

    @property
    def forcing_period(self) -> float:
        """Period of the forcing: T = 2*pi (since omega_drive = 1)."""
        return 2.0 * np.pi

    @property
    def mechanical_energy(self) -> float:
        """Instantaneous mechanical energy: E = 0.5*y^2 + 0.25*x^4.

        This is the energy of the conservative part (no damping, no forcing).
        For delta=0 and B=0, this quantity is conserved.
        """
        x, y = self._state
        return 0.5 * y**2 + 0.25 * x**4

    def compute_lyapunov_exponent(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via Wolf et al. (1985).

        Tracks two nearby trajectories and renormalizes periodically to
        measure the rate of exponential divergence.

        Args:
            n_steps: Number of integration steps.
            dt: Timestep override. Uses config.dt if None.

        Returns:
            Estimated maximum Lyapunov exponent (positive => chaos).
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0.0])
        t = self._t

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance state1
            k1 = self._derivatives(state1, t)
            k2 = self._derivatives(state1 + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self._derivatives(state1 + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self._derivatives(state1 + dt * k3, t + dt)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance state2
            k1 = self._derivatives(state2, t)
            k2 = self._derivatives(state2 + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self._derivatives(state2 + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self._derivatives(state2 + dt * k3, t + dt)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            t += dt

            # Compute distance and renormalize
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def compute_poincare_section(
        self,
        n_transient: int = 500,
        n_points: int = 500,
    ) -> np.ndarray:
        """Sample state stroboscopically at multiples of the forcing period.

        The Poincare section for a forced oscillator strobes at the
        drive period T = 2*pi. A period-1 orbit gives 1 point, period-2
        gives 2 points, chaos gives a fractal cloud.

        Args:
            n_transient: Number of forcing periods to skip as transient.
            n_points: Number of Poincare section points to collect.

        Returns:
            Array of shape (n_points, 2) with [x, y] at each strobe.
        """
        dt = self.config.dt
        T = self.forcing_period
        steps_per_period = int(round(T / dt))

        # Skip transient
        for _ in range(n_transient):
            for _ in range(steps_per_period):
                self._rk4_step()
                self._step_count += 1

        # Collect strobe points
        points = []
        for _ in range(n_points):
            for _ in range(steps_per_period):
                self._rk4_step()
                self._step_count += 1
            points.append(self._state.copy())

        return np.array(points)

    def measure_steady_amplitude(self, n_periods: int = 20) -> float:
        """Measure steady-state oscillation amplitude after transient.

        Args:
            n_periods: Number of forcing periods to measure over.

        Returns:
            Maximum absolute displacement observed.
        """
        dt = self.config.dt
        T = self.forcing_period
        # Skip 200 forcing periods as transient
        transient_steps = int(200 * T / dt)
        for _ in range(transient_steps):
            self._rk4_step()
            self._step_count += 1

        # Measure peak displacement
        x_max = 0.0
        measure_steps = int(n_periods * T / dt)
        for _ in range(measure_steps):
            self._rk4_step()
            self._step_count += 1
            x_max = max(x_max, abs(self._state[0]))

        return float(x_max)

    def lyapunov_sweep(
        self,
        B_values: np.ndarray,
        n_transient: int = 5000,
        n_measure: int = 30000,
    ) -> dict[str, np.ndarray]:
        """Sweep forcing amplitude B and compute Lyapunov exponent at each.

        Args:
            B_values: Array of forcing amplitudes to sweep.
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
                    "delta": self.delta,
                    "B": B_val,
                    "x_0": self.x_0,
                    "y_0": self.y_0,
                },
            )
            sim = UedaSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            lam = sim.compute_lyapunov_exponent(
                n_steps=n_measure, dt=self.config.dt
            )
            lyapunov_exps.append(lam)

            if lam > 0.01:
                atype = "chaotic"
            elif lam < -0.01:
                atype = "periodic"
            else:
                atype = "marginal"
            attractor_types.append(atype)

        return {
            "B": B_values,
            "lyapunov_exponent": np.array(lyapunov_exps),
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
            Dict with mean, std, min, max for x and y components.
        """
        self.reset()

        # Skip transient
        for _ in range(n_transient):
            self.step()

        xs, ys = [], []
        for _ in range(n_steps):
            self.step()
            xs.append(self._state[0])
            ys.append(self._state[1])

        xs = np.array(xs)
        ys = np.array(ys)

        return {
            "x_mean": float(np.mean(xs)),
            "y_mean": float(np.mean(ys)),
            "x_std": float(np.std(xs)),
            "y_std": float(np.std(ys)),
            "x_min": float(np.min(xs)),
            "y_min": float(np.min(ys)),
            "x_max": float(np.max(xs)),
            "y_max": float(np.max(ys)),
        }
