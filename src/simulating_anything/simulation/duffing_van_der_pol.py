"""Duffing-Van der Pol hybrid oscillator simulation.

Combines the Duffing nonlinear restoring force with the Van der Pol
self-excitation mechanism:

    x'' + mu*(x^2 - 1)*x' + alpha*x + beta*x^3 = F*cos(omega*t)

Or as a first-order system:
    dx/dt = y
    dy/dt = -mu*(x^2 - 1)*y - alpha*x - beta*x^3 + F*cos(omega*t)

Target rediscoveries:
- ODE recovery via SINDy (unforced case)
- Unforced limit cycle amplitude ~2 (Van der Pol behavior for small beta)
- Period and amplitude scaling with mu and beta
- Bifurcation structure under periodic forcing
- Lyapunov exponent vs forcing amplitude F
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DuffingVanDerPolSimulation(SimulationEnvironment):
    """Duffing-Van der Pol hybrid: x'' + mu*(x^2-1)*x' + alpha*x + beta*x^3 = F*cos(omega*t).

    State vector: [x, y, t] where x = displacement, y = velocity (dx/dt),
    t = time (included for the forcing term).

    The nonlinear damping mu*(x^2-1)*y provides:
    - Negative damping for |x| < 1 (energy injection, Van der Pol)
    - Positive damping for |x| > 1 (energy dissipation)
    - Stable limit cycle for F=0 and mu > 0

    The cubic stiffness beta*x^3 provides:
    - Hardening spring behavior (Duffing nonlinearity)
    - Amplitude-dependent frequency shift

    Parameters:
        mu: Van der Pol nonlinear damping strength (default 1.0)
        alpha: linear stiffness coefficient (default 1.0)
        beta: cubic Duffing stiffness coefficient (default 0.2)
        F: forcing amplitude (default 0.3)
        omega: forcing angular frequency (default 1.0)
        x_0: initial displacement (default 0.1)
        y_0: initial velocity (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.mu = p.get("mu", 1.0)
        self.alpha = p.get("alpha", 1.0)
        self.beta = p.get("beta", 0.2)
        self.F = p.get("F", 0.3)
        self.omega = p.get("omega", 1.0)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.0)
        self._t = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize displacement, velocity, and time."""
        self._state = np.array([self.x_0, self.y_0, 0.0], dtype=np.float64)
        self._step_count = 0
        self._t = 0.0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with time-dependent forcing."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y, t]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta with explicit time dependence."""
        dt = self.config.dt
        y = self._state[:2]
        t = self._t

        k1 = self._derivatives(y, t)
        k2 = self._derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._derivatives(y + dt * k3, t + dt)

        new_xy = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._t += dt
        self._state = np.array([new_xy[0], new_xy[1], self._t], dtype=np.float64)

    def _derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """Compute dy/dt for the Duffing-Van der Pol equation.

        Args:
            state: State vector [x, y].
            t: Current time (needed for forcing term).

        Returns:
            Derivatives [dx/dt, dy/dt].
        """
        x, y = state
        dx_dt = y
        dy_dt = (
            -self.mu * (x**2 - 1) * y
            - self.alpha * x
            - self.beta * x**3
            + self.F * np.cos(self.omega * t)
        )
        return np.array([dx_dt, dy_dt])

    def compute_energy(self, x: float | None = None, y: float | None = None) -> float:
        """Compute approximate mechanical energy.

        E = y^2/2 + alpha*x^2/2 + beta*x^4/4

        This is the energy of the conservative part (ignoring damping and forcing).
        For mu=0, F=0, this quantity is conserved.

        Args:
            x: Displacement (uses current state if None).
            y: Velocity (uses current state if None).

        Returns:
            Instantaneous energy.
        """
        if x is None:
            x = self._state[0]
        if y is None:
            y = self._state[1]
        return 0.5 * y**2 + 0.5 * self.alpha * x**2 + 0.25 * self.beta * x**4

    def poincare_section(
        self,
        n_periods: int = 200,
        n_transient: int = 100,
    ) -> dict[str, np.ndarray]:
        """Compute Poincare section by stroboscopic sampling at forcing period.

        Samples (x, y) at times t = 2*pi*n/omega, producing a discrete map
        that reveals periodic orbits (finite points), quasiperiodicity
        (closed curves), and chaos (scattered points).

        Args:
            n_periods: Number of forcing periods to sample.
            n_transient: Number of forcing periods to skip as transient.

        Returns:
            Dict with 'x' and 'y' arrays of Poincare points.
        """
        if self.omega == 0:
            return {"x": np.array([]), "y": np.array([])}

        dt = self.config.dt
        T_force = 2 * np.pi / self.omega
        steps_per_period = max(1, int(round(T_force / dt)))

        # Skip transient
        for _ in range(n_transient * steps_per_period):
            self.step()

        # Collect Poincare points
        x_points = []
        y_points = []
        for _ in range(n_periods):
            for _ in range(steps_per_period):
                self.step()
            x_points.append(self._state[0])
            y_points.append(self._state[1])

        return {"x": np.array(x_points), "y": np.array(y_points)}

    def bifurcation_sweep(
        self,
        F_values: np.ndarray,
        n_poincare: int = 100,
        n_transient: int = 200,
    ) -> dict[str, list]:
        """Sweep forcing amplitude F and record Poincare section x-values.

        For each F, runs a fresh simulation through transient then collects
        stroboscopic samples. The resulting data traces bifurcation diagrams.

        Args:
            F_values: Array of forcing amplitudes to sweep.
            n_poincare: Number of Poincare points per F value.
            n_transient: Transient periods to skip.

        Returns:
            Dict with 'F' (list of floats) and 'x' (list of arrays).
        """
        if self.omega == 0:
            return {"F": [], "x": []}

        dt = self.config.dt
        T_force = 2 * np.pi / self.omega
        steps_per_period = max(1, int(round(T_force / dt)))

        all_F = []
        all_x = []

        for F_val in F_values:
            # Create fresh simulation with this F
            params = dict(self.config.parameters)
            params["F"] = float(F_val)
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=self.config.n_steps,
                parameters=params,
            )
            sim = DuffingVanDerPolSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient * steps_per_period):
                sim.step()

            # Collect Poincare points
            x_points = []
            for _ in range(n_poincare):
                for _ in range(steps_per_period):
                    sim.step()
                x_points.append(sim._state[0])

            all_F.append(float(F_val))
            all_x.append(np.array(x_points))

        return {"F": all_F, "x": all_x}

    def compute_lyapunov(
        self,
        n_steps: int = 50000,
        n_transient: int = 10000,
    ) -> float:
        """Compute the maximal Lyapunov exponent via tangent vector evolution.

        Uses the standard algorithm: evolve a reference trajectory and a nearby
        tangent vector, periodically renormalizing the tangent vector and
        accumulating the logarithm of the growth factor.

        Args:
            n_steps: Number of steps for Lyapunov computation.
            n_transient: Steps to skip for transient.

        Returns:
            Estimated maximal Lyapunov exponent (bits/time).
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Initialize tangent vector (in x, y space)
        delta = np.array([1.0, 0.0])
        delta = delta / np.linalg.norm(delta)

        lyap_sum = 0.0

        for _ in range(n_steps):
            x, y = self._state[0], self._state[1]

            # Jacobian of the flow at current point
            # d(dx/dt)/dx = 0, d(dx/dt)/dy = 1
            # d(dy/dt)/dx = -2*mu*x*y - alpha - 3*beta*x^2
            # d(dy/dt)/dy = -mu*(x^2 - 1)
            J = np.array([
                [0.0, 1.0],
                [-2 * self.mu * x * y - self.alpha - 3 * self.beta * x**2,
                 -self.mu * (x**2 - 1)],
            ])

            # Evolve tangent vector with RK4 for linearized equations
            dk1 = J @ delta
            # Approximate: use same Jacobian for all RK4 substeps
            # (valid for small dt)
            dk2 = J @ (delta + 0.5 * dt * dk1)
            dk3 = J @ (delta + 0.5 * dt * dk2)
            dk4 = J @ (delta + dt * dk3)
            delta = delta + (dt / 6.0) * (dk1 + 2 * dk2 + 2 * dk3 + dk4)

            # Renormalize
            norm = np.linalg.norm(delta)
            if norm > 0:
                lyap_sum += np.log(norm)
                delta = delta / norm

            # Advance reference trajectory
            self.step()

        return lyap_sum / (n_steps * dt)

    def compute_limit_cycle_amplitude(self, n_steps: int = 50000) -> float:
        """Measure the limit cycle amplitude of the unforced system.

        For F=0 and mu>0, the system exhibits a Van der Pol-like limit cycle.
        This method runs the simulation through a transient and measures the
        peak |x| value on the attractor.

        Args:
            n_steps: Total steps to run (includes transient).

        Returns:
            Maximum |x| observed after transient.
        """
        # Use roughly half the steps as transient
        n_transient = n_steps // 2

        for _ in range(n_transient):
            self.step()

        x_max = 0.0
        for _ in range(n_steps - n_transient):
            self.step()
            x_max = max(x_max, abs(self._state[0]))

        return float(x_max)

    def frequency_response(
        self,
        omega_values: np.ndarray,
        n_transient_periods: int = 100,
        n_measure_periods: int = 20,
    ) -> dict[str, np.ndarray]:
        """Compute amplitude vs forcing frequency (frequency response curve).

        For each omega, creates a fresh simulation with all other parameters
        held constant, runs through a transient, and measures the steady-state
        peak amplitude.

        Args:
            omega_values: Array of forcing frequencies to sweep.
            n_transient_periods: Number of forcing periods for transient.
            n_measure_periods: Number of periods to measure amplitude over.

        Returns:
            Dict with 'omega' and 'amplitude' arrays.
        """
        dt = self.config.dt
        amplitudes = []

        for omega_val in omega_values:
            if omega_val <= 0:
                amplitudes.append(0.0)
                continue

            params = dict(self.config.parameters)
            params["omega"] = float(omega_val)
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=self.config.n_steps,
                parameters=params,
            )
            sim = DuffingVanDerPolSimulation(config)
            sim.reset()

            T_force = 2 * np.pi / omega_val
            transient_steps = max(
                int(50 / dt), int(n_transient_periods * T_force / dt)
            )
            for _ in range(transient_steps):
                sim.step()

            x_max = 0.0
            measure_steps = int(n_measure_periods * T_force / dt)
            for _ in range(measure_steps):
                sim.step()
                x_max = max(x_max, abs(sim._state[0]))

            amplitudes.append(float(x_max))

        return {
            "omega": omega_values,
            "amplitude": np.array(amplitudes),
        }

    @property
    def approximate_period(self) -> float:
        """Approximate small-amplitude period: T = 2*pi/sqrt(alpha).

        Valid for alpha > 0 and small oscillations where beta*x^2 << alpha.
        """
        if self.alpha <= 0:
            return float("inf")
        return 2 * np.pi / np.sqrt(self.alpha)

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period via upward zero crossings.

        Args:
            n_periods: Number of periods to average over.

        Returns:
            Mean period, or inf if no oscillation detected.
        """
        dt = self.config.dt
        T_approx = self.approximate_period
        if not np.isfinite(T_approx):
            T_approx = 2 * np.pi

        # Let transient die out
        transient_steps = max(int(50 / dt), int(20 * T_approx / dt))
        for _ in range(transient_steps):
            self.step()

        # Detect upward zero crossings of x
        crossings: list[float] = []
        prev_x = self._state[0]
        measure_steps = int(n_periods * T_approx / dt * 2)
        for _ in range(measure_steps):
            self.step()
            x = self._state[0]
            if prev_x < 0 and x >= 0:
                t_cross = (
                    (self._step_count - 1) * dt
                    + dt * (-prev_x) / (x - prev_x)
                )
                crossings.append(t_cross)
            prev_x = x

        if len(crossings) < 2:
            return float("inf")

        periods = np.diff(crossings)
        return float(np.mean(periods))
