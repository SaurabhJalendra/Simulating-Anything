"""Stuart-Landau oscillator simulation.

The Stuart-Landau equation is the normal form of supercritical Hopf bifurcation:

    dz/dt = (mu + i*omega)*z - (1 + i*beta)*|z|^2*z

In real coordinates (x = Re(z), y = Im(z)):

    dx/dt = mu*x - omega*y - (x^2 + y^2)*(x - beta*y)
    dy/dt = omega*x + mu*y - (x^2 + y^2)*(beta*x + y)

Target rediscoveries:
- Hopf bifurcation at mu = 0 (origin stable for mu < 0, limit cycle for mu > 0)
- Limit cycle amplitude: r = sqrt(mu) for mu > 0
- Oscillation frequency: omega_eff = omega - beta*mu (from polar form)
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class StuartLandauSimulation(SimulationEnvironment):
    """Stuart-Landau oscillator: normal form of supercritical Hopf bifurcation.

    State vector: [x, y] where z = x + i*y is the complex amplitude.

    For mu > 0 the origin is unstable and a stable limit cycle exists with
    radius r = sqrt(mu). For mu < 0 the origin is a stable spiral.

    Parameters:
        mu: bifurcation parameter (>0 for limit cycle, <0 for stable origin)
        omega: natural frequency of oscillation
        beta: frequency-amplitude coupling (nonlinear frequency shift)
        x_0: initial x (Re(z))
        y_0: initial y (Im(z))
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.mu = p.get("mu", 1.0)
        self.omega = p.get("omega", 1.0)
        self.beta = p.get("beta", 0.0)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.0)

    @property
    def theoretical_amplitude(self) -> float:
        """Theoretical limit cycle radius: r = sqrt(mu) for mu > 0."""
        if self.mu > 0:
            return np.sqrt(self.mu)
        return 0.0

    @property
    def theoretical_frequency(self) -> float:
        """Effective angular frequency on the limit cycle.

        From the polar form d(theta)/dt = omega - beta*r^2, with r^2 = mu
        on the limit cycle, the effective frequency is omega - beta*mu.
        """
        return self.omega - self.beta * self.mu

    @property
    def is_oscillatory(self) -> bool:
        """True if mu > 0 (supercritical limit cycle exists)."""
        return self.mu > 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize position in the (x, y) plane."""
        self._state = np.array([self.x_0, self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dx/dt, dy/dt for the Stuart-Landau system."""
        x, yc = y
        r2 = x**2 + yc**2
        dx = self.mu * x - self.omega * yc - r2 * (x - self.beta * yc)
        dy = self.omega * x + self.mu * yc - r2 * (self.beta * x + yc)
        return np.array([dx, dy])

    def compute_amplitude(self, state: np.ndarray | None = None) -> float:
        """Compute instantaneous amplitude r = sqrt(x^2 + y^2)."""
        if state is None:
            state = self._state
        return float(np.sqrt(state[0] ** 2 + state[1] ** 2))

    def compute_phase(self, state: np.ndarray | None = None) -> float:
        """Compute instantaneous phase theta = atan2(y, x)."""
        if state is None:
            state = self._state
        return float(np.arctan2(state[1], state[0]))

    def compute_frequency(self, n_periods: int = 5) -> float:
        """Measure angular frequency from zero crossings of x.

        Returns angular frequency (rad/time) after letting transient decay.
        """
        if not self.is_oscillatory:
            return 0.0

        dt = self.config.dt
        # Approximate period for transient estimation
        approx_period = 2 * np.pi / abs(self.theoretical_frequency)

        # Let transient die out
        transient_steps = max(int(50 / dt), int(20 * approx_period / dt))
        for _ in range(transient_steps):
            self.step()

        # Detect upward zero crossings of x
        crossings = []
        prev_x = self._state[0]
        measure_steps = int(n_periods * approx_period / dt * 2)
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
            return 0.0

        periods = np.diff(crossings)
        mean_period = float(np.mean(periods))
        if mean_period <= 0:
            return 0.0
        return 2 * np.pi / mean_period

    def measure_amplitude(self, n_periods: int = 3) -> float:
        """Measure the steady-state amplitude after transient."""
        if not self.is_oscillatory:
            return 0.0

        dt = self.config.dt
        approx_period = 2 * np.pi / abs(self.theoretical_frequency)

        transient_steps = max(int(50 / dt), int(20 * approx_period / dt))
        for _ in range(transient_steps):
            self.step()

        # Track amplitude over several cycles
        r_max = 0.0
        r_min = np.inf
        measure_steps = int(n_periods * approx_period / dt * 2)
        for _ in range(measure_steps):
            self.step()
            r = self.compute_amplitude()
            r_max = max(r_max, r)
            r_min = min(r_min, r)

        # On a perfect limit cycle, r_max ~ r_min ~ sqrt(mu)
        return float((r_max + r_min) / 2)

    def mu_sweep(
        self,
        mu_values: np.ndarray | None = None,
        dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Sweep mu to show Hopf bifurcation at mu = 0.

        Returns dict with 'mu', 'amplitude', and 'frequency' arrays.
        """
        if mu_values is None:
            mu_values = np.linspace(-1.0, 3.0, 30)
        if dt is None:
            dt = self.config.dt

        amplitudes = []
        frequencies = []

        for mu_val in mu_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=dt,
                n_steps=1000,
                parameters={
                    "mu": float(mu_val),
                    "omega": self.omega,
                    "beta": self.beta,
                    "x_0": 0.1,
                    "y_0": 0.0,
                },
            )
            sim = StuartLandauSimulation(config)
            sim.reset()

            if mu_val > 0:
                amp = sim.measure_amplitude(n_periods=3)
                freq = sim.compute_frequency(n_periods=3)
            else:
                # Below bifurcation: run to steady state, measure decay
                for _ in range(int(100 / dt)):
                    sim.step()
                amp = sim.compute_amplitude()
                freq = 0.0

            amplitudes.append(amp)
            frequencies.append(freq)

        return {
            "mu": mu_values,
            "amplitude": np.array(amplitudes),
            "frequency": np.array(frequencies),
        }
