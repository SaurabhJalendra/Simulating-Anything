"""Ring oscillator simulation (3-stage electronic inverter ring).

A ring oscillator consists of an odd number (3) of inverting amplifiers
connected in a loop. Each stage is modeled as:

    dx_i/dt = -x_i + tanh(g * (-x_{i-1}))

where f(x) = tanh(g*x) is the saturating inverting nonlinearity and g is
the gain. With 3 stages:

    dx1/dt = -x1 + tanh(g * (-x3))
    dx2/dt = -x2 + tanh(g * (-x1))
    dx3/dt = -x3 + tanh(g * (-x2))

Target rediscoveries:
- Oscillation onset at gain g > 1 (sufficient loop gain for instability)
- Period T ~ 6*tau where tau = RC time constant (=1 here)
- 3-phase oscillation with 120-degree phase shift between stages
- Amplitude saturates at tanh(g) ~ 1 for large g
- Z3 cyclic symmetry (repressilator-like)
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RingOscillatorSimulation(SimulationEnvironment):
    """3-stage ring oscillator with tanh inverting nonlinearity.

    State vector: [x1, x2, x3] representing voltages at each stage.

    For gain g > 1, the trivial fixed point at the origin becomes unstable
    and the system oscillates with 3-phase symmetry (120-degree shifts).
    For g < 1, the origin is a stable fixed point.

    Parameters:
        gain: inverter gain (>1 for oscillation, default 5.0)
        x1_0: initial voltage at stage 1 (default 0.01)
        x2_0: initial voltage at stage 2 (default 0.0)
        x3_0: initial voltage at stage 3 (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.gain = p.get("gain", 5.0)
        self.x1_0 = p.get("x1_0", 0.01)
        self.x2_0 = p.get("x2_0", 0.0)
        self.x3_0 = p.get("x3_0", 0.0)

    @property
    def is_oscillatory(self) -> bool:
        """True if gain > 1 (sufficient loop gain for instability)."""
        return self.gain > 1.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize stage voltages with small perturbation from origin."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            perturbation = rng.normal(0, 0.01, size=3)
            self._state = np.array(
                [self.x1_0, self.x2_0, self.x3_0], dtype=np.float64
            ) + perturbation
        else:
            self._state = np.array(
                [self.x1_0, self.x2_0, self.x3_0], dtype=np.float64
            )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x1, x2, x3]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for the 3-stage ring oscillator.

        dx1/dt = -x1 + tanh(g * (-x3))
        dx2/dt = -x2 + tanh(g * (-x1))
        dx3/dt = -x3 + tanh(g * (-x2))

        Each stage inverts and saturates the output of the previous stage.
        """
        x1, x2, x3 = state
        g = self.gain

        dx1 = -x1 + np.tanh(g * (-x3))
        dx2 = -x2 + np.tanh(g * (-x1))
        dx3 = -x3 + np.tanh(g * (-x2))

        return np.array([dx1, dx2, dx3])

    def compute_period(
        self, n_transient: int = 1000, n_measure: int = 3000
    ) -> float:
        """Measure oscillation period from x1 upward zero crossings.

        Args:
            n_transient: Steps to discard for transient decay.
            n_measure: Steps to use for period measurement.

        Returns:
            Mean period in time units, or inf if no oscillation detected.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Detect upward zero crossings of x1
        crossings: list[float] = []
        prev_x1 = self._state[0]
        for _ in range(n_measure):
            self.step()
            x1 = self._state[0]
            if prev_x1 < 0 and x1 >= 0:
                # Linear interpolation for crossing time
                t_cross = (
                    (self._step_count - 1) * dt
                    + dt * (-prev_x1) / (x1 - prev_x1)
                )
                crossings.append(t_cross)
            prev_x1 = x1

        if len(crossings) < 2:
            return float("inf")

        periods = np.diff(crossings)
        return float(np.mean(periods))

    def measure_amplitude(
        self, n_transient: int = 1000, n_measure: int = 3000
    ) -> float:
        """Measure steady-state amplitude of x1 after transient.

        Args:
            n_transient: Steps to discard for transient decay.
            n_measure: Steps to measure over.

        Returns:
            Peak amplitude (max |x1|) during measurement window.
        """
        # Skip transient
        for _ in range(n_transient):
            self.step()

        x1_max = 0.0
        for _ in range(n_measure):
            self.step()
            x1_max = max(x1_max, abs(self._state[0]))

        return float(x1_max)

    def measure_amplitudes_all(
        self, n_transient: int = 1000, n_measure: int = 3000
    ) -> tuple[float, float, float]:
        """Measure steady-state amplitudes of all three stages.

        Args:
            n_transient: Steps to discard for transient decay.
            n_measure: Steps to measure over.

        Returns:
            Tuple (amp1, amp2, amp3) of peak amplitudes.
        """
        for _ in range(n_transient):
            self.step()

        x_max = np.zeros(3)
        for _ in range(n_measure):
            self.step()
            for j in range(3):
                x_max[j] = max(x_max[j], abs(self._state[j]))

        return float(x_max[0]), float(x_max[1]), float(x_max[2])

    def compute_phase_shifts(
        self, n_transient: int = 2000, n_measure: int = 5000
    ) -> tuple[float, float]:
        """Compute phase shifts between consecutive stages.

        Measures the time delay between x1 and x2 zero crossings, and
        between x2 and x3, as a fraction of the period.

        Args:
            n_transient: Steps to discard for transient.
            n_measure: Steps for measurement.

        Returns:
            Tuple (shift_12, shift_23) in degrees. For ideal 3-phase
            oscillation, both should be close to 120 degrees.
        """
        if not self.is_oscillatory:
            return 0.0, 0.0

        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Record upward zero crossings for x1 and x2
        crossings_x1: list[float] = []
        crossings_x2: list[float] = []
        crossings_x3: list[float] = []
        prev = self._state.copy()

        for _ in range(n_measure):
            self.step()
            curr = self._state

            for idx, clist in enumerate(
                [crossings_x1, crossings_x2, crossings_x3]
            ):
                if prev[idx] < 0 and curr[idx] >= 0:
                    t_cross = (
                        (self._step_count - 1) * dt
                        + dt * (-prev[idx]) / (curr[idx] - prev[idx])
                    )
                    clist.append(t_cross)
            prev = curr.copy()

        if len(crossings_x1) < 2:
            return 0.0, 0.0

        period = float(np.mean(np.diff(crossings_x1)))
        if period <= 0:
            return 0.0, 0.0

        # Find first x2 crossing after first x1 crossing
        shift_12 = 0.0
        if crossings_x2:
            for t2 in crossings_x2:
                if t2 > crossings_x1[0]:
                    shift_12 = (t2 - crossings_x1[0]) / period * 360.0
                    break

        shift_23 = 0.0
        if crossings_x2 and crossings_x3:
            for t3 in crossings_x3:
                if t3 > crossings_x2[0]:
                    shift_23 = (t3 - crossings_x2[0]) / period * 360.0
                    break

        return float(shift_12), float(shift_23)

    def gain_sweep(
        self,
        gain_range: tuple[float, float] = (0.5, 10.0),
        n_gains: int = 30,
        n_steps: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep gain parameter and record amplitude and period.

        Args:
            gain_range: (min_gain, max_gain) range to sweep.
            n_gains: Number of gain values.
            n_steps: Total integration steps per gain value.

        Returns:
            Dict with 'gain', 'amplitude', and 'period' arrays.
        """
        gain_values = np.linspace(gain_range[0], gain_range[1], n_gains)
        amplitudes = np.empty(n_gains)
        periods = np.empty(n_gains)

        n_transient = n_steps // 2
        n_measure = n_steps - n_transient

        for i, g_val in enumerate(gain_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "gain": float(g_val),
                    "x1_0": 0.01,
                    "x2_0": 0.0,
                    "x3_0": 0.0,
                },
            )
            sim = RingOscillatorSimulation(config)
            sim.reset()

            amp = sim.measure_amplitude(
                n_transient=n_transient, n_measure=n_measure
            )
            amplitudes[i] = amp

            # Fresh sim for period measurement
            sim2 = RingOscillatorSimulation(config)
            sim2.reset()
            period = sim2.compute_period(
                n_transient=n_transient, n_measure=n_measure
            )
            periods[i] = period

        return {
            "gain": gain_values,
            "amplitude": amplitudes,
            "period": periods,
        }
