"""Theta neuron (Ermentrout-Kopell canonical model) simulation.

The theta neuron is the canonical model for Type I neural excitability,
defined on the unit circle:

    dtheta/dt = 1 - cos(theta) + (1 + cos(theta)) * I

where theta is the phase on [-pi, pi] and I is the external current.

Key properties:
- SNIC (saddle-node on invariant circle) bifurcation at I_c = 0
- I < 0: stable fixed point (resting state)
- I > 0: periodic spiking with frequency f = sqrt(I) / pi
- Spike = theta crosses pi (wraps to -pi)
- Mathematically equivalent to the quadratic integrate-and-fire neuron

Target rediscoveries:
- f-I curve: f = sqrt(I) / pi for I > 0
- SNIC bifurcation at I_c = 0
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


def _wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


class ThetaNeuronSimulation(SimulationEnvironment):
    """Theta neuron model: dtheta/dt = 1 - cos(theta) + (1 + cos(theta)) * I.

    State vector: [theta] (1D, phase on circle, range [-pi, pi])

    A spike occurs when theta crosses pi from below (wraps to -pi).
    The model exhibits a SNIC bifurcation at I = 0:
    - I < 0: stable fixed point at theta_s = -arccos((1+I)/(I-1))
    - I > 0: periodic spiking with period T = pi / sqrt(I)

    Parameters:
        I: external current (default -0.1, subthreshold)
        theta_0: initial phase (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.I = p.get("I", -0.1)
        self.theta_0 = p.get("theta_0", 0.0)
        self._spike_count = 0
        self._spike_times: list[float] = []

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize theta at theta_0."""
        self._state = np.array([self.theta_0], dtype=np.float64)
        self._step_count = 0
        self._spike_count = 0
        self._spike_times = []
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with angle wrapping."""
        dt = self.config.dt
        theta_old = self._state[0]

        # RK4 integration
        k1 = self._derivative(self._state)
        k2 = self._derivative(self._state + 0.5 * dt * k1)
        k3 = self._derivative(self._state + 0.5 * dt * k2)
        k4 = self._derivative(self._state + dt * k3)

        theta_new = theta_old + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])

        # Detect spike: theta crosses pi from below
        if theta_old < np.pi and theta_new >= np.pi:
            self._spike_count += 1
            self._spike_times.append((self._step_count + 1) * dt)

        # Wrap to [-pi, pi]
        theta_new = _wrap_angle(theta_new)

        self._state = np.array([theta_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [theta]."""
        return self._state

    def _derivative(self, state: np.ndarray) -> np.ndarray:
        """Compute dtheta/dt = 1 - cos(theta) + (1 + cos(theta)) * I."""
        theta = state[0]
        dtheta = 1.0 - np.cos(theta) + (1.0 + np.cos(theta)) * self.I
        return np.array([dtheta])

    @property
    def is_spiking(self) -> bool:
        """True if I > 0 (superthreshold regime)."""
        return self.I > 0.0

    @staticmethod
    def analytical_frequency(I: float) -> float:
        """Analytical firing frequency for I > 0: f = sqrt(I) / pi.

        Args:
            I: external current.

        Returns:
            Firing frequency (0 if I <= 0).
        """
        if I <= 0:
            return 0.0
        return np.sqrt(I) / np.pi

    @staticmethod
    def analytical_period(I: float) -> float:
        """Analytical period for I > 0: T = pi / sqrt(I).

        Args:
            I: external current.

        Returns:
            Period (inf if I <= 0).
        """
        if I <= 0:
            return float("inf")
        return np.pi / np.sqrt(I)

    @staticmethod
    def stable_fixed_point(I: float) -> float | None:
        """Stable fixed point theta_s for I < 0.

        Setting dtheta/dt = 0:
            (1+I) + (I-1)*cos(theta) = 0
            cos(theta) = (1+I) / (1-I)

        The stable fixed point is at theta_s = -arccos((1+I)/(1-I)).

        Args:
            I: external current.

        Returns:
            Stable fixed point angle, or None if I >= 0.
        """
        if I >= 0:
            return None
        arg = (1.0 + I) / (1.0 - I)
        if abs(arg) > 1.0:
            return None
        return -np.arccos(arg)

    def count_spikes(self, trajectory: np.ndarray) -> int:
        """Count spikes (pi-crossings) in a theta trajectory.

        A spike is detected when theta wraps from near pi to near -pi,
        i.e., theta[i] > pi/2 and theta[i+1] < -pi/2.

        Args:
            trajectory: 1D array of theta values in [-pi, pi].

        Returns:
            Number of spikes detected.
        """
        spikes = 0
        for i in range(1, len(trajectory)):
            # Detect wrap-around from near +pi to near -pi
            if trajectory[i - 1] > np.pi / 2 and trajectory[i] < -np.pi / 2:
                spikes += 1
        return spikes

    def measure_frequency(
        self,
        t_max: float = 50.0,
        transient: float = 10.0,
    ) -> float:
        """Measure firing frequency from simulation.

        Args:
            t_max: total simulation time.
            transient: initial transient to discard.

        Returns:
            Measured firing frequency (Hz), 0 if no spikes.
        """
        dt = self.config.dt
        n_steps = int(t_max / dt)
        transient_steps = int(transient / dt)

        self.reset()

        # Collect theta trace
        theta_trace = np.zeros(n_steps + 1)
        theta_trace[0] = self._state[0]
        for i in range(1, n_steps + 1):
            self.step()
            theta_trace[i] = self._state[0]

        # Count spikes after transient
        steady = theta_trace[transient_steps:]
        n_spikes = self.count_spikes(steady)
        duration = (n_steps - transient_steps) * dt

        if duration <= 0 or n_spikes < 1:
            return 0.0
        return n_spikes / duration

    def compute_fi_curve(
        self,
        I_values: np.ndarray,
        t_max: float = 50.0,
        transient: float = 10.0,
    ) -> dict[str, np.ndarray]:
        """Compute f-I curve: firing frequency vs external current.

        Args:
            I_values: array of current values to sweep.
            t_max: simulation time per current value.
            transient: transient to discard.

        Returns:
            Dict with 'I', 'frequency', 'frequency_theory' arrays.
        """
        frequencies = []
        original_I = self.I

        for I_val in I_values:
            self.I = float(I_val)
            freq = self.measure_frequency(t_max=t_max, transient=transient)
            frequencies.append(freq)

        self.I = original_I

        theory = np.array([self.analytical_frequency(I) for I in I_values])

        return {
            "I": np.array(I_values),
            "frequency": np.array(frequencies),
            "frequency_theory": theory,
        }
