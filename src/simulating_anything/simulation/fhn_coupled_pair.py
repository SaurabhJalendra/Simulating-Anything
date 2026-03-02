"""FitzHugh-Nagumo coupled pair simulation.

Two mutually coupled FHN neurons with diffusive voltage coupling:

    dv1/dt = v1 - v1^3/3 - w1 + I1 + g*(v2 - v1)
    dw1/dt = eps*(v1 + a - b*w1)
    dv2/dt = v2 - v2^3/3 - w2 + I2 + g*(v1 - v2)
    dw2/dt = eps*(v2 + a - b*w2)

The coupling is symmetric and diffusive in the voltage variable.
Depending on coupling strength g and initial conditions, the system exhibits:
- Independent oscillations at g=0
- In-phase synchronization (v1 ~ v2) at moderate-to-strong coupling
- Anti-phase synchronization (v1 ~ -v2) with repulsive coupling (g < 0)
- Frequency locking under heterogeneous input currents

Target rediscoveries:
- Synchronization transition as a function of coupling g
- Critical coupling g_c for sync onset
- Phase-locking under heterogeneous drive (I1 != I2)
- Sync error decay rate vs coupling strength
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FHNCoupledPairSimulation(SimulationEnvironment):
    """Two mutually coupled FitzHugh-Nagumo neurons.

    State vector: [v1, w1, v2, w2]

    Parameters:
        a: recovery parameter (default 0.7)
        b: recovery parameter (default 0.8)
        eps: timescale separation (default 0.08)
        I1: external current to neuron 1 (default 0.5)
        I2: external current to neuron 2 (default 0.5)
        g: coupling strength between neurons (default 0.1)
        v1_0: initial voltage of neuron 1 (default -1.0)
        w1_0: initial recovery of neuron 1 (default -0.5)
        v2_0: initial voltage of neuron 2 (default 0.5)
        w2_0: initial recovery of neuron 2 (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.7)
        self.b_param = p.get("b", 0.8)
        self.eps = p.get("eps", 0.08)
        self.I1 = p.get("I1", 0.5)
        self.I2 = p.get("I2", 0.5)
        self.g = p.get("g", 0.1)
        self.v1_0 = p.get("v1_0", -1.0)
        self.w1_0 = p.get("w1_0", -0.5)
        self.v2_0 = p.get("v2_0", 0.5)
        self.w2_0 = p.get("w2_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize both neurons with specified initial conditions."""
        self._state = np.array(
            [self.v1_0, self.w1_0, self.v2_0, self.w2_0],
            dtype=np.float64,
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [v1, w1, v2, w2]."""
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
        """Compute derivatives for the coupled FHN system.

        dv1/dt = v1 - v1^3/3 - w1 + I1 + g*(v2 - v1)
        dw1/dt = eps*(v1 + a - b*w1)
        dv2/dt = v2 - v2^3/3 - w2 + I2 + g*(v1 - v2)
        dw2/dt = eps*(v2 + a - b*w2)
        """
        v1, w1, v2, w2 = state

        dv1 = v1 - v1**3 / 3 - w1 + self.I1 + self.g * (v2 - v1)
        dw1 = self.eps * (v1 + self.a - self.b_param * w1)
        dv2 = v2 - v2**3 / 3 - w2 + self.I2 + self.g * (v1 - v2)
        dw2 = self.eps * (v2 + self.a - self.b_param * w2)

        return np.array([dv1, dw1, dv2, dw2])

    def compute_sync_error(self, state: np.ndarray | None = None) -> float:
        """Compute instantaneous synchronization error.

        Euclidean distance between the two neurons' (v, w) states.

        Args:
            state: State vector [v1, w1, v2, w2]. Uses current state if None.

        Returns:
            Euclidean distance between neuron states.
        """
        s = state if state is not None else self._state
        diff = np.array([s[0] - s[2], s[1] - s[3]])
        return float(np.linalg.norm(diff))

    def compute_phase_difference(self) -> float:
        """Compute approximate phase difference between the two neurons.

        Uses the arctangent of (w, v) for each neuron to extract an
        instantaneous phase angle, then returns the difference mod 2*pi.

        Returns:
            Phase difference in radians, in the range [0, 2*pi).
        """
        v1, w1, v2, w2 = self._state
        phi1 = np.arctan2(w1, v1)
        phi2 = np.arctan2(w2, v2)
        dphi = (phi1 - phi2) % (2 * np.pi)
        return float(dphi)

    def is_synchronized(self, threshold: float = 0.1) -> bool:
        """Check if the two neurons are approximately synchronized.

        Synchronized means |v1 - v2| < threshold, averaged over the
        current instantaneous state.

        Args:
            threshold: Maximum voltage difference to count as synchronized.

        Returns:
            True if |v1 - v2| < threshold.
        """
        return bool(abs(self._state[0] - self._state[2]) < threshold)

    def is_in_phase(self, threshold: float = 0.3) -> bool:
        """Check if neurons are approximately in phase.

        In-phase means phase difference is near 0 (or 2*pi).

        Args:
            threshold: Maximum phase difference (radians) to count as in-phase.

        Returns:
            True if neurons are approximately synchronized in phase.
        """
        dphi = self.compute_phase_difference()
        return bool(dphi < threshold or dphi > (2 * np.pi - threshold))

    def is_anti_phase(self, threshold: float = 0.3) -> bool:
        """Check if neurons are approximately in anti-phase.

        Anti-phase means phase difference is near pi.

        Args:
            threshold: Maximum deviation from pi to count as anti-phase.

        Returns:
            True if neurons are approximately in anti-phase.
        """
        dphi = self.compute_phase_difference()
        return bool(abs(dphi - np.pi) < threshold)

    def coupling_sweep(
        self,
        g_values: np.ndarray,
        n_steps: int = 10000,
        n_transient: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep coupling strength and measure steady-state sync error.

        For each g value, creates a fresh simulation from the same initial
        conditions, discards transient, and computes the time-averaged
        synchronization error.

        Args:
            g_values: Array of coupling strengths to sweep.
            n_steps: Measurement steps after transient.
            n_transient: Transient steps to discard.

        Returns:
            Dict with 'g', 'mean_error', and 'mean_phase_diff' arrays.
        """
        mean_errors = np.empty(len(g_values))
        mean_phases = np.empty(len(g_values))

        for i, g_val in enumerate(g_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "a": self.a,
                    "b": self.b_param,
                    "eps": self.eps,
                    "I1": self.I1,
                    "I2": self.I2,
                    "g": float(g_val),
                    "v1_0": self.v1_0,
                    "w1_0": self.w1_0,
                    "v2_0": self.v2_0,
                    "w2_0": self.w2_0,
                },
            )
            sim = FHNCoupledPairSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Measure average error and phase difference
            error_sum = 0.0
            phase_sum = 0.0
            for _ in range(n_steps):
                sim.step()
                error_sum += sim.compute_sync_error()
                phase_sum += sim.compute_phase_difference()

            mean_errors[i] = error_sum / n_steps
            mean_phases[i] = phase_sum / n_steps

        return {
            "g": g_values,
            "mean_error": mean_errors,
            "mean_phase_diff": mean_phases,
        }

    def sync_error_trajectory(
        self, n_steps: int = 10000,
    ) -> dict[str, np.ndarray]:
        """Record synchronization error and phase difference over time.

        Args:
            n_steps: Number of integration steps.

        Returns:
            Dict with 'time', 'error', and 'phase_diff' arrays.
        """
        dt = self.config.dt
        errors = np.empty(n_steps + 1)
        phases = np.empty(n_steps + 1)
        times = np.empty(n_steps + 1)

        errors[0] = self.compute_sync_error()
        phases[0] = self.compute_phase_difference()
        times[0] = 0.0

        for i in range(1, n_steps + 1):
            self.step()
            errors[i] = self.compute_sync_error()
            phases[i] = self.compute_phase_difference()
            times[i] = i * dt

        return {"time": times, "error": errors, "phase_diff": phases}

    def measure_firing_frequency(
        self, neuron: int = 1, n_spikes: int = 5,
    ) -> float:
        """Measure firing frequency of a neuron via upward threshold crossings.

        Args:
            neuron: Which neuron to measure (1 or 2).
            n_spikes: Number of spikes to wait for.

        Returns:
            Firing frequency (Hz), or 0.0 if no oscillation detected.
        """
        dt = self.config.dt
        threshold = 0.0
        v_idx = 0 if neuron == 1 else 2

        # Transient
        for _ in range(int(500 / dt)):
            self.step()

        crossings: list[float] = []
        prev_v = self._state[v_idx]
        max_steps = int(n_spikes * 500 / dt)
        for _ in range(max_steps):
            self.step()
            v = self._state[v_idx]
            if prev_v < threshold and v >= threshold:
                crossings.append(self._step_count * dt)
            prev_v = v

        if len(crossings) < 2:
            return 0.0

        periods = np.diff(crossings)
        return float(1.0 / np.mean(periods))

    def is_oscillatory(self, n_test_steps: int = 5000) -> bool:
        """Check if current parameters produce sustained oscillations.

        Tests neuron 1 for oscillatory behavior.

        Args:
            n_test_steps: Number of steps for transient and measurement.

        Returns:
            True if voltage range exceeds threshold for oscillation.
        """
        self.reset()
        # Skip transient
        for _ in range(n_test_steps):
            self.step()
        # Check for oscillation over next period
        v_values = []
        for _ in range(n_test_steps):
            self.step()
            v_values.append(self._state[0])
        v_range = max(v_values) - min(v_values)
        return v_range > 0.5
