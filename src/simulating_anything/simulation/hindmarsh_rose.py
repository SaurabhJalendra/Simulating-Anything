"""Hindmarsh-Rose neuron model simulation.

The Hindmarsh-Rose model is a 3D system that exhibits bursting behavior:
    dx/dt = y - a*x^3 + b*x^2 - z + I_ext
    dy/dt = c - d*x^2 - y
    dz/dt = r * (s*(x - x_rest) - z)

where x = membrane potential, y = fast ion current, z = slow adaptation current.
The slow variable z modulates the fast (x, y) subsystem, producing complex
patterns of spiking and bursting.

Target rediscoveries:
- Bursting dynamics: bursts of spikes separated by quiescent periods
- Behavior transitions: quiescent -> spiking -> bursting -> continuous spiking
- Slow-fast timescale separation via r << 1
- Spike count per burst as a function of parameters
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class HindmarshRoseSimulation(SimulationEnvironment):
    """Hindmarsh-Rose neuron model with slow-fast bursting dynamics.

    State vector: [x, y, z] where
        x = membrane potential (fast variable)
        y = fast ion current (fast variable)
        z = slow adaptation current (slow variable)

    Parameters:
        a: cubic coefficient for x (default 1.0)
        b: quadratic coefficient for x (default 3.0)
        c: y equation constant (default 1.0)
        d: y equation quadratic coefficient (default 5.0)
        r: slow timescale parameter (default 0.001)
        s: coupling strength for slow variable (default 4.0)
        x_rest: resting potential for slow variable (default -1.6)
        I_ext: external current (default 3.25)
        x_0: initial membrane potential
        y_0: initial fast current
        z_0: initial adaptation current
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.0)
        self.b = p.get("b", 3.0)
        self.c = p.get("c", 1.0)
        self.d = p.get("d", 5.0)
        self.r = p.get("r", 0.001)
        self.s = p.get("s", 4.0)
        self.x_rest = p.get("x_rest", -1.6)
        self.I_ext = p.get("I_ext", 3.25)
        self.x_0 = p.get("x_0", -1.5)
        self.y_0 = p.get("y_0", -10.0)
        self.z_0 = p.get("z_0", 2.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize at resting state."""
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
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives of [x, y, z].

        dx/dt = y - a*x^3 + b*x^2 - z + I_ext
        dy/dt = c - d*x^2 - y
        dz/dt = r * (s*(x - x_rest) - z)
        """
        x, y, z = state
        dx = y - self.a * x**3 + self.b * x**2 - z + self.I_ext
        dy = self.c - self.d * x**2 - y
        dz = self.r * (self.s * (x - self.x_rest) - z)
        return np.array([dx, dy, dz])

    def detect_spikes(
        self, x_trace: np.ndarray, threshold: float = 0.0
    ) -> np.ndarray:
        """Find spike indices (upward threshold crossings in x).

        Args:
            x_trace: 1D array of membrane potential values.
            threshold: Voltage threshold for spike detection.

        Returns:
            Array of indices where spikes (upward crossings) occur.
        """
        crossings = []
        for i in range(1, len(x_trace)):
            if x_trace[i - 1] < threshold and x_trace[i] >= threshold:
                crossings.append(i)
        return np.array(crossings, dtype=int)

    def detect_bursts(
        self,
        x_trace: np.ndarray,
        threshold: float = 0.0,
        min_gap_steps: int = 500,
    ) -> list[list[int]]:
        """Detect burst events as groups of spikes separated by quiescent gaps.

        A burst is a sequence of spikes where consecutive spikes are separated
        by fewer than min_gap_steps. Bursts are separated by gaps larger than
        min_gap_steps.

        Args:
            x_trace: 1D array of membrane potential values.
            threshold: Voltage threshold for spike detection.
            min_gap_steps: Minimum gap (in steps) between bursts.

        Returns:
            List of bursts, where each burst is a list of spike indices.
        """
        spike_indices = self.detect_spikes(x_trace, threshold)
        if len(spike_indices) == 0:
            return []

        bursts: list[list[int]] = [[int(spike_indices[0])]]
        for i in range(1, len(spike_indices)):
            gap = spike_indices[i] - spike_indices[i - 1]
            if gap > min_gap_steps:
                # Start a new burst
                bursts.append([int(spike_indices[i])])
            else:
                # Continue current burst
                bursts[-1].append(int(spike_indices[i]))

        return bursts

    def classify_behavior(
        self, t_max: int = 5000, transient: int = 2000
    ) -> str:
        """Classify the current parameter regime as quiescent/spiking/bursting.

        Runs the simulation for t_max steps (after discarding transient steps)
        and classifies:
        - "quiescent": no spikes detected
        - "spiking": regular spiking (no burst structure)
        - "bursting": bursts of spikes separated by quiescent periods
        - "continuous_spiking": high-frequency continuous spiking

        Args:
            t_max: Number of steps to simulate for classification.
            transient: Steps to discard as transient.

        Returns:
            Behavior label string.
        """
        self.reset()
        # Skip transient
        for _ in range(transient):
            self.step()

        x_trace = np.zeros(t_max)
        for i in range(t_max):
            self.step()
            x_trace[i] = self._state[0]

        spike_indices = self.detect_spikes(x_trace, threshold=0.0)
        if len(spike_indices) < 2:
            return "quiescent"

        # Compute inter-spike intervals
        isis = np.diff(spike_indices)

        # Check for burst structure: high variance in ISIs indicates bursting
        if len(isis) < 2:
            return "spiking"

        cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
        mean_isi = np.mean(isis)

        # Bursting: high coefficient of variation (long gaps between bursts)
        if cv > 0.5:
            return "bursting"

        # Continuous spiking: very short regular ISIs
        dt = self.config.dt
        if mean_isi * dt < 5.0:
            return "continuous_spiking"

        return "spiking"

    def compute_burst_profile(
        self,
        I_values: np.ndarray,
        t_max: int = 10000,
        transient: int = 3000,
    ) -> dict[str, list]:
        """Sweep I_ext and measure behavior type and spikes per burst.

        Args:
            I_values: Array of external current values to sweep.
            t_max: Steps to simulate for each I_ext.
            transient: Transient steps to discard.

        Returns:
            Dict with 'I_ext', 'behavior', 'spikes_per_burst', 'n_bursts'.
        """
        original_I = self.I_ext
        behaviors = []
        spikes_per_burst = []
        n_bursts_list = []

        for i_val in I_values:
            self.I_ext = float(i_val)
            self.reset()

            # Skip transient
            for _ in range(transient):
                self.step()

            x_trace = np.zeros(t_max)
            for i in range(t_max):
                self.step()
                x_trace[i] = self._state[0]

            behavior = self.classify_behavior(t_max=t_max, transient=transient)
            behaviors.append(behavior)

            bursts = self.detect_bursts(x_trace, threshold=0.0)
            n_bursts_list.append(len(bursts))

            if len(bursts) > 0:
                mean_spikes = np.mean([len(b) for b in bursts])
                spikes_per_burst.append(float(mean_spikes))
            else:
                spikes_per_burst.append(0.0)

        self.I_ext = original_I

        return {
            "I_ext": list(I_values),
            "behavior": behaviors,
            "spikes_per_burst": spikes_per_burst,
            "n_bursts": n_bursts_list,
        }

    def interspike_interval(
        self,
        x_trace: np.ndarray,
        threshold: float = 0.0,
    ) -> dict[str, float]:
        """Compute inter-spike interval statistics.

        Args:
            x_trace: 1D array of membrane potential values.
            threshold: Voltage threshold for spike detection.

        Returns:
            Dict with 'mean_isi', 'std_isi', 'cv_isi', 'n_spikes'.
        """
        spike_indices = self.detect_spikes(x_trace, threshold)
        n_spikes = len(spike_indices)

        if n_spikes < 2:
            return {
                "mean_isi": 0.0,
                "std_isi": 0.0,
                "cv_isi": 0.0,
                "n_spikes": n_spikes,
            }

        isis = np.diff(spike_indices) * self.config.dt
        mean_isi = float(np.mean(isis))
        std_isi = float(np.std(isis))
        cv_isi = std_isi / mean_isi if mean_isi > 0 else 0.0

        return {
            "mean_isi": mean_isi,
            "std_isi": std_isi,
            "cv_isi": cv_isi,
            "n_spikes": n_spikes,
        }
