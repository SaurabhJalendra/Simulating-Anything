"""Izhikevich neuron model simulation.

The Izhikevich model combines the biological plausibility of Hodgkin-Huxley
with the computational efficiency of integrate-and-fire via a 2D ODE with reset:

    dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
    if v >= 30 mV: v <- c, u <- u + d

Parameters (a, b, c, d) select from 20+ known firing patterns:
- Regular Spiking (RS):        a=0.02, b=0.2, c=-65, d=8
- Intrinsically Bursting (IB): a=0.02, b=0.2, c=-55, d=4
- Chattering (CH):             a=0.02, b=0.2, c=-50, d=2
- Fast Spiking (FS):           a=0.1,  b=0.2, c=-65, d=2
- Thalamo-Cortical (TC):       a=0.02, b=0.25, c=-65, d=0.05
- Resonator (RZ):              a=0.1,  b=0.26, c=-65, d=2
- Low-Threshold Spiking (LTS): a=0.02, b=0.25, c=-65, d=2

Target rediscoveries:
- f-I curve: firing frequency vs injected current
- Pattern classification across (a, b, c, d) parameter space
- Spike timing analysis and inter-spike interval statistics
- Reset mechanism verification (v->c, u->u+d at v>=30)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

# Named parameter presets for known firing patterns
PATTERN_PRESETS: dict[str, dict[str, float]] = {
    "RS": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
    "IB": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
    "CH": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
    "FS": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
    "TC": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 0.05},
    "RZ": {"a": 0.1, "b": 0.26, "c": -65.0, "d": 2.0},
    "LTS": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},
}


class IzhikevichSimulation(SimulationEnvironment):
    """Izhikevich neuron model with spike-reset dynamics.

    State vector: [v, u] where
        v = membrane potential (mV)
        u = recovery variable (dimensionless)

    The model uses Euler integration (standard for Izhikevich due to the
    discontinuous reset rule at v >= 30 mV).

    Parameters:
        a: recovery time constant (default 0.02)
        b: sensitivity of u to v (default 0.2)
        c: post-spike reset voltage (default -65.0)
        d: post-spike recovery increment (default 8.0)
        I: external current (default 10.0)
        v_0: initial membrane potential (default -65.0)
        u_0: initial recovery variable (default -14.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.02)
        self.b = p.get("b", 0.2)
        self.c = p.get("c", -65.0)
        self.d = p.get("d", 8.0)
        self.I = p.get("I", 10.0)
        self.v_0 = p.get("v_0", -65.0)
        self.u_0 = p.get("u_0", -14.0)
        self._spike_count = 0
        self._spike_times: list[float] = []

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize at resting state with v=v_0, u=u_0."""
        self._state = np.array([self.v_0, self.u_0], dtype=np.float64)
        self._step_count = 0
        self._spike_count = 0
        self._spike_times = []
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using Euler integration with spike reset.

        The Izhikevich model uses Euler (not RK4) because the reset rule
        introduces a discontinuity that higher-order integrators cannot handle
        cleanly. The original paper uses 1ms steps with Euler.
        """
        dt = self.config.dt
        v, u = self._state

        # Euler integration of continuous dynamics
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + self.I
        du = self.a * (self.b * v - u)
        v_new = v + dt * dv
        u_new = u + dt * du

        # Spike detection and reset
        if v_new >= 30.0:
            v_new = self.c
            u_new = u_new + self.d
            self._spike_count += 1
            self._spike_times.append((self._step_count + 1) * dt)

        self._state = np.array([v_new, u_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [v, u]."""
        return self._state

    def detect_spikes(
        self, V_trace: np.ndarray, threshold: float = 0.0
    ) -> np.ndarray:
        """Find spike times (indices of upward threshold crossings).

        Args:
            V_trace: 1D array of membrane voltages.
            threshold: Voltage threshold for spike detection.

        Returns:
            Array of indices where spikes occur.
        """
        crossings = []
        for i in range(1, len(V_trace)):
            if V_trace[i - 1] < threshold and V_trace[i] >= threshold:
                crossings.append(i)
        return np.array(crossings, dtype=int)

    def compute_firing_rate(
        self, t_max: float = 1000.0, transient: float = 200.0
    ) -> float:
        """Compute firing rate (spikes/ms) for the current parameters.

        Args:
            t_max: Total simulation time (ms).
            transient: Initial transient to discard (ms).

        Returns:
            Firing rate in spikes per ms.
        """
        dt = self.config.dt
        n_steps = int(t_max / dt)
        transient_steps = int(transient / dt)

        self.reset()

        # Collect voltage trace
        V_trace = np.zeros(n_steps + 1)
        V_trace[0] = self._state[0]
        for i in range(1, n_steps + 1):
            self.step()
            V_trace[i] = self._state[0]

        # Count spikes after transient
        V_steady = V_trace[transient_steps:]
        spike_indices = self.detect_spikes(V_steady, threshold=0.0)
        n_spikes = len(spike_indices)
        duration_ms = (n_steps - transient_steps) * dt

        if duration_ms <= 0:
            return 0.0
        return n_spikes / duration_ms

    def compute_fi_curve(
        self,
        I_values: np.ndarray,
        t_max: float = 1000.0,
        transient: float = 200.0,
    ) -> dict[str, np.ndarray]:
        """Compute firing frequency vs injected current (f-I curve).

        Args:
            I_values: Array of external current values to sweep.
            t_max: Total simulation time (ms) for each current.
            transient: Time to discard as transient (ms).

        Returns:
            Dict with 'I' and 'frequency' arrays (frequency in Hz = spikes/s).
        """
        dt = self.config.dt
        n_steps = int(t_max / dt)
        transient_steps = int(transient / dt)
        frequencies = []
        original_I = self.I

        for i_val in I_values:
            self.I = float(i_val)
            self.reset()

            # Collect voltage trace
            V_trace = np.zeros(n_steps + 1)
            V_trace[0] = self._state[0]
            for i in range(1, n_steps + 1):
                self.step()
                V_trace[i] = self._state[0]

            # Count spikes after transient
            V_steady = V_trace[transient_steps:]
            spike_indices = self.detect_spikes(V_steady, threshold=0.0)

            if len(spike_indices) >= 2:
                isis = np.diff(spike_indices) * dt
                mean_isi = np.mean(isis)
                freq = 1000.0 / mean_isi  # Convert to Hz (spikes/s)
            else:
                freq = 0.0

            frequencies.append(freq)

        self.I = original_I

        return {
            "I": np.array(I_values),
            "frequency": np.array(frequencies),
        }

    def classify_pattern(
        self, t_max: float = 1000.0, transient: float = 200.0
    ) -> str:
        """Classify the firing pattern of the current parameter set.

        Runs the simulation and analyzes spike timing to determine the
        pattern type: regular_spiking, fast_spiking, bursting, chattering,
        or quiescent.

        Args:
            t_max: Total simulation time (ms).
            transient: Transient to discard (ms).

        Returns:
            Pattern label string.
        """
        dt = self.config.dt
        n_steps = int(t_max / dt)
        transient_steps = int(transient / dt)

        self.reset()

        V_trace = np.zeros(n_steps + 1)
        V_trace[0] = self._state[0]
        for i in range(1, n_steps + 1):
            self.step()
            V_trace[i] = self._state[0]

        V_steady = V_trace[transient_steps:]
        spike_indices = self.detect_spikes(V_steady, threshold=0.0)

        if len(spike_indices) < 2:
            return "quiescent"

        isis = np.diff(spike_indices) * dt

        if len(isis) < 2:
            return "regular_spiking"

        cv = float(np.std(isis) / np.mean(isis)) if np.mean(isis) > 0 else 0.0
        mean_isi = float(np.mean(isis))

        # Bursting: high coefficient of variation from alternating short/long ISIs
        if cv > 0.5:
            # Distinguish bursting from chattering by inter-burst interval
            # Chattering has shorter inter-burst gaps than intrinsic bursting
            long_isis = isis[isis > 1.5 * np.median(isis)]
            if len(long_isis) > 0 and np.mean(long_isis) < 50.0 * dt:
                return "chattering"
            return "bursting"

        # Fast spiking: short, regular ISIs (high frequency)
        if mean_isi < 10.0:
            return "fast_spiking"

        return "regular_spiking"

    def interspike_interval(
        self,
        V_trace: np.ndarray,
        threshold: float = 0.0,
    ) -> dict[str, float]:
        """Compute inter-spike interval statistics.

        Args:
            V_trace: 1D array of membrane voltages.
            threshold: Voltage threshold for spike detection.

        Returns:
            Dict with 'mean_isi', 'std_isi', 'cv_isi', 'n_spikes'.
        """
        spike_indices = self.detect_spikes(V_trace, threshold)
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

    @staticmethod
    def get_preset(name: str) -> dict[str, float]:
        """Get parameter preset by name.

        Args:
            name: One of RS, IB, CH, FS, TC, RZ, LTS.

        Returns:
            Dict with a, b, c, d parameters.

        Raises:
            ValueError: If preset name is not recognized.
        """
        name_upper = name.upper()
        if name_upper not in PATTERN_PRESETS:
            valid = ", ".join(sorted(PATTERN_PRESETS.keys()))
            raise ValueError(
                f"Unknown preset '{name}'. Valid presets: {valid}"
            )
        return dict(PATTERN_PRESETS[name_upper])
