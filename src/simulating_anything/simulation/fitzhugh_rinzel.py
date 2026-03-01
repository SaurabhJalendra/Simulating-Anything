"""FitzHugh-Rinzel neuron model simulation.

The FitzHugh-Rinzel model extends FitzHugh-Nagumo with a slow variable y
that produces bursting dynamics -- clusters of rapid spikes separated by
quiescent periods:

    dv/dt = v - v^3/3 - w + y + I_ext   (fast membrane potential)
    dw/dt = delta * (a + v - b*w)        (recovery, fast timescale)
    dy/dt = mu * (c - v - d*y)           (slow modulation, mu << delta)

Three timescales: v (fast spikes), w (spike recovery), y (burst modulation).
For mu=0 the model reduces to standard FitzHugh-Nagumo.

Target rediscoveries:
- Bursting: clusters of spikes separated by quiescent periods
- Burst statistics: duration, interburst interval, spikes per burst
- ODE recovery via SINDy
- mu sweep: burst frequency as a function of ultraslow timescale
- FHN limit when mu=0
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FitzHughRinzelSimulation(SimulationEnvironment):
    """FitzHugh-Rinzel: 3-variable extension of FHN with bursting dynamics.

    State vector: [v, w, y] where
        v = membrane potential (fast variable)
        w = recovery variable (fast timescale, but slower than v)
        y = slow modulation variable (ultraslow, controls bursting)

    Parameters:
        a: recovery shift (default 0.7)
        b: recovery coupling (default 0.8)
        c: slow variable equilibrium (default -0.775)
        d: slow variable self-coupling (default 1.0)
        delta: fast/slow timescale ratio for w (default 0.08)
        mu: ultraslow timescale ratio for y (default 0.0001)
        I_ext: external current (default 0.0)
        v_0: initial membrane potential
        w_0: initial recovery
        y_0: initial slow variable
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.7)
        self.b_param = p.get("b", 0.8)
        self.c = p.get("c", -0.775)
        self.d = p.get("d", 1.0)
        self.delta = p.get("delta", 0.08)
        self.mu = p.get("mu", 0.0001)
        self.I_ext = p.get("I_ext", 0.0)
        self.v_0 = p.get("v_0", -1.0)
        self.w_0 = p.get("w_0", -0.5)
        self.y_0 = p.get("y_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize membrane potential, recovery, and slow variable."""
        self._state = np.array(
            [self.v_0, self.w_0, self.y_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [v, w, y]."""
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
        """Compute derivatives of [v, w, y].

        dv/dt = v - v^3/3 - w + y + I_ext
        dw/dt = delta * (a + v - b*w)
        dy/dt = mu * (c - v - d*y)
        """
        v, w, y = state
        dv = v - v**3 / 3.0 - w + y + self.I_ext
        dw = self.delta * (self.a + v - self.b_param * w)
        dy = self.mu * (self.c - v - self.d * y)
        return np.array([dv, dw, dy])

    def detect_spikes(
        self, v_trace: np.ndarray, threshold: float = 0.0
    ) -> np.ndarray:
        """Find spike indices (upward threshold crossings in v).

        Args:
            v_trace: 1D array of membrane potential values.
            threshold: Voltage threshold for spike detection.

        Returns:
            Array of indices where spikes (upward crossings) occur.
        """
        crossings = []
        for i in range(1, len(v_trace)):
            if v_trace[i - 1] < threshold and v_trace[i] >= threshold:
                crossings.append(i)
        return np.array(crossings, dtype=int)

    def count_bursts(
        self, n_steps: int = 100000, transient: int = 20000
    ) -> int:
        """Count bursting episodes in a trajectory.

        A burst is a group of spikes separated from other groups by a
        quiescent gap longer than min_gap_steps.

        Args:
            n_steps: Number of steps to simulate after transient.
            transient: Steps to discard as transient.

        Returns:
            Number of detected bursts.
        """
        self.reset()
        for _ in range(transient):
            self.step()

        v_trace = np.zeros(n_steps)
        for i in range(n_steps):
            self.step()
            v_trace[i] = self._state[0]

        bursts = self._group_spikes_into_bursts(v_trace)
        return len(bursts)

    def measure_burst_statistics(
        self,
        n_steps: int = 200000,
        transient: int = 50000,
        min_gap_steps: int = 2000,
    ) -> dict[str, float]:
        """Measure burst duration, interburst interval, spikes per burst.

        Args:
            n_steps: Steps to simulate after transient.
            transient: Steps to discard.
            min_gap_steps: Minimum gap between bursts (in steps).

        Returns:
            Dict with burst_duration, interburst_interval,
            spikes_per_burst, n_bursts.
        """
        self.reset()
        for _ in range(transient):
            self.step()

        v_trace = np.zeros(n_steps)
        for i in range(n_steps):
            self.step()
            v_trace[i] = self._state[0]

        bursts = self._group_spikes_into_bursts(
            v_trace, min_gap_steps=min_gap_steps
        )

        if len(bursts) < 2:
            return {
                "burst_duration": 0.0,
                "interburst_interval": 0.0,
                "spikes_per_burst": float(len(bursts[0])) if bursts else 0.0,
                "n_bursts": len(bursts),
            }

        dt = self.config.dt

        # Burst duration: time from first spike to last spike in each burst
        durations = []
        for burst in bursts:
            if len(burst) > 1:
                durations.append((burst[-1] - burst[0]) * dt)
            else:
                durations.append(0.0)

        # Interburst interval: time between last spike of one burst
        # and first spike of the next
        intervals = []
        for i in range(len(bursts) - 1):
            gap = (bursts[i + 1][0] - bursts[i][-1]) * dt
            intervals.append(gap)

        spikes_per_burst = [len(b) for b in bursts]

        return {
            "burst_duration": float(np.mean(durations)),
            "interburst_interval": float(np.mean(intervals)),
            "spikes_per_burst": float(np.mean(spikes_per_burst)),
            "n_bursts": len(bursts),
        }

    def slow_variable_dynamics(
        self, n_steps: int = 100000, transient: int = 10000
    ) -> dict[str, np.ndarray]:
        """Return y(t) trajectory showing slow drift during bursting.

        Args:
            n_steps: Steps to simulate after transient.
            transient: Steps to discard.

        Returns:
            Dict with time, v, w, y arrays.
        """
        self.reset()
        for _ in range(transient):
            self.step()

        dt = self.config.dt
        v_arr = np.zeros(n_steps)
        w_arr = np.zeros(n_steps)
        y_arr = np.zeros(n_steps)

        for i in range(n_steps):
            self.step()
            v_arr[i] = self._state[0]
            w_arr[i] = self._state[1]
            y_arr[i] = self._state[2]

        return {
            "time": np.arange(n_steps) * dt,
            "v": v_arr,
            "w": w_arr,
            "y": y_arr,
        }

    def mu_sweep(
        self,
        mu_values: np.ndarray,
        n_steps: int = 200000,
        transient: int = 50000,
    ) -> dict[str, list]:
        """Sweep ultraslow timescale and measure burst frequency.

        Args:
            mu_values: Array of mu values to sweep.
            n_steps: Steps per simulation.
            transient: Transient steps to discard.

        Returns:
            Dict with mu, n_bursts, burst_frequency lists.
        """
        original_mu = self.mu
        mus = []
        n_bursts_list = []
        burst_freqs = []

        dt = self.config.dt
        total_time = n_steps * dt

        for mu_val in mu_values:
            self.mu = float(mu_val)
            n_bursts = self.count_bursts(
                n_steps=n_steps, transient=transient
            )
            freq = n_bursts / total_time if total_time > 0 else 0.0

            mus.append(float(mu_val))
            n_bursts_list.append(n_bursts)
            burst_freqs.append(freq)

        self.mu = original_mu

        return {
            "mu": mus,
            "n_bursts": n_bursts_list,
            "burst_frequency": burst_freqs,
        }

    def current_sweep(
        self,
        I_ext_values: np.ndarray,
        n_steps: int = 100000,
        transient: int = 20000,
    ) -> dict[str, list]:
        """Sweep external current and measure burst count and spike count.

        Args:
            I_ext_values: Array of I_ext values to sweep.
            n_steps: Steps per simulation.
            transient: Transient steps to discard.

        Returns:
            Dict with I_ext, n_bursts, n_spikes lists.
        """
        original_I = self.I_ext
        i_list = []
        n_bursts_list = []
        n_spikes_list = []

        for I_val in I_ext_values:
            self.I_ext = float(I_val)
            self.reset()
            for _ in range(transient):
                self.step()

            v_trace = np.zeros(n_steps)
            for i in range(n_steps):
                self.step()
                v_trace[i] = self._state[0]

            spikes = self.detect_spikes(v_trace, threshold=0.0)
            bursts = self._group_spikes_into_bursts(v_trace)

            i_list.append(float(I_val))
            n_bursts_list.append(len(bursts))
            n_spikes_list.append(len(spikes))

        self.I_ext = original_I

        return {
            "I_ext": i_list,
            "n_bursts": n_bursts_list,
            "n_spikes": n_spikes_list,
        }

    def _group_spikes_into_bursts(
        self,
        v_trace: np.ndarray,
        threshold: float = 0.0,
        min_gap_steps: int = 2000,
    ) -> list[list[int]]:
        """Group detected spikes into bursts based on inter-spike gaps.

        Spikes separated by fewer than min_gap_steps belong to the same
        burst. Gaps larger than min_gap_steps separate distinct bursts.

        Args:
            v_trace: 1D array of membrane potential values.
            threshold: Voltage threshold for spike detection.
            min_gap_steps: Minimum gap (in steps) between bursts.

        Returns:
            List of bursts, each burst a list of spike indices.
        """
        spike_indices = self.detect_spikes(v_trace, threshold)
        if len(spike_indices) == 0:
            return []

        bursts: list[list[int]] = [[int(spike_indices[0])]]
        for i in range(1, len(spike_indices)):
            gap = spike_indices[i] - spike_indices[i - 1]
            if gap > min_gap_steps:
                bursts.append([int(spike_indices[i])])
            else:
                bursts[-1].append(int(spike_indices[i]))

        return bursts
