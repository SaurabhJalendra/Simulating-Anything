"""Repressilator synthetic genetic oscillatory network simulation.

The repressilator is a 3-gene synthetic regulatory circuit where each gene
represses the next in a cycle: Gene 1 -> Gene 2 -> Gene 3 -> Gene 1.

ODEs (dimensionless Elowitz & Leibler 2000):
    dm1/dt = -m1 + alpha/(1 + p3^n) + alpha0
    dm2/dt = -m2 + alpha/(1 + p1^n) + alpha0
    dm3/dt = -m3 + alpha/(1 + p2^n) + alpha0
    dp1/dt = beta*(m1 - p1)
    dp2/dt = beta*(m2 - p2)
    dp3/dt = beta*(m3 - p3)

Target rediscoveries:
- Oscillation onset: critical alpha depends on n and beta
- Z3 cyclic symmetry: 120-degree phase shift between genes
- Period and amplitude as functions of alpha, n, beta
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class RepressilatorSimulation(SimulationEnvironment):
    """Repressilator: 3-gene mutual repression oscillator.

    State vector: [m1, m2, m3, p1, p2, p3] where m_i are mRNA and p_i are
    protein concentrations.

    The system exhibits sustained oscillations when alpha is sufficiently large.
    Due to Z3 cyclic symmetry, all three genes oscillate with the same amplitude
    and period, phase-shifted by 120 degrees (T/3).

    Parameters:
        alpha: maximal transcription rate (default 216.0)
        alpha0: leakiness / basal transcription rate (default 0.001)
        n: Hill coefficient for repression (default 2.0)
        beta: protein/mRNA ratio timescale (default 5.0)
        m1_0, m2_0, m3_0: initial mRNA concentrations
        p1_0, p2_0, p3_0: initial protein concentrations
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 216.0)
        self.alpha0 = p.get("alpha0", 0.001)
        self.n = p.get("n", 2.0)
        self.beta = p.get("beta", 5.0)
        # Initial conditions -- small asymmetric perturbation breaks symmetry
        self.m1_0 = p.get("m1_0", 1.0)
        self.m2_0 = p.get("m2_0", 0.5)
        self.m3_0 = p.get("m3_0", 0.2)
        self.p1_0 = p.get("p1_0", 1.0)
        self.p2_0 = p.get("p2_0", 0.5)
        self.p3_0 = p.get("p3_0", 0.2)

    def _hill_repression(self, p: float) -> float:
        """Hill repression function: alpha / (1 + p^n) + alpha0."""
        return self.alpha / (1.0 + p ** self.n) + self.alpha0

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute time derivatives for the repressilator ODEs."""
        m1, m2, m3, p1, p2, p3 = y

        dm1 = -m1 + self._hill_repression(p3)
        dm2 = -m2 + self._hill_repression(p1)
        dm3 = -m3 + self._hill_repression(p2)
        dp1 = self.beta * (m1 - p1)
        dp2 = self.beta * (m2 - p2)
        dp3 = self.beta * (m3 - p3)

        return np.array([dm1, dm2, dm3, dp1, dp2, dp3])

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize mRNA and protein concentrations."""
        self._state = np.array(
            [self.m1_0, self.m2_0, self.m3_0, self.p1_0, self.p2_0, self.p3_0],
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
        """Return current state [m1, m2, m3, p1, p2, p3]."""
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

    @property
    def symmetric_fixed_point(self) -> np.ndarray:
        """Compute the symmetric fixed point where m1=m2=m3=m*, p1=p2=p3=p*.

        At fixed point: m* = alpha/(1+p*^n) + alpha0, and p* = m* (from dp/dt=0).
        So we solve: g(p) = alpha/(1+p^n) + alpha0 - p = 0.

        Uses bisection since the simple iteration can oscillate for large alpha.
        """
        def g(p: float) -> float:
            return self.alpha / (1.0 + p ** self.n) + self.alpha0 - p

        # Bisection: g(0+) > 0 (large positive), g(alpha+alpha0) < 0
        lo, hi = 1e-6, self.alpha + self.alpha0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-14:
                break

        p_star = 0.5 * (lo + hi)
        m_star = p_star  # At equilibrium, m* = p*
        return np.array([m_star, m_star, m_star, p_star, p_star, p_star])

    @property
    def is_oscillatory(self) -> bool:
        """Check if parameters are in the oscillatory regime.

        For n=2, approximate condition: alpha > (beta+1) * ((beta+1)/beta)^(1/n).
        This is a necessary condition from linearization around the symmetric
        fixed point. The actual threshold depends on all parameters.
        """
        ratio = (self.beta + 1.0) / self.beta
        threshold = (self.beta + 1.0) * ratio ** (1.0 / self.n)
        return self.alpha > threshold

    def compute_period(self, n_transient: int = 2000, n_measure: int = 3000) -> float:
        """Measure oscillation period from protein p1 peak-to-peak intervals.

        Args:
            n_transient: Steps to skip for transient decay.
            n_measure: Steps to collect for period measurement.

        Returns:
            Mean period in time units, or inf if no oscillation detected.
        """
        self.reset()
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect p1 values
        p1_values = []
        for _ in range(n_measure):
            self.step()
            p1_values.append(self._state[3])  # p1 index

        p1_arr = np.array(p1_values)

        # Find peaks: points higher than both neighbors
        peaks = []
        for i in range(1, len(p1_arr) - 1):
            if p1_arr[i] > p1_arr[i - 1] and p1_arr[i] > p1_arr[i + 1]:
                peaks.append(i)

        if len(peaks) < 2:
            return float("inf")

        # Compute peak-to-peak intervals
        peak_times = np.array(peaks) * dt
        intervals = np.diff(peak_times)

        # Filter out spurious short intervals (less than half the median)
        if len(intervals) > 2:
            median_interval = np.median(intervals)
            valid = intervals > 0.5 * median_interval
            if np.sum(valid) > 0:
                intervals = intervals[valid]

        return float(np.mean(intervals))

    def parameter_sweep(
        self,
        param_name: str,
        param_range: np.ndarray,
        n_steps: int = 10000,
    ) -> dict[str, np.ndarray]:
        """Sweep a parameter and measure oscillation amplitude and period.

        Args:
            param_name: Name of parameter to sweep (e.g., "alpha", "n", "beta").
            param_range: Array of parameter values to sweep.
            n_steps: Total integration steps per parameter value.

        Returns:
            Dict with 'param_values', 'amplitudes', and 'periods' arrays.
        """
        amplitudes = []
        periods = []

        n_transient = n_steps // 2
        n_measure = n_steps - n_transient

        for val in param_range:
            params = dict(self.config.parameters)
            params[param_name] = float(val)
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters=params,
            )
            sim = RepressilatorSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Measure oscillation
            p1_vals = []
            for _ in range(n_measure):
                sim.step()
                p1_vals.append(sim.observe()[3])  # p1 index

            p1_arr = np.array(p1_vals)
            amp = float(np.max(p1_arr) - np.min(p1_arr))
            amplitudes.append(amp)

            # Measure period from this sim instance
            period = sim.compute_period(n_transient=n_transient, n_measure=n_measure)
            periods.append(period)

        return {
            "param_values": param_range,
            "amplitudes": np.array(amplitudes),
            "periods": np.array(periods),
        }

    def verify_z3_symmetry(
        self, n_transient: int = 5000, n_measure: int = 5000
    ) -> dict[str, float]:
        """Verify Z3 cyclic symmetry: genes oscillate with same amplitude,
        phase-shifted by 120 degrees.

        Returns dict with amplitude ratios and phase differences.
        """
        self.reset()
        dt = self.config.dt

        for _ in range(n_transient):
            self.step()

        # Collect all protein time series
        p_series = [[], [], []]
        for _ in range(n_measure):
            self.step()
            for j in range(3):
                p_series[j].append(self._state[3 + j])  # p1, p2, p3

        p_arr = [np.array(s) for s in p_series]

        # Amplitude: max - min for each protein
        amps = [float(np.max(p) - np.min(p)) for p in p_arr]

        # Phase differences via cross-correlation
        phase_diffs = []
        for i in range(3):
            j = (i + 1) % 3
            # Cross-correlate to find lag
            p_i = p_arr[i] - np.mean(p_arr[i])
            p_j = p_arr[j] - np.mean(p_arr[j])
            corr = np.correlate(p_i, p_j, mode="full")
            lags = np.arange(-len(p_i) + 1, len(p_i))
            # Find first positive peak
            center = len(p_i) - 1
            # Search positive lags only
            pos_corr = corr[center:]
            pos_lags = lags[center:]
            if len(pos_corr) > 2:
                # Find first local maximum after lag=0
                for k in range(1, len(pos_corr) - 1):
                    if pos_corr[k] > pos_corr[k - 1] and pos_corr[k] > pos_corr[k + 1]:
                        phase_diffs.append(float(pos_lags[k] * dt))
                        break
                else:
                    phase_diffs.append(float("nan"))
            else:
                phase_diffs.append(float("nan"))

        return {
            "amplitudes": amps,
            "amplitude_ratio_12": amps[0] / amps[1] if amps[1] > 0 else float("nan"),
            "amplitude_ratio_23": amps[1] / amps[2] if amps[2] > 0 else float("nan"),
            "amplitude_ratio_31": amps[2] / amps[0] if amps[0] > 0 else float("nan"),
            "phase_diffs": phase_diffs,
        }
