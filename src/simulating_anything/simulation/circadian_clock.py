"""Circadian clock simulation (Gonze-Goodwin model).

The Gonze simplified circadian oscillator models the core negative feedback loop
in the mammalian/Drosophila circadian clock with 3 variables:

    dM/dt  = v_s * K_I^n / (K_I^n + Fn^n) - v_m * M / (K_m + M)
    dFc/dt = k_s * M - v_d * Fc / (K_d + Fc) - k1 * Fc + k2 * Fn
    dFn/dt = k1 * Fc - k2 * Fn

where:
    M  = clock gene mRNA (e.g. per/tim)
    Fc = cytoplasmic clock protein (e.g. PER/TIM)
    Fn = nuclear clock protein (represses transcription)

Key features:
- Hill repression of transcription by nuclear protein (cooperativity n)
- Michaelis-Menten degradation of mRNA and cytoplasmic protein
- Nuclear-cytoplasmic shuttling with rates k1, k2
- Circadian period ~24h with default parameters

Target rediscoveries:
- Circadian period ~24h (tunable via parameters)
- Oscillation onset depends on Hill coefficient n (n >= 2 typically needed)
- Phase relationships: M leads Fc leads Fn
- ODE recovery via SINDy

Reference: Gonze et al., Biophys J 89:120-129 (2005)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CircadianClockSimulation(SimulationEnvironment):
    """Gonze-Goodwin circadian clock oscillator.

    State vector: [M, Fc, Fn] where:
        M  = clock gene mRNA concentration
        Fc = cytoplasmic protein concentration
        Fn = nuclear protein concentration

    The nuclear protein Fn represses its own mRNA transcription through a
    Hill function, creating the delayed negative feedback loop essential
    for circadian oscillation. Michaelis-Menten kinetics govern mRNA and
    protein degradation.

    Parameters:
        v_s: maximal transcription rate (default 1.6 nM/h)
        K_I: Hill function half-saturation for repression (default 1.0 nM)
        n: Hill coefficient for cooperativity (default 4)
        v_m: maximal mRNA degradation rate (default 0.505 nM/h)
        K_m: Michaelis constant for mRNA degradation (default 0.5 nM)
        k_s: translation rate (default 0.5 h^-1)
        v_d: maximal protein degradation rate (default 1.4 nM/h)
        K_d: Michaelis constant for protein degradation (default 0.13 nM)
        k1: nuclear import rate (default 0.33 h^-1)
        k2: nuclear export rate (default 0.43 h^-1)
        M_0: initial mRNA concentration (default 0.5)
        Fc_0: initial cytoplasmic protein concentration (default 0.5)
        Fn_0: initial nuclear protein concentration (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Transcription / Hill repression
        self.v_s = p.get("v_s", 1.6)
        self.K_I = p.get("K_I", 1.0)
        self.n = p.get("n", 4.0)

        # mRNA degradation (Michaelis-Menten)
        self.v_m = p.get("v_m", 0.505)
        self.K_m = p.get("K_m", 0.5)

        # Translation
        self.k_s = p.get("k_s", 0.5)

        # Protein degradation (Michaelis-Menten)
        self.v_d = p.get("v_d", 1.4)
        self.K_d = p.get("K_d", 0.13)

        # Nuclear transport
        self.k1 = p.get("k1", 0.33)
        self.k2 = p.get("k2", 0.43)

        # Initial conditions
        self.M_0 = p.get("M_0", 0.5)
        self.Fc_0 = p.get("Fc_0", 0.5)
        self.Fn_0 = p.get("Fn_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize concentrations [M, Fc, Fn]."""
        self._state = np.array([self.M_0, self.Fc_0, self.Fn_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        # Enforce non-negativity (concentrations cannot go below zero)
        self._state = np.maximum(self._state, 0.0)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [M, Fc, Fn]."""
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

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dM/dt, dFc/dt, dFn/dt for the Gonze circadian model."""
        M, Fc, Fn = y

        # Use max(0, x) for safety inside rate expressions
        M_pos = max(M, 0.0)
        Fc_pos = max(Fc, 0.0)
        Fn_pos = max(Fn, 0.0)

        # Hill repression: v_s * K_I^n / (K_I^n + Fn^n)
        KI_n = self.K_I ** self.n
        Fn_n = Fn_pos ** self.n
        hill = self.v_s * KI_n / (KI_n + Fn_n)

        # Michaelis-Menten mRNA degradation
        mm_mrna = self.v_m * M_pos / (self.K_m + M_pos)

        # dM/dt = transcription - degradation
        dM = hill - mm_mrna

        # Michaelis-Menten protein degradation
        mm_prot = self.v_d * Fc_pos / (self.K_d + Fc_pos)

        # dFc/dt = translation - degradation - nuclear import + nuclear export
        dFc = self.k_s * M_pos - mm_prot - self.k1 * Fc_pos + self.k2 * Fn_pos

        # dFn/dt = nuclear import - nuclear export
        dFn = self.k1 * Fc_pos - self.k2 * Fn_pos

        return np.array([dM, dFc, dFn])

    def is_oscillating(
        self,
        transient_steps: int = 5000,
        measure_steps: int = 5000,
        threshold: float = 0.01,
    ) -> bool:
        """Check whether the system exhibits sustained oscillations.

        Runs the simulation, discards transient, then checks if the
        peak-to-peak amplitude of M exceeds the threshold.

        Args:
            transient_steps: steps to skip for transient decay.
            measure_steps: steps to monitor for oscillation.
            threshold: minimum peak-to-peak amplitude to classify as oscillating.

        Returns:
            True if oscillations are detected.
        """
        # Skip transient
        for _ in range(transient_steps):
            self.step()

        # Collect M values
        m_vals: list[float] = []
        for _ in range(measure_steps):
            self.step()
            m_vals.append(self._state[0])

        amplitude = max(m_vals) - min(m_vals)
        return amplitude > threshold

    def measure_period(
        self,
        transient_steps: int = 5000,
        measure_steps: int = 10000,
        n_periods_min: int = 2,
    ) -> float:
        """Measure the oscillation period via upward zero crossings of M - M_mean.

        Uses the mean of M over the measurement window as the crossing level.

        Args:
            transient_steps: steps to skip for transient decay.
            measure_steps: steps to use for period measurement.
            n_periods_min: minimum number of complete periods required.

        Returns:
            Mean period in time units, or inf if oscillation not detected.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(transient_steps):
            self.step()

        # Collect M values
        m_vals: list[float] = []
        for _ in range(measure_steps):
            self.step()
            m_vals.append(self._state[0])

        m_arr = np.array(m_vals)
        m_mean = np.mean(m_arr)

        # Detect upward crossings of M = M_mean
        crossings: list[float] = []
        for i in range(1, len(m_arr)):
            if m_arr[i - 1] < m_mean and m_arr[i] >= m_mean:
                # Linear interpolation for fractional crossing
                if m_arr[i] != m_arr[i - 1]:
                    frac = (m_mean - m_arr[i - 1]) / (m_arr[i] - m_arr[i - 1])
                else:
                    frac = 0.5
                t_cross = (i - 1 + frac) * dt
                crossings.append(t_cross)

        if len(crossings) < n_periods_min + 1:
            return float("inf")

        periods = np.diff(crossings)
        return float(np.mean(periods))

    def measure_amplitude(
        self,
        transient_steps: int = 5000,
        measure_steps: int = 5000,
    ) -> tuple[float, float, float]:
        """Measure peak-to-peak amplitudes of all three variables.

        Args:
            transient_steps: steps to skip for transient decay.
            measure_steps: steps to use for amplitude measurement.

        Returns:
            Tuple of (M_amplitude, Fc_amplitude, Fn_amplitude).
        """
        # Skip transient
        for _ in range(transient_steps):
            self.step()

        # Collect all variable values
        m_vals: list[float] = []
        fc_vals: list[float] = []
        fn_vals: list[float] = []
        for _ in range(measure_steps):
            self.step()
            m_vals.append(self._state[0])
            fc_vals.append(self._state[1])
            fn_vals.append(self._state[2])

        amp_m = max(m_vals) - min(m_vals)
        amp_fc = max(fc_vals) - min(fc_vals)
        amp_fn = max(fn_vals) - min(fn_vals)
        return (amp_m, amp_fc, amp_fn)

    def measure_phase_lag(
        self,
        transient_steps: int = 5000,
        measure_steps: int = 10000,
    ) -> tuple[float, float]:
        """Measure phase lags between M, Fc, and Fn using cross-correlation.

        Args:
            transient_steps: steps to skip for transient decay.
            measure_steps: steps for measurement.

        Returns:
            Tuple of (lag_M_to_Fc, lag_Fc_to_Fn) in time units.
            Positive lag means the second variable lags the first.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(transient_steps):
            self.step()

        # Collect values
        m_vals: list[float] = []
        fc_vals: list[float] = []
        fn_vals: list[float] = []
        for _ in range(measure_steps):
            self.step()
            m_vals.append(self._state[0])
            fc_vals.append(self._state[1])
            fn_vals.append(self._state[2])

        m_arr = np.array(m_vals) - np.mean(m_vals)
        fc_arr = np.array(fc_vals) - np.mean(fc_vals)
        fn_arr = np.array(fn_vals) - np.mean(fn_vals)

        # Cross-correlation to find lag
        def _find_lag(sig1: np.ndarray, sig2: np.ndarray) -> float:
            """Find lag of sig2 relative to sig1 via cross-correlation."""
            corr = np.correlate(sig1, sig2, mode="full")
            # The center of the full cross-correlation is at index len(sig1) - 1
            center = len(sig1) - 1
            # Search within one expected period (~240 steps at dt=0.1, period~24)
            max_lag = min(len(sig1) // 2, int(30.0 / dt))
            search_range = corr[center:center + max_lag]
            if len(search_range) == 0:
                return 0.0
            # Find the first peak after zero lag (skip the center peak)
            # Use the argmax which gives the lag with highest correlation
            peak_idx = np.argmax(search_range)
            return float(peak_idx * dt)

        lag_m_fc = _find_lag(m_arr, fc_arr)
        lag_fc_fn = _find_lag(fc_arr, fn_arr)

        return (lag_m_fc, lag_fc_fn)
