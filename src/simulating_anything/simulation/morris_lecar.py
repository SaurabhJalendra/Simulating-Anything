"""Morris-Lecar neuron model simulation.

A 2D conductance-based model bridging Hodgkin-Huxley (biophysical) and
FitzHugh-Nagumo (reduced). Exhibits Type I vs Type II excitability depending
on parameters:

    C * dV/dt = I_ext - g_L*(V - V_L) - g_Ca*m_ss(V)*(V - V_Ca) - g_K*w*(V - V_K)
    dw/dt = phi * (w_ss(V) - w) / tau_w(V)

where:
    m_ss(V) = 0.5 * (1 + tanh((V - V1) / V2))
    w_ss(V) = 0.5 * (1 + tanh((V - V3) / V4))
    tau_w(V) = 1 / cosh((V - V3) / (2 * V4))

Target rediscoveries:
- f-I curve: firing frequency vs input current
- Excitability type classification (Type I vs Type II)
- Nullcline structure and equilibrium point
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class MorrisLecarSimulation(SimulationEnvironment):
    """Morris-Lecar 2D conductance-based neuron model.

    State vector: [V, w] where
        V = membrane potential (mV)
        w = fraction of open K+ channels (slow recovery)

    Parameters:
        C: membrane capacitance (uF/cm^2, default 20)
        g_L: leak conductance (mS/cm^2, default 2)
        g_Ca: Ca2+ conductance (mS/cm^2, default 4)
        g_K: K+ conductance (mS/cm^2, default 8)
        V_L: leak reversal potential (mV, default -60)
        V_Ca: Ca2+ reversal potential (mV, default 120)
        V_K: K+ reversal potential (mV, default -84)
        V1: half-activation voltage for m_ss (mV, default -1.2)
        V2: slope factor for m_ss (mV, default 18)
        V3: half-activation voltage for w_ss (mV, default 2)
        V4: slope factor for w_ss (mV, default 30)
        phi: timescale parameter for w (default 0.04)
        I_ext: external current (uA/cm^2, default 0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.C = p.get("C", 20.0)
        self.g_L = p.get("g_L", 2.0)
        self.g_Ca = p.get("g_Ca", 4.0)
        self.g_K = p.get("g_K", 8.0)
        self.V_L = p.get("V_L", -60.0)
        self.V_Ca = p.get("V_Ca", 120.0)
        self.V_K = p.get("V_K", -84.0)
        self.V1 = p.get("V1", -1.2)
        self.V2 = p.get("V2", 18.0)
        self.V3 = p.get("V3", 2.0)
        self.V4 = p.get("V4", 30.0)
        self.phi = p.get("phi", 0.04)
        self.I_ext = p.get("I_ext", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize near the resting potential with w at steady state."""
        V_rest = -60.0
        w_rest = self.w_ss(V_rest)
        self._state = np.array([V_rest, w_rest], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [V, w]."""
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
        # Clamp w to [0, 1] -- it is a gating variable
        self._state[1] = np.clip(self._state[1], 0.0, 1.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute derivatives of [V, w]."""
        V, w = y

        # Ionic currents
        I_L = self.g_L * (V - self.V_L)
        I_Ca = self.g_Ca * self.m_ss(V) * (V - self.V_Ca)
        I_K = self.g_K * w * (V - self.V_K)

        # Membrane voltage
        dV = (self.I_ext - I_L - I_Ca - I_K) / self.C

        # Recovery variable
        dw = self.phi * (self.w_ss(V) - w) / self.tau_w(V)

        return np.array([dV, dw])

    @staticmethod
    def m_ss(V: float | np.ndarray) -> float | np.ndarray:
        """Steady-state Ca2+ activation: m_ss(V) = 0.5*(1 + tanh((V-V1)/V2)).

        Uses default V1=-1.2, V2=18 for the static method.
        Instance method _m_ss uses instance parameters.
        """
        return 0.5 * (1.0 + np.tanh((V - (-1.2)) / 18.0))

    @staticmethod
    def w_ss(V: float | np.ndarray) -> float | np.ndarray:
        """Steady-state K+ activation: w_ss(V) = 0.5*(1 + tanh((V-V3)/V4)).

        Uses default V3=2, V4=30 for the static method.
        Instance method _w_ss uses instance parameters.
        """
        return 0.5 * (1.0 + np.tanh((V - 2.0) / 30.0))

    @staticmethod
    def tau_w(V: float | np.ndarray) -> float | np.ndarray:
        """Time constant for w: tau_w(V) = 1/cosh((V-V3)/(2*V4)).

        Uses default V3=2, V4=30.
        """
        return 1.0 / np.cosh((V - 2.0) / (2.0 * 30.0))

    def _m_ss_inst(self, V: float | np.ndarray) -> float | np.ndarray:
        """Instance version of m_ss using self.V1, self.V2."""
        return 0.5 * (1.0 + np.tanh((V - self.V1) / self.V2))

    def _w_ss_inst(self, V: float | np.ndarray) -> float | np.ndarray:
        """Instance version of w_ss using self.V3, self.V4."""
        return 0.5 * (1.0 + np.tanh((V - self.V3) / self.V4))

    def _tau_w_inst(self, V: float | np.ndarray) -> float | np.ndarray:
        """Instance version of tau_w using self.V3, self.V4."""
        return 1.0 / np.cosh((V - self.V3) / (2.0 * self.V4))

    def ionic_currents(self, state: np.ndarray | None = None) -> dict[str, float]:
        """Decompose the membrane current into ionic components.

        Returns:
            Dict with keys 'I_L', 'I_Ca', 'I_K' (positive = outward).
        """
        if state is None:
            state = self._state
        V, w = state
        return {
            "I_L": float(self.g_L * (V - self.V_L)),
            "I_Ca": float(self.g_Ca * self._m_ss_inst(V) * (V - self.V_Ca)),
            "I_K": float(self.g_K * w * (V - self.V_K)),
        }

    @property
    def nullcline_v(self):
        """V-nullcline: I_ext = g_L*(V-V_L) + g_Ca*m_ss(V)*(V-V_Ca) + g_K*w*(V-V_K).

        Solved for w: w = (I_ext - g_L*(V-V_L) - g_Ca*m_ss(V)*(V-V_Ca)) /
                          (g_K*(V-V_K))
        """
        def _nullcline(V: float | np.ndarray) -> float | np.ndarray:
            num = (self.I_ext - self.g_L * (V - self.V_L)
                   - self.g_Ca * self._m_ss_inst(V) * (V - self.V_Ca))
            den = self.g_K * (V - self.V_K)
            return num / den
        return _nullcline

    @property
    def nullcline_w(self):
        """W-nullcline: w = w_ss(V)."""
        return self._w_ss_inst

    def compute_nullclines(
        self,
        V_range: tuple[float, float] = (-80.0, 60.0),
        n_points: int = 500,
    ) -> dict[str, np.ndarray]:
        """Compute V- and w-nullclines over a voltage range.

        Returns:
            Dict with 'V', 'w_v_nullcline', 'w_w_nullcline' arrays.
        """
        V_arr = np.linspace(V_range[0], V_range[1], n_points)

        # V-nullcline: solve dV/dt=0 for w
        w_v = self.nullcline_v(V_arr)

        # w-nullcline: w = w_ss(V)
        w_w = self.nullcline_w(V_arr)

        return {"V": V_arr, "w_v_nullcline": w_v, "w_w_nullcline": w_w}

    def find_equilibrium(
        self,
        V_range: tuple[float, float] = (-80.0, 60.0),
        n_search: int = 1000,
    ) -> tuple[float, float]:
        """Find equilibrium by locating V- and w-nullcline intersection.

        Returns:
            (V_eq, w_eq) tuple at the equilibrium point.
        """
        V_arr = np.linspace(V_range[0], V_range[1], n_search)
        w_v = self.nullcline_v(V_arr)
        w_w = self.nullcline_w(V_arr)

        # Find zero crossing of (w_v - w_w)
        diff = w_v - w_w
        # Handle NaN/Inf from division by zero near V_K
        valid = np.isfinite(diff)

        best_idx = None
        best_val = np.inf
        for i in range(len(diff) - 1):
            if valid[i] and valid[i + 1] and diff[i] * diff[i + 1] <= 0:
                # Linear interpolation for sign change
                if abs(diff[i]) < best_val:
                    best_val = abs(diff[i])
                    best_idx = i

        if best_idx is not None:
            # Refine with linear interpolation
            V0, V1 = V_arr[best_idx], V_arr[best_idx + 1]
            d0, d1 = diff[best_idx], diff[best_idx + 1]
            if abs(d1 - d0) > 1e-14:
                V_eq = V0 - d0 * (V1 - V0) / (d1 - d0)
            else:
                V_eq = 0.5 * (V0 + V1)
            w_eq = float(self._w_ss_inst(V_eq))
            return float(V_eq), w_eq

        # Fallback: find minimum absolute difference
        valid_diff = np.abs(diff)
        valid_diff[~valid] = np.inf
        idx = int(np.argmin(valid_diff))
        return float(V_arr[idx]), float(w_w[idx])

    def is_oscillatory(self, n_test_steps: int = 5000) -> bool:
        """Check if current parameters produce sustained oscillations."""
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
        # Morris-Lecar spikes are ~50-100 mV in amplitude
        return v_range > 10.0

    def measure_firing_frequency(self, n_spikes: int = 5) -> float:
        """Measure firing frequency by counting upward threshold crossings.

        Uses a threshold of -20 mV (midway between rest and spike peak).
        """
        dt = self.config.dt
        threshold = -20.0

        # Transient: run for 500 ms
        transient_steps = int(500.0 / dt)
        for _ in range(transient_steps):
            self.step()

        crossings = []
        prev_v = self._state[0]
        max_steps = int(n_spikes * 500.0 / dt)
        for _ in range(max_steps):
            self.step()
            v = self._state[0]
            if prev_v < threshold and v >= threshold:
                crossings.append(self._step_count * dt)
            prev_v = v

        if len(crossings) < 2:
            return 0.0

        periods = np.diff(crossings)
        mean_period_ms = float(np.mean(periods))
        # Return frequency in Hz (1000 ms/s / period_ms)
        return 1000.0 / mean_period_ms

    def compute_fi_curve(
        self,
        I_values: np.ndarray,
        t_max: float = 2000.0,
    ) -> dict[str, np.ndarray]:
        """Compute firing frequency vs injected current (f-I curve).

        Args:
            I_values: Array of external current values to sweep.
            t_max: Total simulation time (ms) for each current.

        Returns:
            Dict with 'I' and 'frequency' arrays.
        """
        dt = self.config.dt
        n_steps = int(t_max / dt)
        threshold = -20.0
        frequencies = []
        original_I = self.I_ext

        for i_val in I_values:
            self.I_ext = float(i_val)
            self.reset()

            # Collect voltage trace
            V_trace = np.zeros(n_steps + 1)
            V_trace[0] = self._state[0]
            for i in range(1, n_steps + 1):
                self.step()
                V_trace[i] = self._state[0]

            # Discard transient (first 500 ms)
            transient_steps = int(500.0 / dt)
            V_steady = V_trace[transient_steps:]

            # Detect spikes via upward threshold crossings
            spike_indices = []
            for i in range(1, len(V_steady)):
                if V_steady[i - 1] < threshold and V_steady[i] >= threshold:
                    spike_indices.append(i)

            if len(spike_indices) >= 2:
                isis = np.diff(spike_indices) * dt  # ms
                mean_isi = np.mean(isis)
                freq = 1000.0 / mean_isi  # Hz
            else:
                freq = 0.0

            frequencies.append(freq)

        self.I_ext = original_I

        return {
            "I": np.array(I_values),
            "frequency": np.array(frequencies),
        }
