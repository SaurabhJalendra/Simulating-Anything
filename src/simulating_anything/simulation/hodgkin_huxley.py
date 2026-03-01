"""Hodgkin-Huxley neuron model simulation.

The HH model describes action potential generation in neurons using 4 coupled ODEs
for membrane voltage V and 3 gating variables (n, m, h):

    C_m * dV/dt = I_ext - g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L)
    dn/dt = alpha_n(V)*(1-n) - beta_n(V)*n
    dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m
    dh/dt = alpha_h(V)*(1-h) - beta_h(V)*h

Target rediscoveries:
- Action potential shape (spike ~100mV amplitude, ~1ms duration)
- Threshold behavior: subthreshold vs suprathreshold stimulation
- Refractory period (absolute and relative)
- f-I curve (firing frequency vs injected current)
- Gating variable dynamics during spike
- Ionic current decomposition (Na+, K+, leak)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


def _safe_alpha_n(V: float | np.ndarray) -> float | np.ndarray:
    """Compute alpha_n with safe handling of singularity at V=-55.

    alpha_n(V) = 0.01*(V+55) / (1 - exp(-(V+55)/10))
    At V=-55, L'Hopital gives alpha_n = 0.1.
    """
    x = V + 55.0
    # Use L'Hopital limit where |x| is very small
    if np.isscalar(V):
        if abs(x) < 1e-7:
            return 0.1
        return 0.01 * x / (1.0 - np.exp(-x / 10.0))
    result = np.where(
        np.abs(x) < 1e-7,
        0.1,
        0.01 * x / (1.0 - np.exp(-x / 10.0)),
    )
    return result


def _beta_n(V: float | np.ndarray) -> float | np.ndarray:
    """beta_n(V) = 0.125 * exp(-(V+65)/80)."""
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def _safe_alpha_m(V: float | np.ndarray) -> float | np.ndarray:
    """Compute alpha_m with safe handling of singularity at V=-40.

    alpha_m(V) = 0.1*(V+40) / (1 - exp(-(V+40)/10))
    At V=-40, L'Hopital gives alpha_m = 1.0.
    """
    x = V + 40.0
    if np.isscalar(V):
        if abs(x) < 1e-7:
            return 1.0
        return 0.1 * x / (1.0 - np.exp(-x / 10.0))
    result = np.where(
        np.abs(x) < 1e-7,
        1.0,
        0.1 * x / (1.0 - np.exp(-x / 10.0)),
    )
    return result


def _beta_m(V: float | np.ndarray) -> float | np.ndarray:
    """beta_m(V) = 4 * exp(-(V+65)/18)."""
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def _alpha_h(V: float | np.ndarray) -> float | np.ndarray:
    """alpha_h(V) = 0.07 * exp(-(V+65)/20)."""
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def _beta_h(V: float | np.ndarray) -> float | np.ndarray:
    """beta_h(V) = 1 / (1 + exp(-(V+35)/10))."""
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


class HodgkinHuxleySimulation(SimulationEnvironment):
    """Hodgkin-Huxley neuron model with 4 coupled ODEs.

    State vector: [V, n, m, h] where
        V = membrane potential (mV)
        n = K+ activation gating variable
        m = Na+ activation gating variable
        h = Na+ inactivation gating variable

    Parameters:
        g_Na: maximal Na+ conductance (mS/cm^2, default 120)
        g_K: maximal K+ conductance (mS/cm^2, default 36)
        g_L: leak conductance (mS/cm^2, default 0.3)
        E_Na: Na+ reversal potential (mV, default 50)
        E_K: K+ reversal potential (mV, default -77)
        E_L: leak reversal potential (mV, default -54.387)
        C_m: membrane capacitance (uF/cm^2, default 1)
        I_ext: external current (uA/cm^2, default 10)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.g_Na = p.get("g_Na", 120.0)
        self.g_K = p.get("g_K", 36.0)
        self.g_L = p.get("g_L", 0.3)
        self.E_Na = p.get("E_Na", 50.0)
        self.E_K = p.get("E_K", -77.0)
        self.E_L = p.get("E_L", -54.387)
        self.C_m = p.get("C_m", 1.0)
        self.I_ext = p.get("I_ext", 10.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize at resting potential with steady-state gating variables."""
        V_rest = -65.0
        n_inf, m_inf, h_inf = self.steady_state_gating(V_rest)
        self._state = np.array([V_rest, n_inf, m_inf, h_inf], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [V, n, m, h]."""
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
        # Clamp gating variables to [0, 1]
        self._state[1:] = np.clip(self._state[1:], 0.0, 1.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute derivatives of [V, n, m, h]."""
        V, n, m, h = y

        # Ionic currents
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K = self.g_K * n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # Membrane voltage
        dV = (self.I_ext - I_Na - I_K - I_L) / self.C_m

        # Gating variables
        dn = _safe_alpha_n(V) * (1.0 - n) - _beta_n(V) * n
        dm = _safe_alpha_m(V) * (1.0 - m) - _beta_m(V) * m
        dh = _alpha_h(V) * (1.0 - h) - _beta_h(V) * h

        return np.array([dV, dn, dm, dh])

    @staticmethod
    def steady_state_gating(V: float) -> tuple[float, float, float]:
        """Compute steady-state gating variables at a given voltage.

        Returns:
            (n_inf, m_inf, h_inf) where x_inf = alpha_x / (alpha_x + beta_x).
        """
        an = _safe_alpha_n(V)
        bn = _beta_n(V)
        am = _safe_alpha_m(V)
        bm = _beta_m(V)
        ah = _alpha_h(V)
        bh = _beta_h(V)

        n_inf = float(an / (an + bn))
        m_inf = float(am / (am + bm))
        h_inf = float(ah / (ah + bh))
        return n_inf, m_inf, h_inf

    def ionic_currents(self, state: np.ndarray | None = None) -> dict[str, float]:
        """Decompose the membrane current into ionic components.

        Returns:
            Dict with keys 'I_Na', 'I_K', 'I_L' (positive = outward).
        """
        if state is None:
            state = self._state
        V, n, m, h = state
        return {
            "I_Na": float(self.g_Na * m**3 * h * (V - self.E_Na)),
            "I_K": float(self.g_K * n**4 * (V - self.E_K)),
            "I_L": float(self.g_L * (V - self.E_L)),
        }

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

    def compute_fi_curve(
        self,
        I_values: np.ndarray,
        t_max: float = 500.0,
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

            # Discard initial transient (first 100 ms)
            transient_steps = int(100.0 / dt)
            V_steady = V_trace[transient_steps:]

            # Detect spikes
            spike_indices = self.detect_spikes(V_steady, threshold=0.0)
            if len(spike_indices) >= 2:
                # Mean inter-spike interval in ms
                isis = np.diff(spike_indices) * dt
                mean_isi = np.mean(isis)
                freq = 1000.0 / mean_isi  # Convert to Hz (spikes/s)
            else:
                freq = 0.0

            frequencies.append(freq)

        # Restore original I_ext
        self.I_ext = original_I

        return {
            "I": np.array(I_values),
            "frequency": np.array(frequencies),
        }
