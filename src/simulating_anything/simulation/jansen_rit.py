"""Jansen-Rit neural mass model simulation.

The Jansen-Rit model describes the dynamics of a cortical column with three
neural populations: pyramidal cells, excitatory interneurons, and inhibitory
interneurons. The model produces EEG-like output as the difference between
excitatory and inhibitory postsynaptic potentials (PSPs).

The system consists of 6 first-order ODEs (3 pairs of position + velocity):

    dy0/dt = y3
    dy3/dt = A*a*S(y1 - y2) - 2*a*y3 - a^2*y0
    dy1/dt = y4
    dy4/dt = A*a*(p + C2*S(C1*y0)) - 2*a*y4 - a^2*y1
    dy2/dt = y5
    dy5/dt = B*b*C4*S(C3*y0) - 2*b*y5 - b^2*y2

where S(v) = vmax / (1 + exp(r*(v0 - v))) is a sigmoid function.
Output (EEG-like) = y1 - y2 (excitatory minus inhibitory PSP).

Target rediscoveries:
- Alpha rhythm (~10 Hz) for intermediate external input p
- Bifurcation structure: Hopf bifurcation as p varies
- EEG output amplitude characterization
- Oscillation onset and offset vs p
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class JansenRitSimulation(SimulationEnvironment):
    """Jansen-Rit cortical column model producing EEG-like oscillations.

    State vector: [y0, y1, y2, y3, y4, y5] where
    - y0: output of pyramidal population (PSP -> firing rate)
    - y1: excitatory interneuron PSP (from excitatory feedback)
    - y2: inhibitory interneuron PSP (from inhibitory feedback)
    - y3: derivative of y0
    - y4: derivative of y1
    - y5: derivative of y2

    The EEG-like output is y1 - y2 (excitatory minus inhibitory PSP).

    Parameters:
        A: excitatory synaptic gain (mV), default 3.25
        B: inhibitory synaptic gain (mV), default 22.0
        a: inverse excitatory time constant (s^-1), default 100.0
        b: inverse inhibitory time constant (s^-1), default 50.0
        C: connectivity constant, default 135.0
            C1 = C, C2 = 0.8*C, C3 = 0.25*C, C4 = 0.25*C
        vmax: maximum firing rate (s^-1), default 5.0
        v0: sigmoid midpoint voltage (mV), default 6.0
        r: sigmoid steepness (mV^-1), default 0.56
        p: external input rate (Hz), default 120.0
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.A = p.get("A", 3.25)
        self.B = p.get("B", 22.0)
        self.a = p.get("a", 100.0)
        self.b = p.get("b", 50.0)
        self.C = p.get("C", 135.0)
        self.vmax = p.get("vmax", 5.0)
        self.v0 = p.get("v0", 6.0)
        self.r = p.get("r", 0.56)
        self.p_ext = p.get("p", 120.0)

    @property
    def C1(self) -> float:
        """Pyramidal to excitatory interneuron connectivity."""
        return self.C

    @property
    def C2(self) -> float:
        """Excitatory interneuron to pyramidal connectivity."""
        return 0.8 * self.C

    @property
    def C3(self) -> float:
        """Pyramidal to inhibitory interneuron connectivity."""
        return 0.25 * self.C

    @property
    def C4(self) -> float:
        """Inhibitory interneuron to pyramidal connectivity."""
        return 0.25 * self.C

    def sigmoid(self, v: float | np.ndarray) -> float | np.ndarray:
        """Sigmoid firing rate function.

        S(v) = vmax / (1 + exp(r * (v0 - v)))
             = vmax * 2 / (1 + exp(r * (v0 - v))) with vmax scaled

        Note: The standard Jansen-Rit sigmoid uses vmax/(1+exp(r*(v0-v))).
        At v = v0, S(v0) = vmax/2 (half-maximum firing rate).
        """
        return self.vmax / (1.0 + np.exp(self.r * (self.v0 - v)))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state with small random perturbation.

        State: [y0, y1, y2, y3, y4, y5]
        """
        rng = np.random.RandomState(seed if seed is not None else 42)
        # Small perturbation around zero
        self._state = rng.randn(6) * 0.1
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [y0, y1, y2, y3, y4, y5]."""
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
        """Compute derivatives of the 6D Jansen-Rit system.

        dy0/dt = y3
        dy3/dt = A*a*S(y1 - y2) - 2*a*y3 - a^2*y0
        dy1/dt = y4
        dy4/dt = A*a*(p + C2*S(C1*y0)) - 2*a*y4 - a^2*y1
        dy2/dt = y5
        dy5/dt = B*b*C4*S(C3*y0) - 2*b*y5 - b^2*y2
        """
        y0, y1, y2, y3, y4, y5 = y

        dy0 = y3
        dy3 = (
            self.A * self.a * self.sigmoid(y1 - y2)
            - 2.0 * self.a * y3
            - self.a**2 * y0
        )

        dy1 = y4
        dy4 = (
            self.A * self.a * (self.p_ext + self.C2 * self.sigmoid(self.C1 * y0))
            - 2.0 * self.a * y4
            - self.a**2 * y1
        )

        dy2 = y5
        dy5 = (
            self.B * self.b * self.C4 * self.sigmoid(self.C3 * y0)
            - 2.0 * self.b * y5
            - self.b**2 * y2
        )

        return np.array([dy0, dy1, dy2, dy3, dy4, dy5])

    def compute_eeg_output(self, states: np.ndarray) -> np.ndarray:
        """Compute EEG-like output from state trajectory.

        The EEG output is y1 - y2 (excitatory minus inhibitory PSP).

        Args:
            states: Array of shape (n_steps, 6) with columns [y0..y5].

        Returns:
            1D array of EEG output values.
        """
        return states[:, 1] - states[:, 2]

    def compute_frequency(
        self,
        n_transient: int = 1000,
        n_measure: int = 3000,
    ) -> float:
        """Compute dominant oscillation frequency via FFT of EEG output.

        Args:
            n_transient: Number of steps to skip as transient.
            n_measure: Number of steps to use for frequency measurement.

        Returns:
            Dominant frequency in Hz (0.0 if no clear oscillation).
        """
        self.reset()

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect EEG signal
        eeg = np.zeros(n_measure)
        for i in range(n_measure):
            self.step()
            eeg[i] = self._state[1] - self._state[2]

        # Detrend
        eeg_detrend = eeg - np.mean(eeg)

        # Check for sufficient amplitude
        if np.std(eeg_detrend) < 0.01:
            return 0.0

        dt = self.config.dt
        fft_vals = np.fft.rfft(eeg_detrend)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(eeg_detrend), d=dt)

        # Find peak frequency (skip DC component)
        if len(power) > 1:
            peak_idx = np.argmax(power[1:]) + 1
            peak_freq = freqs[peak_idx]
        else:
            peak_freq = 0.0

        return float(peak_freq)

    def input_sweep(
        self,
        p_range: tuple[float, float] = (50.0, 300.0),
        n_p: int = 30,
        n_steps: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep external input p and record oscillation amplitude and frequency.

        Args:
            p_range: (min_p, max_p) range of external input values.
            n_p: Number of p values to sweep.
            n_steps: Number of simulation steps per p value.

        Returns:
            Dict with 'p_values', 'amplitude', 'frequency', and 'mean_eeg'.
        """
        p_values = np.linspace(p_range[0], p_range[1], n_p)
        amplitudes = np.zeros(n_p)
        frequencies = np.zeros(n_p)
        mean_eeg = np.zeros(n_p)

        n_transient = n_steps // 3
        n_measure = n_steps - n_transient

        for i, p_val in enumerate(p_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "A": self.A,
                    "B": self.B,
                    "a": self.a,
                    "b": self.b,
                    "C": self.C,
                    "vmax": self.vmax,
                    "v0": self.v0,
                    "r": self.r,
                    "p": float(p_val),
                },
            )
            sim = JansenRitSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Collect EEG signal
            eeg = np.zeros(n_measure)
            for j in range(n_measure):
                sim.step()
                eeg[j] = sim._state[1] - sim._state[2]

            amplitudes[i] = np.max(eeg) - np.min(eeg)
            mean_eeg[i] = np.mean(eeg)

            # Compute frequency via FFT
            eeg_detrend = eeg - np.mean(eeg)
            if np.std(eeg_detrend) > 0.01:
                fft_vals = np.fft.rfft(eeg_detrend)
                power = np.abs(fft_vals) ** 2
                freqs = np.fft.rfftfreq(len(eeg_detrend), d=self.config.dt)
                if len(power) > 1:
                    peak_idx = np.argmax(power[1:]) + 1
                    frequencies[i] = freqs[peak_idx]

        return {
            "p_values": p_values,
            "amplitude": amplitudes,
            "frequency": frequencies,
            "mean_eeg": mean_eeg,
        }
